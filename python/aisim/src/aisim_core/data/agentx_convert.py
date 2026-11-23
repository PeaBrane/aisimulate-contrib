# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert AgentX benchmark bundles (official and self-collected) to ``aisim-e2e/v1`` configurations.

A bundle is the directory the AgentX normalizers write (``manifest.json``, ``configuration.json``,
``observations.json``, ``workload.json``, ``features.json``; schema ``aisimulate.agentx.benchmark.v1``).
One bundle is one observation; bundles with the same serving configuration (same identity) become one
configuration with one observation per bundle, so lanes 1, 4, 8, ... of a campaign sit side by side.

Where things land (see docs/ unified data, e2e schema):

* identity and serving facts (model, hardware, framework, parallelism, precision, effective runtime
  settings with ``observed`` authority) -> manifest body;
* lanes, corpus, boundary, warmup -> ``workload`` and the observation row;
* per-request rows (phase, times, token counts, cached tokens, lineage keys) -> the requests table;
* server-side series (running, waiting, KV use, ...) -> the series table;
* run-level ``ttft``/``tpot``/``e2el`` statistics and ``output_throughput`` over the successful profile
  requests, derived from the request rows (``source_name`` says so) -> the metrics table;
* the AgentX feature list and other AgentX-only facts -> ``ext["agentx/1"]``.

Names are canonical (:mod:`aisim_core.data.names`); the AgentX spellings stay in ``source_ids``. Offline
only; nothing is read from the network and raw prompts, streams and traces are never copied.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aisim_core.data.e2e import (
    E2eConfigurationData,
    E2eManifestBody,
    MetricRow,
    ObservationRow,
    RequestRow,
    SeriesRow,
    Setting,
    Workload,
    _canonical,
    make_metric,
    missing_required_metrics,
    producer_run_id,
)
from aisim_core.data.names import normalize

BUNDLE_SCHEMA = "aisimulate.agentx.benchmark.v1"
EXT_NAMESPACE = "agentx/1"
LATENCY_STATISTICS = ("mean", "median", "p90")
# Gaps that belong to one bundle's observation; they are kept per observation when bundles are merged.
_PER_OBSERVATION_GAPS = frozenset({"requests.lineage", "device_count"})
_SETTING_SECTIONS = ("cache_config", "scheduler", "compilation", "model", "parallelism")


class AgentxBundleError(ValueError):
    """A bundle is missing a file or declares an unsupported schema."""


def _load(bundle: Path, name: str, *, required: bool = True) -> Any:
    path = bundle / name
    if not path.is_file():
        if required:
            raise AgentxBundleError(f"{bundle}: missing {name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), ``q`` in 0..100."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _statistics(values: Iterable[float | None]) -> dict[str, float] | None:
    kept = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not kept:
        return None
    return {
        "mean": sum(kept) / len(kept),
        "median": _percentile(kept, 50),
        "p90": _percentile(kept, 90),
    }


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _feature_parameters(features: Mapping[str, Any], feature_id: str) -> dict[str, Any] | None:
    for feature in features.get("features", []):
        if feature.get("id") == feature_id:
            return {p["name"]: p["value"] for p in feature.get("parameters", [])} | {"_state": feature.get("state")}
    return None


def _observed_settings(effective: Sequence[Mapping[str, Any]]) -> dict[str, Setting]:
    """Rank-0 runtime settings, ``section.name`` keys, authority ``observed``."""
    if not effective:
        return {}
    rank0 = effective[0]
    settings = {}
    for section in _SETTING_SECTIONS:
        for name, value in (rank0.get(section) or {}).items():
            settings[f"{section}.{name}"] = Setting(value=value, authority="observed")
    for name in ("num_blocks",):
        if rank0.get(name) is not None:
            settings[name] = Setting(value=rank0[name], authority="observed")
    return settings


# vLLM parallelism sizes the bundle records per rank, keyed by the registry dimension they set. The first
# four multiply the GPU count; ``decode_context_parallel_size`` reuses TP ranks and only enters the identity.
_WORLD_DEGREES = (
    ("tensor_parallel_size", "tp"),
    ("pipeline_parallel_size", "pp"),
    ("data_parallel_size", "attention_dp"),
    ("prefill_context_parallel_size", "cp"),
)
_WORLD_DIMENSIONS = tuple(dimension for _, dimension in _WORLD_DEGREES)


def _observed_parallelism(effective: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Every valid rank-0 parallelism degree the bundle records, by registry dimension (possibly none).

    ``manifest["tensor_parallel"]`` is only the TP degree. With attention DP, PP or context parallelism the run
    uses more GPUs, and bundles that differ in a recorded degree are other serving configurations even when
    the GPU count itself cannot be derived.
    """
    parallel = (effective[0].get("parallelism") if effective else None) or {}
    dims: dict[str, int] = {}
    for source, dimension in _WORLD_DEGREES:
        value = parallel.get(source)
        if type(value) is int and value >= 1:
            dims[dimension] = value
    decode_cp = parallel.get("decode_context_parallel_size")
    if type(decode_cp) is int and decode_cp > 1:
        dims["dcp"] = decode_cp
    return dims


def _device_count(effective: Sequence[Mapping[str, Any]]) -> int | None:
    """GPU count the bundle records (the product of the observed GPU-multiplying degrees), else ``None``."""
    dims = _observed_parallelism(effective)
    if any(dimension not in dims for dimension in _WORLD_DIMENSIONS):
        return None
    count = 1
    for dimension in _WORLD_DIMENSIONS:
        count *= dims[dimension]
    return count


def _body(
    manifest: Mapping[str, Any], configuration: Mapping[str, Any], workload: Mapping[str, Any], features
) -> E2eManifestBody:
    effective = configuration.get("effective") or []
    observed = _observed_parallelism(effective)
    gaps: list[dict[str, Any]] = [{"field": "settings", "reason": g} for g in configuration.get("evidence_gaps", [])]
    precision_text = manifest.get("precision")
    precision = {"weights": normalize("precision", precision_text)} if precision_text else {}
    if not precision_text:
        gaps.append({"field": "precision", "reason": "the bundle does not record the weight precision"})
    speculation = None
    identity_settings: dict[str, Any] = {}
    spec = _feature_parameters(features, "speculation")
    if spec is not None:
        state = spec.pop("_state")
        if state == "enabled":
            speculation = {k: v for k, v in spec.items() if v is not None}
            identity_settings["spec_method"] = spec.get("method", "unknown")
            # Runs that differ in a reported speculation parameter (draft length, ...) are not one configuration.
            for name, value in sorted(speculation.items()):
                if name != "method":
                    identity_settings[f"spec_{name}"] = value
        elif state == "disabled":
            identity_settings["spec_method"] = "none"
    variant = workload.get("variant", {})
    weka = _feature_parameters(features, "agentx.weka") or {}
    # ``other_silicon_settings_and_observations`` carries the dataset facts when ``features`` omits them.
    for item in workload.get("other_silicon_settings_and_observations", []):
        if item.get("id") == "agentx.weka":
            weka = {p["name"]: p["value"] for p in item.get("parameters", [])}
    dataset = {"corpus_id": manifest.get("corpus_id")}
    if weka.get("context_limit") is not None:
        dataset["context_limit"] = weka["context_limit"]
    implementation = dict(configuration.get("implementation_identity") or {})
    if effective:
        implementation["ranks_observed"] = len(effective)
    return E2eManifestBody(
        origin=normalize("origin", manifest["category"]),
        model_id=normalize("model", manifest["model"]),
        deployment="aggregated",
        hardware=normalize("hardware", manifest["hardware"]),
        framework=normalize("framework", manifest["backend"]),
        framework_version=manifest["measured_backend_version"],
        # Complete only when every GPU-multiplying degree was observed: otherwise the others are unknown, not 1.
        parallelism={"aggregated": {"tp": manifest["tensor_parallel"], **observed}},
        parallelism_complete=all(dimension in observed for dimension in _WORLD_DIMENSIONS),
        precision=precision,
        speculation=speculation,
        identity_settings=identity_settings,
        settings=_observed_settings(effective),
        source_kind="agentx",
        workload=Workload(
            kind="agentic_trace",
            dataset=dataset,
            boundary=variant.get("boundary") or manifest.get("boundary"),
            seed=weka.get("seed"),
        ),
        implementation=implementation,
        source_ids={"agentx": {k: manifest[k] for k in ("model", "hardware", "backend") if k in manifest}},
        ext={
            EXT_NAMESPACE: {
                "schema": EXT_NAMESPACE,
                "features": [
                    {"id": f["id"], "state": f["state"], "parameters": f.get("parameters", [])}
                    for f in features.get("features", [])
                ],
            }
        },
        evidence_gaps=gaps,
    )


def _requests(observation_id: str, rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]):
    out = []
    for row in rows:
        ref = reference.get(row["key"], {})
        attrs = {k: ref[k] for k in ("session_id",) if ref.get(k) is not None}
        out.append(
            RequestRow(
                observation_id=observation_id,
                request_id=row["key"],
                status=normalize("request_status", row["status"]),
                phase=normalize("phase", row["phase"]),
                start_ms=row.get("start_ms"),
                end_ms=row.get("end_ms"),
                input_tokens=row.get("input_length"),
                output_tokens=row.get("output_length"),
                cached_tokens=row.get("cached_tokens"),
                ttft_ms=row.get("ttft_ms"),
                tpot_ms=row.get("tpot_ms"),
                e2el_ms=row.get("e2e_ms"),
                error=row.get("failure_reason"),
                identity_valid=row.get("identity_valid"),
                conversation_id=ref.get("conversation_id"),
                source_trace_id=ref.get("source_trace_id"),
                source_outer_idx=ref.get("source_outer_idx"),
                source_inner_idx=ref.get("source_inner_idx"),
                agent_depth=ref.get("agent_depth"),
                attrs=_canonical(attrs).decode() if attrs else None,
            )
        )
    return out


def _run_metrics(
    observation_id: str, profile: Sequence[Mapping[str, Any]], device_count: int | None
) -> list[MetricRow]:
    """``ttft``/``tpot``/``e2el`` statistics and total output throughput of the successful profile requests."""
    successful = [r for r in profile if r["status"] == "completed"]
    rows: list[MetricRow] = []
    source = {"ttft": "ttft_ms", "tpot": "tpot_ms", "e2el": "e2e_ms"}
    for metric, column in source.items():
        stats = _statistics(r.get(column) for r in successful)
        for statistic in LATENCY_STATISTICS:
            if stats is None:
                continue
            rows.append(
                make_metric(
                    observation_id,
                    metric,
                    statistic,
                    stats[statistic],
                    "ms",
                    source_name=f"derived:observations.{column}",
                    observer="client",
                )
            )
    if successful and all(_finite(r.get(k)) for r in successful for k in ("start_ms", "end_ms", "output_length")):
        span = max(r["end_ms"] for r in successful) - min(r["start_ms"] for r in successful)
        if span > 0:
            rows.append(
                make_metric(
                    observation_id,
                    "output_throughput",
                    "value",
                    sum(r["output_length"] for r in successful) * 1000 / span,
                    "tokens/s",
                    normalization="total",
                    device_count=device_count,
                    source_name="derived:observations.output_length/span",
                    observer="client",
                )
            )
    rows.extend(missing_required_metrics(observation_id, rows))
    return rows


def _series(observation_id: str, workload: Mapping[str, Any]) -> list[SeriesRow]:
    silicon = (workload.get("silicon_reference") or {}).get("silicon") or {}
    definitions = silicon.get("metric_definitions", {})
    out = []
    for phase, body in (silicon.get("phases") or {}).items():
        for source_name, intervals in (body.get("series") or {}).items():
            metric = normalize("metric", source_name)
            for item in intervals:
                out.append(
                    SeriesRow(
                        observation_id=observation_id,
                        metric=metric,
                        start_ms=item["start"] * 1000.0,
                        end_ms=item["end"] * 1000.0,
                        value=item.get("value"),
                        coverage=item.get("coverage"),
                        unit=(definitions.get(source_name) or {}).get("unit"),
                        phase=normalize("phase", phase),
                    )
                )
    return out


_BUNDLE_FILES = ("manifest.json", "configuration.json", "workload.json", "features.json", "observations.json")


def _bundle_digest(bundle: Path, observations_bytes: bytes) -> str:
    """Revision of everything the published rows are derived from, not only the request observations.

    A corrected lineage in ``workload.json`` changes the stored rows without touching ``observations.json``, so the
    digest covers every bundle file; an input snapshot then tells the two versions apart.
    """
    digest = hashlib.sha256()
    for name in _BUNDLE_FILES:
        data = (
            observations_bytes
            if name == "observations.json"
            else (bundle / name).read_bytes()
            if (bundle / name).is_file()
            else None
        )
        if data is not None:
            digest.update(f"{name}\0{len(data)}\0".encode())
            digest.update(data)
    return digest.hexdigest()


def agentx_bundle_to_configuration(bundle: Path | str) -> E2eConfigurationData:
    """One AgentX bundle directory to one configuration with one observation."""
    bundle = Path(bundle)
    manifest = _load(bundle, "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise AgentxBundleError(f"{bundle}: unsupported bundle schema {manifest.get('schema')!r}")
    configuration = _load(bundle, "configuration.json")
    workload = _load(bundle, "workload.json")
    features = _load(bundle, "features.json", required=False) or {"features": []}
    observations_bytes = (bundle / "observations.json").read_bytes()
    rows = json.loads(observations_bytes)

    body = _body(manifest, configuration, workload, features)
    observation_id = manifest["id"]
    reference_rows = (workload.get("silicon_reference") or {}).get("reference", [])
    reference = {r["key"]: r for r in reference_rows if "key" in r}
    if reference_rows and not reference:
        # Official bundles key their reference rows by request ID, not by the observation key, so the
        # lineage fields cannot be joined to the request rows without guessing.
        body.evidence_gaps.append(
            {
                "field": "requests.lineage",
                "observation_id": observation_id,
                "reason": "the bundle's reference rows carry no observation key; conversation, trace and agent "
                "depth are not joined to the request rows",
            }
        )
    requests = _requests(observation_id, rows, reference)
    # Profile selection, counts and metrics use the same canonical phase and status as the stored request rows.
    profile = [
        dict(row, status=request.status)
        for row, request in zip(rows, requests, strict=True)
        if request.phase == "profile"
    ]
    variant = workload.get("variant", {})
    silicon_provenance = ((workload.get("silicon_reference") or {}).get("silicon") or {}).get("provenance", {})
    attrs = {
        k: v
        for k, v in {
            "corpus_id": manifest.get("corpus_id"),
            "profile_seconds": variant.get("profile_seconds") or manifest.get("profile_seconds"),
            "repeat": variant.get("repeat"),
            "measured_requests": manifest.get("measured_requests"),
            "benchmark_url": manifest.get("benchmark_url"),
        }.items()
        if v is not None
    }
    device_count = _device_count(configuration.get("effective") or [])
    if device_count is None:
        body.evidence_gaps.append(
            {
                "field": "device_count",
                "observation_id": observation_id,
                "reason": "the bundle records no observed parallelism, so the GPU count is unknown; "
                "per-device throughput cannot be derived",
            }
        )
    completed = sum(r["status"] == "completed" for r in profile)
    observation = ObservationRow(
        observation_id=observation_id,
        configuration_id=body.configuration_id(),
        record_sha256=_bundle_digest(bundle, observations_bytes),
        source_kind="agentx",
        source_observation_id=manifest["id"],
        producer_kind="github-actions" if manifest.get("source_run_url") else None,
        producer_id=producer_run_id(manifest.get("source_run_url")),
        benchmark_tool="aiperf",
        benchmark_tool_version=silicon_provenance.get("aiperf_version"),
        benchmark_type="agentic_trace",
        concurrency=manifest.get("lanes"),
        cache_state=variant.get("cache_state"),
        outcome="complete",
        requests_total=len(profile),
        requests_completed=completed,
        requests_failed=len(profile) - completed,
        device_count=device_count,
        # Self-collected runs set aiperf's unsafe override, so they are never submittable.
        valid_for_submission=False if body.origin == "self-collected" else None,
        observer="client",
        time_origin="first_request",
        attrs=_canonical(attrs).decode() if attrs else None,
    )
    metrics = _run_metrics(observation_id, profile, device_count)
    return E2eConfigurationData(body, [observation], metrics, requests, [], _series(observation_id, workload))


def agentx_bundles_to_configurations(bundles: Iterable[Path | str]) -> list[E2eConfigurationData]:
    """Convert bundles and merge those that share a serving configuration into one configuration.

    Bundles whose manifest bodies differ only in per-point facts (lanes, run) share a ``configuration_id``;
    their observations, metrics, requests and series are concatenated in bundle order. Differing
    runtime ``settings`` between merged bundles are kept as a gap on the merged body, never silently picked.
    """
    merged: dict[str, E2eConfigurationData] = {}
    for bundle in bundles:
        data = agentx_bundle_to_configuration(bundle)
        key = data.body.configuration_id()
        first = merged.get(key)
        if first is None:
            merged[key] = data
            continue
        if first.body.settings != data.body.settings:
            differing = sorted(
                name
                for name in set(first.body.settings) | set(data.body.settings)
                if first.body.settings.get(name) != data.body.settings.get(name)
            )
            first.body.evidence_gaps.append(
                {
                    "field": "settings",
                    "reason": f"merged points differ in runtime settings {differing}; kept those of the first point",
                }
            )
        for gap in data.body.evidence_gaps:
            if gap.get("field") in _PER_OBSERVATION_GAPS:
                first.body.evidence_gaps.append(gap)  # already labeled with its observation
            elif gap not in first.body.evidence_gaps:
                first.body.evidence_gaps.append(gap)
        first.observations.extend(data.observations)
        first.metrics.extend(data.metrics)
        first.requests.extend(data.requests)
        first.series.extend(data.series)
    return list(merged.values())
