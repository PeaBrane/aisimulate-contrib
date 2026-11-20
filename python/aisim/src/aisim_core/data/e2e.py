# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim-e2e/v1`` schema family: one E2E layout for InferenceX, AgentX and self-collected data.

**Evolution contract.** The ``/v1`` in each schema ID is the major version; additive changes bump
:data:`SCHEMA_MINOR` and never break an older reader. Models accept unknown fields (``extra="allow"``) and
keep them, tables are matched by column name (a missing optional column reads as null, an unknown column is
carried through), and consumers read only known fields. Stored names are canonical
(:mod:`aisim_core.data.names`); a non-canonical name is reported by :func:`name_findings`, never translated
by a reader. Must-have fields are the few every source can fill; everything else is optional.

Source adapters normalize; evaluators never branch on source. The family
specializes the shared envelope (``aisim-artifact-manifest/v1``, category
``e2e``, purpose ``e2e-configuration``) instead of adding a second one:

* **Manifest body** (``aisim-e2e-manifest/v1``) lives in ``provenance["e2e"]`` of
  the envelope. A fixed identity core (model, deployment, hardware, framework,
  parallelism, precision) and the deployment facts that tell otherwise equal
  deployments apart (``dp_attention`` per role as a boolean, ``multinode``,
  ``offload_mode``; identity hash v3, see :func:`aisim_core.data.configuration_id`)
  is mirrored into the envelope ``Identity`` so ``configuration_id`` and the
  existing catalog checks keep working. Parallelism degrees a tool reports as
  metrics (``pp``, ``pcp_size`` -> ``cp``, ``dcp_size`` -> ``dcp``) are identity
  dimensions too. An open ``settings`` map carries every other effective setting
  with an authority tag and never changes identity; ``identity_settings`` is the
  versioned open part of the identity (speculation method, worker counts).
* **Observations** (``aisim-e2e-observations/v1``): one row per benchmark point
  per producer run. Source fields without a canonical home are typed columns, so a
  source-agnostic reader never needs the raw artifact: ``recipe_fingerprint``,
  ``source_config_json`` (the upstream configuration record) and
  ``unmapped_metrics_json`` (source metrics without a metric row), verbatim.
* **Metrics** (``aisim-e2e-metrics/v1``): one row per metric/statistic/scope in
  canonical units (ms, tokens/s) with explicit normalization and a
  ``valid|missing|invalid`` status. Missing is a null value, never zero.
* **Requests** (``aisim-e2e-requests/v1``): optional per-request rows.
* **Input snapshot** (``aisim-e2e-input-snapshot/v1``): frozen membership.
* **Provenance artifacts**: raw tool output and logs, gzip-compressed,
  hash-linked from the manifest, tier 3, size-capped, with a redaction flag.
  A configuration with many observations stores its raw records in one
  ``raw-output-bundle`` (one record per line); each observation keeps the byte
  range (``raw_offset``, ``raw_length``) and the SHA-256 (``record_sha256``) of
  its own record, so every row stays individually auditable.

All tables are Parquet+zstd written through :mod:`aisim_core.data.parquet`.
Offline only; no network access.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import types
import typing
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aisim_core.data import Artifact, Digest, Identity, Manifest, _file, _verify, configuration_id
from aisim_core.data.loaders import E2eLoader, E2eTables, LoaderError
from aisim_core.data.names import NameFinding, canonical_name, canonical_unit, name_findings, normalize, split_framework
from aisim_core.data.parquet import read_parquet, write_parquet

MANIFEST_BODY_SCHEMA = "aisim-e2e-manifest/v1"
OBSERVATIONS_SCHEMA = "aisim-e2e-observations/v1"
METRICS_SCHEMA = "aisim-e2e-metrics/v1"
REQUESTS_SCHEMA = "aisim-e2e-requests/v1"
SERIES_SCHEMA = "aisim-e2e-series/v1"
SNAPSHOT_SCHEMA = "aisim-e2e-input-snapshot/v1"
SNAPSHOT_MEMBERS_SCHEMA = "aisim-e2e-snapshot-members/v1"
SNAPSHOT_ACQUISITION_SCHEMA = "aisim-e2e-snapshot-acquisition/v1"
E2E_TABLE_SCHEMAS = (OBSERVATIONS_SCHEMA, METRICS_SCHEMA, REQUESTS_SCHEMA, SERIES_SCHEMA)
PURPOSE = "e2e-configuration"
# Minor version of the /v1 family. Bump when an optional field, column or registered name is added; a reader
# of the same major accepts any minor and warns when a file is newer than it knows.
SCHEMA_MINOR = 1

# Canonical units. Latency statistics are milliseconds, throughput tokens/s.
UNIT_MS = "ms"
UNIT_TOKENS_PER_S = "tokens/s"
UNIT_TOKENS_PER_S_PER_USER = "tokens/s/user"
UNIT_REQUESTS_PER_S = "requests/s"
# Statistics the tool must report to qualify; absent ones are stored as ``missing``.
REQUIRED_LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")
REQUIRED_STATISTICS = ("mean", "median", "p90")
CANONICAL_UNIT = {
    "ttft": UNIT_MS,
    "tpot": UNIT_MS,
    "itl": UNIT_MS,
    "e2el": UNIT_MS,
    "output_throughput": UNIT_TOKENS_PER_S,
    "input_throughput": UNIT_TOKENS_PER_S,
    "total_throughput": UNIT_TOKENS_PER_S,
    "interactivity": UNIT_TOKENS_PER_S_PER_USER,
    "request_throughput": UNIT_REQUESTS_PER_S,
}

# Provenance artifact limits (tier 3). Logs keep a head/tail excerpt past the cap;
# raw tool output is never truncated and is rejected instead.
LOG_CAP_BYTES = 4 * 1024 * 1024
LOG_EXCERPT_BYTES = 64 * 1024
RAW_CAP_BYTES = 16 * 1024 * 1024
# One bundle holds every raw record of a configuration (each record is still capped and never truncated).
BUNDLE_CAP_BYTES = 256 * 1024 * 1024
TRUNCATION_MARKER = b"\n...[truncated by aisim-e2e size cap; full SHA-256 in manifest]...\n"

Authority = Literal["observed", "producer_declared", "source_verified_default", "unknown"]
Origin = Literal["official", "self-collected"]
Status = Literal["valid", "missing", "invalid"]
Normalization = Literal["none", "per_device", "per_user", "total"]
Outcome = Literal["complete", "partial", "failed", "unknown"]


class _Record(BaseModel):
    """Core fields are strict; unknown fields are kept (``model_extra``) and written back unchanged."""

    model_config = ConfigDict(extra="allow", strict=True, populate_by_name=True)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- units


_UNIT_ALIASES = {
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "msec": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "µs": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
    "tokens/s": "tokens/s",
    "tokens/sec": "tokens/s",
    "tokens/second": "tokens/s",
    "tok/s": "tokens/s",
    "tokens/s/user": "tokens/s/user",
    "tokens/sec/user": "tokens/s/user",
    "tokens/second/user": "tokens/s/user",
    "tok/s/user": "tokens/s/user",
    "requests/s": "requests/s",
    "requests/sec": "requests/s",
    "requests/second": "requests/s",
    "req/s": "requests/s",
}
_TO_CANONICAL = {"s": ("ms", 1000.0), "ms": ("ms", 1.0), "us": ("ms", 1e-3), "ns": ("ms", 1e-6)}


def normalize_unit(unit: str) -> str:
    """Map a reported unit spelling to its short form; unknown units raise."""
    key = unit.strip().lower()
    if key not in _UNIT_ALIASES:
        raise ValueError(f"unknown unit: {unit!r}")
    return _UNIT_ALIASES[key]


def to_canonical(value: float, unit: str, canonical_unit: str) -> float:
    """Convert ``value`` in ``unit`` to ``canonical_unit`` (time units and exact throughput aliases)."""
    short = normalize_unit(unit)
    if short in _TO_CANONICAL:
        target, factor = _TO_CANONICAL[short]
        if target != canonical_unit:
            raise ValueError(f"cannot convert {unit!r} to {canonical_unit!r}")
        return value * factor
    if short != canonical_unit:
        raise ValueError(f"cannot convert {unit!r} to {canonical_unit!r}")
    return value


# --------------------------------------------------------------------------- manifest body


class Setting(_Record):
    """One effective setting with the authority of its value."""

    value: JsonValue
    authority: Authority


class WorkerGroup(_Record):
    """Workers of one role; ``None`` means unknown, never a default."""

    num_workers: int | None = Field(default=None, ge=0)
    gpus_per_worker: int | None = Field(default=None, ge=0)
    total_gpus: int | None = Field(default=None, ge=0)


class ProvenanceInfo(_Record):
    """Tier-3 evidence file linked from the manifest by artifact ID and hash."""

    artifact_id: str = Field(min_length=1)
    observation_id: str | None = None
    kind: Literal["raw-output", "raw-output-bundle", "log"]
    tier: Literal[3] = 3
    compression: Literal["gzip"] = "gzip"
    redacted: bool
    truncated: bool
    full_sha256: Digest  # SHA-256 of the original, uncompressed bytes
    full_bytes: int = Field(ge=0)
    stored_bytes: int = Field(ge=0)  # uncompressed bytes kept (excerpt size when truncated)
    cap_bytes: int = Field(ge=1)
    source_tool: str | None = None


class Workload(_Record):
    """What was run against the server. Only ``kind`` is required; everything else is optional."""

    kind: str = Field(min_length=1)  # canonical workload_kind: single_turn, agentic_trace, synthetic, trace_replay
    concurrency: int | None = Field(default=None, ge=0)  # client concurrency, or lanes for agentic traces
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    dataset: dict[str, JsonValue] | None = None  # {repo, revision, sha256, filter}
    duration_s: float | None = Field(default=None, ge=0)  # profile span
    grace_s: float | None = Field(default=None, ge=0)
    warmup: dict[str, JsonValue] | None = None
    boundary: str | None = None  # canonical workload_boundary
    cache_state: str | None = None
    arrival_policy: str | None = None
    seed: int | None = None


class E2eManifestBody(_Record):
    """E2E specialization stored in ``provenance["e2e"]`` of the shared envelope."""

    schema_version: Literal["aisim-e2e-manifest/v1"] = MANIFEST_BODY_SCHEMA
    schema_minor: int = Field(default=SCHEMA_MINOR, ge=0)
    origin: Origin  # official (InferenceX, AgentX official) or self-collected; the producing tool is in source_kind
    # Fixed identity core (mirrored into the envelope Identity).
    model_id: str = Field(min_length=1)
    model_revision: str | None = None
    deployment: Literal["aggregated", "disaggregated"]
    hardware: str = Field(min_length=1)  # canonical system name, e.g. b300_sxm
    framework: str = Field(min_length=1)
    framework_version: str = Field(min_length=1)
    parallelism: dict[str, dict[str, int]]
    parallelism_complete: bool
    precision: dict[str, str]
    # Deployment facts mirrored into the envelope Identity (identity hash v3). None means not reported.
    dp_attention: dict[str, bool] | None = None  # per worker role; a boolean, never a degree
    multinode: bool | None = None
    offload_mode: str | None = None
    orchestrator: str | None = None  # canonical orchestrator (dynamo, llm-d, none); hashed when set and not none
    speculation: dict[str, JsonValue] | None = None  # method, draft length, acceptance mode
    # Identity-relevant open settings (versioned identity rule); changes the config ID.
    identity_settings: dict[str, JsonValue] = Field(default_factory=dict)
    # Open effective settings; a new runtime option is a new key, never a schema change.
    settings: dict[str, Setting] = Field(default_factory=dict)
    requested_settings: dict[str, JsonValue] | None = None
    workers: dict[str, WorkerGroup] = Field(default_factory=dict)
    # Upstream grouping kept for parity with the source.
    source_kind: str = Field(min_length=1)
    source_config_id: str | None = None
    cohort_group_id: str | None = None
    # Source fields with no mapping yet, kept verbatim (e.g. dp_attention booleans).
    source_fields: dict[str, JsonValue] = Field(default_factory=dict)
    workload: Workload | None = None
    # Image digest, git SHAs, model and config revisions: what the numbers were measured on.
    implementation: dict[str, JsonValue] = Field(default_factory=dict)
    # Spellings the source used (InferenceX key, AgentX point ID, system name); aliases, never identity.
    source_ids: dict[str, JsonValue] = Field(default_factory=dict)
    # Namespaced extensions, ``{namespace: {schema, ...}}`` (agentx/1, inferencex/1). Core readers ignore them.
    ext: dict[str, JsonValue] = Field(default_factory=dict)
    provenance_artifacts: list[ProvenanceInfo] = Field(default_factory=list)
    evidence_gaps: list[dict[str, JsonValue]] = Field(default_factory=list)

    def identity(self) -> Identity:
        effective = dict(self.identity_settings)
        if self.orchestrator and self.orchestrator != "none":
            effective["orchestrator"] = self.orchestrator
        return Identity(
            hardware=self.hardware,
            framework=self.framework,
            framework_version=self.framework_version,
            operation_family=None,
            model=self.model_id,
            effective_settings=dict(sorted(effective.items())),
            unresolved_identity=None,
            parallelism=self.parallelism,
            parallelism_complete=self.parallelism_complete,
            precision=self.precision,
            deployment=self.deployment,
            dp_attention=self.dp_attention,
            multinode=self.multinode,
            offload_mode=self.offload_mode,
        )

    def configuration_id(self) -> str:
        return configuration_id("e2e", PURPOSE, self.identity())

    @model_validator(mode="after")
    def valid_parallelism(self):
        # Shared normalization (degree 1 dropped, keys sorted) so body and identity agree.
        self.parallelism = self.identity().parallelism
        return self


def public_release_problems(body: E2eManifestBody) -> list[str]:
    """Tier-3 artifacts must be redacted before any public use."""
    return [f"unredacted tier-3 artifact: {p.artifact_id}" for p in body.provenance_artifacts if not p.redacted]


# --------------------------------------------------------------------------- row models


_GITHUB_RUN = re.compile(
    r"^https://github\.com/(?P<repo>[\w.-]+/[\w.-]+)/actions/runs/(?P<run>\d+)(?:/attempts/\d+)?/?$"
)


def producer_run_id(run_url: str | None) -> str | None:
    """``owner/repo#run`` for a GitHub Actions run URL, else ``None``.

    Run URLs stay out of stored rows (dataset rule: no run URLs); the run identity is enough to tell producers
    apart, and the attempt has its own ``producer_attempt`` column.
    """
    found = _GITHUB_RUN.match(run_url or "")
    return None if found is None else f"{found['repo']}#{found['run']}"


class ObservationRow(_Record):
    observation_id: str = Field(min_length=1)
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    record_sha256: Digest  # versions content: corrections keep the ID and change this
    source_kind: str
    source_observation_id: str | None = None
    source_config_id: str | None = None
    cohort_group_id: str | None = None
    producer_kind: str | None = None
    producer_id: str | None = None  # source and producer identities stay separate
    producer_attempt: int | None = None
    producer_revision: str | None = None
    benchmark_tool: str | None = None
    benchmark_tool_version: str | None = None
    collected_at: str | None = None
    image: str | None = None
    benchmark_type: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    concurrency: int | None = Field(default=None, ge=0)
    request_rate: float | None = None
    arrival_policy: str | None = None
    cache_state: str | None = None
    num_prompts: int | None = Field(default=None, ge=0)
    outcome: Outcome = "unknown"
    requests_total: int | None = Field(default=None, ge=0)
    requests_completed: int | None = Field(default=None, ge=0)
    requests_failed: int | None = Field(default=None, ge=0)
    device_count: int | None = Field(default=None, ge=0)
    raw_artifact_id: str | None = None
    # Member of a raw-output bundle: byte range of this observation's record inside the uncompressed
    # bundle. Both null when the artifact holds exactly this observation's record. ``record_sha256`` is
    # always the hash of that record, so every row can be audited on its own.
    raw_offset: int | None = Field(default=None, ge=0)
    raw_length: int | None = Field(default=None, ge=0)
    log_artifact_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    # Source fields without a canonical home, verbatim (canonical JSON text), so a source-agnostic reader
    # does not need the raw artifact: the producer's recipe hash, the upstream configuration record this
    # observation belongs to, and the source metrics that have no metric row (power, qps, cache hit rates,
    # parallelism degrees, ...). Typed columns and identity fields stay authoritative where they overlap.
    recipe_fingerprint: str | None = None
    source_config_json: str | None = None
    unmapped_metrics_json: str | None = None
    # Added in minor 1, all optional. ``observer`` says who timed the observation's metrics (client or server),
    # ``time_origin`` what request times count from, ``attrs`` is canonical JSON text for scalars with no column
    # yet (a promoted attr becomes a column in a later minor).
    valid_for_submission: bool | None = None
    qualification: str | None = None
    observer: str | None = None
    time_origin: str | None = None
    attrs: str | None = None


# JSON Schema twin of ``MetricRow.status_matches_value`` so external readers reject what AISim rejects (a
# non-finite number cannot appear in JSON, so finiteness needs no clause).
_METRIC_VALUE_RULE = {
    "if": {"properties": {"status": {"const": "valid"}}, "required": ["status"]},
    "then": {"required": ["value"], "properties": {"value": {"type": "number"}}},
    "else": {"properties": {"value": {"type": "null"}}},
}


class MetricRow(_Record):
    model_config = ConfigDict(json_schema_extra={"allOf": [_METRIC_VALUE_RULE]})

    observation_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)  # ttft, tpot, itl, e2el, output_throughput, ...
    statistic: str = Field(min_length=1)  # mean, median, p90, p99, std, min, max, value
    scope: str = "run"
    normalization: Normalization = "none"
    device_count: int | None = Field(default=None, ge=1)  # divisor for per_device
    unit: str = Field(min_length=1)  # canonical unit
    value: float | None = None  # canonical units; null unless status == "valid"
    status: Status
    timing_boundary: str | None = None
    source_name: str | None = None  # the tool's own name, kept for audit
    source_unit: str | None = None
    source_value: float | None = None  # as reported, before conversion
    observer: str | None = None  # client or server: who timed it (never mixed, e.g. client vs engine TTFT)
    attrs: str | None = None  # canonical JSON text for fields with no column yet

    @model_validator(mode="after")
    def status_matches_value(self):
        if self.status == "valid" and (self.value is None or not math.isfinite(self.value)):
            raise ValueError("a valid metric needs a finite value")
        if self.status != "valid" and self.value is not None:
            raise ValueError("a missing or invalid metric must not carry a value; missing is never zero")
        # A registered metric is stored in its one canonical unit; an unregistered (extension) metric keeps its own.
        registered = CANONICAL_UNIT.get(self.metric) or canonical_unit(self.metric)
        if registered is not None and self.unit != registered:
            raise ValueError(f"metric {self.metric!r} must be stored in {registered!r}, not {self.unit!r}")
        return self


class RequestRow(_Record):
    """Per-request row; latencies in ms, null when not measured. Times count from ``time_origin``."""

    observation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    status: Literal["completed", "failed", "cancelled"] = "completed"
    phase: str | None = None  # canonical phase: warmup, kernel_warmup, profile
    start_ms: float | None = None
    end_ms: float | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    e2el_ms: float | None = None
    queue_ms: float | None = None
    error: str | None = None
    identity_valid: bool | None = None  # False when the request cannot be matched to its trace position
    # Lineage keys for agentic traces; all optional.
    conversation_id: str | None = None
    turn_index: int | None = Field(default=None, ge=0)
    parent_request_id: str | None = None
    source_trace_id: str | None = None
    source_outer_idx: int | None = Field(default=None, ge=0)
    source_inner_idx: int | None = Field(default=None, ge=0)
    agent_depth: int | None = Field(default=None, ge=0)
    attrs: str | None = None  # canonical JSON text: think time, dependency IDs, worker ID, ...


class SeriesRow(_Record):
    """One interval of a server-side time series (running requests, KV use, ...).

    An absent series means unavailable; an observed zero is a row with value 0. ``source`` names the one
    reporting rank or pool: replicas of a tensor-parallel group are never summed.
    """

    observation_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    start_ms: float
    end_ms: float
    value: float | None = None
    coverage: float | None = Field(default=None, ge=0, le=1)  # fraction of the interval with samples
    unit: str | None = None
    source: str | None = None
    phase: str | None = None  # canonical phase; times are ms from the start of that phase


class SnapshotMember(_Record):
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    observation_id: str = Field(min_length=1)
    record_sha256: Digest
    manifest_path: str = Field(min_length=1)


class SnapshotAcquisition(_Record):
    """How an input reached the snapshot; no measured data."""

    observation_id: str = Field(min_length=1)
    acquired_at: str | None = None
    source_uri: str | None = None
    status: Literal["acquired", "inherited", "missing"] = "acquired"


class E2eInputSnapshot(_Record):
    """Frozen input selection. Contains inputs only (membership and acquisition)."""

    schema_version: Literal["aisim-e2e-input-snapshot/v1"] = SNAPSHOT_SCHEMA
    snapshot_id: Digest
    source_scope: dict[str, JsonValue]
    selection_policy: str = Field(min_length=1)
    completeness: Literal["complete", "partial"]
    counts: dict[str, int]
    members: Artifact
    acquisition: Artifact | None = None


def _arrow_type(annotation: Any) -> pa.DataType:
    """Arrow column type of a model field annotation (``X | None`` is ``X``; unknown shapes are strings)."""
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if origin in (typing.Union, types.UnionType):
        return _arrow_type(args[0]) if len(args) == 1 else pa.string()
    if origin is list and args == [str]:
        return pa.list_(pa.string())
    if annotation is bool:
        return pa.bool_()
    if annotation is int:
        return pa.int64()
    if annotation is float:
        return pa.float64()
    return pa.string()


def _arrow_schema(model: type[BaseModel]) -> pa.Schema:
    return pa.schema([pa.field(name, _arrow_type(f.annotation)) for name, f in model.model_fields.items()])


def _required_columns(model: type[BaseModel]) -> set[str]:
    return {name for name, f in model.model_fields.items() if f.is_required()}


OBSERVATIONS_ARROW = _arrow_schema(ObservationRow)
METRICS_ARROW = _arrow_schema(MetricRow)
REQUESTS_ARROW = _arrow_schema(RequestRow)
SERIES_ARROW = _arrow_schema(SeriesRow)
MEMBERS_ARROW = _arrow_schema(SnapshotMember)
ACQUISITION_ARROW = _arrow_schema(SnapshotAcquisition)
_TABLES = {
    OBSERVATIONS_SCHEMA: (ObservationRow, OBSERVATIONS_ARROW),
    METRICS_SCHEMA: (MetricRow, METRICS_ARROW),
    REQUESTS_SCHEMA: (RequestRow, REQUESTS_ARROW),
    SERIES_SCHEMA: (SeriesRow, SERIES_ARROW),
    SNAPSHOT_MEMBERS_SCHEMA: (SnapshotMember, MEMBERS_ARROW),
    SNAPSHOT_ACQUISITION_SCHEMA: (SnapshotAcquisition, ACQUISITION_ARROW),
}
MINOR_METADATA_KEY = b"aisim.schema_minor"


def _extra_column(values: list[Any]) -> pa.Array:
    """Arrow column for an unknown field: its inferred type, or a string column when it has none yet."""
    column = pa.array(values)
    return column.cast(pa.string()) if pa.types.is_null(column.type) else column


def rows_to_table(schema_id: str, rows: Iterable[BaseModel]) -> pa.Table:
    """Rows to a table with every known column plus the unknown fields the rows carry."""
    model, arrow = _TABLES[schema_id]
    dumped = []
    for row in rows:
        if not isinstance(row, model):
            raise TypeError(f"{schema_id} rows must be {model.__name__}")
        dumped.append(row.model_dump())
    table = pa.Table.from_pylist(dumped, schema=arrow)
    extras = [name for name in dict.fromkeys(k for row in dumped for k in row) if name not in arrow.names]
    for name in extras:
        table = table.append_column(name, _extra_column([row.get(name) for row in dumped]))
    return table


def table_to_rows(schema_id: str, table: pa.Table) -> list[Any]:
    """Typed rows from a table matched by column name.

    A missing optional column reads as null, an unknown column is kept on the row (``model_extra``), and a
    missing required column is an error. A table written by a newer minor still loads, with a warning.
    """
    model, _arrow = _TABLES[schema_id]
    missing = _required_columns(model) - set(table.schema.names)
    if missing:
        raise ValueError(f"{schema_id}: missing required columns {sorted(missing)}")
    _warn_if_newer(schema_id, (table.schema.metadata or {}).get(MINOR_METADATA_KEY))
    return [model.model_validate(row) for row in table.to_pylist()]


def _warn_if_newer(what: str, minor: bytes | int | str | None) -> None:
    try:
        value = int(minor) if minor is not None else 0
    except ValueError:
        return
    if value > SCHEMA_MINOR:
        warnings.warn(
            f"{what} was written by e2e schema minor {value}, newer than this reader's {SCHEMA_MINOR}; "
            "fields it does not know are kept and ignored",
            stacklevel=3,
        )


# --------------------------------------------------------------------------- metric helpers


def make_metric(
    observation_id: str,
    metric: str,
    statistic: str,
    value: Any,
    unit: str | None,
    *,
    normalization: Normalization = "none",
    device_count: int | None = None,
    scope: str = "run",
    timing_boundary: str | None = None,
    source_name: str | None = None,
    observer: str | None = None,
) -> MetricRow:
    """Build one metric row, converting ``value`` from ``unit`` to the canonical unit.

    ``None`` becomes ``missing`` (never zero); NaN/inf, booleans, non-numbers and
    negative latencies/throughputs become ``invalid``. A genuine 0 stays a valid 0.
    """
    # A registered metric has one canonical unit; an unregistered one keeps the unit it was reported in
    # (it is flagged non-canonical by name_findings until it is registered).
    registered = CANONICAL_UNIT.get(metric) or canonical_unit(metric)
    if registered is not None:
        canonical = registered
    else:
        short = _UNIT_ALIASES.get((unit or "").strip().lower(), unit or "")
        canonical = UNIT_MS if short in _TO_CANONICAL else short
    base = dict(
        observation_id=observation_id,
        metric=metric,
        statistic=statistic,
        scope=scope,
        normalization=normalization,
        device_count=device_count,
        unit=canonical,
        timing_boundary=timing_boundary,
        source_name=source_name,
        source_unit=unit,
        observer=observer,
    )
    if value is None:
        return MetricRow(**base, status="missing")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        source = float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        return MetricRow(
            **base, status="invalid", source_value=source if source is not None and math.isfinite(source) else None
        )
    if registered is None and (unit or "").strip().lower() not in _UNIT_ALIASES:
        converted = float(value)  # a unit the registry does not know is kept as reported
    else:
        converted = to_canonical(float(value), unit or canonical, canonical)
    return MetricRow(**base, status="valid", value=converted, source_value=float(value))


def missing_required_metrics(observation_id: str, present: Iterable[MetricRow]) -> list[MetricRow]:
    """Explicit ``missing`` rows for required (metric, statistic) pairs the tool did not report."""
    have = {(m.metric, m.statistic) for m in present if m.scope == "run" and m.normalization == "none"}
    return [
        make_metric(observation_id, metric, stat, None, None)
        for metric in REQUIRED_LATENCY_METRICS
        for stat in REQUIRED_STATISTICS
        if (metric, stat) not in have
    ]


def throughput_per_device(row: MetricRow) -> float | None:
    """Per-device tokens/s of a throughput row; never silently total vs per device."""
    if row.status != "valid":
        return None
    if row.normalization == "per_device":
        return row.value
    if row.normalization == "total":
        if row.device_count is None:
            raise ValueError(f"{row.metric}: total throughput needs device_count to normalize")
        return row.value / row.device_count
    raise ValueError(f"{row.metric}: normalization {row.normalization!r} is not a device throughput")


# --------------------------------------------------------------------------- provenance artifacts


_SECRET = re.compile(
    rb"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|(?i:bearer)\s+[A-Za-z0-9._\-]{16,})"
)


def redact(data: bytes) -> bytes:
    """Mask obvious credentials. A best-effort first pass, not a guarantee; set ``redacted`` only after review."""
    return _SECRET.sub(b"[REDACTED]", data)


@dataclass(frozen=True)
class ProvenanceFile:
    """Original bytes of a tool output or log, to be stored compressed and hash-linked."""

    kind: Literal["raw-output", "raw-output-bundle", "log"]
    data: bytes
    observation_id: str | None = None
    name: str | None = None
    redacted: bool = False
    source_tool: str | None = None
    cap_bytes: int | None = None


def _gzip(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6, mtime=0)  # fixed mtime keeps bytes deterministic


def store_provenance(root: Path, directory: str, item: ProvenanceFile) -> tuple[Artifact, ProvenanceInfo]:
    """Write one compressed provenance artifact under ``root/directory/provenance/``.

    Raw tool output is stored unchanged or rejected past its cap; a log past its
    cap keeps a head/tail excerpt plus the full-content hash and size.
    """
    cap = item.cap_bytes or {"raw-output": RAW_CAP_BYTES, "raw-output-bundle": BUNDLE_CAP_BYTES}.get(
        item.kind, LOG_CAP_BYTES
    )
    full = item.data
    kept, truncated = full, False
    if len(full) > cap:
        if item.kind != "log":
            raise ValueError(f"raw tool output is {len(full)} bytes, over the {cap} byte cap; it is never truncated")
        kept = full[:LOG_EXCERPT_BYTES] + TRUNCATION_MARKER + full[-LOG_EXCERPT_BYTES:]
        truncated = True
    full_sha = _sha(full)
    name = item.name or f"{item.kind}-{(item.observation_id or full_sha[:12]).replace(':', '_').replace('/', '_')}"
    relative = f"{directory}/provenance/{name}.gz"
    payload = _gzip(kept)
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    artifact_id = f"{item.kind}:{name}"
    artifact = Artifact(
        artifact_id=artifact_id,
        path=relative,
        role=item.kind,
        format="gzip",
        schema="aisim-e2e-provenance/v1",
        sha256=_sha(payload),
        bytes=len(payload),
    )
    info = ProvenanceInfo(
        artifact_id=artifact_id,
        observation_id=item.observation_id,
        kind=item.kind,
        redacted=item.redacted,
        truncated=truncated,
        full_sha256=full_sha,
        full_bytes=len(full),
        stored_bytes=len(kept),
        cap_bytes=cap,
        source_tool=item.source_tool,
    )
    return artifact, info


def read_provenance(root: Path, artifact: Artifact) -> bytes:
    """Verified, decompressed content of a provenance artifact (the excerpt when truncated)."""
    return gzip.decompress(_verify(Path(root), artifact).read_bytes())


def bundle_raw_records(records: Sequence[tuple[str, bytes]]) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Concatenate raw records (one per line) into one bundle.

    Returns the uncompressed bundle bytes and ``observation_id -> (offset, length)`` of each record
    (the newline separator is not part of the record). Every record is capped like a single raw
    artifact and never truncated.
    """
    out, ranges = bytearray(), {}
    for observation_id, data in records:
        if observation_id in ranges:
            raise ValueError(f"duplicate raw record: {observation_id}")
        if len(data) > RAW_CAP_BYTES:
            raise ValueError(
                f"raw tool output of {observation_id} is {len(data)} bytes, over the {RAW_CAP_BYTES} byte cap; "
                "it is never truncated"
            )
        ranges[observation_id] = (len(out), len(data))
        out += data + b"\n"
    return bytes(out), ranges


def read_raw_records(root: Path, manifest: Manifest, observations: Iterable[ObservationRow]) -> dict[str, bytes]:
    """Verified raw record bytes by observation ID, decompressing each artifact once.

    Each record is checked against the observation's ``record_sha256`` (and the artifact against its
    manifest hash), so a bundled and a one-file-per-observation dataset read identically.
    """
    artifacts = {a.artifact_id: a for a in manifest.artifacts}
    infos = {p.artifact_id: p for p in body_from_manifest(manifest).provenance_artifacts}
    cache: dict[str, bytes] = {}
    out = {}
    for obs in observations:
        ref = obs.raw_artifact_id
        if ref is None:
            raise LoaderError(f"observation {obs.observation_id} has no raw artifact")
        artifact, info = artifacts.get(ref), infos.get(ref)
        if artifact is None or info is None:
            raise LoaderError(f"observation {obs.observation_id} references unknown artifact {ref}")
        if info.truncated:
            raise LoaderError(f"raw artifact {ref} is truncated")
        if ref not in cache:
            cache[ref] = read_provenance(root, artifact)
        data = cache[ref]
        if (obs.raw_offset is None) != (obs.raw_length is None):
            raise LoaderError(f"observation {obs.observation_id} has a partial raw record range")
        if obs.raw_offset is not None:
            if info.kind != "raw-output-bundle" or obs.raw_offset + obs.raw_length > len(data):
                raise LoaderError(f"observation {obs.observation_id} has an invalid raw record range")
            data = data[obs.raw_offset : obs.raw_offset + obs.raw_length]
        elif info.kind == "raw-output-bundle":
            raise LoaderError(f"observation {obs.observation_id} needs a record range in a bundle")
        if _sha(data) != obs.record_sha256:
            raise LoaderError(f"raw record hash mismatch: {obs.observation_id}")
        out[obs.observation_id] = data
    return out


def read_raw_record(root: Path, dataset: E2eDataset, observation_id: str) -> bytes:
    """Verified raw record of one observation of a loaded configuration."""
    obs = next((o for o in dataset.observations if o.observation_id == observation_id), None)
    if obs is None:
        raise LoaderError(f"unknown observation: {observation_id}")
    return read_raw_records(Path(root), dataset.manifest, [obs])[observation_id]


# --------------------------------------------------------------------------- configuration write/read


@dataclass
class E2eConfigurationData:
    """Everything for one configuration directory, before it is written."""

    body: E2eManifestBody
    observations: list[ObservationRow]
    metrics: list[MetricRow]
    requests: list[RequestRow] = field(default_factory=list)
    provenance: list[ProvenanceFile] = field(default_factory=list)
    series: list[SeriesRow] = field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", text).strip("-") or "unknown"


def parallelism_label(parallelism: Mapping[str, Mapping[str, int]]) -> str:
    """Readable path label such as ``tp4-ep8`` (prefill/decode roles are prefixed); ``tp1`` when all 1."""
    parts = []
    for role, dims in parallelism.items():
        prefix = "" if role == "aggregated" else role[0]
        parts.extend(f"{prefix}{name}{degree}" for name, degree in dims.items())
    return "-".join(parts) or "tp1"


def configuration_dir(body: E2eManifestBody) -> str:
    """``e2e/<model>/<hardware>/<framework>/<version>/<parallelism>/<configuration-id>``."""
    return "/".join(
        [
            "e2e",
            _slug(body.model_id),
            _slug(body.hardware),
            _slug(body.framework),
            _slug(body.framework_version),
            _slug(parallelism_label(body.parallelism)),
            body.configuration_id(),
        ]
    )


def _write_table(root: Path, directory: str, name: str, schema_id: str, rows: Sequence[BaseModel], role: str):
    relative = f"{directory}/{name}.parquet" if directory not in ("", ".") else f"{name}.parquet"
    write_parquet(
        rows_to_table(schema_id, rows),
        Path(root) / relative,
        schema_id,
        extra_metadata={MINOR_METADATA_KEY.decode(): str(SCHEMA_MINOR)},
    )
    path = Path(root) / relative
    return Artifact(
        artifact_id=name,
        path=relative,
        role=role,
        format="parquet",
        schema=schema_id,
        sha256=_sha(path.read_bytes()),
        bytes=path.stat().st_size,
        rows=len(rows),
    )


def _publish_staged(stage: Path, root: Path, relatives: Sequence[str]) -> None:
    """Move staged files into ``root`` in order; if one move fails, restore every file already replaced.

    The previous version of each file is kept as a hard link until all moves succeeded, so a failure in the middle
    (a full disk on the second table) leaves the old manifest and the old tables it digests exactly as they were.
    """
    done: list[tuple[Path, Path | None]] = []
    try:
        for relative in relatives:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if destination.exists():
                backup = stage / ".previous" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.link(destination, backup)
            done.append((destination, backup))
            os.replace(stage / relative, destination)
    except BaseException:
        for destination, backup in reversed(done):
            if backup is not None:
                os.replace(backup, destination)
            else:
                destination.unlink(missing_ok=True)
        raise


def write_configuration(root: Path, data: E2eConfigurationData, *, prefix: str = "") -> str:
    """Write tables, provenance artifacts and the envelope manifest; return the manifest path.

    ``prefix`` (e.g. ``"data"``) is prepended to the configuration directory, so
    every path stays relative to the dataset root.
    """
    root = Path(root)
    body = data.body
    cid = body.configuration_id()
    directory = f"{prefix.strip('/')}/{configuration_dir(body)}" if prefix else configuration_dir(body)
    if not data.observations:
        raise ValueError("a configuration needs at least one observation")
    known = {o.observation_id for o in data.observations}
    if len(known) != len(data.observations):
        raise ValueError("duplicate observation_id in a configuration")
    for o in data.observations:
        if o.configuration_id != cid:
            raise ValueError(f"observation {o.observation_id} belongs to a different configuration")
    for rows in (data.metrics, data.requests, data.series):
        stray = {r.observation_id for r in rows} - known
        if stray:
            raise ValueError(f"rows reference unknown observations: {sorted(stray)}")
    # Everything is built in a staging directory first: a failure after the tables (an oversize raw artifact, say)
    # must leave an existing configuration readable, so published files are replaced only once all succeeded.
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e2e-stage-", dir=root))
    try:
        artifacts = [
            _write_table(stage, directory, "observations", OBSERVATIONS_SCHEMA, data.observations, "e2e-observations"),
            _write_table(stage, directory, "metrics", METRICS_SCHEMA, data.metrics, "e2e-metrics"),
        ]
        if data.requests:
            artifacts.append(_write_table(stage, directory, "requests", REQUESTS_SCHEMA, data.requests, "e2e-requests"))
        if data.series:
            artifacts.append(_write_table(stage, directory, "series", SERIES_SCHEMA, data.series, "e2e-series"))
        infos = list(body.provenance_artifacts)
        for item in data.provenance:
            artifact, info = store_provenance(stage, directory, item)
            artifacts.append(artifact)
            infos.append(info)
        body = body.model_copy(update={"provenance_artifacts": infos})
        manifest = Manifest(
            schema_version="aisim-artifact-manifest/v1",
            category="e2e",
            purpose=PURPOSE,
            configuration_id=cid,
            identity=body.identity(),
            requested_settings=body.requested_settings,
            artifacts=artifacts,
            provenance={"e2e": body.model_dump(mode="json")},
            evidence_gaps=body.evidence_gaps,
        )
        artifact_ids = {a.artifact_id for a in artifacts}
        # The loader's provenance rule, applied before anything is published: a retained entry whose artifact was
        # not regenerated here (or a stored artifact without an entry) would make the new configuration unreadable.
        info_ids = {p.artifact_id for p in infos}
        if info_ids - artifact_ids or {a.artifact_id for a in artifacts if a.format == "gzip"} - info_ids:
            raise ValueError("provenance artifacts and manifest body disagree")
        for o in data.observations:
            for ref in (o.raw_artifact_id, o.log_artifact_id, *o.evidence_refs):
                if ref is not None and ref not in artifact_ids:
                    raise ValueError(f"observation {o.observation_id} references unknown artifact {ref}")
        relative = f"{directory}/manifest.json"
        (stage / relative).write_text(
            json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        _publish_staged(stage, root, [*(a.path for a in artifacts), relative])  # the manifest last
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return relative


def body_from_manifest(manifest: Manifest) -> E2eManifestBody:
    raw = manifest.provenance.get("e2e")
    if not isinstance(raw, dict):
        raise LoaderError("manifest lacks the e2e manifest body")
    body = E2eManifestBody.model_validate(raw)
    _warn_if_newer("e2e manifest", body.schema_minor)
    if body.identity() != manifest.identity:
        raise LoaderError("e2e manifest body disagrees with the envelope identity")
    return body


@dataclass(frozen=True)
class E2eDataset:
    """Verified typed rows for one configuration, identical for every source."""

    tables: E2eTables
    body: E2eManifestBody
    observations: list[ObservationRow]
    metrics: list[MetricRow]
    requests: list[RequestRow]
    series: list[SeriesRow] = field(default_factory=list)

    @property
    def manifest(self) -> Manifest:
        return self.tables.manifest


def load_e2e(root: Path, manifest_path: str, *, config: Mapping[str, Any] | None = None) -> E2eDataset:
    """Load one configuration through :class:`E2eLoader` and cross-check its tables."""
    root = Path(root)
    tables = E2eLoader(accepted_schemas=E2E_TABLE_SCHEMAS).load(root, manifest_path, config=config)
    body = body_from_manifest(tables.manifest)
    parquet_schemas = [a.artifact_schema for a in tables.manifest.artifacts if a.format == "parquet"]
    duplicated = sorted({schema for schema in parquet_schemas if parquet_schemas.count(schema) > 1})
    if duplicated:
        raise LoaderError(f"manifest declares more than one table for {duplicated}; one table per schema")
    by_schema = {a.artifact_schema: a.artifact_id for a in tables.manifest.artifacts if a.format == "parquet"}
    if OBSERVATIONS_SCHEMA not in by_schema or METRICS_SCHEMA not in by_schema:
        raise LoaderError("manifest must declare observation and metric tables")
    observations = table_to_rows(OBSERVATIONS_SCHEMA, tables[by_schema[OBSERVATIONS_SCHEMA]])
    metrics = table_to_rows(METRICS_SCHEMA, tables[by_schema[METRICS_SCHEMA]])
    requests = (
        table_to_rows(REQUESTS_SCHEMA, tables[by_schema[REQUESTS_SCHEMA]]) if REQUESTS_SCHEMA in by_schema else []
    )
    series = table_to_rows(SERIES_SCHEMA, tables[by_schema[SERIES_SCHEMA]]) if SERIES_SCHEMA in by_schema else []
    known = {o.observation_id for o in observations}
    if len(known) != len(observations):
        raise LoaderError("duplicate observation_id")
    artifact_ids = {a.artifact_id for a in tables.manifest.artifacts}
    for o in observations:
        if o.configuration_id != tables.manifest.configuration_id:
            raise LoaderError(f"observation {o.observation_id} has a different configuration_id")
        for ref in (o.raw_artifact_id, o.log_artifact_id, *o.evidence_refs):
            if ref is not None and ref not in artifact_ids:
                raise LoaderError(f"observation {o.observation_id} references unknown artifact {ref}")
    for rows in (metrics, requests, series):
        if {r.observation_id for r in rows} - known:
            raise LoaderError("rows reference unknown observations")
    info_ids = {p.artifact_id for p in body.provenance_artifacts}
    if info_ids - artifact_ids or {a.artifact_id for a in tables.manifest.artifacts if a.format == "gzip"} - info_ids:
        raise LoaderError("provenance artifacts and manifest body disagree")
    return E2eDataset(tables, body, observations, metrics, requests, series)


# --------------------------------------------------------------------------- evaluator view


@dataclass(frozen=True)
class EvaluationPoint:
    """One benchmark point in canonical units; the same shape for every source."""

    observation_id: str
    configuration_id: str
    input_tokens: int | None
    output_tokens: int | None
    concurrency: int | None
    # (metric, statistic) -> canonical value, valid metrics only (latency ms, throughput tokens/s).
    metrics: dict[tuple[str, str], float]
    # (metric, statistic) pairs that are missing or invalid; never zero-filled.
    unavailable: frozenset[tuple[str, str]]
    # Per-device tokens/s by throughput metric, normalized explicitly.
    per_device_throughput: dict[str, float]
    # Which timing boundary (``client`` or ``server``) reported each ``metrics`` key (``None``: the row is silent).
    observers: dict[tuple[str, str], str | None] = field(default_factory=dict)


def evaluation_points(dataset: E2eDataset, *, observer: str | None = None) -> list[EvaluationPoint]:
    """One point per observation.

    Client and server measurements of one metric are different timing boundaries and never mix: with
    ``observer`` only that observer's rows are used; without it a (metric, statistic) reported twice is
    ambiguous and raises ``ValueError`` instead of letting row order decide.
    """
    metrics: dict[str, list[MetricRow]] = {}
    for row in dataset.metrics:
        metrics.setdefault(row.observation_id, []).append(row)
    points = []
    for obs in dataset.observations:
        values, unavailable, device = {}, set(), {}
        reported: dict[tuple[str, str], str | None] = {}  # evaluator output key -> observer
        for row in metrics.get(obs.observation_id, []):
            if row.scope != "run" or (observer is not None and row.observer != observer):
                continue
            # Consumers read canonical names only: a known alias (``p50``) or a stored spelling the registry
            # rejects is ignored until the dataset is fixed (see name_findings_for).
            if name_findings("metric", "", row.metric) or name_findings("statistic", "", row.statistic):
                continue
            key = (row.metric, row.statistic)
            # Uniqueness is checked on the key the point actually returns: throughput keys carry their
            # normalization (total and per-device never mix); every other metric is keyed by (metric, statistic),
            # so rows that differ only in normalization or observer would otherwise overwrite one another.
            out_key = (f"{row.metric}:{row.normalization}", row.statistic) if row.metric.endswith("throughput") else key
            if out_key in reported:
                raise ValueError(
                    f"observation {obs.observation_id}: {out_key[0]}/{row.statistic} is reported by observers "
                    f"{reported[out_key]!r} and {row.observer!r}; select one with observer="
                )
            reported[out_key] = row.observer
            if row.status != "valid":
                unavailable.add(key)
            elif row.metric.endswith("throughput"):
                values[out_key] = row.value
                if row.normalization == "total" and row.device_count is None and obs.device_count:
                    row = row.model_copy(update={"device_count": obs.device_count})
                if row.normalization == "per_device" or (row.normalization == "total" and row.device_count):
                    device[row.metric] = throughput_per_device(row)
            else:
                values[key] = row.value
        points.append(
            EvaluationPoint(
                obs.observation_id,
                obs.configuration_id,
                obs.input_tokens,
                obs.output_tokens,
                obs.concurrency,
                values,
                frozenset(unavailable),
                device,
                reported,
            )
        )
    return points


# --------------------------------------------------------------------------- snapshot


def snapshot_id(members: Sequence[SnapshotMember], source_scope: Mapping[str, Any], selection_policy: str) -> str:
    ordered = sorted((m.model_dump() for m in members), key=lambda m: (m["configuration_id"], m["observation_id"]))
    return _sha(_canonical({"members": ordered, "source_scope": source_scope, "selection_policy": selection_policy}))


def write_snapshot(
    root: Path,
    path: str,
    members: Sequence[SnapshotMember],
    *,
    source_scope: Mapping[str, JsonValue],
    selection_policy: str,
    completeness: Literal["complete", "partial"],
    acquisition: Sequence[SnapshotAcquisition] = (),
) -> E2eInputSnapshot:
    """Write membership (and optional acquisition) tables plus the snapshot document at ``path``.

    The request is validated and everything staged before any live file is touched, and the document is
    published last, so a rejected or failed replacement leaves the existing snapshot readable.
    """
    root = Path(root)
    directory = str(Path(path).parent)
    ordered = sorted(members, key=lambda m: (m.configuration_id, m.observation_id))
    if len({m.observation_id for m in ordered}) != len(ordered):
        raise ValueError("duplicate observation in snapshot membership")
    # Tables are named after the document, so sibling snapshots never overwrite each other's frozen members.
    stem = Path(path).stem
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e2e-stage-", dir=root))
    try:
        member_artifact = _write_table(
            stage, directory, f"{stem}-members", SNAPSHOT_MEMBERS_SCHEMA, ordered, "e2e-snapshot-members"
        )
        acquisition_artifact = (
            _write_table(
                stage,
                directory,
                f"{stem}-acquisition",
                SNAPSHOT_ACQUISITION_SCHEMA,
                list(acquisition),
                "e2e-snapshot-acquisition",
            )
            if acquisition
            else None
        )
        snapshot = E2eInputSnapshot(
            snapshot_id=snapshot_id(ordered, source_scope, selection_policy),
            source_scope=dict(source_scope),
            selection_policy=selection_policy,
            completeness=completeness,
            counts={
                "observations": len(ordered),
                "configurations": len({m.configuration_id for m in ordered}),
            },
            members=member_artifact,
            acquisition=acquisition_artifact,
        )
        (stage / path).parent.mkdir(parents=True, exist_ok=True)
        (stage / path).write_text(
            json.dumps(snapshot.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True) + "\n"
        )
        tables = [a.path for a in (member_artifact, acquisition_artifact) if a is not None]
        _publish_staged(stage, root, [*tables, path])  # the document last
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return snapshot


def read_snapshot(root: Path, path: str) -> tuple[E2eInputSnapshot, list[SnapshotMember]]:
    """Verify a snapshot document, its tables, counts and ID."""
    root = Path(root)
    snapshot = E2eInputSnapshot.model_validate_json(_file(root, path).read_bytes())
    members = table_to_rows(
        SNAPSHOT_MEMBERS_SCHEMA, read_parquet(_verify(root, snapshot.members), SNAPSHOT_MEMBERS_SCHEMA)
    )
    if snapshot.acquisition is not None:
        table_to_rows(
            SNAPSHOT_ACQUISITION_SCHEMA, read_parquet(_verify(root, snapshot.acquisition), SNAPSHOT_ACQUISITION_SCHEMA)
        )
    if snapshot.counts.get("observations") != len(members):
        raise ValueError("snapshot observation count mismatch")
    if snapshot.counts.get("configurations") != len({m.configuration_id for m in members}):
        raise ValueError("snapshot configuration count mismatch")
    if snapshot.snapshot_id != snapshot_id(members, snapshot.source_scope, snapshot.selection_policy):
        raise ValueError("snapshot ID does not match its membership")
    return snapshot, members


# --------------------------------------------------------------------------- InferenceX converter

_IFX_LATENCY = re.compile(r"^(?P<stat>mean|median|std|min|max|p\d+(?:[._]\d+)?)_(?P<metric>ttft|tpot|itl|e2el)$")
_IFX_INTERACTIVITY = re.compile(r"^(?P<stat>mean|median|std|min|max|p\d+(?:[._]\d+)?)_intvty$")
_IFX_THROUGHPUT = {
    "tput_per_gpu": "total_throughput",
    "output_tput_per_gpu": "output_throughput",
    "input_tput_per_gpu": "input_throughput",
}
INFERENCEX_CONFIG_FIELDS = (
    "model",
    "hardware",
    "framework",
    "precision",
    "spec_method",
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_prefill_gpu",
    "num_decode_gpu",
)
INFERENCEX_LATENCY_UNIT = "s"  # InferenceX reports latencies in seconds


def _stat(name: str) -> str:
    return name.replace("_", ".") if re.fullmatch(r"p\d+_\d+", name) else name  # p99_9 and p99.9 both -> p99.9


def inferencex_metrics(
    observation_id: str, metrics: Mapping[str, Any], device_count: int | None
) -> tuple[list[MetricRow], list[str]]:
    """Normalize one InferenceX ``metrics`` dict; returns (rows, unmapped source keys).

    Latencies are seconds in the source and milliseconds here. ``*_tput_per_gpu``
    are per-device rates, ``*_intvty`` are per-user. Required statistics the
    source did not report are stored explicitly as ``missing``.
    """
    rows, unmapped = [], []
    for key in sorted(metrics):
        value = metrics[key]
        if match := _IFX_LATENCY.match(key):
            rows.append(
                make_metric(
                    observation_id,
                    match["metric"],
                    _stat(match["stat"]),
                    value,
                    INFERENCEX_LATENCY_UNIT,
                    source_name=key,
                )
            )
        elif match := _IFX_INTERACTIVITY.match(key):
            rows.append(
                make_metric(
                    observation_id,
                    "interactivity",
                    _stat(match["stat"]),
                    value,
                    UNIT_TOKENS_PER_S_PER_USER,
                    normalization="per_user",
                    source_name=key,
                )
            )
        elif key in _IFX_THROUGHPUT:
            rows.append(
                make_metric(
                    observation_id,
                    _IFX_THROUGHPUT[key],
                    "value",
                    value,
                    UNIT_TOKENS_PER_S,
                    normalization="per_device",
                    device_count=device_count or None,
                    source_name=key,
                )
            )
        elif re.fullmatch(r"(?:(?:prefill|decode)_)?(?:pp|pcp_size|dcp_size)", key):
            continue  # a parallelism degree: mapped by inferencex_metric_parallelism, kept in the raw artifact
        else:
            unmapped.append(key)
    rows.extend(missing_required_metrics(observation_id, rows))
    return rows, unmapped


# InferenceX reports pipeline/context parallel degrees inside the ``metrics`` dict, not in the
# configuration row. They are deployment facts, so they map to parallelism dimensions.
_IFX_METRIC_DIMENSIONS = {"pp": "pp", "pcp_size": "cp", "dcp_size": "dcp"}


def inferencex_metric_parallelism(raw: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Parallelism degrees an InferenceX record reports under ``metrics`` (aggregated or per role)."""
    metrics = raw.get("metrics") or {}
    out: dict[str, dict[str, int]] = {}
    roles = ("prefill", "decode") if raw.get("disagg") else ("aggregated",)
    for role in roles:
        dims = {}
        for key, dim in _IFX_METRIC_DIMENSIONS.items():
            # Aggregated records use the plain key, or the decode_* key when only that is reported.
            names = (key, f"decode_{key}") if role == "aggregated" else (f"{role}_{key}",)
            for name in names:
                degree = metrics.get(name)
                if isinstance(degree, int) and not isinstance(degree, bool) and degree > 0:
                    dims[dim] = degree
                    break
        if dims:
            out[role] = dims
    return out


def _canonical_kind(value: Any) -> str | None:
    """Canonical workload kind of a source ``benchmark_type`` (unknown spellings stay as reported)."""
    return None if value is None else (canonical_name("workload_kind", str(value)) or str(value))


def _ifx_body(
    config: Mapping[str, Any], gaps: list[dict], metric_parallelism: Mapping[str, Mapping[str, int]] | None = None
) -> E2eManifestBody:
    disagg = bool(config["disagg"])
    roles = ("prefill", "decode") if disagg else ("decode",)
    # Aggregated InferenceX records carry the serving worker under the decode_* fields.
    parallelism: dict[str, dict[str, int]] = {}
    for role in roles:
        label = "aggregated" if not disagg else role
        dims = {}
        for dim in ("tp", "ep"):
            degree = config[f"{role}_{dim}"]
            if isinstance(degree, int) and not isinstance(degree, bool) and degree > 0:
                dims[dim] = degree
            else:
                # A zero or missing degree is "not reported", never a real degree: keep it out of the identity.
                gaps.append({"field": f"parallelism.{label}.{dim}", "reason": f"source reported {degree!r}"})
        dims.update((metric_parallelism or {}).get(label, {}))
        parallelism[label] = dims
    workers = {
        ("aggregated" if not disagg else role): WorkerGroup(
            num_workers=config.get(f"{role}_num_workers"),
            total_gpus=config.get("num_prefill_gpu" if role == "prefill" else "num_decode_gpu"),
        )
        for role in roles
    }
    gaps.extend(
        [
            {"field": "framework_version", "reason": "InferenceX reports an image, not a framework version"},
            {"field": "model_revision", "reason": "not reported by the source"},
            {
                "field": "parallelism",
                "reason": "pp, cp and dcp come from the record's metrics keys when present; dp degrees are not "
                "reported; dp_attention is the source boolean, not a degree",
            },
        ]
    )
    settings = {"spec_method": Setting(value=config["spec_method"], authority="producer_declared")}
    if "workers" in config:
        settings["workers"] = Setting(value=config["workers"], authority="producer_declared")
    # Stored names are canonical (see aisim_core.data.names): the source spellings are kept under source_ids.
    engine = split_framework(config["framework"])
    for name, value in engine.get("settings", {}).items():
        settings[name] = Setting(value=value, authority="producer_declared")
    return E2eManifestBody(
        origin="official",
        model_id=normalize("model", config["model"]),
        deployment="disaggregated" if disagg else "aggregated",
        hardware=normalize("hardware", config["hardware"]),
        framework=engine["framework"],
        orchestrator=engine.get("orchestrator"),
        speculation={"method": config["spec_method"]},
        framework_version="unknown",
        parallelism=parallelism,
        parallelism_complete=False,
        precision={"weights": normalize("precision", config["precision"])},
        source_ids={
            "inferencex": {k: config[k] for k in ("model", "hardware", "framework") if k in config},
        },
        # Every source field that distinguishes upstream deployments is identity, so distinct source
        # configurations never share a configuration ID: attention DP (a boolean, never a degree),
        # multinode and KV offload are first-class identity fields (hash v3); the speculation method
        # and worker counts are identity settings. Run telemetry (``workers`` GPU stats) is not identity.
        dp_attention={role: config[f"{role}_dp_attention"] for role in ("prefill", "decode")},
        multinode=config["is_multinode"],
        offload_mode=config.get("offload_mode"),
        identity_settings={
            "spec_method": config["spec_method"],
            "workers": {k: v.model_dump(exclude_none=True) for k, v in sorted(workers.items())},
            # A fused spelling such as ``mori-sglang`` carries a KV transfer backend that plain ``sglang`` lacks.
            **{k: v for k, v in engine.get("settings", {}).items() if k == "kv_transfer_backend"},
        },
        settings=settings,
        workers=workers,
        source_kind="inferencex",
        source_config_id=str(config.get("config_id")) if config.get("config_id") is not None else None,
        evidence_gaps=gaps,
    )


def inferencex_to_configurations(
    records: Iterable[Mapping[str, Any]], runs: Mapping[str, Mapping[str, Any]] | None = None
) -> list[E2eConfigurationData]:
    """Convert InferenceX benchmark records (the cache's raw record shape) to E2E configurations.

    Each raw record is kept unchanged (canonical JSON) in one hash-linked
    raw-output bundle per configuration (every observation keeps the hash and byte
    range of its own record), so a mapping error is fixed by re-normalizing. Source
    fields without a canonical home (recipe fingerprint, upstream configuration
    record, unmapped metrics) are also typed observation columns. ``runs`` optionally
    maps a ``run_url`` to producer metadata (``head_sha``, ``run_attempt``,
    ``run_started_at``, ``head_branch``, ...); it is kept verbatim under
    ``source_fields["producer_runs"]`` because ordering and freshness decisions need it.

    Upstream source configurations (content hash ``hf:<sha256>``, kept per
    observation as ``source_config_id``) that differ only in run telemetry share one
    ``configuration_id`` and are merged into one configuration directory.
    """
    runs = runs or {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw in records:
        grouped.setdefault(_sha(_canonical(_source_config(raw))), []).append(raw)
    merged: dict[str, E2eConfigurationData] = {}
    raw_members: dict[str, list[tuple[str, bytes]]] = {}
    subgroups: list[tuple[str, str, dict, list[Mapping[str, Any]]]] = []
    for source_config, group in sorted(grouped.items()):
        by_dims: dict[str, tuple[dict, list[Mapping[str, Any]]]] = {}
        for raw in group:
            dims = inferencex_metric_parallelism(raw)
            by_dims.setdefault(_sha(_canonical(dims)), (dims, []))[1].append(raw)
        subgroups.extend((source_config, key, dims, rows) for key, (dims, rows) in sorted(by_dims.items()))
    for source_config, _dims_key, dims, group in subgroups:
        first = group[0]
        body = _ifx_body({**first, "config_id": "hf:" + source_config}, [], dims)
        cid = body.configuration_id()
        observations, metrics = [], []
        members = raw_members.setdefault(cid, [])
        unmapped_keys: set[str] = set()
        producer_runs: dict[str, Any] = {}
        gpus = (first.get("num_prefill_gpu") or 0) + (first.get("num_decode_gpu") or 0)
        source_config_json = _canonical(_source_config(first)).decode()
        for raw in sorted(group, key=lambda r: int(r["id"])):
            canonical = _canonical(raw)
            oid = f"inferencex:{raw['id']}"
            rows, unmapped = inferencex_metrics(oid, raw.get("metrics") or {}, gpus or None)
            metrics.extend(rows)
            unmapped_keys.update(unmapped)
            mapped = {m.source_name for m in rows}
            extras = {k: v for k, v in (raw.get("metrics") or {}).items() if k not in mapped}
            run = runs.get(raw.get("run_url") or "", {})
            if run:
                producer_runs[raw["run_url"]] = dict(run)
            members.append((oid, canonical))
            observations.append(
                ObservationRow(
                    observation_id=oid,
                    configuration_id=cid,
                    record_sha256=_sha(canonical),
                    source_kind="inferencex",
                    source_observation_id=str(raw["id"]),
                    source_config_id="hf:" + source_config,
                    producer_kind="github-actions" if raw.get("run_url") else None,
                    producer_id=producer_run_id(raw.get("run_url")),
                    producer_attempt=run.get("run_attempt"),
                    producer_revision=run.get("head_sha"),
                    benchmark_tool="inferencex",
                    collected_at=raw.get("date"),
                    image=raw.get("image"),
                    benchmark_type=_canonical_kind(raw.get("benchmark_type")),
                    input_tokens=raw.get("isl"),
                    output_tokens=raw.get("osl"),
                    concurrency=raw.get("conc"),
                    outcome="complete",
                    device_count=gpus or None,
                    recipe_fingerprint=raw.get("recipe_fingerprint"),
                    source_config_json=source_config_json,
                    unmapped_metrics_json=_canonical(extras).decode() if extras else None,
                )
            )
        existing = merged.get(cid)
        if existing is None:
            body.source_config_id = "hf:" + source_config
            if unmapped_keys:
                body.source_fields["unmapped_metrics"] = sorted(unmapped_keys)
            if producer_runs:
                body.source_fields["producer_runs"] = producer_runs  # type: ignore[assignment]
            merged[cid] = E2eConfigurationData(body, observations, metrics, [], [])
            continue
        target = existing.body
        ids = set(target.source_fields.get("source_config_ids", [target.source_config_id]))  # type: ignore[arg-type]
        ids.add("hf:" + source_config)
        target.source_config_id = None
        target.source_fields["source_config_ids"] = sorted(ids)  # type: ignore[assignment]
        if unmapped_keys:
            union = set(target.source_fields.get("unmapped_metrics", [])) | unmapped_keys  # type: ignore[arg-type]
            target.source_fields["unmapped_metrics"] = sorted(union)  # type: ignore[assignment]
        if producer_runs:
            target.source_fields.setdefault("producer_runs", {}).update(producer_runs)  # type: ignore[union-attr]
        existing.observations.extend(observations)
        existing.metrics.extend(metrics)
    for cid, data in merged.items():
        # One bundle of raw records per configuration, in observation order; every observation keeps the
        # hash and byte range of its own record.
        data.observations.sort(key=lambda o: int(o.source_observation_id))
        order = {o.observation_id: i for i, o in enumerate(data.observations)}
        data.metrics.sort(key=lambda m: order[m.observation_id])  # stable: keeps each observation's row order
        bundle, ranges = bundle_raw_records(sorted(raw_members[cid], key=lambda m: order[m[0]]))
        item = ProvenanceFile("raw-output-bundle", bundle, name=f"inferencex-raw-{cid[4:16]}", source_tool="inferencex")
        data.provenance.append(item)
        ref = f"raw-output-bundle:{item.name}"
        for obs in data.observations:
            obs.raw_artifact_id = ref
            obs.raw_offset, obs.raw_length = ranges[obs.observation_id]
            obs.evidence_refs = [ref]
    return [merged[cid] for cid in sorted(merged)]


def _source_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The upstream configuration record of an InferenceX row (its content hash is the source config ID)."""
    config = {key: raw[key] for key in INFERENCEX_CONFIG_FIELDS}
    config.update({key: raw[key] for key in ("offload_mode", "workers") if key in raw})
    return config


# --------------------------------------------------------------------------- developer (AIPerf) converter

_AIPERF_METRICS = {
    "time_to_first_token": ("ttft", "run", "none"),
    # AIPerf's inter_token_latency is a per-request mean gap, i.e. TPOT; ITL stays missing.
    "inter_token_latency": ("tpot", "run", "none"),
    "request_latency": ("e2el", "run", "none"),
    "output_token_throughput": ("output_throughput", "run", "total"),
    "request_throughput": ("request_throughput", "run", "total"),
    "output_token_throughput_per_user": ("interactivity", "run", "per_user"),
}
_AIPERF_STATS = {"avg": "mean", "p50": "median", "std": "std", "min": "min", "max": "max"}


def _aiperf_stat(name: str) -> str | None:
    if name in _AIPERF_STATS:
        return _AIPERF_STATS[name]
    return name if re.fullmatch(r"p\d+", name) else None


def aiperf_to_configuration(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    body: E2eManifestBody,
    *,
    observation_id: str,
    device_count: int | None = None,
    log: bytes | None = None,
    log_redacted: bool = False,
    producer_id: str | None = None,
    collected_at: str | None = None,
    tool_version: str | None = None,
) -> E2eConfigurationData:
    """Convert an AIPerf summary (``profile_export_aiperf.json``) and per-request rows.

    Every reported statistic is stored (``avg``/``p50``/``p90``/``p99``/``std``/...),
    converted to canonical units; required statistics AIPerf did not report are
    ``missing``. The raw summary is kept as a raw-output artifact and an optional
    log as a size-capped, compressed log artifact.
    """
    cid = body.configuration_id()
    rows: list[MetricRow] = []
    for name, (metric, scope, normalization) in _AIPERF_METRICS.items():
        entry = summary.get(name)
        if not isinstance(entry, Mapping):
            continue
        unit = entry.get("unit")
        for stat_name, value in entry.items():
            if (stat := _aiperf_stat(stat_name)) is None:
                continue
            rows.append(
                make_metric(
                    observation_id,
                    metric,
                    stat,
                    value,
                    unit,
                    normalization=normalization,
                    device_count=device_count if normalization == "total" else None,
                    scope=scope,
                    source_name=f"{name}.{stat_name}",
                    observer="client",
                )
            )
    rows.extend(missing_required_metrics(observation_id, rows))
    requests = []
    for index, record in enumerate(records):
        metric_map = record.get("metrics", {})

        def ms(name: str) -> float | None:
            entry = metric_map.get(name)
            if not isinstance(entry, Mapping) or entry.get("value") is None:
                return None
            return to_canonical(float(entry["value"]), entry.get("unit") or "ms", UNIT_MS)

        def tokens(name: str) -> int | None:
            entry = metric_map.get(name)
            return int(entry["value"]) if isinstance(entry, Mapping) and entry.get("value") is not None else None

        failed = record.get("error") is not None
        requests.append(
            RequestRow(
                observation_id=observation_id,
                request_id=str(record.get("request_id", index)),
                status="failed" if failed else "completed",
                input_tokens=tokens("input_sequence_length"),
                output_tokens=tokens("output_sequence_length"),
                ttft_ms=ms("time_to_first_token"),
                tpot_ms=ms("inter_token_latency"),
                e2el_ms=ms("request_latency"),
                error=str(record["error"]) if failed else None,
            )
        )
    summary_bytes = _canonical(summary)
    count = summary.get("request_count", {}).get("avg")
    errors = summary.get("error_request_count", {}).get("avg", 0)
    raw_item = ProvenanceFile("raw-output", summary_bytes, observation_id, name=f"aiperf-{_sha(summary_bytes)[:12]}")
    provenance = [raw_item]
    log_id = None
    if log is not None:
        log_item = ProvenanceFile(
            "log", log, observation_id, name=f"log-{_sha(log)[:12]}", redacted=log_redacted, source_tool="aiperf"
        )
        provenance.append(log_item)
        log_id = f"log:{log_item.name}"
    raw_id = f"raw-output:{raw_item.name}"
    obs = ObservationRow(
        observation_id=observation_id,
        configuration_id=cid,
        record_sha256=_sha(summary_bytes),
        source_kind="developer",
        producer_kind="aiperf",
        observer="client",
        producer_id=producer_id,
        benchmark_tool="aiperf",
        benchmark_tool_version=tool_version,
        collected_at=collected_at,
        input_tokens=_mean_int(requests, "input_tokens"),
        output_tokens=_mean_int(requests, "output_tokens"),
        outcome="complete" if summary.get("is_complete") and not summary.get("was_cancelled") else "partial",
        # AIPerf's request_count counts the successful requests; error_request_count is reported separately.
        requests_total=int(count) + int(errors) if count is not None else None,
        requests_failed=int(errors) if count is not None else None,
        requests_completed=int(count) if count is not None else None,
        device_count=device_count,
        raw_artifact_id=raw_id,
        log_artifact_id=log_id,
        evidence_refs=[raw_id] + ([log_id] if log_id else []),
    )
    return E2eConfigurationData(body, [obs], rows, requests, provenance)


def _mean_int(requests: Sequence[RequestRow], name: str) -> int | None:
    values = {getattr(r, name) for r in requests if getattr(r, name) is not None}
    return values.pop() if len(values) == 1 else None


# --------------------------------------------------------------------------- canonical-name validation


def name_findings_for(
    body: E2eManifestBody,
    observations: Sequence[ObservationRow] = (),
    metrics: Sequence[MetricRow] = (),
    requests: Sequence[RequestRow] = (),
    series: Sequence[SeriesRow] = (),
) -> list[NameFinding]:
    """Stored names that are not canonical, with the spelling to use.

    Readers never translate; this is the check contributors (and their agents) run before a pull request.
    Consumers such as the dashboard and gym ignore a non-canonical name until it is registered.
    """
    out: list[NameFinding] = []
    out += name_findings("origin", "manifest.origin", body.origin)
    out += name_findings("model", "manifest.model_id", body.model_id)
    out += name_findings("hardware", "manifest.hardware", body.hardware)
    out += name_findings("framework", "manifest.framework", body.framework)
    out += name_findings("deployment", "manifest.deployment", body.deployment)
    out += name_findings("orchestrator", "manifest.orchestrator", body.orchestrator)
    for key, value in body.precision.items():
        out += name_findings("precision", f"manifest.precision.{key}", value)
    if body.workload is not None:
        out += name_findings("workload_kind", "manifest.workload.kind", body.workload.kind)
        out += name_findings("workload_boundary", "manifest.workload.boundary", body.workload.boundary)
    for i, obs in enumerate(observations):
        out += name_findings("workload_kind", f"observations[{i}].benchmark_type", obs.benchmark_type)
        out += name_findings("observer", f"observations[{i}].observer", obs.observer)
    seen: set[tuple[str, str, str]] = set()
    for i, row in enumerate(metrics):
        for kind, field_name, value in (
            ("metric", "metric", row.metric),
            ("statistic", "statistic", row.statistic),
            ("observer", "observer", row.observer),
        ):
            if (kind, field_name, value) in seen:
                continue
            found = name_findings(kind, f"metrics[{i}].{field_name}", value)
            if found:
                seen.add((kind, field_name, value))
                out += found
    for i, req in enumerate(requests):
        for kind, field_name, value in (("request_status", "status", req.status), ("phase", "phase", req.phase)):
            if (kind, field_name, value) in seen:
                continue
            found = name_findings(kind, f"requests[{i}].{field_name}", value)
            if found:
                seen.add((kind, field_name, value))
                out += found
    for i, point in enumerate(series):
        for kind, field_name, value in (("metric", "metric", point.metric), ("phase", "phase", point.phase)):
            if (kind, f"series.{field_name}", value) in seen:
                continue
            found = name_findings(kind, f"series[{i}].{field_name}", value)
            if found:
                seen.add((kind, f"series.{field_name}", value))
                out += found
    return out


def name_findings_for_dataset(dataset: E2eDataset) -> list[NameFinding]:
    return name_findings_for(dataset.body, dataset.observations, dataset.metrics, dataset.requests, dataset.series)
