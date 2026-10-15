# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe measured-table DSpark coverage through the canonical performance API."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

import aisimulate._runtime as native
import aisimulate_core
from aisimulate_core.sdk import RustForwardPassPerfModel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def error_record(error):
    return {"type": type(error).__name__, "message": str(error)}


def config_snapshot(value):
    """Keep checkpoint identity without copying upstream config contents."""
    value = json.loads(json.dumps(value))
    params = value.get("speculation", {}).get("params", {})
    draft = params.pop("draft_config", None)
    if draft is not None:
        params["draft_config_canonical_json_sha256"] = digest(json.dumps(draft, sort_keys=True).encode())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-cache", type=Path, default=Path(tempfile.gettempdir()) / "dspark-op-coverage-configs")
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[2]
    systems = root / "python/aisimulate/src/aisimulate_core/systems"
    refs = json.loads((evidence / "checkpoint-references.json").read_text())
    pins = {ref["model"]: ref for ref in refs}
    paths = {}
    for ref in refs:
        path = args.config_cache / ref["model"].replace("/", "--") / ref["revision"] / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(ref["url"], timeout=60) as response:
                path.write_bytes(response.read())
        if digest(path.read_bytes()) != ref["config_sha256"]:
            raise ValueError(f"Pinned config digest mismatch: {path}")
        paths[ref["model"]] = path

    result = {
        "schema": "dspark-op-coverage-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "CPU-only op-table query coverage; no paired serving accuracy or silicon measurements",
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "working_diff_sha256": digest(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)),
            "native_library_sha256": digest(Path(native.__file__).read_bytes()),
            "script_sha256": digest(Path(__file__).read_bytes()),
            "checkpoint_references_sha256": digest((evidence / "checkpoint-references.json").read_bytes()),
        },
        "policy": {
            "constructor": "aisimulate_core.sdk.RustForwardPassPerfModel.best_available",
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "database_mode": "SILICON",
            "backend_version": "Rust default, recorded per model; not the campaign runtime revision",
            "notes": [
                "SILICON prevents empirical table-miss fallback; built-in empirical primitives are reported.",
                "Shared-source and shape/quantization transfer defaults are visible in resolved configuration.",
                "Construction readiness does not establish query coverage or runtime kernel equivalence.",
                "Static diagnostic SOL fields are comparisons only, never the selected timing model.",
                "Latency source shares classify top-level operations, not individual measured kernel coverage.",
                "Decode diagnostics use context_length=1023 because the API evaluates context_length+1.",
            ],
        },
        "checkpoint_references": refs,
        "models": [],
    }
    for key, model, system, tp, depth in [
        ("v4-pro", "deepseek-ai/DeepSeek-V4-Pro-0813", "b200_sxm", 8, 6),
        ("v41-flash", "deepseek-ai/DeepSeek-V4.1-Flash", "b200_sxm", 4, 5),
        ("kimi-k3", "moonshotai/Kimi-K3", "b300_sxm", 8, 2),
    ]:
        draft = "Inferact/Kimi-K3-DSpark" if key == "kimi-k3" else model
        target_path = paths[model].parent
        target_identity = str(target_path)
        if key == "v41-flash":
            # decoder_replay requires this public identity; use it only after
            # proving the bundled target is byte-identical to the immutable pin.
            bundled = systems.parent / "model_configs" / "deepseek-ai--DeepSeek-V4.1-Flash_config.json"
            if digest(bundled.read_bytes()) != pins[model]["config_sha256"]:
                raise ValueError("Bundled V4.1 config no longer matches campaign pin")
            target_identity = model
        params = {
            "num_speculative_tokens": depth,
            "adaptive_verification": False,
            "draft_config": json.loads(paths[draft].read_text()),
        }
        if key == "kimi-k3":
            params.update(draft_model_path=draft, draft_revision=pins[draft]["revision"])
        config = {
            "model": target_identity,
            "system": system,
            "backend": "vllm",
            "worker_type": "aggregated",
            "tp": tp,
            "moe_tp_size": tp,
            "moe_ep_size": 1,
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "database_mode": "SILICON",
            "kv_cache_dtype": "fp8",
            "decoder_replay": key == "v41-flash",
            "speculation": {"kind": "dspark", "params": params},
        }
        record = {
            "model_key": key,
            "model": model,
            "revision": pins[model]["revision"],
            "requested_config": config_snapshot(config),
        }
        record["normalized_config"] = config_snapshot(RustForwardPassPerfModel.normalize_config(config))
        record["phases"] = [
            {"phase": "prefill", "sequence_count": 1, "input_tokens": 1024, "prefix_tokens": 0},
            {
                "phase": "decode",
                "sequence_count": 1,
                "total_past_kv_tokens": 1024,
                "draft_width": depth,
                "verify_width": depth + 1,
            },
        ]
        try:
            perf = RustForwardPassPerfModel.best_available(config)
        except Exception as error:
            record["construction"] = {"status": "failed", "error": error_record(error)}
            for phase in record["phases"]:
                phase["status"] = "blocked_by_construction"
        else:
            record["construction"] = {"status": "ready"}
            diagnostics = perf.diagnostics()
            diagnostics["provenance"]["config"] = config_snapshot(diagnostics["provenance"]["config"])
            record["diagnostics"] = diagnostics
            for phase in record["phases"]:
                prefill = phase["phase"] == "prefill"
                try:
                    phase["latency_ms"] = (
                        perf.predict_prefill_latency(1, 1024) if prefill else perf.predict_decode_latency_total(1, 1024)
                    )
                    phase["status"] = "query_succeeded"
                except Exception as error:
                    phase.update(status="query_failed", error=error_record(error))
                try:
                    ops = perf.static_phase_diagnostics(
                        batch_size=1, context_length=1024 if prefill else 1023, prefill=prefill
                    )
                    phase["operations"] = ops
                    latency = defaultdict(float)
                    for op in ops:
                        latency[op["source"]] += op["latency_ms"]
                    phase["top_level_source_counts"] = dict(Counter(op["source"] for op in ops))
                    phase["top_level_source_latency_ms"] = dict(latency)
                    phase["diagnostic_total_latency_ms"] = sum(latency.values())
                except Exception as error:
                    phase["diagnostic_error"] = error_record(error)
        result["models"].append(record)
        print(
            key,
            record["construction"],
            [(p["phase"], p["status"], p.get("latency_ms")) for p in record["phases"]],
            flush=True,
        )

    result["relevant_table_inventory"] = []
    for system, families in [("b200_sxm", ["dsv41", "sparse_attention"]), ("b300_sxm", ["mla", "kda"])]:
        for family in families:
            for path in sorted((systems / "data" / system / family).rglob("*.parquet")):
                if "/vllm/" not in str(path) and family not in ("dsv41", "mla"):
                    continue
                meta = path.parent / "collection_meta.yaml"
                result["relevant_table_inventory"].append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": digest(path.read_bytes()),
                        "sidecar": str(meta.relative_to(root)) if meta.exists() else None,
                        "sidecar_sha256": digest(meta.read_bytes()) if meta.exists() else None,
                        "scope": "available packaged table, not proof this source was selected",
                    }
                )
    manifest = systems / "perf_data_reuse_manifest.yaml"
    groups = yaml.safe_load(manifest.read_text())["groups"]
    result["reuse_audit"] = {
        "manifest": str(manifest.relative_to(root)),
        "manifest_sha256": digest(manifest.read_bytes()),
        "policy": "Primary, declared donors, earlier same-backend versions, then kernel-gated cross-backend fill",
        "resolution_reports": [],
        "physical_table_axes": [],
    }
    for system, table in [("b200_sxm", "dsv41_module_perf.parquet"), ("b300_sxm", "generation_mla_perf.parquet")]:
        result["reuse_audit"]["resolution_reports"].append(
            {
                "system": system,
                "backend": "vllm",
                "version": "0.24.0",
                "table": table,
                "manifest_groups": [g for g in groups if g["op_file"] == table],
                "report": json.loads(
                    aisimulate_core.resolve_op_sources_report_json(
                        str(systems),
                        str(systems / "data" / system),
                        "vllm",
                        "0.24.0",
                        table,
                        enable_shared_layer=True,
                        strict=False,
                    )
                ),
            }
        )
    tables = list((systems / "data/b200_sxm/dsv41").rglob("*.parquet"))
    tables += list((systems / "data/b300_sxm/mla/vllm/0.25.0").glob("*.parquet"))
    tables += list((systems / "data/b300_sxm/mla/sglang/0.5.14").glob("generation_mla_perf.parquet"))
    for path in tables:
        table = pq.read_table(path)
        axes = {
            name: table[name].unique().to_pylist()
            for name in table.column_names
            if name
            in (
                "kernel_source",
                "mla_dtype",
                "kv_cache_dtype",
                "num_heads",
                "measurement_scope",
                "execution_profile",
            )
        }
        result["reuse_audit"]["physical_table_axes"].append(
            {
                "path": str(path.relative_to(root)),
                "row_count": table.num_rows,
                "columns": table.column_names,
                "axes": axes,
                "has_explicit_query_width_axis": any(
                    name in table.column_names for name in ("query_width", "query_length", "verify_width")
                ),
            }
        )
    result["summary"] = {
        "models_constructed": sum(m["construction"]["status"] == "ready" for m in result["models"]),
        "queries_succeeded": sum(p["status"] == "query_succeeded" for m in result["models"] for p in m["phases"]),
        "planned_phase_queries": 6,
        "paired_measurement_records": 0,
        "mape": None,
    }
    (args.output or evidence / "op-coverage.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
