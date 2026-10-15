# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit downloaded Pro P/D windows; preserve incomplete capacity evidence."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from acceptance import measured_acceptance
from reconcile_pro_cache import reconcile_pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def samples(snapshot):
    result = {}
    for endpoint, text in snapshot.items():
        role = "prefill" if ":8010/" in endpoint else "decode"
        result[role] = {
            line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith("vllm:") and "_created{" not in line and "_bucket{" not in line
        }
    return result


def completed(sample):
    result = {}
    for role, series in sample.items():
        result[role] = {}
        for key, value in series.items():
            if key.startswith("vllm:request_success_total"):
                engine = re.search(r'engine="(\d+)"', key)[1]
                if 'finished_reason="length"' in key:
                    result[role][engine] = int(value)
                else:
                    check(value == 0, "unexpected stop/abort/error/repetition")
    return result


def audit(job, window, window_index):
    run = ROOT / "raw" / f"pd-production-{job}"
    path = run / window
    out = Path(__file__).parent / f"{job}-{window}"
    out.mkdir(exist_ok=True)
    summary, identity = read(path / "summary.json"), read(path / "identity.json")
    cell = identity["cell"]
    c = cell["concurrency"]
    rows, warmup, workload = (
        lines(path / name) for name in ("request_records.jsonl", "warmup.jsonl", "workload.jsonl")
    )
    check(len(rows) == len(workload) == 128 and len(warmup) == 8, "wrong request count")
    check(summary["status"] == "success" and summary["expected_requests"] == 128, "incomplete summary")
    check(summary["metrics"]["completed_requests"] == 128 and summary["metrics"]["failed_requests"] == 0, "failures")
    check(
        hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == summary["identity_sha256"],
        "identity hash",
    )
    check(sha(run / "effective-cache.json") == identity["runtime_cache_sha256"], "cache hash")
    check(sha(run / "source/collect_fixed.py") == identity["client_source_sha256"], "client source hash")
    check(identity["runtime_revision"] == "ac9126e58aa7bbab1856ba6593ba4d5003fea516", "runtime revision")
    check(identity["model"]["revision"] == "72e1d3230f6c080a530b0a1d46f8eb4602340597", "checkpoint revision")
    for role, launch in identity["roles"].items():
        check(launch == read(run / f"{role}-identity.json"), "role identity mismatch")
        command = launch["command"]
        spec = json.loads(command[command.index("--speculative-config") + 1])
        check(spec["method"] == "dspark" and spec["num_speculative_tokens"] == 6, "speculation identity")
        check(spec["enable_adaptive_verification"] is False and launch["prefix_cache"] is False, "adaptive/prefix")
        check(command[command.index("--max-model-len") + 1] == "16384", "context")
        check(command[command.index("--max-num-seqs") + 1] == "64", "maxseq")
        check(command[command.index("--max-num-batched-tokens") + 1] == "8192", "batch")
        check("--no-enable-prefix-caching" in command and "--cpu-offload-gb" not in command, "cache/offload")
        check(command[command.index("--attention-backend") + 1] == "FLASHMLA_SPARSE_DSV4", "attention")
        check(command[command.index("--kv-cache-dtype") + 1] == "fp8", "KV dtype")
        check(launch["graph_policy"] == "runtime-default", "graph policy")
        check(
            launch["checkpoint"]["config_sha256"] == "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0",
            "config hash",
        )
        if cell["acceptance"] == "synthetic":
            check(
                spec["rejection_sample_method"] == "synthetic" and spec["synthetic_acceptance_length"] == 3.77,
                "synthetic config",
            )
            check(launch["synthetic_control"] == identity["synthetic_control"], "synthetic control role mismatch")
        else:
            check(spec["rejection_sample_method"] == "block" and launch["synthetic_control"] is None, "real config")
    for prefix, records in (("measure", rows), ("warmup", warmup)):
        for i, row in enumerate(records):
            check(row["request_id"] == f"{prefix}-{i}" and row["status"] == "success", "request ID/status")
            check(
                len(row["prompt_token_ids"]) == len(row["output_token_ids"]) == row["expected_output_length"] == 1024,
                "token length",
            )
            check(row["usage"]["prompt_tokens"] == row["usage"]["completion_tokens"] == 1024, "usage counts")
            events = row["stream_events"]
            check(sum(e["token_count"] for e in events) == 1024, "stream count")
            check(math.isclose(row["ttft_ms"], events[0]["elapsed_ns"] / 1e6), "TTFT definition")
            check(
                math.isclose(row["tpot_ms"], (events[-1]["elapsed_ns"] - events[0]["elapsed_ns"]) / 1e6 / 1023),
                "TPOT definition",
            )
        active = peak = 0
        for _, delta in sorted([(r["arrival_ns"], 1) for r in records] + [(r["completion_ns"], -1) for r in records]):
            active += delta
            peak = max(active, peak)
            check(0 <= active <= c, "closed-loop overlap exceeds concurrency")
        check(active == 0 and peak == c, "closed-loop concurrency mismatch")
    check([r["prompt_token_ids"] for r in rows] == [r["prompt_token_ids"] for r in workload], "workload order")
    prompts = [r["prompt_token_ids"] for r in warmup + rows]
    check(hashlib.sha256(json.dumps(prompts).encode()).hexdigest() == identity["workload_sha256"], "workload hash")
    for key in ["mean_ttft_ms", "mean_tpot_ms"]:
        field = key.removeprefix("mean_")
        check(math.isclose(summary["metrics"][key], sum(r[field] for r in rows) / 128, rel_tol=1e-12), "summary means")
    check(
        math.isclose(summary["metrics"]["output_throughput_tok_s"], 128 * 1024 / summary["duration_s"], rel_tol=1e-12),
        "throughput",
    )
    before, after = (read(path / f"metrics-{p}.json") for p in ["before", "after"])
    acceptance = measured_acceptance(
        {e: t for e, t in before.items() if ":8020/" in e}, {e: t for e, t in after.items() if ":8020/" in e}, 6
    )
    saved = read(path / "acceptance.json")
    check(all(saved[k] == v for k, v in acceptance.items()), "acceptance counters")
    initial, final = samples(before), samples(after)
    counts = {"before": completed(initial), "after": completed(final)}
    for role, engines in (("prefill", 8), ("decode", 1)):
        expected_empty = {str(i): 0.0 for i in range(engines)}
        for snapshot, name in (
            (initial, "vllm:num_requests_running"),
            (initial, "vllm:num_requests_waiting"),
            (final, "vllm:num_preemptions_total"),
        ):
            values = {}
            for key, value in snapshot[role].items():
                if key.startswith(name + "{"):
                    engine = re.search(r'engine="(\d+)"', key)[1]
                    check(engine not in values, "duplicate engine metric")
                    values[engine] = value
            check(values == expected_empty, f"{role} {name}: nonzero or incomplete engine coverage")
    check(sum(counts["after"]["prefill"].values()) - sum(counts["before"]["prefill"].values()) == 128, "P completions")
    check(counts["after"]["decode"]["0"] - counts["before"]["decode"]["0"] == 128, "D completions")
    for role, series in final.items():
        for key, val in series.items():
            if key.startswith("vllm:nixl_") and any(s in key for s in ["failed_", "expired_"]):
                check(val == 0, "NIXL failure")
    before_gauges = {}
    for key, value in initial["prefill"].items():
        if key.startswith("vllm:kv_cache_usage_perc{"):
            engine = re.search(r'engine="(\d+)"', key)[1]
            before_gauges[engine] = value
    check(before_gauges == {str(i): 0.0 for i in range(8)}, "exact empty P pool boundary")
    check(
        counts["before"]["prefill"] == {str(i): 1 + 17 * window_index for i in range(8)},
        "all prior issued requests completed before boundary",
    )
    measured_by_engine = {k: v - counts["before"]["prefill"][k] for k, v in counts["after"]["prefill"].items()}
    check(measured_by_engine == {str(i): 16 for i in range(8)}, "per-engine measured admissions")
    proof = ROOT / "pd-drained-pool-boundary-source-audit.json"
    check(proof.is_file(), "missing independent installed-source boundary proof")
    source = {
        "kind": "conservative",
        "max_resident_requests_by_dp_rank": measured_by_engine,
        "includes_active_and_finished_holds": True,
        "source": (
            "Exact-zero physical P pool gauges after all prior requests and warmup completed, "
            "bound to synchronous source-audited metric publication; retain all 16 new measured requests "
            "per P engine with no release assumption during this window."
        ),
        "boundary_gauges": before_gauges,
        "boundary_completions": counts["before"],
        "measured_completions_by_prefill_engine": measured_by_engine,
        "installed_source_proof": {"path": "../../pd-drained-pool-boundary-source-audit.json", "sha256": sha(proof)},
        "files": {
            str((path / f"metrics-{p}.json").relative_to(ROOT)): sha(path / f"metrics-{p}.json")
            for p in ["before", "after"]
        },
    }
    write(out / "source-residency.json", source)
    cache = reconcile_pd(
        read(run / "effective-cache.json"),
        k=6,
        concurrency=c,
        input_length=1024,
        output_length=1024,
        source_residency_evidence=source,
    )
    cache["runtime_receipt_sha256"] = sha(run / "effective-cache.json")
    cache["adapter_sha256"] = sha(ROOT / "reconcile_pro_cache.py")
    write(out / "cache-reconciliation.json", cache)
    observers = [read(p) for p in sorted((run / "source-pool-observations").glob("*.json"))]
    reasons = (
        []
        if cache["native_capacity_proven_nonbinding"] and cache["replay_capacity_proven_nonbinding"]
        else [
            (
                "Startup-only observer receipts; conservative native source bound exceeds physical pool. "
                "Exact held-source bound remains unresolved."
            )
        ]
    )
    audit_record = {
        "schema": "dspark-request-audit-v1",
        "identity_sha256": summary["identity_sha256"],
        "measured_requests": 128,
        "drained_warmup_requests": 8,
        "input_tokens_per_request": 1024,
        "output_tokens_per_request": 1024,
        "client_usage_and_stream_ids_agree": True,
        "closed_loop_concurrency": c,
        "frozen_workload_hash_matches": True,
        "client_source_hash_matches": True,
        "metrics_recomputed_from_records": True,
        "acceptance_recomputed_from_counters": True,
        "completion_counters": counts,
        "observer_complete_engines": sum(o["complete"] for o in observers),
        "observer_engines": len(observers),
        "files": {
            str(p.relative_to(ROOT)): sha(p)
            for p in path.glob("*")
            if p.is_file() and p.suffix in [".json", ".jsonl"] and p.name != "qualified-summary.json"
        },
    }
    write(out / "request-audit.json", audit_record)
    qualification = {
        "schema": "dspark-runtime-qualification-v1",
        "eligible": not reasons,
        "reasons": reasons,
        "measurement_summary_sha256": sha(path / "summary.json"),
        "identity_sha256": summary["identity_sha256"],
        "runtime_cache_sha256": sha(run / "effective-cache.json"),
        "request_audit_sha256": sha(out / "request-audit.json"),
        "cache_reconciliation_sha256": sha(out / "cache-reconciliation.json"),
        "source_residency_sha256": sha(out / "source-residency.json"),
        "installed_source_audit": {
            "path": "../../pd-installed-source-audit.json",
            "sha256": sha(ROOT / "pd-installed-source-audit.json"),
        },
        "validated": {
            "measured_requests": 128,
            "drained_warmup_requests": 8,
            "exact_input_and_client_visible_output_tokens": 1024,
            "closed_loop_concurrency": c,
            "adaptive_verification": False,
            "prefix_caching": False,
            "external_offload": False,
            "replay_capacity_nonbinding": cache["replay_capacity_proven_nonbinding"],
            "native_capacity_nonbinding": cache["native_capacity_proven_nonbinding"],
        },
        "limitations": [
            "Observer shutdown receipts are incomplete and are not used as a peak proof.",
            "Transfer latency is unmodeled; rank histogram sums are not request critical-path timing.",
            "Unequal per-rank P work and framework/table differences retain previous approximations.",
            "Synthetic uses frozen 2.77 accepted drafts; real uses measured-window conditional rates.",
        ],
    }
    write(out / "qualification.json", qualification)
    updated = dict(summary)
    updated["original_summary_sha256"] = sha(path / "summary.json")
    updated["qualification"] = {
        "eligible": not reasons,
        "reasons": reasons,
        "receipt": f"../../../qualification-20261015/{job}-{window}/qualification.json",
        "receipt_sha256": sha(out / "qualification.json"),
    }
    write(path / "qualified-summary.json", updated)
    return {
        "job_id": job,
        "window": window,
        "qualified": not reasons,
        "audit_directory": str(out.relative_to(ROOT)),
        "summary": str((path / "qualified-summary.json").relative_to(ROOT)),
        "reasons": reasons,
    }


if __name__ == "__main__":
    result = [
        audit(job, p.name, index)
        for job in ["2258652", "2258677"]
        for index, p in enumerate(sorted((ROOT / "raw" / f"pd-production-{job}").glob("initial-*")))
    ]
    write(
        Path(__file__).with_name("index.json"),
        {"schema": "dspark-window-qualification-v1", "windows": result, "audit_source_sha256": sha(Path(__file__))},
    )
    print(json.dumps(result, indent=2))
