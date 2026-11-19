#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize an `aisim recommend` output directory.

Reads recommendation.json (recommendation.csv stores configs and metrics as
JSON blobs). Prints selected candidates in rank order, optionally the best
feasible candidate of each deployment mode, and the candidates that were not
selected, always saying how many rows were left out.

Usage:
  summarize_recommendation.py OUT_DIR [--top N] [--best-per-mode] [--json]
  summarize_recommendation.py OUT_DIR --export CANDIDATE_ID --to FILE.yaml

This file is shipped identically in aisim-deployment-search and
aisim-gpu-selection-and-sizing. Stdlib only, except --export (needs PyYAML,
which aisim installs).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REJECTED_SHOWN = 10


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def shape(cfg: dict) -> str:
    if cfg.get("deployment_mode") == "disagg":

        def role(r):
            return (
                f"tp{cfg.get(r + '_tp')} dp{cfg.get(r + '_attention_dp')} "
                f"moe_tp{cfg.get(r + '_moe_tp')} ep{cfg.get(r + '_moe_ep')} x{cfg.get(r + '_replicas')}"
            )

        return f"disagg P[{role('prefill')}] D[{role('decode')}]"
    return (
        f"agg tp{cfg.get('tp')} dp{cfg.get('attention_dp')} moe_tp{cfg.get('moe_tp')} "
        f"ep{cfg.get('moe_ep')} x{cfg.get('replicas')}"
    )


def row(c: dict, rank: int | None, config_file: str | None) -> dict:
    m = c.get("metrics") or {}
    cfg = c.get("config") or {}
    gpus = c.get("used_gpus")
    out_tok_s = _num(m.get("output_throughput_tok_s"))
    good_tok_s = _num(m.get("goodput_output_throughput_tok_s"))
    per_gpu = out_tok_s / gpus if out_tok_s is not None and isinstance(gpus, int) and gpus > 0 else None
    return {
        "rank": rank,
        "candidate_id": c.get("candidate_id"),
        "config_file": config_file,
        "deployment_mode": cfg.get("deployment_mode"),
        "backend": f"{cfg.get('backend')} {cfg.get('backend_version')}",
        "shape": shape(cfg),
        "used_gpus": gpus,
        "score": _num(c.get("score")),
        "completed_requests": _num(m.get("completed_requests")),
        "sla_compliant_requests": _num(m.get("goodput_completed_requests")),
        "request_rps": _num(m.get("request_throughput_rps")),
        "goodput_rps": _num(m.get("goodput_request_throughput_rps")),
        "output_tok_s": out_tok_s,
        "output_tok_s_per_gpu": per_gpu,
        "sla_compliant_output_tok_s": good_tok_s,
        "mean_ttft_ms": _num(m.get("mean_ttft_ms")),
        "mean_tpot_ms": _num(m.get("mean_tpot_ms")),
        "mean_e2e_ms": _num(m.get("mean_e2e_latency_ms")),
    }


def fmt(v, p=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def line(r: dict) -> str:
    head = f"#{r['rank']}" if r["rank"] is not None else "(not in selected view)"
    where = r["config_file"] or f"candidate {r['candidate_id']}"
    return (
        f"{head} {where} | {r['backend']} | {r['shape']} | gpus={r['used_gpus']} | score={fmt(r['score'])}"
        f" | out_tok_s={fmt(r['output_tok_s'])} ({fmt(r['output_tok_s_per_gpu'])}/gpu)"
        f" | ttft={fmt(r['mean_ttft_ms'])}ms tpot={fmt(r['mean_tpot_ms'], 2)}ms e2e={fmt(r['mean_e2e_ms'])}ms"
        f" | completed={fmt(r['completed_requests'], 0)}"
        f" sla_ok={fmt(r['sla_compliant_requests'], 0)} goodput_rps={fmt(r['goodput_rps'], 2)}"
    )


def validate_result(result) -> None:
    """Validate the consumed ledger and view fields without requiring AISim imports."""
    if not isinstance(result, dict):
        raise ValueError("top level is not an object")
    candidates, views = result.get("candidates"), result.get("views")
    if not isinstance(candidates, list) or not isinstance(views, dict):
        raise ValueError("'candidates' must be a list and 'views' must be an object")
    counts = result.get("counts", {})
    if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("'counts' must contain nonnegative integer counts")
    by_id = {}
    statuses = {"feasible", "infeasible", "unsupported", "timed_out", "failed", "resource_limited"}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("a candidate is not an object")
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid or cid in by_id:
            raise ValueError("candidate IDs must be nonempty unique strings")
        status = candidate.get("status")
        if not isinstance(status, str) or status not in statuses:
            raise ValueError(f"{cid}: missing or invalid status")
        config = candidate.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{cid}: config must be an object")
        for key in ("deployment_mode", "backend", "backend_version"):
            if key in config and config[key] is not None and not isinstance(config[key], str):
                raise ValueError(f"{cid}: config.{key} must be a string or null")
        metrics = candidate.get("metrics", {})
        if not isinstance(metrics, dict) or any(v is not None and _num(v) is None for v in metrics.values()):
            raise ValueError(f"{cid}: metrics must contain finite numbers or null")
        gpus, score = candidate.get("used_gpus"), candidate.get("score")
        if gpus is not None and (type(gpus) is not int or gpus < 0):
            raise ValueError(f"{cid}: used_gpus must be a nonnegative integer or null")
        if score is not None and _num(score) is None:
            raise ValueError(f"{cid}: score must be a finite number or null")
        if status == "feasible" and (gpus is None or score is None):
            raise ValueError(f"{cid}: feasible candidate needs used_gpus and score")
        for key in ("reason", "reason_category"):
            value = candidate.get(key)
            if status != "feasible" and (not isinstance(value, str) or not value):
                raise ValueError(f"{cid}: non-feasible candidate needs {key}")
            if status == "feasible" and value is not None:
                raise ValueError(f"{cid}: feasible candidate cannot have {key}")
        prediction = candidate.get("prediction_config")
        if prediction is not None and not isinstance(prediction, dict):
            raise ValueError(f"{cid}: prediction_config must be an object or null")
        by_id[cid] = candidate
    for name in ("top_n", "pareto_front"):
        view = views.get(name, [])
        if not isinstance(view, list) or any(not isinstance(cid, str) for cid in view):
            raise ValueError(f"{name} must be a list of candidate IDs")
        if len(view) != len(set(view)):
            raise ValueError(f"{name} contains duplicate candidate IDs")
        if any(cid not in by_id or by_id[cid]["status"] != "feasible" for cid in view):
            raise ValueError(f"{name} must reference retained feasible candidates")
    if views.get("top_n") and views.get("pareto_front"):
        raise ValueError("top_n and pareto_front cannot both be populated")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--best-per-mode",
        action="store_true",
        help="also print the best feasible candidate of each deployment mode, ranked by score",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--export", metavar="CANDIDATE_ID", help="write this candidate's concrete prediction config")
    ap.add_argument("--to", metavar="FILE", help="destination for --export")
    a = ap.parse_args()

    path = Path(a.output_dir) / "recommendation.json"
    if not path.is_file():
        print(f"error: {path} not found (the run did not finish or wrote elsewhere)", file=sys.stderr)
        return 2
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    try:
        validate_result(d)
    except ValueError as exc:
        print(f"error: {path} is not a valid AISim recommendation result: {exc}", file=sys.stderr)
        return 2
    cands = d["candidates"]
    by_id = {c["candidate_id"]: c for c in cands}

    if a.export:
        if not a.to:
            ap.error("--export needs --to")
        c = by_id.get(a.export)
        if c is None or c.get("status") != "feasible" or not c.get("prediction_config"):
            print(f"error: {a.export} is not a feasible candidate with a prediction config", file=sys.stderr)
            return 2
        import yaml  # provided by aisim's dependencies

        Path(a.to).write_text(yaml.safe_dump(c["prediction_config"], sort_keys=False))
        print(f"wrote {a.to}")
        return 0

    views = d.get("views") or {}
    view_name = "top_n" if views.get("top_n") else ("pareto_front" if views.get("pareto_front") else None)
    order = views.get(view_name) or []
    selected = [row(by_id[cid], i, f"recommendations/{i:04d}.yaml") for i, cid in enumerate(order, 1)]
    shown = selected[: a.top]
    selected_ids = set(order)
    feasible_unselected = [c for c in cands if c.get("status") == "feasible" and c["candidate_id"] not in selected_ids]
    rejected = [c for c in cands if c.get("status") != "feasible"]

    best_per_mode = {}
    if a.best_per_mode:
        rank_of = {cid: i for i, cid in enumerate(order, 1)}
        for c in cands:
            if c.get("status") != "feasible" or _num(c.get("score")) is None:
                continue
            mode = (c.get("config") or {}).get("deployment_mode")
            if mode not in best_per_mode or c["score"] > best_per_mode[mode]["score"]:
                best_per_mode[mode] = c
        best_per_mode = {
            mode: row(
                c,
                rank_of.get(c["candidate_id"]),
                f"recommendations/{rank_of[c['candidate_id']]:04d}.yaml" if c["candidate_id"] in rank_of else None,
            )
            for mode, c in best_per_mode.items()
        }

    summary = {
        "counts": d.get("counts", {}),
        "selected_view": view_name,
        "selected_total": len(selected),
        "selected_shown": len(shown),
        "feasible_not_in_selected_view": len(feasible_unselected),
        "not_feasible_total": len(rejected),
        "not_feasible_shown": min(len(rejected), REJECTED_SHOWN),
    }
    not_feasible = [
        {
            "candidate_id": c["candidate_id"],
            "status": c.get("status"),
            "reason_category": c.get("reason_category"),
            "reason": c.get("reason"),
        }
        for c in rejected[:REJECTED_SHOWN]
    ]

    if a.json:
        json.dump(
            {
                "summary": summary,
                "selected": shown,
                "best_per_mode": best_per_mode or None,
                "not_feasible": not_feasible,
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print("counts:", ", ".join(f"{k}={v}" for k, v in summary["counts"].items()))
    print(f"selected view: {view_name or 'none'}; showing {len(shown)} of {len(selected)} selected")
    for r in shown:
        print(line(r))
    if best_per_mode:
        print("best feasible candidate per mode:")
        for mode, r in best_per_mode.items():
            print(f"  {mode}: {line(r)}")
            if r["config_file"] is None:
                print(f"    no YAML was written for it; export with --export {r['candidate_id']} --to best-{mode}.yaml")
    if feasible_unselected:
        print(f"{len(feasible_unselected)} feasible candidate(s) are not in the selected view")
    print(f"not feasible: {len(rejected)} (showing {min(len(rejected), REJECTED_SHOWN)})")
    for r in not_feasible:
        print(f"  {r['candidate_id']} {r['status']} {r['reason_category']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
