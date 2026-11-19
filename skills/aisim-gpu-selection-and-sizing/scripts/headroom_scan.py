#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-predict one sized configuration at higher request rates.

For each tested load (a multiple of the config's own request rate) this runs
`aisim predict` and classifies the result:

  pass     every configured SLA bound holds on aggregate means (AISim's own
           inclusive check, which fails closed on missing or non-finite
           metrics and on zero samples), AND at least --min-compliant-share
           of all requests completed within the per-request SLA, AND
           SLA-compliant goodput is at least --min-goodput-share of the
           offered rate. Goodput is measured over the whole run including
           drain, so it reads slightly below the offered rate even when the
           deployment keeps up; arrival_window_s and duration_s are reported
           so the user can see that effect.
  fail     the prediction ran and one of those checks failed
  unknown  the prediction did not run or its output could not be read

It reports the highest tested passing load: the largest multiplier such that
it and every smaller tested multiplier passed. That is a statement about the
tested points only, not a continuous capacity boundary.

Scope: request-rate traffic only (traffic.load.type poisson or
constant_rate). Concurrency and trace loads are rejected. Configure the load,
stop conditions and SLA in CONFIG.yaml; --set overrides of those sections
(including their parent mappings) are rejected before any prediction.

Exit status: 0 when every tested load is pass or fail; 2 for unusable input;
3 when any load is unknown (execution error or timeout).

Usage:
  headroom_scan.py CONFIG.yaml --min-compliant-share 0.99 --min-goodput-share 0.95 \
      [--steps 1.0,1.25,1.5,2.0] [--stack engine] [--timeout 900] \
      [--set PATH=VALUE ...] --workdir ./headroom
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REQUEST_RATE_LOADS = {"poisson", "constant_rate"}


def load_config(path: Path) -> tuple[dict, str]:
    """Return (mapping, parser) using AISim's validator when available."""
    try:
        import aisim.config.cli as aisim_config_cli  # type: ignore

        core_prediction_config = aisim_config_cli.CorePredictionConfig
    except Exception:
        core_prediction_config = None
    import yaml  # PyYAML ships with aisim

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config is not a YAML mapping")
    if core_prediction_config is not None:
        core_prediction_config.from_yaml(path)  # raises on an invalid prediction config
        return raw, "aisim"
    return raw, "yaml"


def sla_checker():
    try:
        from aisim.sweeper.config import SLATarget  # type: ignore
        from aisim.sweeper.score import aggregate_sla_violations  # type: ignore

        def check(report, sla):
            return list(aggregate_sla_violations(report, SLATarget(**sla)))

        return check, "aisim.sweeper.score.aggregate_sla_violations"
    except Exception:

        def check(report, sla):
            out = []
            done = _finite(report.get("completed_requests"))
            if done is None or done <= 0:
                out.append("completed_requests is missing, non-finite or zero")
            for label, metric, samples, key in (
                ("ttft", "mean_ttft_ms", "num_ttft_samples", "ttft_ms"),
                ("tpot", "mean_tpot_ms", "num_tpot_samples", "itl_ms"),
                ("e2e", "mean_e2e_latency_ms", "num_e2e_latency_samples", "e2e_ms"),
            ):
                bound = sla.get(key)
                if bound is None:
                    continue
                n = _finite(report.get(samples))
                if n is None or n <= 0:
                    out.append(f"{label} has no qualifying samples")
                v = _finite(report.get(metric))
                if v is None:
                    out.append(f"{label} metric {metric} is missing or non-finite")
                elif v > bound:
                    out.append(f"{label} {v:g}ms > {bound:g}ms")
            return out

        return check, "built-in fallback (same rules)"


# AISim log lines that mean some operations were priced from substituted data
# or templates. Successful runs can emit them; the user must be told.
FALLBACK_MARKERS = (
    "Loading low-fidelity fallback rows",
    "excluded from implicit nearest-earlier fallback",
    "No exact CLI args template",
)


def keep_log(out: Path, stdout: str, stderr: str) -> dict:
    """Save a run's stdout and stderr next to its output directory; return the log path and fallback lines."""
    log = out.parent / f"{out.name}.aisim.log"
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n")
    except OSError:
        log = None
    seen = []
    for line in f"{stdout}\n{stderr}".splitlines():
        line = line.strip()
        if any(marker in line for marker in FALLBACK_MARKERS) and line not in seen:
            seen.append(line[:400])
    return {"log_file": str(log) if log else None, "fallback_warnings": seen}


def _finite(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--min-goodput-share",
        type=float,
        required=True,
        help="required SLA-compliant goodput as a share of the offered rate at each load (0-1], agreed with the user",
    )
    ap.add_argument(
        "--min-compliant-share",
        type=float,
        required=True,
        help="required share of all requests that complete within the per-request SLA (0-1], agreed with the user",
    )
    ap.add_argument("--steps", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--stack", default="engine")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per prediction")
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="extra aisim --set override applied to every run (e.g. a memory cap)",
    )
    ap.add_argument("--workdir", type=Path, required=True, help="must not exist or be empty")
    a = ap.parse_args()

    for name in ("min_goodput_share", "min_compliant_share"):
        if not 0 < getattr(a, name) <= 1:
            print(f"error: --{name.replace('_', '-')} must be in (0, 1]", file=sys.stderr)
            return 2
    try:
        steps = [float(s) for s in a.steps.split(",")]
    except ValueError:
        print("error: --steps must be comma-separated numbers", file=sys.stderr)
        return 2
    if not steps or any(s <= 0 or not math.isfinite(s) for s in steps) or steps != sorted(set(steps)):
        print("error: --steps must be positive, finite, strictly ascending", file=sys.stderr)
        return 2
    if a.workdir.exists() and any(a.workdir.iterdir()):
        print(f"error: {a.workdir} is not empty; use a fresh directory", file=sys.stderr)
        return 2

    try:
        for assignment in a.sets:
            path = assignment.partition("=")[0]
            for managed in ("traffic.load", "traffic.stop", "evaluation.sla"):
                if path == managed or path.startswith(managed + ".") or managed.startswith(path + "."):
                    raise ValueError(
                        f"--set {path!r} conflicts with the scan; "
                        "set traffic.load, traffic.stop and evaluation.sla in the input YAML"
                    )
        cfg, parser = load_config(a.config)
    except Exception as exc:
        print(f"error: cannot use {a.config}: {exc}", file=sys.stderr)
        return 2
    load = (cfg.get("traffic") or {}).get("load") or {}
    if load.get("type") not in REQUEST_RATE_LOADS:
        print(
            f"error: traffic.load.type is {load.get('type')!r}; this scan supports request-rate loads "
            f"({', '.join(sorted(REQUEST_RATE_LOADS))}) only",
            file=sys.stderr,
        )
        return 2
    base_rps = _finite(load.get("requests_per_second"))
    if base_rps is None or base_rps <= 0:
        print("error: traffic.load.requests_per_second is missing or not positive", file=sys.stderr)
        return 2
    sla = {k: v for k, v in (((cfg.get("evaluation") or {}).get("sla")) or {}).items() if v is not None}
    if not sla:
        print("error: evaluation.sla has no bound; headroom against an SLA needs at least one", file=sys.stderr)
        return 2
    check, checker_name = sla_checker()
    stop = (cfg.get("traffic") or {}).get("stop") or {}
    arrival_window_s = max(300.0, (stop.get("requests") or 0) / base_rps)

    rows = []
    for step in steps:
        rate = base_rps * step
        out = a.workdir / f"x{step:g}"
        cmd = [
            "aisim",
            "predict",
            "--stack",
            a.stack,
            "--config",
            str(a.config),
            "--set",
            f"traffic.load.requests_per_second={rate!r}",
            "--set",
            f"traffic.stop.requests={math.ceil(rate * arrival_window_s)}",
            "--set",
            "traffic.stop.requests_per_load_unit=null",
            "--format",
            "json",
            "--output-dir",
            str(out),
        ]
        for s in a.sets:
            cmd += ["--set", s]
        r = {"multiplier": step, "offered_rps": rate, "output_dir": str(out)}
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            rows.append({**r, "status": "unknown", "error": f"timed out after {a.timeout:g}s"})
            continue
        r.update(keep_log(out, p.stdout, p.stderr))
        if p.returncode != 0:
            tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            rows.append({**r, "status": "unknown", "error": f"aisim exit {p.returncode}: {tail}"})
            continue
        try:
            m = json.loads(p.stdout)
        except json.JSONDecodeError:
            rows.append({**r, "status": "unknown", "error": "could not parse aisim JSON output"})
            continue
        violations = check(m, sla)
        good = _finite(m.get("goodput_request_throughput_rps"))
        share = (good / rate) if good is not None else None
        if good is None:
            violations.append("goodput_request_throughput_rps is missing or non-finite")
        elif share < a.min_goodput_share:
            violations.append(
                f"goodput {good:.3g} rps is {share:.1%} of offered {rate:.3g} rps < required {a.min_goodput_share:.0%}"
            )
        total = _finite(m.get("num_requests"))
        compliant = _finite(m.get("goodput_completed_requests"))
        compliant_share = (compliant / total) if (compliant is not None and total) else None
        if compliant_share is None:
            violations.append("SLA-compliant request count is missing")
        elif compliant_share < a.min_compliant_share:
            violations.append(
                f"{compliant:g} of {total:g} requests met the per-request SLA "
                f"({compliant_share:.1%} < required {a.min_compliant_share:.0%})"
            )
        duration_ms = _finite(m.get("duration_ms"))
        rows.append(
            {
                **r,
                "status": "pass" if not violations else "fail",
                "compliant_share": compliant_share,
                "arrival_window_s": (total / rate) if total else None,
                "duration_s": duration_ms / 1000 if duration_ms is not None else None,
                "violations": violations,
                "num_requests": _finite(m.get("num_requests")),
                "completed_requests": _finite(m.get("completed_requests")),
                "sla_compliant_requests": _finite(m.get("goodput_completed_requests")),
                "goodput_rps": good,
                "goodput_share": share,
                "mean_ttft_ms": _finite(m.get("mean_ttft_ms")),
                "mean_tpot_ms": _finite(m.get("mean_tpot_ms")),
                "mean_e2e_ms": _finite(m.get("mean_e2e_latency_ms")),
            }
        )

    highest = None
    for row in rows:
        if row["status"] != "pass":
            break
        highest = row["multiplier"]
    first_not_pass = next((row for row in rows if row["status"] != "pass"), None)
    result = {
        "config": str(a.config),
        "config_parser": parser,
        "sla_checker": checker_name,
        "sla": sla,
        "base_offered_rps": base_rps,
        "min_goodput_share": a.min_goodput_share,
        "min_compliant_share": a.min_compliant_share,
        "rows": rows,
        "highest_tested_passing_multiplier": highest,
        "first_non_passing": None
        if first_not_pass is None
        else {"multiplier": first_not_pass["multiplier"], "status": first_not_pass["status"]},
        "note": "Only the listed loads were tested; nothing is claimed between or beyond them.",
    }
    json.dump(result, sys.stdout, indent=2)
    print()
    unknown = [row for row in rows if row["status"] == "unknown"]
    if unknown:
        print(f"{len(unknown)} load(s) could not be evaluated; see 'error' fields", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
