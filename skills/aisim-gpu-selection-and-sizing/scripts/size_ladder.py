#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find the smallest qualifying GPU count found for one GPU type.

Workaround for an aisim 0.14.0 issue: min_gpus overshoots in a single
large-budget search. Fixed on main after 0.14.0
(https://github.com/ai-dynamo/aisim/pull/913). Remove this rising-budget
ladder once the skill pins a release with that fix, where a single
`aisim recommend` with `optimization.target: min_gpus` returns the smallest
qualifying shape.

Steps, each `aisim recommend` in its own fresh output directory:
  1. preflight: validate the config, require min_gpus, a request-rate load
     (poisson or constant_rate) and an explicit min_goodput_rps
  2. coverage probe: a one-request `aisim predict` per backend; stop if AISim
     has no timing data or does not model the architecture
  3. budget ladder (default 1,2,4,8,16): raise the budget only when the run
     shows genuine infeasibility (every evaluated candidate missed the SLA,
     load or fit constraints, or no shape fits the budget); stop on
     configuration, data, resource or evaluation errors
  4. ascending refinement: after a qualifying result using U GPUs, try each
     budget from (last failing budget + 1) to U - 1 in ascending order

The answer is the smallest qualifying configuration found in the tested
search space. Each budget is a sampled search (optimizer.max_trials), so a
smaller configuration can still exist.

Hardware and maximum GPU budget are managed by --hardware and --budgets.
User --set overrides of those paths or their parent mappings are rejected.

Exit status: 0 found; 1 nothing qualified up to the largest budget;
2 input or configuration error; 3 resource, timeout or evaluation error;
4 no AISim coverage for this model, GPU type and backend.

Usage:
  size_ladder.py SIZING.yaml --hardware h200_sxm --out-root ./sizing-h200 \
      [--budgets 1,2,4,8,16] [--no-refine] [--timeout 1800] [--set PATH=VALUE ...]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REQUEST_RATE_LOADS = {"poisson", "constant_rate"}
NO_COVERAGE_MARKERS = ("no timing data for", "is not supported by the op-level model")
NO_VIABLE_SHAPE = "no deployment_mode has a viable parallel config"
CAPACITY_REASONS = {"sla_constraint", "load_constraint"}


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    """Mirror aisim's --set (dot path, YAML value) so preflight sees the effective config."""
    import yaml

    for item in sets:
        path, _, value = item.partition("=")
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(value)
    return cfg


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


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout:g}s"
    return p.returncode, p.stdout, p.stderr


def last_line(text):
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def read_result(out: Path):
    path = out / "recommendation.json"
    return json.loads(path.read_text()) if path.is_file() else None


def best_selected(result):
    order = (result.get("views") or {}).get("top_n") or []
    by_id = {c["candidate_id"]: c for c in result.get("candidates", [])}
    return by_id.get(order[0]) if order else None


def probe(cfg, hardware, backend, workdir, sets, timeout):
    import yaml

    engine = cfg["engine"]
    traffic = cfg["traffic"]
    pconf = {
        "traffic": {
            "source": traffic["source"],
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
        "engine": {
            "mode": "aggregated",
            "model": engine["model"],
            "hardware": hardware,
            "backend": backend,
            "workers": {"aggregated": {}},
        },
    }
    for key in ("context_length", "backend_version"):
        if key in engine:
            pconf["engine"][key] = engine[key]
    if "execution" in cfg:
        pconf["execution"] = cfg["execution"]
    path = workdir / f"probe-{backend}.yaml"
    path.write_text(yaml.safe_dump(pconf, sort_keys=False))
    cmd = ["aisim", "predict", "--config", str(path), "--output-dir", str(workdir / f"probe-{backend}")]
    for s in sets:
        cmd += ["--set", s]
    rc, _, err = run(cmd, timeout)
    failed = [ln for ln in err.splitlines() if "failed:" in ln or "error:" in ln]
    msg = (failed[-1] if failed else last_line(err))[:400]
    if rc is None:
        return "timeout", msg
    if rc == 0:
        return "covered", ""
    if any(marker in err for marker in NO_COVERAGE_MARKERS):
        return "no_coverage", msg
    if rc == 3:
        return "resource_error", msg
    return "covered_but_probe_shape_failed", msg  # e.g. the model does not fit one GPU


def classify(rc, err, out: Path):
    """Return (status, detail, result) for one recommend run."""
    result = read_result(out)
    if rc is None:
        return "timeout", err, result
    if rc == 0:
        return "qualified", "", result
    if rc == 3:
        if result and best_selected(result):
            return "qualified_partial", "some candidates were resource-limited; result covers evaluated ones", result
        return "resource_error", last_line(err), result
    if rc == 1:
        if not result:
            return "evaluation_error", last_line(err), result
        cands = result.get("candidates", [])
        evaluated = [c for c in cands if c.get("status") != "resource_limited"]
        capacity = [
            c for c in evaluated if c.get("status") == "infeasible" and c.get("reason_category") in CAPACITY_REASONS
        ]
        if evaluated and len(capacity) == len(evaluated):
            return "infeasible", f"{len(capacity)} evaluated candidate(s) missed SLA/load constraints", result
        unsupported = [c for c in evaluated if c.get("status") == "unsupported"]
        if capacity and len(capacity) + len(unsupported) == len(evaluated):
            return (
                "infeasible_partial_coverage",
                f"{len(capacity)} candidate(s) missed SLA/load constraints; {len(unsupported)} could not be "
                "evaluated (unsupported), so this budget was only partly covered",
                result,
            )
        other = sorted(
            {f"{c.get('status')}/{c.get('reason_category')}" for c in evaluated}
            - {"infeasible/sla_constraint", "infeasible/load_constraint"}
        )
        return "evaluation_error", f"non-capacity outcomes: {', '.join(other) or 'none evaluated'}", result
    if rc == 2 and NO_VIABLE_SHAPE in err:
        return "no_shape_fits_budget", last_line(err), result
    return "config_error", last_line(err), result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--out-root", type=Path, required=True, help="must not exist or be empty")
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per aisim run")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="PATH=VALUE")
    a = ap.parse_args()

    import yaml

    if a.out_root.exists() and any(a.out_root.iterdir()):
        print(f"error: {a.out_root} is not empty; use a fresh directory", file=sys.stderr)
        return 2
    a.out_root.mkdir(parents=True, exist_ok=True)
    try:
        budgets = [int(b) for b in a.budgets.split(",")]
    except ValueError:
        print("error: --budgets must be comma-separated integers", file=sys.stderr)
        return 2
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] < 1:
        print("error: --budgets must be positive and strictly ascending", file=sys.stderr)
        return 2

    try:
        for assignment in a.sets:
            path = assignment.partition("=")[0]
            for managed in ("engine.hardware", "optimization.constraints.max_candidate_gpus"):
                if path == managed or path.startswith(managed + ".") or managed.startswith(path + "."):
                    raise ValueError(f"--set {path!r} targets a value managed by --hardware or --budgets")
        # Match the first recommendation step's effective configuration and override order.
        cfg = apply_sets(
            yaml.safe_load(a.config.read_text()),
            [
                f"engine.hardware={a.hardware}",
                f"optimization.constraints.max_candidate_gpus={budgets[0]}",
                *a.sets,
            ],
        )
        try:
            from aisim.config.cli import CoreRecommendationConfig  # validates the full schema

        except ImportError:
            pass
        else:
            CoreRecommendationConfig.model_validate(cfg)
    except Exception as exc:
        print(f"error: invalid recommendation config: {exc}", file=sys.stderr)
        return 2
    opt = cfg.get("optimization") or {}
    load = (cfg.get("traffic") or {}).get("load") or {}
    floor = (opt.get("constraints") or {}).get("min_goodput_rps")
    problems = []
    if opt.get("target") != "min_gpus":
        problems.append("optimization.target must be min_gpus")
    if load.get("type") not in REQUEST_RATE_LOADS:
        problems.append(f"traffic.load.type {load.get('type')!r} is not a request-rate load (poisson, constant_rate)")
    if floor is None:
        problems.append("optimization.constraints.min_goodput_rps must be set explicitly")
    if problems:
        print("error: " + "; ".join(problems), file=sys.stderr)
        return 2
    try:
        rate = float(load["requests_per_second"])
        if isinstance(load["requests_per_second"], bool) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("traffic.load.requests_per_second must be positive and finite")
        stop_requests = ((cfg.get("traffic") or {}).get("stop") or {}).get("requests")
        window = (stop_requests / rate) if stop_requests else None
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: invalid recommendation config: {exc}", file=sys.stderr)
        return 2

    report = {
        "config": str(a.config),
        "hardware": a.hardware,
        "offered_rps": rate,
        "min_goodput_rps": floor,
        "arrival_window_s": window,
        "warnings": [],
        "probe": {},
        "steps": [],
        "answer": None,
    }
    if window is not None and window < 300:
        report["warnings"].append(
            f"arrival window is {window:.0f}s; use at least 300s (traffic.stop.requests >= "
            f"{int(rate * 300)}) so drain and warm-up do not dominate"
        )

    backends = cfg["engine"]["backend"]
    backends = backends.get("choices", []) if isinstance(backends, dict) else [backends]
    probe_dir = a.out_root / "probe"
    probe_dir.mkdir()
    for backend in backends:
        state, msg = probe(cfg, a.hardware, backend, probe_dir, [], a.timeout)  # cfg already has --set applied
        report["probe"][backend] = {"state": state, "message": msg}
        if state == "timeout":
            report["stopped"] = "timeout during the coverage probe"
            print(json.dumps(report, indent=2))
            return 3
    states = [v["state"] for v in report["probe"].values()]
    if "resource_error" in states:
        report["stopped"] = "local resource limit during the coverage probe"
        print(json.dumps(report, indent=2))
        return 3
    if all(s == "no_coverage" for s in states):
        report["stopped"] = "AISim has no timing data or architecture support for this model, GPU type and backend"
        print(json.dumps(report, indent=2))
        return 4

    def step(budget, phase):
        out = a.out_root / f"{phase}-b{budget}"
        cmd = [
            "aisim",
            "recommend",
            "--stack",
            "engine",
            "--config",
            str(a.config),
            "--set",
            f"engine.hardware={a.hardware}",
            "--set",
            f"optimization.constraints.max_candidate_gpus={budget}",
            "--output-dir",
            str(out),
        ]
        for s in a.sets:
            cmd += ["--set", s]
        rc, stdout, err = run(cmd, a.timeout)
        status, detail, result = classify(rc, err, out)
        entry = {
            "phase": phase,
            "budget": budget,
            "exit": rc,
            "status": status,
            "detail": detail,
            "output_dir": str(out),
            **keep_log(out, stdout, err),
        }
        if result:
            entry["counts"] = result.get("counts")
            best = best_selected(result)
            if best:
                entry["used_gpus"] = best.get("used_gpus")
                entry["config_file"] = str(out / "recommendations" / "0001.yaml")
        report["steps"].append(entry)
        return entry

    last_fail = 0
    found = None
    for budget in budgets:
        e = step(budget, "ladder")
        if e["status"] in ("qualified", "qualified_partial"):
            found = e
            break
        if e["status"] in ("infeasible", "infeasible_partial_coverage", "no_shape_fits_budget"):
            if e["status"] == "infeasible_partial_coverage":
                report["warnings"].append(f"budget {budget}: {e['detail']}")
            last_fail = budget
            continue
        report["stopped"] = f"{e['status']} at budget {budget}: {e['detail']}"
        print(json.dumps(report, indent=2))
        return 2 if e["status"] == "config_error" else 3

    if found is None:
        report["stopped"] = f"no qualifying configuration found up to {budgets[-1]} GPUs in the sampled search"
        print(json.dumps(report, indent=2))
        return 1

    if int(found["used_gpus"]) <= last_fail:
        report["warnings"].append(
            f"the budget-{last_fail} search found no qualifying configuration, but the budget-{found['budget']} "
            f"search found one using {found['used_gpus']} GPUs; each budget step is a sampled search, so a "
            "failed step is not proof that nothing fits"
        )
    if not a.no_refine:
        for budget in range(last_fail + 1, int(found["used_gpus"])):
            e = step(budget, "refine")
            if e["status"] in ("qualified", "qualified_partial"):
                found = e
                break
            if e["status"] not in ("infeasible", "infeasible_partial_coverage", "no_shape_fits_budget"):
                report["warnings"].append(f"refinement stopped at budget {budget}: {e['status']} {e['detail']}")
                break

    report["answer"] = {
        "statement": "smallest qualifying configuration found in the tested search space",
        "used_gpus": found["used_gpus"],
        "budget": found["budget"],
        "phase": found["phase"],
        "status": found["status"],
        "output_dir": found["output_dir"],
        "config_file": found["config_file"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
