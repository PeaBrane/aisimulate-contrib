#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect source-only SLA parity and historical fake-runner negative controls.

Run with PYTHONPATH=python/aisim/src and --scratch outside the checkout.
Nothing here runs a native prediction or establishes prediction accuracy.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from aisim.sweeper.config import SLATarget
from aisim.sweeper.score import aggregate_sla_violations

ROOT = Path(__file__).resolve().parents[3]
AREA = "skills/aisim-gpu-selection-and-sizing/scripts"


def load(path):
    spec = importlib.util.spec_from_file_location("headroom", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_results():
    module = load(ROOT / AREA / "headroom_scan.py")
    original_import = builtins.__import__

    def without_score(name, *args, **kwargs):
        if name == "aisim.sweeper.score":
            raise ImportError("exercise standalone fallback")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", without_score):
        fallback, _ = module.sla_checker()
    base = json.loads((Path(__file__).parent / "consumer-fixture.json").read_text())["prediction_report"]
    results = []
    for changes, passes in (
        ({}, True),
        ({"mean_ttft_ms": 500}, True),
        ({"mean_ttft_ms": 500.01}, False),
        ({"mean_tpot_ms": 50}, True),
        ({"mean_tpot_ms": 50.01}, False),
        ({"mean_e2e_latency_ms": 3000}, True),
        ({"mean_e2e_latency_ms": 3000.01}, False),
        ({"completed_requests": 0}, False),
        ({"completed_requests": None}, False),
        ({"num_ttft_samples": 0}, False),
        ({"num_tpot_samples": None}, False),
        ({"mean_ttft_ms": None}, False),
    ):
        sla = {"e2e_ms": 3000} if any("e2e" in key for key in changes) else {"ttft_ms": 500, "itl_ms": 50}
        report = {**base, **changes}
        actual = list(aggregate_sla_violations(report, SLATarget(**sla)))
        standalone = fallback(report, sla)
        assert (not actual) is passes and (not standalone) is passes
        results.append(
            {
                "changes": changes,
                "sla": sla,
                "expected_pass": passes,
                "aisim_violations": actual,
                "fallback_violations": standalone,
            }
        )
    return results


def negative_controls(scratch):
    source = (ROOT / "tests/sweeper/test_catalog_size_ladder.py").read_text()
    cases = (
        ("bdb0a4a0", "headroom_scan", "headroom_preserves_arrival_window_at_double_rate"),
        ("92c2564e", "size_ladder", "probe_timeout_stops_before_next_probe_or_recommend"),
        ("16da056e", "size_ladder", "invalid_effective_config_stops_before_execution"),
        ("f49e7925", "size_ladder", "retry_above_rejected_two_gpu_answer"),
    )
    results = []
    for revision, script, selected in cases:
        for version in ("before", "after"):
            ref = revision + "^" if version == "before" else "HEAD"
            archived = scratch / f"{revision}-{version}-{script}.py"
            archived.write_text(
                subprocess.check_output(["git", "show", f"{ref}:{AREA}/{script}.py"], cwd=ROOT, text=True)
            )
            test_file = scratch / f"test_{revision}_{version}.py"
            modified = source.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
            modified = modified.replace(f'path = ROOT / "{AREA}/{script}.py"', f"path = Path({str(archived)!r})")
            test_file.write_text(modified)
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(ROOT / "pytest.ini"),
                str(test_file),
                "-k",
                selected,
                "-q",
                "--tb=line",
                "--no-header",
                "--color=no",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(scratch / f"tmp-{revision}-{version}"),
            ]
            run = subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True
            )
            assert run.returncode == (1 if version == "before" else 0), run.stdout + run.stderr
            output = (run.stdout + run.stderr).replace(str(scratch), "$SCRATCH").replace(str(ROOT), "$REPO")
            results.append(
                {
                    "historical_fix": revision,
                    "source": ref,
                    "version": version,
                    "test_filter": selected,
                    "exit_code": run.returncode,
                    "output": output,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if scratch.is_relative_to(ROOT):
        parser.error("--scratch must be outside the checkout")
    scratch.mkdir(parents=True, exist_ok=True)
    result = {
        "evidence_type": "local synthetic/fake-runner; not hosted CI, model predictions, or agent evals",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "oracle_comparison": oracle_results(),
        "negative_controls": negative_controls(scratch),
        "diagnostics": "Both checkers agree on verdicts; missing/zero-sample message wording differs. "
        "The fallback rejects numeric strings and booleans; these are outside the numeric report contract.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
