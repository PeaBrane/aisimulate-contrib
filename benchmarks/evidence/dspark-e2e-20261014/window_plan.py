# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select canonical fixed-length cells and collect serial windows on one server."""

import json
import subprocess
import sys
from pathlib import Path

from campaign import MODELS, cells


def add_selection_arguments(parser):
    parser.add_argument("--input-length", type=int, choices=[1024, 8192], action="append")
    parser.add_argument("--cell-id", choices=[row["cell_id"] for row in cells()], action="append")
    parser.add_argument("--completed-runs", type=Path)
    parser.add_argument("--selection-only", action="store_true", help="Print selected cells without starting workers")


def select_runs(args, deployment, acceptance):
    canonical = {row["cell_id"]: row for row in cells()}
    if args.k not in MODELS[args.model_key]["k"]:
        raise ValueError("K is outside the approved model matrix")
    if args.mode == "pilot" and acceptance != "real":
        raise ValueError("acceptance pilots require real acceptance")
    selected = set(args.cell_id or [])
    if len(selected) != len(args.cell_id or []):
        raise ValueError("duplicate selected cell")
    candidates = [
        row
        for row in canonical.values()
        if row["model_key"] == args.model_key
        and row["deployment"] == deployment
        and row["num_speculative_tokens"] == args.k
        and row["acceptance"] == acceptance
    ]
    if selected - {row["cell_id"] for row in candidates}:
        raise ValueError("selected cell differs from the launched model/K/deployment/acceptance")
    windows = getattr(args, "window", None)
    if windows and (args.concurrency or args.repetition):
        raise ValueError("explicit windows cannot be combined with concurrency/repetition filters")
    pairs = [tuple(pair) for pair in windows] if windows else None
    if pairs and (
        len(set(pairs)) != len(pairs) or any(c not in [1, 4, 8, 16, 64] or r not in [1, 2, 3] for c, r in pairs)
    ):
        raise ValueError("invalid or duplicate explicit window")
    lengths = args.input_length or ([1024, 8192] if args.mode == "sweep" else [1024])
    concurrencies = args.concurrency or (
        [1, 4, 8, 16, 64] if args.mode == "sweep" else [1, 4] if args.mode == "initial" else [1]
    )
    repetitions = args.repetition or ([1, 2, 3] if args.mode in ["initial", "sweep"] else [1])
    if len(set(repetitions)) != len(repetitions):
        raise ValueError("duplicate repetition")
    if args.mode != "sweep" and (lengths != [1024] or any(c not in [1, 4] for c in concurrencies)):
        raise ValueError("expanded lengths/concurrency require mode sweep")
    done = set()
    if args.completed_runs and args.mode not in ["initial", "sweep"]:
        raise ValueError("completed production runs must not suppress independent pilots or smokes")
    if args.completed_runs:
        receipt = json.loads(args.completed_runs.read_text())
        if receipt.get("schema") != "dspark-completed-runs-v1":
            raise ValueError("unsupported completed-runs receipt")
        for row in receipt["completed"]:
            key = (row["cell_id"], row["repetition"])
            if key[0] not in canonical or type(key[1]) is not int or key[1] not in [1, 2, 3] or key in done:
                raise ValueError("invalid or duplicate completed run")
            done.add(key)
    result = []
    matched = set()
    for cell in sorted(candidates, key=lambda r: (r["input_length"], r["concurrency"])):
        if (selected and cell["cell_id"] not in selected) or cell["input_length"] not in lengths:
            continue
        allowed = [r for c, r in pairs if c == cell["concurrency"]] if pairs else repetitions
        if not pairs and cell["concurrency"] not in concurrencies:
            continue
        if allowed:
            matched.add(cell["cell_id"])
        for repetition in allowed:
            if (cell["cell_id"], repetition) not in done:
                result.append({"cell": cell, "repetition": repetition})
    if selected - matched:
        raise ValueError("explicit cell excluded by length/concurrency/window filters")
    return result


def execute_runs(
    plan,
    *,
    mode,
    output,
    source_dir,
    model_key,
    k,
    identity_path,
    endpoint,
    metrics_urls,
    tokenizer_endpoint=None,
    acceptance_endpoint=None,
):
    """A failed startup smoke prevents production; retain each completed window."""
    expected_rows = {row["cell_id"]: row for row in cells()}
    frozen_rows = {row["cell_id"]: row for row in json.loads((source_dir / "matrix.json").read_text())["cells"]}
    if frozen_rows != expected_rows:
        raise ValueError("frozen matrix differs from canonical campaign")
    rows = []
    for row in plan:
        cell, rep = row["cell"], row["repetition"]
        name = (
            f"sweep-i{cell['input_length']}-o{cell['output_length']}-c{cell['concurrency']}-r{rep}"
            if mode == "sweep"
            else f"{mode}-c{cell['concurrency']}-r{rep}"
        )
        rows.append({**row, "output": name, "status": "pending", "smoke": mode == "smoke"})
    if mode == "sweep" and plan:
        rows.insert(0, {**plan[0], "output": "startup-smoke", "status": "pending", "smoke": True})
    state_path = output / "window-status.json"

    def save():
        state_path.write_text(json.dumps({"schema": "dspark-window-execution-v1", "windows": rows}, indent=2) + "\n")

    save()
    for row in rows:
        target = output / row["output"]
        command = [
            sys.executable,
            str(source_dir / "collect_fixed.py"),
            "--matrix",
            str(source_dir / "matrix.json"),
            "--cell-id",
            row["cell"]["cell_id"],
            "--identity",
            str(identity_path),
            "--endpoint",
            endpoint,
            "--model",
            MODELS[model_key]["model"],
            "--output",
            str(target),
            "--repetition",
            str(row["repetition"]),
        ]
        for url in metrics_urls:
            command += ["--metrics-url", url]
        if tokenizer_endpoint:
            command += ["--tokenizer-endpoint", tokenizer_endpoint]
        if row["smoke"]:
            command.append("--smoke")
        row["status"] = "running"
        save()
        try:
            subprocess.run(command, check=True)
            summary = json.loads((target / "summary.json").read_text())
            expected = 1 if row["smoke"] else row["cell"]["expected_requests"]
            if (
                summary["status"] != "success"
                or summary["metrics"]["completed_requests"] != expected
                or summary["metrics"]["failed_requests"] != 0
            ):
                raise ValueError("window did not complete the exact requested tokens/requests")
            command = [sys.executable, str(source_dir / "acceptance.py"), "--run", str(target), "--k", str(k)]
            if acceptance_endpoint:
                command += ["--endpoint", acceptance_endpoint]
            subprocess.run(command, check=True)
            row["status"] = "success"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save()
    return rows
