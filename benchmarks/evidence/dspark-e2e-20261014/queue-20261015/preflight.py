# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the exact queued selections and actual-image CLI without GPU execution."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import vllm
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--queue", type=Path, required=True)
parser.add_argument("--platform", choices=["b200", "b300"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.bundle))
from window_plan import select_runs  # Import the explicitly selected frozen bundle.

plan = json.loads((args.queue / "plan.json").read_text())
manifest = json.loads((args.bundle / "bundle-manifest.json").read_text())
for name, digest in manifest["files"].items():
    if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f"bundle source changed: {name}")
args.output.mkdir(parents=True, exist_ok=False)
tokenizer_root = Path(vllm.__file__).parent / "tokenizers"
shutil.copytree(tokenizer_root, args.output / "native-tokenizer-sources", ignore=shutil.ignore_patterns("__pycache__"))
selected, checked = [], set()
cli_parser = None
cli_status = "passed"
try:
    cli_parser = make_arg_parser(FlexibleArgumentParser())
except RuntimeError as error:
    if "Failed to infer device type" not in str(error):
        raise
    cli_status = "unavailable_without_gpu; validated by each GPU startup smoke"
for job in plan["jobs"]:
    if job["platform"] != args.platform:
        continue
    selection = SimpleNamespace(
        model_key=job["model_key"],
        k=job["k"],
        mode=job["kind"],
        cell_id=list(dict.fromkeys(w["cell_id"] for w in job["windows"])) or None,
        input_length=[job["input_length"]],
        concurrency=None,
        repetition=None,
        completed_runs=args.bundle / "completed-runs.json" if job["kind"] == "sweep" else None,
    )
    rows = select_runs(selection, job["deployment"], job["acceptance"])
    if job["kind"] == "sweep":
        actual = {(r["cell"]["cell_id"], r["repetition"]) for r in rows}
        expected = {(r["cell_id"], r["repetition"]) for r in job["windows"]}
        if actual != expected:
            raise ValueError(f"selection mismatch: {job['key']}")
    elif len(rows) != 1 or rows[0]["cell"]["expected_requests"] != 128:
        raise ValueError("pilot must select one independent 128-request window")
    selected.append({"key": job["key"], "selected_windows": len(rows)})
    roles = ["agg"] if job["deployment"] == "agg" else ["prefill", "decode"]
    for role in roles:
        key = (job["model_key"], job["k"], role)
        if key in checked:
            continue
        checked.add(key)
        destination = args.output / ("-".join(map(str, key)))
        command = [
            sys.executable,
            str(args.bundle / "serve_role.py"),
            "--preflight-only",
            "--model-key",
            job["model_key"],
            "--k",
            str(job["k"]),
            "--role",
            role,
            "--port",
            "8010",
            "--output",
            str(destination),
            "--graph-policy",
            "runtime-default",
            "--load-strategy",
            "eager",
            "--startup-timeout",
            "6000",
        ]
        environment = {**os.environ, "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.2", "VLLM_NIXL_SIDE_CHANNEL_PORT": "5557"}
        subprocess.run(command, check=True, env=environment)
        identity = json.loads((destination / f"{role}-preflight.json").read_text())
        if cli_parser is not None:
            cli_parser.parse_args(identity["command"][3:])
result = {
    "status": "passed_cpu_preflight",
    "platform": args.platform,
    "bundle_manifest_sha256": hashlib.sha256((args.bundle / "bundle-manifest.json").read_bytes()).hexdigest(),
    "queued_selections": selected,
    "actual_image_role_commands": len(checked),
    "cli_parsing": cli_status,
    "scope": "Source hashes, exact checkpoints and selections; CPU CLI parsing when available. No GPU smoke.",
}
(args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
