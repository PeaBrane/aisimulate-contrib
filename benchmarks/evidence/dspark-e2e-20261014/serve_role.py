# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one P/D role inside its assigned Slurm step, with identical DSpark settings."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from campaign import MODELS, RUNTIME_REVISION
from serve_smoke import server_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--role", choices=["agg", "prefill", "decode"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    acceptance = parser.add_mutually_exclusive_group()
    acceptance.add_argument("--synthetic-al", type=float)
    acceptance.add_argument("--synthetic-control", type=Path)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--graph-policy",
        choices=["eager", "runtime-default"],
        default="runtime-default",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Resolve identity without GPU access")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k not in MODELS[args.model_key]["k"]:
        parser.error("K must belong to the approved matrix")
    model = MODELS[args.model_key]
    control = None
    if args.synthetic_control is not None:
        control = json.loads(args.synthetic_control.read_text())
        accepted = control["accepted_tokens"]
        if (
            control.get("model_key") != args.model_key
            or control.get("model_revision") != model["revision"]
            or control.get("k") != args.k
            or control.get("frozen_before_measurement") is not True
            or not math.isfinite(accepted)
            or not 0 <= accepted <= args.k
        ):
            parser.error("frozen control identity or accepted mean does not match model/K")
        args.synthetic_al = accepted + 1
    elif args.synthetic_al is not None:
        if args.k != model["reference_k"] or args.synthetic_al != model["reference_al"]:
            parser.error("non-reference synthetic AL requires a frozen pilot control")
        control = {
            "accepted_tokens": args.synthetic_al - 1,
            "synthetic_acceptance_length": args.synthetic_al,
            "frozen_before_measurement": True,
            "source": f"AgentX result {model['reference_result']}",
            "producer": f"vllm/config/speculative.py::_acceptance_length_to_rates@{RUNTIME_REVISION}",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    command, env, checkpoint = server_command(
        args.model_key,
        args.k,
        args.port,
        args.role,
        args.synthetic_al,
        args.graph_policy,
        prefix_cache=args.prefix_cache,
        context_length=args.context_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    receipt_name = f"{args.role}-preflight.json" if args.preflight_only else f"{args.role}-identity.json"
    (args.output / receipt_name).write_text(
        json.dumps(
            {
                "command": command,
                "checkpoint": checkpoint,
                "graph_policy": args.graph_policy,
                "synthetic_control": control,
                "prefix_cache": args.prefix_cache,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "runtime_environment": {
                    name: env[name]
                    for name in (
                        "VLLM_NIXL_SIDE_CHANNEL_HOST",
                        "VLLM_NIXL_SIDE_CHANNEL_PORT",
                        "VLLM_SERVER_DEV_MODE",
                        "VLLM_USE_V2_MODEL_RUNNER",
                        "VLLM_USE_RUST_FRONTEND",
                    )
                    if name in env
                },
                "receipt_rpc_scope": "Private allocation host; runtime inspection before measurement.",
                "cache_receipt_extension_sha256": hashlib.sha256(
                    Path(__file__).with_name("runtime_receipt.py").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    if args.preflight_only:
        return
    with (args.output / f"{args.role}-gpu.csv").open("w") as output:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version,power.limit,clocks.sm",
                "--format=csv",
            ],
            stdout=output,
            check=True,
        )
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
