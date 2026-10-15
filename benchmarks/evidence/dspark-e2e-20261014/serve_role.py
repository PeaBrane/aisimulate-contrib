# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one P/D role inside its assigned Slurm step, with identical DSpark settings."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from campaign import MODELS
from serve_smoke import server_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--synthetic-al", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k not in MODELS[args.model_key]["k"]:
        parser.error("K must belong to the approved matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    command, env, checkpoint = server_command(args.model_key, args.k, args.port, args.role, args.synthetic_al)
    (args.output / f"{args.role}-identity.json").write_text(
        json.dumps({"command": command, "checkpoint": checkpoint}, indent=2) + "\n"
    )
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
