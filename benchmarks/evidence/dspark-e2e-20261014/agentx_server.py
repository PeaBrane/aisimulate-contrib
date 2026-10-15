# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect identity from existing prefix-enabled campaign servers; no GPU allocation."""

import argparse
import json
from pathlib import Path

from agentx_phaseb import digest, read_json
from campaign import IMAGE, MODELS, RUNTIME_REVISION
from collect_fixed import fetch
from pd_driver import collect_role_receipts


def validate_role(identity, model_key, k, role):
    command = identity["command"]
    speculation = json.loads(command[command.index("--speculative-config") + 1])
    if (
        identity.get("prefix_cache") is not True
        or "--enable-prefix-caching" not in command
        or "--no-enable-prefix-caching" in command
        or command[command.index("--served-model-name") + 1] != MODELS[model_key]["model"]
        or speculation.get("method") != "dspark"
        or speculation.get("num_speculative_tokens") != k
        or speculation.get("enable_adaptive_verification") is not False
        or speculation.get("rejection_sample_method") != "block"
        or identity.get("graph_policy") != "runtime-default"
    ):
        raise ValueError(
            "server must use matching fixed DSpark, real acceptance, prefix cache and runtime graph policy"
        )
    tp = 1 if model_key == "v4-pro" and role == "prefill" else MODELS[model_key]["gpus_per_role"]
    if command[command.index("--tensor-parallel-size") + 1] != str(tp):
        raise ValueError("server role tensor parallel size differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--deployment", choices=("agg", "disagg"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--role",
        action="append",
        nargs=3,
        metavar=("ROLE", "ENDPOINT", "IDENTITY"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roles = {role: (endpoint, Path(identity)) for role, endpoint, identity in args.role}
    if len(roles) != len(args.role) or set(roles) != ({"agg"} if args.deployment == "agg" else {"prefill", "decode"}):
        parser.error("provide exactly the expected serving roles")
    if args.k not in MODELS[args.model_key]["k"]:
        parser.error("K is outside the approved matrix")
    for role, (_, path) in roles.items():
        validate_role(read_json(path), args.model_key, args.k, role)
    limits = {
        int(read_json(path)["command"][read_json(path)["command"].index("--max-model-len") + 1])
        for _, path in roles.values()
    }
    if len(limits) != 1:
        raise ValueError("serving roles have different context limits")
    args.output.mkdir(parents=True, exist_ok=False)
    cache, launch = [], []
    for role, (endpoint, path) in roles.items():
        if role == "agg":
            with fetch(
                endpoint + "/collective_rpc",
                {"method": "dspark_campaign_receipt", "timeout": 120},
                timeout=150,
            ) as response:
                ranks = json.load(response)["results"]
            expected = MODELS[args.model_key]["gpus_per_role"]
            if len(ranks) != expected or {r["rank"] for r in ranks} != set(range(expected)):
                raise ValueError("aggregated cache receipt omits or duplicates TP workers")
        else:
            ranks = collect_role_receipts(endpoint, args.output.resolve() / role, args.model_key, role)
        destination = args.output / f"{role}-cache.json"
        destination.write_text(json.dumps(ranks, indent=2) + "\n")
        cache.append({"path": destination.name, "sha256": digest(destination)})
        copied = args.output / f"{role}-launch.json"
        copied.write_bytes(path.read_bytes())
        launch.append({"path": copied.name, "sha256": digest(copied)})
    identity = {
        "schema": "dspark-agentx-server-v1",
        "model": MODELS[args.model_key],
        "deployment": args.deployment,
        "k": args.k,
        "image": IMAGE,
        "runtime_revision": RUNTIME_REVISION,
        "max_model_len": limits.pop(),
        "prefix_cache": True,
        "adaptive_verification": False,
        "external_kv_offload": False,
        "cache_receipts": cache,
        "launch_receipts": launch,
        "scope": (
            "Declared pinned launcher identity and read-only native cache inspection; "
            "retain allocation/image receipts separately."
        ),
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
    main()
