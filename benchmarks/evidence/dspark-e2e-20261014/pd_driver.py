# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the unmodified pinned vLLM NIXL proxy to collect a controlled P/D smoke."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from campaign import IMAGE, MODELS, RUNTIME_REVISION
from collect_fixed import fetch

PROXY_SHA256 = "1210e5525cc0c57f891734b682d3f8939aa7456a672bea16b44f03b6707136ff"
PROXY_URL = f"https://raw.githubusercontent.com/vllm-project/vllm/{RUNTIME_REVISION}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"


def wait_ready(endpoint, deadline):
    while time.monotonic() < deadline:
        try:
            with fetch(endpoint, timeout=5):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10)
    raise TimeoutError(f"endpoint not ready: {endpoint}")


def load_role_receipts(root, model_key, role):
    """Validate every DP/TP worker, not only the HTTP client's first DP result."""
    dp, tp = (8, 1) if model_key == "v4-pro" and role == "prefill" else (1, MODELS[model_key]["gpus_per_role"])
    expected = {(d, t) for d in range(dp) for t in range(tp)}
    ranks = [json.loads(path.read_text()) for path in sorted(root.glob("dp-*-tp-*.json"))]
    keys = [(row["data_parallel_index"], row["tensor_parallel_rank"]) for row in ranks]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"{role} cache receipt is missing or duplicates DP/TP workers")
    for row in ranks:
        parallel = row["parallelism"]
        if (
            parallel["data_parallel_size"] != dp
            or parallel["tensor_parallel_size"] != tp
            or parallel["pipeline_parallel_size"] != 1
            or parallel["enable_expert_parallel"] != (dp == 8)
            or row["schema"] != "dspark-native-cache-v1"
            or not row["layer_specs"]
            or not row["transfer_group_ids"]
        ):
            raise ValueError(f"{role} cache receipt differs from the approved physical topology")
    return ranks


def collect_role_receipts(endpoint, root, model_key, role):
    root.mkdir(parents=True, exist_ok=False)
    # DPLBAsyncMPClient joins every engine but only returns engine zero's
    # result. The worker extension writes one unique atomic file per worker.
    with fetch(
        endpoint + "/collective_rpc",
        {"method": "dspark_campaign_receipt", "args": [str(root)], "timeout": 120},
        timeout=150,
    ) as response:
        rpc_result = json.load(response)
    (root / "rpc-result.json").write_text(json.dumps(rpc_result, indent=2) + "\n")
    return load_role_receipts(root, model_key, role)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "pilot", "initial"], default="smoke")
    parser.add_argument("--concurrency", type=int, choices=[1, 4], action="append")
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.proxy_script.read_bytes()).hexdigest() != PROXY_SHA256:
        raise ValueError("NIXL proxy differs from pinned unmodified upstream source")
    prefill, decode = (
        f"http://{args.prefill_host}:8010",
        f"http://{args.decode_host}:8020",
    )
    deadline = time.monotonic() + 3600
    for endpoint in [prefill, decode]:
        wait_ready(endpoint + "/health", deadline)
    role_receipts = {
        role: collect_role_receipts(endpoint, args.output / "cache-receipts" / role, args.model_key, role)
        for role, endpoint in (("prefill", prefill), ("decode", decode))
    }
    cache_path = args.output / "effective-cache.json"
    cache_bytes = (json.dumps(role_receipts, indent=2, sort_keys=True) + "\n").encode()
    cache_path.write_bytes(cache_bytes)
    identity = {
        "model": MODELS[args.model_key],
        "image": IMAGE,
        "runtime_revision": RUNTIME_REVISION,
        "deployment": "disagg",
        "proxy_url": PROXY_URL,
        "proxy_sha256": PROXY_SHA256,
        "runtime_cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "transfer_accounting": {
            "simulator_bytes": "rank-local logical target KV plus draft state",
            "bandwidth_scope": "effective bandwidth per modeled handoff, not total wire throughput",
            "physical_bytes": "Physical page/group sizes retained per worker; request wire bytes are unmeasured.",
            "pro_mla_fanout": (
                "Each of 8 D TP ranks pulls replicated MLA from rank 0 of the selected P DP engine. "
                "Parallel copies do not imply 8x latency; shared-source NIC contention is unmodeled."
                if args.model_key == "v4-pro"
                else None
            ),
        },
        "roles": {
            role: json.loads((args.output / f"{role}-identity.json").read_text()) for role in ["prefill", "decode"]
        },
        "qualification": {
            "eligible": False,
            "reasons": ["P/D connector smoke; transfer and replay qualification pending"],
        },
    }
    controls = [role.get("synthetic_control") for role in identity["roles"].values()]
    if controls[0] != controls[1]:
        raise ValueError("P/D acceptance controls must match")
    acceptance = "synthetic" if controls[0] is not None else "real"
    if args.mode == "pilot" and acceptance != "real":
        raise ValueError("pilots require real acceptance")
    identity["phase"] = args.mode
    if controls[0] is not None:
        identity["synthetic_control"] = controls[0]
    identity_path = args.output / "identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    proxy_command = [
        sys.executable,
        str(args.proxy_script),
        "--host",
        "127.0.0.1",
        "--port",
        "8030",
        "--prefiller-hosts",
        args.prefill_host,
        "--prefiller-ports",
        "8010",
        "--decoder-hosts",
        args.decode_host,
        "--decoder-ports",
        "8020",
    ]
    with (args.output / "proxy.log").open("w") as output:
        proxy = subprocess.Popen(proxy_command, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_ready("http://127.0.0.1:8030/healthcheck", time.monotonic() + 120)
            ready = {
                "identity_path": str(identity_path),
                "effective_config_path": str(cache_path),
                "cache_capacity_by_role_rank": role_receipts,
                "log_paths": [str(args.output / f"{role}.log") for role in ["prefill", "decode"]],
                "cell_id": f"{args.model_key}-disagg-k{args.k}-i1024-o1024-c1-{acceptance}",
                "endpoint": "http://127.0.0.1:8030",
                "metrics_urls": [prefill + "/metrics", decode + "/metrics"],
            }
            (args.output / "server-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            concurrencies = args.concurrency or ([1, 4] if args.mode == "initial" else [1])
            repetitions = args.repetition or ([1, 2, 3] if args.mode == "initial" else [1])
            for concurrency in concurrencies:
                for repetition in repetitions:
                    cell_id = f"{args.model_key}-disagg-k{args.k}-i1024-o1024-c{concurrency}-{acceptance}"
                    run_output = args.output / f"{args.mode}-c{concurrency}-r{repetition}"
                    client = [
                        sys.executable,
                        str(Path(__file__).with_name("collect_fixed.py")),
                        "--matrix",
                        str(Path(__file__).with_name("matrix.json")),
                        "--cell-id",
                        cell_id,
                        "--identity",
                        str(identity_path),
                        "--endpoint",
                        ready["endpoint"],
                        "--tokenizer-endpoint",
                        prefill,
                        "--metrics-url",
                        prefill + "/metrics",
                        "--metrics-url",
                        decode + "/metrics",
                        "--model",
                        MODELS[args.model_key]["model"],
                        "--output",
                        str(run_output),
                        "--repetition",
                        str(repetition),
                    ]
                    if args.mode == "smoke":
                        client.append("--smoke")
                    subprocess.run(client, check=True)
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("acceptance.py")),
                            "--run",
                            str(run_output),
                            "--k",
                            str(args.k),
                            "--endpoint",
                            decode + "/metrics",
                        ],
                        check=True,
                    )
        finally:
            proxy.terminate()
            try:
                proxy.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()


if __name__ == "__main__":
    main()
