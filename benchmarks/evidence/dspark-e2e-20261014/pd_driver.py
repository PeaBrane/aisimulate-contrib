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
from window_plan import add_selection_arguments, execute_runs, select_runs

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


def selected_windows(mode, concurrencies=None, repetitions=None, windows=None):
    """Select distinct measurement windows without restarting loaded workers."""
    if windows and (concurrencies or repetitions):
        raise ValueError("explicit windows cannot be combined with concurrency/repetition filters")
    pairs = (
        [tuple(pair) for pair in windows]
        if windows
        else [
            (concurrency, repetition)
            for concurrency in (concurrencies or ([1, 4] if mode == "initial" else [1]))
            for repetition in (repetitions or ([1, 2, 3] if mode == "initial" else [1]))
        ]
    )
    if any(concurrency not in (1, 4) or repetition not in (1, 2, 3) for concurrency, repetition in pairs):
        raise ValueError("windows require concurrency 1/4 and repetition 1/2/3")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate measurement window")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--prefill-host")
    parser.add_argument("--decode-host")
    parser.add_argument("--proxy-script", type=Path)
    parser.add_argument("--mode", choices=["smoke", "pilot", "initial", "sweep"], default="smoke")
    parser.add_argument("--concurrency", type=int, choices=[1, 4, 8, 16, 64], action="append")
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], action="append")
    parser.add_argument("--window", type=int, nargs=2, action="append", metavar=("CONCURRENCY", "REPETITION"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance", choices=["real", "synthetic"])
    parser.add_argument("--startup-timeout", type=int, default=3600)
    add_selection_arguments(parser)
    args = parser.parse_args()
    if args.selection_only:
        print(json.dumps(select_runs(args, "disagg", args.acceptance or "real"), indent=2))
        return
    if not args.prefill_host or not args.decode_host or not args.proxy_script or args.startup_timeout <= 0:
        parser.error("P/D collection requires role hosts, proxy script and positive startup timeout")
    if hashlib.sha256(args.proxy_script.read_bytes()).hexdigest() != PROXY_SHA256:
        raise ValueError("NIXL proxy differs from pinned unmodified upstream source")
    prefill, decode = (
        f"http://{args.prefill_host}:8010",
        f"http://{args.decode_host}:8020",
    )
    deadline = time.monotonic() + args.startup_timeout
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
        "serving_run_id": args.output.name,
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
    if args.acceptance is not None and args.acceptance != acceptance:
        raise ValueError("selected acceptance differs from launched P/D controls")
    plan = select_runs(args, "disagg", acceptance)
    if not plan:
        raise ValueError("no uncompleted windows selected")
    if args.mode == "sweep" and any(role.get("prefix_cache") for role in identity["roles"].values()):
        raise ValueError("fixed-length sweep requires prefix caching disabled")
    identity["window_selection"] = [{"cell_id": r["cell"]["cell_id"], "repetition": r["repetition"]} for r in plan]
    identity["completed_runs_sha256"] = (
        hashlib.sha256(args.completed_runs.read_bytes()).hexdigest() if args.completed_runs else None
    )
    identity["phase"] = args.mode
    if controls[0] is not None:
        identity["synthetic_control"] = controls[0]
    if args.completed_runs:
        (args.output / "completed-runs.json").write_bytes(args.completed_runs.read_bytes())
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
            execute_runs(
                plan,
                mode=args.mode,
                output=args.output,
                source_dir=Path(__file__).parent,
                model_key=args.model_key,
                k=args.k,
                identity_path=identity_path,
                endpoint=ready["endpoint"],
                metrics_urls=ready["metrics_urls"],
                tokenizer_endpoint=prefill,
                acceptance_endpoint=decode + "/metrics",
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
