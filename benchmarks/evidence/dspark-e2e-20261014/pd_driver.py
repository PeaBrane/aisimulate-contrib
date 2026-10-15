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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--prefill-host", required=True)
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--proxy-script", type=Path, required=True)
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
    identity = {
        "model": MODELS[args.model_key],
        "image": IMAGE,
        "runtime_revision": RUNTIME_REVISION,
        "deployment": "disagg",
        "proxy_url": PROXY_URL,
        "proxy_sha256": PROXY_SHA256,
        "roles": {
            role: json.loads((args.output / f"{role}-identity.json").read_text()) for role in ["prefill", "decode"]
        },
        "qualification": {
            "eligible": False,
            "reasons": ["P/D connector smoke; transfer and replay qualification pending"],
        },
    }
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
                "effective_config_path": str(identity_path),
                "cache_capacity_by_role_rank": None,
                "log_paths": [str(args.output / f"{role}.log") for role in ["prefill", "decode"]],
                "cell_id": f"{args.model_key}-disagg-k{args.k}-i1024-o1024-c1-real",
                "endpoint": "http://127.0.0.1:8030",
                "metrics_urls": [prefill + "/metrics", decode + "/metrics"],
            }
            (args.output / "server-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("collect_fixed.py")),
                    "--matrix",
                    str(Path(__file__).with_name("matrix.json")),
                    "--cell-id",
                    ready["cell_id"],
                    "--identity",
                    str(identity_path),
                    "--endpoint",
                    ready["endpoint"],
                    "--tokenizer-endpoint",
                    prefill,
                    "--metrics-url",
                    ready["metrics_urls"][0],
                    "--metrics-url",
                    ready["metrics_urls"][1],
                    "--model",
                    MODELS[args.model_key]["model"],
                    "--output",
                    str(args.output / "smoke"),
                    "--repetition",
                    "1",
                    "--smoke",
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("acceptance.py")),
                    "--run",
                    str(args.output / "smoke"),
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
