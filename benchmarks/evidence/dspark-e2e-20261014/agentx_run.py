# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run gated AgentX clients against assigned roles, retaining the pinned NIXL proxy."""

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

from agentx_phaseb import digest, fixed_gate, profile_gate
from campaign import MODELS
from pd_driver import PROXY_SHA256, wait_ready


def client_windows(phase, selected=None):
    if selected is not None:
        allowed = client_windows(phase)
        if len(selected) != len(set(selected)) or any(window not in allowed for window in selected):
            raise ValueError("invalid or duplicate AgentX window")
        return selected
    return [
        (c, r) for c in ((1, 4) if phase == "pilot" else (1, 4, 16)) for r in ((1,) if phase == "pilot" else (1, 2, 3))
    ]


def client_command(args, concurrency, repetition, endpoint, metrics):
    command = [
        sys.executable,
        str(Path(__file__).with_name("agentx_phaseb.py")),
        "--model-key",
        args.model_key,
        "--deployment",
        args.deployment,
        "--k",
        str(MODELS[args.model_key]["reference_k"]),
        "--acceptance",
        "synthetic",
        "--phase",
        args.phase,
        "--concurrency",
        str(concurrency),
        "--repetition",
        str(repetition),
        "--collection-only",
        "--server-identity",
        str(args.output / "server" / "identity.json"),
        "--context-audit",
        str(args.context_audit),
        "--producer-checkout",
        str(args.producer_checkout),
        "--aiperf",
        str(args.aiperf),
        "--endpoint",
        endpoint,
        "--output",
        str(args.output / f"{args.phase}-c{concurrency}-r{repetition}"),
        "--execute",
    ]
    for anchor in args.fixed_measurement:
        command += ["--fixed-measurement", str(anchor)]
    for check in args.pilot_check:
        command += ["--pilot-check", str(check)]
    for url in metrics:
        command += ["--metrics-url", url]
    return command


def preflight(args):
    k = MODELS[args.model_key]["reference_k"]
    fixed_gate(
        [(path, None) for path in args.fixed_measurement],
        args.model_key,
        args.deployment,
        k,
        required_concurrencies=(1, 4)
        if args.phase == "profile" or not args.window
        else tuple(w[0] for w in args.window),
        collection_only=True,
        acceptance="synthetic",
    )
    if args.phase == "profile":
        profile_gate(
            args.pilot_check, args.model_key, args.deployment, k, max_model_len=1048576, acceptance="synthetic"
        )
    if args.deployment == "disagg" and (
        not args.decode_host or args.proxy_script is None or digest(args.proxy_script) != PROXY_SHA256
    ):
        raise ValueError("P/D requires the decode host and unmodified pinned proxy")
    if not args.context_audit.is_file() or not args.aiperf.is_file():
        raise ValueError("native formatter context receipt and installed producer are required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--deployment", choices=("agg", "disagg"), required=True)
    parser.add_argument("--phase", choices=("pilot", "profile"), required=True)
    parser.add_argument("--window", nargs=2, type=int, action="append", metavar=("CONCURRENCY", "REPETITION"))
    parser.add_argument("--prefill-host", default="127.0.0.1")
    parser.add_argument("--decode-host")
    parser.add_argument("--proxy-script", type=Path)
    parser.add_argument("--proxy-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--fixed-measurement", type=Path, action="append", required=True)
    parser.add_argument("--pilot-check", type=Path, action="append", default=[])
    parser.add_argument("--context-audit", type=Path, required=True)
    parser.add_argument("--producer-checkout", type=Path, required=True)
    parser.add_argument("--aiperf", type=Path, required=True)
    parser.add_argument("--roles-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=6000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    windows = client_windows(args.phase, [tuple(w) for w in args.window] if args.window else None)
    preflight(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "native_client_prerequisites_present",
                    "windows": windows,
                    "scope": "Runtime readiness and server identity are checked again before each client.",
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    prefill = "http://127.0.0.1:8010" if args.deployment == "agg" else f"http://{args.prefill_host}:8010"
    roles = [("agg" if args.deployment == "agg" else "prefill", prefill)]
    if args.deployment == "disagg":
        roles.append(("decode", f"http://{args.decode_host}:8020"))
    deadline = time.monotonic() + args.startup_timeout
    for _, endpoint in roles:
        wait_ready(endpoint + "/health", deadline)
    command = [
        sys.executable,
        str(Path(__file__).with_name("agentx_server.py")),
        "--model-key",
        args.model_key,
        "--deployment",
        args.deployment,
        "--k",
        str(MODELS[args.model_key]["reference_k"]),
        "--acceptance",
        "synthetic",
        "--output",
        str(args.output / "server"),
    ]
    for role, endpoint in roles:
        command += ["--role", role, endpoint, str(args.roles_output / f"{role}-identity.json")]
    subprocess.run(command, check=True)
    proxy = None
    state = {"schema": "dspark-agentx-client-windows-v1", "windows": []}
    try:
        with contextlib.ExitStack() as stack:
            endpoint = prefill
            if args.deployment == "disagg":
                command = [
                    str(args.proxy_python),
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
                log = stack.enter_context((args.output / "proxy.log").open("w"))
                proxy = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["proxy"] = {"command": command, "source_sha256": digest(args.proxy_script)}
                endpoint = "http://127.0.0.1:8030"
                wait_ready(endpoint + "/healthcheck", time.monotonic() + 120)
            for concurrency, repetition in windows:
                command = client_command(
                    args, concurrency, repetition, endpoint, [url + "/metrics" for _, url in roles]
                )
                row = {"concurrency": concurrency, "repetition": repetition, "command": command, "status": "running"}
                state["windows"].append(row)
                (args.output / "windows.json").write_text(json.dumps(state, indent=2) + "\n")
                result = subprocess.run(command, check=False)
                row.update(
                    status="completed_client_run" if result.returncode == 0 else "failed", returncode=result.returncode
                )
                if result.returncode:
                    raise RuntimeError(
                        f"AgentX client failed at C{concurrency}/r{repetition}; retain its raw artifacts"
                    )
    finally:
        (args.output / "windows.json").write_text(json.dumps(state, indent=2) + "\n")
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()


if __name__ == "__main__":
    main()
