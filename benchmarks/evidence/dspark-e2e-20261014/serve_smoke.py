# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch an aggregated DSpark smoke, acceptance pilot, or initial fixed cells."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from campaign import IMAGE, MODELS, RUNTIME_REVISION
from collect_fixed import fetch
from inspect_runtime import checkpoint_path, inspect_checkpoint


def server_command(model_key, k, port, role="agg", synthetic_al=None, graph_policy="eager"):
    model = MODELS[model_key]
    checkpoint = inspect_checkpoint(model["model"], model["revision"])
    speculation = {
        "method": "dspark",
        "num_speculative_tokens": k,
        "enable_adaptive_verification": False,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block",
    }
    if synthetic_al is not None:
        if not 1 <= synthetic_al <= k + 1:
            raise ValueError("synthetic AL must include bonus and be within [1, K+1]")
        speculation["rejection_sample_method"] = "synthetic"
        speculation["synthetic_acceptance_length"] = synthetic_al
    if "draft_model" in model:
        speculation["model"] = str(checkpoint_path(model["draft_model"], model["draft_revision"]))
        speculation["attention_backend"] = "FLASHINFER_MLA"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        checkpoint["path"],
        "--served-model-name",
        model["model"],
        "--tensor-parallel-size",
        str(model["gpus_per_role"]),
        "--trust-remote-code",
        "--language-model-only",
        "--kv-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
        "--seed",
        "42",
        "--speculative-config",
        json.dumps(speculation),
    ]
    if graph_policy == "eager":
        command.append("--enforce-eager")
    if model_key == "v4-pro":
        command += ["--attention-backend", "FLASHMLA_SPARSE_DSV4"]
    if model_key == "kimi-k3":
        command += ["--attention-config", '{"use_prefill_query_quantization":true}']
    if model_key == "v41-flash":
        command += [
            "--tokenizer-mode",
            "deepseek_v41",
            "--engram-config",
            '{"cpu_offload":false}',
        ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_RUST_FRONTEND": "0",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    }
    if role == "agg":
        command += ["--host", "127.0.0.1", "--worker-extension-cls", "runtime_receipt.DsparkReceiptWorker"]
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if role != "agg":
        command += [
            "--kv-transfer-config",
            json.dumps({"kv_connector": "NixlConnector", "kv_role": "kv_both"}),
        ]
        if role == "prefill" and model_key == "v4-pro":
            command[command.index("--tensor-parallel-size") + 1] = "1"
            command += ["--data-parallel-size", "8", "--enable-expert-parallel"]
    return command, env, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=list(MODELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mode", choices=["smoke", "pilot", "initial"], default="smoke")
    parser.add_argument("--graph-policy", choices=["eager", "runtime-default"], default="eager")
    parser.add_argument("--concurrency", type=int, choices=[1, 4], action="append")
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], action="append")
    acceptance = parser.add_mutually_exclusive_group()
    acceptance.add_argument("--synthetic-al", type=float)
    acceptance.add_argument("--synthetic-control", type=Path)
    args = parser.parse_args()
    model = MODELS[args.model_key]
    if args.k not in model["k"]:
        parser.error("K must be in the approved matrix")
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
        args.synthetic_al = 1 + accepted
    if args.mode == "pilot" and args.synthetic_al is not None:
        parser.error("pilots must measure real acceptance")
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output / "source"
    source_dir.mkdir()
    source_hashes = {}
    for name in (
        "serve_smoke.py",
        "campaign.py",
        "collect_fixed.py",
        "acceptance.py",
        "runtime_receipt.py",
        "inspect_runtime.py",
        "matrix.json",
    ):
        source = Path(__file__).with_name(name)
        shutil.copy2(source, source_dir / name)
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    command, env, checkpoint = server_command(
        args.model_key, args.k, args.port, synthetic_al=args.synthetic_al, graph_policy=args.graph_policy
    )
    env["PYTHONPATH"] = str(source_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    identity = {
        "model": model,
        "checkpoint": checkpoint,
        "image": IMAGE,
        "runtime_revision": RUNTIME_REVISION,
        "server_command": command,
        "launcher_source_sha256": source_hashes["serve_smoke.py"],
        "source_hashes": source_hashes,
        "runtime_environment": {
            name: env[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_USE_RUST_FRONTEND",
                "VLLM_SERVER_DEV_MODE",
            )
        },
        "graph_policy": args.graph_policy,
        "prefix_cache": False,
        "external_kv_offload": False,
        "engram_cpu_offload": False,
        "phase": args.mode,
        "qualification": {
            "eligible": False,
            "reasons": ["effective cache and runtime qualification pending"],
        },
    }
    if control is not None:
        identity["synthetic_control"] = control
    elif args.synthetic_al is not None:
        if args.k != model["reference_k"] or args.synthetic_al != model["reference_al"]:
            parser.error("non-reference synthetic AL needs a separately frozen pilot control")
        identity["synthetic_control"] = {
            "accepted_tokens": args.synthetic_al - 1,
            "synthetic_acceptance_length": args.synthetic_al,
            "frozen_before_measurement": True,
            "source": f"AgentX result {model['reference_result']}",
            "producer": f"vllm/config/speculative.py::_acceptance_length_to_rates@{RUNTIME_REVISION}",
        }
    identity_path = args.output / "identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version,power.limit,clocks.sm",
            "--format=csv",
        ],
        stdout=(args.output / "gpu.csv").open("w"),
        check=True,
    )
    status = {"status": "starting", "model_key": args.model_key, "k": args.k}
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 3600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup exceeded one hour")
                try:
                    with fetch(f"http://127.0.0.1:{args.port}/health", timeout=5):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(10)
            with fetch(
                f"http://127.0.0.1:{args.port}/collective_rpc",
                {"method": "dspark_campaign_receipt", "timeout": 120},
                timeout=150,
            ) as response:
                cache_receipt = json.load(response)
            ranks = cache_receipt["results"]
            if len(ranks) != model["gpus_per_role"] or {r["rank"] for r in ranks} != set(range(model["gpus_per_role"])):
                raise ValueError("runtime cache receipt is missing TP ranks")
            cache_path = args.output / "effective-cache.json"
            cache_bytes = (json.dumps(cache_receipt, indent=2, sort_keys=True) + "\n").encode()
            cache_path.write_bytes(cache_bytes)
            identity["runtime_cache_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
            identity["cache_receipt_extension_sha256"] = hashlib.sha256(
                (source_dir / "runtime_receipt.py").read_bytes()
            ).hexdigest()
            identity_path.write_text(json.dumps(identity, indent=2) + "\n")
            acceptance = "synthetic" if args.synthetic_al is not None else "real"
            ready = {
                "identity_path": str(identity_path),
                "effective_config_path": str(cache_path),
                "log_paths": [str(args.output / "server.log")],
                "cache_capacity_by_role_rank": {"agg": ranks},
                "model_config_paths": [str(Path(checkpoint["path"]) / "config.json")],
                "cell_id": f"{args.model_key}-agg-k{args.k}-i1024-o1024-c1-{acceptance}",
                "endpoint": f"http://127.0.0.1:{args.port}",
                "metrics_urls": [f"http://127.0.0.1:{args.port}/metrics"],
                "qualification_note": (
                    "resolved cache capacity and actual attention geometry must be read from server log before pairing"
                ),
            }
            (args.output / "server-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            concurrencies = args.concurrency or ([1, 4] if args.mode == "initial" else [1])
            repetitions = args.repetition or ([1, 2, 3] if args.mode == "initial" else [1])
            for concurrency in concurrencies:
                for repetition in repetitions:
                    cell_id = f"{args.model_key}-agg-k{args.k}-i1024-o1024-c{concurrency}-{acceptance}"
                    run_output = args.output / f"{args.mode}-c{concurrency}-r{repetition}"
                    client = [
                        sys.executable,
                        str(source_dir / "collect_fixed.py"),
                        "--matrix",
                        str(source_dir / "matrix.json"),
                        "--cell-id",
                        cell_id,
                        "--identity",
                        str(identity_path),
                        "--endpoint",
                        ready["endpoint"],
                        "--metrics-url",
                        ready["metrics_urls"][0],
                        "--model",
                        model["model"],
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
                            str(source_dir / "acceptance.py"),
                            "--run",
                            str(run_output),
                            "--k",
                            str(args.k),
                        ],
                        check=True,
                    )
            status["status"] = "success"
        except Exception as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
