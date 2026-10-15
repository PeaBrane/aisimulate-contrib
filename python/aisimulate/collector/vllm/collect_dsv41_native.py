# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect experimental native V4.1 site observations on pinned vLLM, TP4 K5.

Original integration using public LLM.generate/collective_rpc APIs at
vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516 (Apache-2.0).
Only complete matching rank observations can produce the native-stage table.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

from collector.vllm.dsv41_native_contract import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    RUNTIME_REVISION,
    config_identity_digest,
    stage_table_rows,
)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "dsv41-native-observations-v1",
        "status": "failed",
        "import_eligible": False,
        "runtime_revision": RUNTIME_REVISION,
        "runtime_image_digest": args.image_digest,
        "target_revision": MODEL_REVISION,
        "k": 5,
        "tp": 4,
        "ep": 1,
        "graph_policy": "eager",
        "warmup_requests": 2,
        "repetitions": args.repetitions,
        "scope": "native site timing with real serving metadata; no serving accuracy claim",
        "remaining_gate": "complete native observations and matching TP-rank ownership",
    }
    llm = None
    installed = False
    try:
        config_bytes = (args.model / "config.json").read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != MODEL_CONFIG_SHA256:
            raise ValueError("checkpoint config does not match the pinned V4.1 campaign")
        receipt["config_identity_sha256"] = config_identity_digest(json.loads(config_bytes))
        tokens = json.loads(args.prompt_tokens.read_text())
        if not isinstance(tokens, list) or len(tokens) != 1024:
            raise ValueError("initial observer requires an explicit 1024-token prompt")
        if any(type(token) is not int or token < 0 or token > 2**32 - 1 for token in tokens):
            raise ValueError("prompt token IDs must be uint32 integers")
        source_hashes = json.loads(Path(__file__).with_name("dsv41_native_sources.json").read_text())
        receipt["source_hashes"] = source_hashes
        receipt["prompt_sha256"] = hashlib.sha256(args.prompt_tokens.read_bytes()).hexdigest()
        (args.output / "prompt-token-ids.json").write_text(json.dumps(tokens) + "\n")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        # Framework import is deliberately inside the executable GPU boundary;
        # CPU identity tests and --help do not require torch or vLLM installed.
        vllm = importlib.import_module("vllm")
        engine_args = {
            "model": str(args.model),
            "tensor_parallel_size": 4,
            "trust_remote_code": True,
            "language_model_only": True,
            "tokenizer_mode": "deepseek_v41",
            "kv_cache_dtype": "fp8",
            "engram_config": {"cpu_offload": False},
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "max_model_len": 16384,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "seed": 42,
            "worker_extension_cls": "collector.vllm.dsv41_native_observer.Dsv41ObserverWorker",
            "speculative_config": {
                "method": "dspark",
                "num_speculative_tokens": 5,
                "enable_adaptive_verification": False,
                "draft_sample_method": "probabilistic",
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": 3.51,
            },
        }
        receipt["engine_args"] = engine_args
        receipt["vllm_version"] = vllm.__version__
        llm = vllm.LLM(**engine_args)
        sampling = vllm.SamplingParams(temperature=0.0, max_tokens=16, min_tokens=16, ignore_eos=True, seed=42)
        prompt = {"prompt_token_ids": tokens}
        for _ in range(2):
            llm.generate([prompt], sampling, use_tqdm=False)
        receipt["worker_receipts"] = llm.collective_rpc(
            "dsv41_observer_install", args=(str(args.output / "raw"), source_hashes)
        )
        installed = True
        receipt["requests"] = []
        site_kinds = ("layer", "attention", "mhc", "engram", "finalize", "context_preparation", "input_preparation")
        for kind in site_kinds:
            for repetition in range(args.repetitions):
                sample = f"{kind}-{repetition}"
                llm.collective_rpc("dsv41_observer_begin", args=(sample, kind))
                try:
                    results = llm.generate([prompt], sampling, use_tqdm=False)
                    output_ids = list(results[0].outputs[0].token_ids)
                    receipt["requests"].append({"sample_id": sample, "output_token_ids": output_ids})
                    if len(output_ids) != 16:
                        raise ValueError("native request did not complete all 16 requested tokens")
                finally:
                    receipt["ranks"] = llm.collective_rpc("dsv41_observer_finish")
        if len(receipt["ranks"]) != 4 or {r["rank"] for r in receipt["ranks"]} != set(range(4)):
            raise ValueError("missing native TP-rank observations")
        if any(
            r["coverage"]["failure_count"]
            or r["coverage"]["missing_initial_shapes"]
            or r["coverage"]["observed_site_kinds"] != sorted(site_kinds)
            for r in receipt["ranks"]
        ):
            raise ValueError("native observation contains missing or unsupported shapes; preserve partial records")
        payloads = [{**json.loads(Path(rank["path"]).read_text()), "rank": rank["rank"]} for rank in receipt["ranks"]]
        rows = stage_table_rows(
            payloads, config_digest=receipt["config_identity_sha256"], image_digest=args.image_digest
        )
        arrow = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
        table = args.output / "dsv41_native_stage_perf.parquet"
        temporary = table.with_suffix(".tmp")
        parquet.write_table(arrow.Table.from_pylist(rows), temporary)
        temporary.replace(table)
        receipt["status"] = "collected_native_stages"
        receipt["import_eligible"] = True
        receipt["table_rows"] = len(rows)
        receipt["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
        receipt["remaining_gate"] = (
            "Exact observed queries only; mixed batches and wider replay KV coverage remain unqualified."
        )
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if installed:
                llm.collective_rpc("dsv41_observer_restore")
        finally:
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned checkpoint directory, already available")
    parser.add_argument("--prompt-tokens", type=Path, required=True, help="JSON array of exactly 1024 input IDs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory shared by four local workers")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or not args.image_digest.startswith("sha256:") or len(args.image_digest) != 71:
        parser.error("positive repetitions and immutable sha256 image digest are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
