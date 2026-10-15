# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect native Kimi MLA for C1/C4, prefill1024/verify3/draft2.

Original vLLM public API integration at ac9126e58aa7bbab1856ba6593ba4d5003fea516.
No synthetic framework metadata, backend substitution or legacy MLA table reuse.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

from collector.vllm.kimi_native_contract import (
    DRAFT_CONFIG_SHA256,
    RUNTIME_REVISION,
    TARGET_CONFIG_SHA256,
    table_rows,
)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "runtime_revision": RUNTIME_REVISION,
        "image_digest": args.image_digest,
        "status": "failed",
        "scope": "native MLA operation measurement, not E2E accuracy",
        "tp": 8,
        "ep": 1,
        "k": 2,
        "warmup_requests": 2,
        "repetitions": args.repetitions,
        "output_tokens": args.output_tokens,
        "sequence_counts": args.sequence_counts,
        "target_config_sha256": TARGET_CONFIG_SHA256,
        "draft_config_sha256": DRAFT_CONFIG_SHA256,
        "precision_policy": {
            "projection_weights_and_outputs": "bfloat16",
            "prefill_query_after_projection": "fp8",
            "kv_cache": "fp8",
            "prefill_query_quantization": True,
        },
    }
    llm, installed = None, False
    try:
        for path, expected in ((args.model, TARGET_CONFIG_SHA256), (args.draft, DRAFT_CONFIG_SHA256)):
            if hashlib.sha256((path / "config.json").read_bytes()).hexdigest() != expected:
                raise ValueError(f"checkpoint config does not match pinned identity: {path}")
        tokens = json.loads(args.prompt_tokens.read_text())
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1024
            or any(type(t) is not int or t < 0 or t >= 163840 for t in tokens)
        ):
            raise ValueError("explicit 1024-token Kimi prompt required")
        sources = json.loads(Path(__file__).with_name("kimi_native_sources.json").read_text())
        receipt["source_hashes"] = sources
        receipt["prompt_sha256"] = hashlib.sha256(args.prompt_tokens.read_bytes()).hexdigest()
        (args.output / "prompt-token-ids.json").write_text(json.dumps(tokens) + "\n")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] = "6000"
        receipt["startup_timeout_seconds"] = 6000
        # Framework imports occur only at the GPU executable boundary.
        vllm = importlib.import_module("vllm")
        engine_args = {
            "model": str(args.model),
            "tensor_parallel_size": 8,
            "trust_remote_code": True,
            "language_model_only": True,
            "safetensors_load_strategy": "eager",
            "skip_tokenizer_init": True,
            "kv_cache_dtype": "fp8",
            "attention_config": {"use_prefill_query_quantization": True},
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "max_model_len": 16384,
            "max_num_seqs": max(args.sequence_counts),
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "seed": 42,
            "worker_extension_cls": "collector.vllm.kimi_native_observer.KimiObserverWorker",
            "speculative_config": {
                "model": str(args.draft),
                "method": "dspark",
                "num_speculative_tokens": 2,
                "enable_adaptive_verification": False,
                "draft_sample_method": "probabilistic",
                "rejection_sample_method": "synthetic",
                "synthetic_acceptance_length": 3.0,
            },
        }
        receipt["engine_args"], receipt["vllm_version"] = engine_args, vllm.__version__
        llm = vllm.LLM(**engine_args)
        params = vllm.SamplingParams(
            temperature=0, max_tokens=args.output_tokens, min_tokens=args.output_tokens, ignore_eos=True, seed=42
        )
        warmup_params = vllm.SamplingParams(temperature=0, max_tokens=16, min_tokens=16, ignore_eos=True, seed=42)
        prompt = {"prompt_token_ids": tokens}
        for _ in range(2):
            llm.generate([prompt] * max(args.sequence_counts), warmup_params, use_tqdm=False)
        receipt["worker_receipts"] = llm.collective_rpc(
            "kimi_observer_install", args=(str(args.output / "raw"), sources)
        )
        installed = True
        receipt["requests"] = []
        for repetition in range(args.repetitions):
            for count in args.sequence_counts:
                llm.collective_rpc("kimi_observer_begin", args=(f"{repetition}-c{count}",))
                try:
                    result = llm.generate([prompt] * count, params, use_tqdm=False)
                    ids = [list(r.outputs[0].token_ids) for r in result]
                    receipt["requests"].append({"sequence_count": count, "token_ids": ids})
                    if len(ids) != count or any(len(tokens) != args.output_tokens for tokens in ids):
                        raise ValueError("native serving did not complete requested output length")
                finally:
                    receipt["ranks"] = llm.collective_rpc("kimi_observer_finish")
        payloads = [json.loads(Path(r["path"]).read_text()) for r in receipt["ranks"]]
        rows = table_rows(payloads, args.sequence_counts)
        for row in rows:
            row["image_digest"] = args.image_digest
        pandas = importlib.import_module("pandas")
        pandas.DataFrame(rows).to_parquet(args.output / "mla_native_module_perf.parquet", index=False)
        receipt["status"], receipt["physical_rows"] = "collected", len(rows)
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if installed:
                llm.collective_rpc("kimi_observer_restore")
        finally:
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "draft", "prompt-tokens", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--sequence-counts", type=int, nargs="+", choices=(1, 4), default=[1, 4])
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=1056,
        help="Operator-only probe; 32-token margin covers the 1024-token E2E window",
    )
    args = parser.parse_args()
    if (
        args.repetitions < 1
        or args.output_tokens < 16
        or args.output_tokens > 15360
        or not args.image_digest.startswith("sha256:")
        or len(args.image_digest) != 71
    ):
        parser.error("positive repetitions and immutable image digest required")
    run(args)


if __name__ == "__main__":
    main()
