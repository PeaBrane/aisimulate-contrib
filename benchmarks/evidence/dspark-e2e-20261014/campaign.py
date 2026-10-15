# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the approved DSpark collection matrix without submitting jobs."""

import argparse
import itertools
import json
from pathlib import Path

MODELS = {
    "v4-pro": {
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "revision": "72e1d3230f6c080a530b0a1d46f8eb4602340597",
        "hardware": "b200",
        "gpus_per_role": 8,
        "k": [3, 6],
        "reference_k": 6,
        "reference_al": 3.77,
        "reference_result": "442741",
        "reference_run": "35434571418/attempts/2",
    },
    "v41-flash": {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": "5254448103b6254fc646dacbca1cbbf20e9216c1",
        "hardware": "b200",
        "gpus_per_role": 4,
        "k": [3, 5],
        "reference_k": 5,
        "reference_al": 3.51,
        "reference_result": "444090",
        "reference_run": "37070984585/attempts/1",
        "reference_engram_weight_cpu_offload": True,
        "engram_weight_cpu_offload": False,
    },
    "kimi-k3": {
        "model": "moonshotai/Kimi-K3",
        "revision": "f831ab66814297da540d832a5235f8e904f29d06",
        "draft_model": "Inferact/Kimi-K3-DSpark",
        "draft_revision": "cf6b8244620e7ea4b0651d214f28e89eac75bed6",
        "hardware": "b300",
        "gpus_per_role": 8,
        "k": [2, 3],
        "reference_k": 2,
        "reference_al": 2.51,
        "reference_result": "439539",
        "reference_run": "31509160691/attempts/2",
    },
}
IMAGE = (
    "vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7"
    "@sha256:3d17635c5aa0340e2f2502c067ce4b587d8e7dc678195e90b71142490dc6d71c"
)
RUNTIME_REVISION = "ac9126e58aa7bbab1856ba6593ba4d5003fea516"


def cells():
    rows = []
    for key, model in MODELS.items():
        for deployment, k, isl, acceptance in itertools.product(
            ["agg", "disagg"], model["k"], [1024, 8192], ["synthetic", "real"]
        ):
            concurrencies = [1, 4, 16, 64]
            if key == "v4-pro" and deployment == "disagg" and isl == 8192:
                concurrencies += [8]
            for concurrency in sorted(concurrencies):
                width = model["gpus_per_role"]
                tp = 1 if key == "v4-pro" and deployment == "disagg" else width
                rows.append(
                    {
                        "cell_id": f"{key}-{deployment}-k{k}-i{isl}-o1024-c{concurrency}-{acceptance}",
                        "model_key": key,
                        "deployment": deployment,
                        "hardware": model["hardware"],
                        "num_speculative_tokens": k,
                        "input_length": isl,
                        "output_length": 1024,
                        "concurrency": concurrency,
                        "acceptance": acceptance,
                        "prefill": {
                            "tp": tp,
                            "attention_dp": width if tp == 1 else 1,
                            "ep": width if tp == 1 else 1,
                        },
                        "decode": {"tp": width, "attention_dp": 1, "ep": 1},
                        "total_gpus": width * (2 if deployment == "disagg" else 1),
                        "repetitions": 3,
                        "expected_requests": max(128, 10 * concurrency),
                        "warmup_requests": max(8, concurrency),
                        "stage": "initial" if isl == 1024 and concurrency in [1, 4] else "expanded",
                        "status": "pending_runtime_qualification",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = cells()
    payload = {
        "schema": "dspark-collection-matrix-v1",
        "models": MODELS,
        "candidate_image": IMAGE,
        "candidate_runtime_revision": RUNTIME_REVISION,
        "runtime_replacement": (
            "New controlled runtime for Pro/Kimi; historical Kimi image tag returns 404. "
            "Flash reference runtime retained."
        ),
        "adaptive_verification": False,
        "external_kv_offload": False,
        "prefix_cache": False,
        "repetitions": 3,
        "metrics_contract": "dspark-e2e-client-v1",
        "counts": {
            "configurations": len(rows),
            "initial_configurations": sum(r["stage"] == "initial" for r in rows),
            "runs": sum(r["repetitions"] for r in rows),
            "measured_requests": sum(r["repetitions"] * r["expected_requests"] for r in rows),
        },
        "cells": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["counts"]))


if __name__ == "__main__":
    main()
