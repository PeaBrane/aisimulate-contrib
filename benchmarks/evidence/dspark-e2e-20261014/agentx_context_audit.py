# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify pinned corpus bytes and inspect every source request without GPU work.

Original format consumer. Source contract: SemiAnalysisAI/aiperf commit
754356e9a39acc6cc6afb242d123bb57c3fb6f75, dataset/loader/weka_trace_models.py
and weka_trace.py (_trace_peak_context_length). See COLLECTION.md.
"""

import argparse
import json
from pathlib import Path

from agentx_phaseb import CORPUS, digest
from campaign import MODELS


def inspect_requests(requests, state):
    for request in requests:
        if request["type"] == "subagent":
            state["subagent_groups"] += 1
            inspect_requests(request["requests"], state)
            continue
        if request["type"] not in ("n", "s"):
            raise ValueError("unknown corpus request type")
        inp, out = request["in"], max(request["out"], 1)
        if type(inp) is not int or inp < 0 or type(out) is not int:
            raise ValueError("invalid request token bounds")
        state["requests"] += 1
        state["input_tokens"] += inp
        state["recorded_output_tokens"] += request["out"]
        state["max_input"] = max(state["max_input"], inp)
        state["max_requested_output"] = max(state["max_requested_output"], out)
        state["max_input_plus_requested_output"] = max(state["max_input_plus_requested_output"], inp + out)
        state["over_16384"] += inp + out > 16384
        state["over_1048576"] += inp + out > 1048576


def audit(path):
    if path.stat().st_size != CORPUS["bytes"] or digest(path) != CORPUS["sha256"]:
        raise ValueError("corpus bytes do not match the immutable LFS content identity")
    state = dict.fromkeys(
        (
            "traces",
            "requests",
            "subagent_groups",
            "input_tokens",
            "recorded_output_tokens",
            "max_input",
            "max_requested_output",
            "max_input_plus_requested_output",
            "over_16384",
            "over_1048576",
        ),
        0,
    )
    trace_ids = set()
    with path.open() as stream:
        for line in stream:
            trace = json.loads(line)
            if trace["id"] in trace_ids or trace["hash_id_scope"] != "local" or trace["block_size"] != 64:
                raise ValueError("duplicate trace or unsupported hash-block identity")
            trace_ids.add(trace["id"])
            state["traces"] += 1
            inspect_requests(trace["requests"], state)
    if state["traces"] != CORPUS["rows"]:
        raise ValueError("incomplete corpus")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-config", nargs=2, action="append", metavar=("MODEL_KEY", "CONFIG"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = audit(args.corpus)
    models = {}
    for key, config_path in args.model_config:
        model = MODELS[key]
        path = Path(config_path)
        config = json.loads(path.read_text())
        text = config.get("text_config", config)
        reference = next(
            r
            for r in json.loads(Path(__file__).with_name("checkpoint-references.json").read_text())
            if r["model"] == model["model"] and r["revision"] == model["revision"]
        )
        if digest(path) != reference["config_sha256"]:
            raise ValueError("target config differs from immutable checkpoint reference")
        limit = text["max_position_embeddings"]
        models[key] = {
            "model": model["model"],
            "revision": model["revision"],
            "config_sha256": digest(path),
            "max_position_embeddings": limit,
            "source_request_bounds_fit": state["max_input_plus_requested_output"] <= limit,
        }
    result = {
        "schema": "dspark-agentx-source-context-v1",
        "corpus": CORPUS,
        "local_content_verified": True,
        "license": "Apache-2.0",
        "source_url": f"https://huggingface.co/datasets/{CORPUS['repository']}/tree/{CORPUS['revision']}",
        "statistics": state,
        "models": models,
        "recommended_context_length": 1048576,
        "output_policy": "recorded_out_minimum_one_ignore_eos",
        "truncation_or_filtering": False,
        "all_generated_payloads_verified": False,
        "limitations": [
            "Source bounds are exact; tokenizer/chat-template expansion and generated prefix identities "
            "still require per-model producer audit.",
            "A new prefix-disabled fixed anchor with the same context limit is required before Phase B.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
