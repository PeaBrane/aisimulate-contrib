# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format normal captured Pro chat payloads offline with the exact installed runtime.

Run inside the pinned vLLM image on CPU. This original adapter imports serving
code without copying it. See COLLECTION.md for the immutable API/installed hashes.
"""

import argparse
import hashlib
import json
from pathlib import Path

from agentx_capture import jsonl
from agentx_import import materialize_raw_record
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding

MODEL = "deepseek-ai/DeepSeek-V4-Pro-0813"
REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
WRAPPER_SHA256 = "0f8a23226a1190bba2fc422fd11730ff96e98f9f46356a91c620660f49817bc3"
ENCODER_SHA256 = "aa407637bd2e58f20a16f781ea0bc519d06f74bed49deadaecb5aa5756d7230e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2_000_000)
    args = parser.parse_args()
    for module, expected in [(deepseek_v4, WRAPPER_SHA256), (deepseek_v4_encoding, ENCODER_SHA256)]:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("installed native formatter differs from the audited image")
    tokenizer = deepseek_v4.DeepseekV4Tokenizer.from_pretrained(MODEL, revision=REVISION)
    events = [row for path in args.capture.glob("events-*.jsonl") for _, row in jsonl(path)]
    initial = [event for event in events if (event["phase"], event["event"]) == ("warmup", "setup")]
    if len(initial) != 1:
        raise ValueError("one initial warmup setup is required")
    lanes = {}
    for lane, trajectory in enumerate(initial[0]["trajectories"]):
        for state in trajectory["snapshot"]["states"]:
            root = state.get("root_correlation_id") or (
                state["x_correlation_id"] if state.get("agent_depth", 0) == 0 else state.get("parent_correlation_id")
            )
            if root in lanes and lanes[root] != lane:
                raise ValueError("producer root aliases multiple lanes")
            lanes[root] = lane
    if args.max_input_tokens <= 0:
        raise ValueError("max-input-tokens must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_name(args.output.name + ".partial")
    seen, excluded, input_count = set(), [], 0
    with temporary.open("x") as stream:
        for path in args.raw:
            for line, raw in jsonl(path):
                meta = raw["metadata"]
                root = meta.get("root_correlation_id") or meta["x_correlation_id"]
                if root not in lanes:
                    excluded.append(
                        dict(
                            request_id=meta["x_request_id"],
                            reason="not_initial_tree",
                            raw_file=str(path),
                            raw_line=line,
                        )
                    )
                    continue
                if meta["x_request_id"] in seen:
                    raise ValueError("duplicate raw request ID")
                seen.add(meta["x_request_id"])
                payload = raw["payload"]
                if (
                    payload.get("model") != MODEL
                    or payload.get("chat_template")
                    or payload.get("continue_final_message")
                    or payload.get("truncate_prompt_tokens") is not None
                    or payload.get("add_special_tokens", False)
                ):
                    raise ValueError("unsupported model or custom chat formatting")
                if any(
                    set(message) - {"role", "content"}
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    for message in payload["messages"]
                ):
                    raise ValueError(
                        "only plain text role/content messages are qualified; normalize other shapes explicitly"
                    )
                options = dict(payload.get("chat_template_kwargs") or {})
                if set(options) - {"thinking", "enable_thinking", "drop_thinking", "reasoning_effort"}:
                    raise ValueError("unqualified chat-template options")
                if payload.get("reasoning_effort") is not None:
                    options["reasoning_effort"] = payload["reasoning_effort"]
                tokens = tokenizer.apply_chat_template(
                    payload["messages"], tools=payload.get("tools"), tokenize=True, **options
                )
                input_count += len(tokens)
                if input_count > args.max_input_tokens:
                    raise ValueError("selected token budget exceeded; partial output remains unqualified")
                row = materialize_raw_record(raw, tokens, WRAPPER_SHA256, args.runtime_image_digest)
                row.update(
                    lane_id=lanes[root],
                    tokenizer_revision=REVISION,
                    encoder_sha256=ENCODER_SHA256,
                    raw_file=str(path),
                    raw_line=line,
                )
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    args.output.with_suffix(".scope.json").write_text(
        json.dumps(
            dict(
                initial_requests=len(seen),
                input_tokens=input_count,
                max_input_tokens=args.max_input_tokens,
                excluded=excluded,
                scoring_eligible=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
