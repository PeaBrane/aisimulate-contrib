# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect actual pinned producer tokenizers; this is not a full payload audit."""

import argparse
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path

from aiperf.common.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", required=True, metavar=("HF_ID", "REVISION"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for model, revision in args.model:
        row = {"model": model, "revision": revision}
        try:
            tokenizer = Tokenizer.from_pretrained(model, trust_remote_code=True, revision=revision, resolve_alias=False)
            inner = tokenizer._tokenizer
            row.update(
                tokenizer_loaded=True,
                tokenizer_class=type(inner).__name__,
                resolved_name=tokenizer.resolved_name,
                chat_template_present=bool(inner.chat_template),
            )
            messages = [{"role": "user", "content": "Probe input."}]
            try:
                tokens = inner.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if isinstance(tokens, Mapping):
                    tokens = tokens["input_ids"]
                row["hf_single_user_probe_tokens"] = len(tokens)
            except Exception as error:
                row["hf_template_error"] = f"{type(error).__name__}: {error}"
        except Exception as error:
            row.update(tokenizer_loaded=False, error=f"{type(error).__name__}: {error}")
        row["serving_native_formatter_verified"] = False
        row["all_generated_payloads_verified"] = False
        rows.append(row)
    result = {
        "producer_revision": "754356e9a39acc6cc6afb242d123bb57c3fb6f75",
        "scope": (
            "Real CPU tokenizer load and one HF chat-template probe; no full-corpus or serving-formatter equivalence."
        ),
        "dependencies": {name: importlib.metadata.version(name) for name in ("aiperf", "transformers", "tokenizers")},
        "models": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
