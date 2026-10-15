# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce one maximum-source-context request using actual producer code.

Original CPU harness. Imports (does not copy) AIPerf754356e9 APIs and the pinned
vLLM installed native encoder. No corpus modification or serving work.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import time
from pathlib import Path

from aiperf.cli import app
from aiperf.common import random_generator as rng
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace

KIMI_RUNTIME_SOURCE_HASHES = {
    "renderers/kimi_k3.py": "1b51e2bc6af29466d1f15c2f8211990903a947da5995e95e9bc68ed1104ecdb4",
    "renderers/params.py": "ad38852674ebc5bbe69aa9df91602847d518382cc2ceb0a25aac1d07be66b504",
    "renderers/registry.py": "5ea03ecad81f8911c72112656f2456c05e0e57122be99af05c27464466596844",
    "tokenizers/registry.py": "3d557b6c19bb7d8f2ede8b0d4f9fac942d976bc82c293a25e52f6bd73ffc6b13",
    "tokenizers/hf.py": "36314bc67ab94552f6d99acb55cdf66045ce5d92553b06f52b4b3ebc7de6f641",
    "entrypoints/openai/chat_completion/protocol.py": (
        "f23a9cb2ebc044fdfae3eef866ea7f8846d81d978f3f5303fa0d084fea97940e"
    ),
}

MODEL_PINS = {
    "kimi-k3": (
        "moonshotai/Kimi-K3",
        "f831ab66814297da540d832a5235f8e904f29d06",
        KIMI_RUNTIME_SOURCE_HASHES["renderers/kimi_k3.py"],
        "../renderers/kimi_k3.py",
    ),
    "v4-pro": (
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "72e1d3230f6c080a530b0a1d46f8eb4602340597",
        "aa407637bd2e58f20a16f781ea0bc519d06f74bed49deadaecb5aa5756d7230e",
        "deepseek_v4_encoding.py",
    ),
    "v41-flash": (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "5254448103b6254fc646dacbca1cbbf20e9216c1",
        "477f22ac250b43c12bd19bb3c93f934d6637b06410b8cb7ccba0862d1049dee8",
        "deepseek_v41_encoding.py",
    ),
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--native-encoder", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODEL_PINS, default="v4-pro")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    model_pin, revision_pin, encoder_sha256, encoder_name = MODEL_PINS[args.model_key]
    if args.corpus.stat().st_size != manifest["corpus"]["bytes"] or sha256(args.corpus) != manifest["corpus"]["sha256"]:
        raise ValueError("corpus content mismatch")
    if sha256(args.native_encoder) != encoder_sha256:
        raise ValueError("native encoder differs from the selected audited pinned-image source")
    best = None
    peak = -1
    outer = None
    with args.corpus.open() as stream:
        for line in stream:
            trace = json.loads(line)
            for i, request in enumerate(trace["requests"]):
                if request["type"] == "subagent":
                    continue
                bound = request["in"] + max(request["out"], 1)
                if bound > peak:
                    peak = bound
                    best = trace
                    outer = i
    # This diagnostic selects the maximum parent/flat request. The independent
    # full-corpus source audit includes child requests and must agree on peak.
    source_audit = json.loads((args.manifest.parent / manifest["context_audit"]["path"]).read_text())
    if peak != source_audit["statistics"]["max_input_plus_requested_output"]:
        raise ValueError("global maximum is a child request; this narrow probe cannot select it")
    command = list(next(c["command"] for c in manifest["cpu_cli_checks"] if c["phase"] == "profile"))
    for flag, value in (("--model", model_pin), ("--tokenizer", model_pin), ("--tokenizer-revision", revision_pin)):
        command[command.index(flag) + 1] = value
    _, parsed, _ = app.parse_args(command[1:], exit_on_error=False)
    plan = build_benchmark_plan(resolve_config(parsed.arguments["cli_config"], None))
    run = BenchmarkRun(
        benchmark_id="dspark-cpu-context-probe",
        cfg=plan.configs[0],
        artifact_dir=args.output.parent / "unused-producer-artifacts",
        random_seed=42,
    )
    model = run.cfg.tokenizer.name
    revision = run.cfg.tokenizer.revision
    if model != model_pin or revision != revision_pin:
        raise ValueError("probe model differs from the selected tokenizer/encoder pin")
    rng.reset()
    rng.init(42)
    tokenizer = Tokenizer.from_pretrained(model, trust_remote_code=True, revision=revision, resolve_alias=False)
    composer = PublicDatasetComposer(run=run, tokenizer=tokenizer)
    loader = WekaTraceLoader(filename=None, run=run, prompt_generator=composer.prompt_generator, default_block_size=64)
    trace = WekaTrace.model_validate(best)
    started = time.monotonic()
    conversations = loader.convert_to_conversations({trace.id: [trace]})
    reconstruction_seconds = time.monotonic() - started
    messages = None
    selected = None
    for conversation in conversations:
        history = []
        for turn in conversation.turns:
            if turn.reset_context:
                history = []
            history.extend(turn.raw_messages or [])
            if turn.source_outer_idx == outer and turn.source_kind in ("weka_main", "weka_flat"):
                if selected is not None:
                    raise ValueError("ambiguous source request mapping")
                selected = turn
                messages = list(history)
    if selected is None:
        raise ValueError("producer omitted the selected source request")
    if any(
        set(message) - {"role", "content"}
        or message.get("role") not in ("system", "user", "assistant")
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("native witness is limited to plain-text messages with identity normalization")
    source_contract = None
    if args.model_key == "kimi-k3":
        source_root = args.native_encoder.parent.parent
        for path, expected in KIMI_RUNTIME_SOURCE_HASHES.items():
            if sha256(source_root / path) != expected:
                raise ValueError("Kimi native renderer/request contract differs from the audited image")
        # For this plain-text/no-tools witness, Kimi's renderer leaves messages
        # unchanged and eagerly invokes the pinned HF remote-code tokenizer.
        # The API defaults add a generation prompt and leave thinking at the
        # tokenizer's True/max defaults; no Jinja template is involved.
        kwargs = {
            "add_generation_prompt": True,
            "continue_final_message": False,
            "return_dict": False,
            "tokenize": True,
        }
        native_tokenizer = tokenizer._tokenizer
        tokens = native_tokenizer.apply_chat_template(messages, **kwargs)
        native_file = Path(inspect.getfile(type(native_tokenizer)))
        source_contract = {
            "runtime_sources": KIMI_RUNTIME_SOURCE_HASHES,
            "call": "Pinned tokenizer.apply_chat_template matching audited plain-text renderer boundary",
            "kwargs": kwargs,
            "remote_code": {path.name: sha256(path) for path in native_file.parent.glob("*.py")},
            "scope": "No tools, media, request reasoning override, or session cache-bust marker.",
        }
    else:
        spec = importlib.util.spec_from_file_location("audited_native_encoder", args.native_encoder)
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)
        text = encoder.encode_messages(messages, thinking_mode="thinking", drop_thinking=True, reasoning_effort="high")
        tokens = tokenizer.encode(text, add_special_tokens=False)
    result = {
        "schema": "dspark-agentx-native-context-probe-v1",
        "model": model,
        "revision": revision,
        "producer_revision": manifest["producer_revision"],
        "corpus": manifest["corpus"],
        "native_encoder_sha256": encoder_sha256,
        "native_encoder_source": (
            "vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516/vllm/"
            + ("renderers/kimi_k3.py" if args.model_key == "kimi-k3" else "tokenizers/" + encoder_name)
        ),
        "installed_image_export_job": "2267423" if args.model_key == "kimi-k3" else "2257688",
        "native_call_contract": source_contract,
        "source_trace_id": trace.id,
        "source_outer_index": outer,
        "source_kind": selected.source_kind,
        "source_input": best["requests"][outer]["in"],
        "requested_output": selected.max_tokens,
        "native_prompt_tokens_without_session_marker": len(tokens),
        "native_prompt_plus_output_without_session_marker": len(tokens) + selected.max_tokens,
        "reconstructed_conversations": len(conversations),
        "reconstructed_turns": sum(len(c.turns) for c in conversations),
        "reconstruction_seconds": reconstruction_seconds,
        "messages": len(messages),
        "scoring_eligible": False,
        "all_generated_payloads_verified": False,
        "limitations": [
            "One source-maximum request, not a full-corpus audit.",
            "Session cache-bust marker is excluded; no actual serving request was issued.",
            "Snapshot/frontier and Replay token-prefix equality are not established.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ["AIPERF_DATASET_WEKA_PARALLEL_WORKERS"] = "1"
    main()
