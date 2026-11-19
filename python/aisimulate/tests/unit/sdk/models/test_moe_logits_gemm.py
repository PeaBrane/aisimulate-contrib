# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE-family prefill pays for the vocabulary projection, as every family does.

Prefill emits the first output token, so its forward pass ends with the same
vocabulary-sharded LM-head GEMM that each decode step runs. The resident
weight inventory is also derived from the context graph, so a prefill graph
without that GEMM under-counts both prefill latency and LM-head weights.
"""

from __future__ import annotations

import pytest

from aisimulate_core.sdk import common, config
from aisimulate_core.sdk.afd_partition import build_afd_ops_partition
from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

_BF16 = {
    "gemm_quant_mode": common.GEMMQuantMode.bfloat16,
    "moe_quant_mode": common.MoEQuantMode.bfloat16,
    "kvcache_quant_mode": common.KVCacheQuantMode.bfloat16,
    "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
}


def _vocab_heads(op_list):
    return [
        (op._name, op._n, op._k, op._quant_mode, op._scale_factor) for op in op_list if op._name.endswith("logits_gemm")
    ]


@pytest.mark.parametrize("backend", ["trtllm", "vllm", "sglang"])
def test_moe_vocab_head_matches_the_dense_family(backend):
    # Qwen3-235B-A22B (MOE family) and Qwen3-8B (LLAMA family) share
    # hidden_size 4096 and the untied 151936-token vocabulary, so their
    # LM-head work must be identical in both phases.
    moe = get_model(
        "Qwen/Qwen3-235B-A22B",
        config.ModelConfig(tp_size=2, moe_tp_size=1, moe_ep_size=2, **_BF16),
        backend,
    )
    dense = get_model("Qwen/Qwen3-8B", config.ModelConfig(tp_size=2, **_BF16), backend)

    assert _vocab_heads(dense.context_ops) == [
        ("context_logits_gemm", 151936 // 2, 4096, common.GEMMQuantMode.bfloat16, 1.0)
    ]
    assert _vocab_heads(moe.context_ops) == _vocab_heads(dense.context_ops)
    assert _vocab_heads(moe.generation_ops) == _vocab_heads(dense.generation_ops)


@pytest.mark.parametrize(
    ("model_path", "backend", "parallel"),
    [
        pytest.param(
            "openai/gpt-oss-20b",
            "trtllm",
            {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2},
            id="gpt-oss-trtllm",
        ),
        pytest.param(
            "mistralai/Mixtral-8x7B-v0.1",
            "vllm",
            {"tp_size": 2, "moe_tp_size": 2, "moe_ep_size": 1},
            id="mixtral-vllm",
        ),
        pytest.param(
            "MiniMaxAI/MiniMax-M2.5",
            "vllm",
            {"tp_size": 4, "pp_size": 2, "moe_tp_size": 1, "moe_ep_size": 4},
            id="minimax-m2-vllm-pp2",
        ),
        pytest.param(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "vllm",
            {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2},
            id="qwen3-vl-moe-vllm",
        ),
        pytest.param(
            "Qwen/Qwen3-30B-A3B",
            "sglang",
            {"cp_size": 2, "cp_style": "allgather", "moe_tp_size": 1, "moe_ep_size": 2},
            id="qwen3-moe-sglang-cp2",
        ),
        pytest.param(
            "Qwen/Qwen3-235B-A22B",
            "sglang",
            {
                "attention_dp_size": 8,
                "moe_tp_size": 1,
                "moe_ep_size": 8,
                "moe_backend": "deepep_moe",
                "moe_comm_backend": {"context": "deepep_ht", "generation": "deepep_ll"},
                "num_gpus_per_node": 8,
            },
            id="qwen3-moe-sglang-large-ep",
        ),
    ],
)
def test_moe_prefill_projects_to_the_decode_vocab_shard(model_path, backend, parallel):
    model = get_model(model_path, config.ModelConfig(**_BF16, **parallel), backend)

    context_heads = _vocab_heads(model.context_ops)
    generation_heads = _vocab_heads(model.generation_ops)
    assert len(context_heads) == len(generation_heads) == 1
    (_, *context_shape, context_scale) = context_heads[0]
    (_, *generation_shape, _) = generation_heads[0]
    # Same vocab shard, hidden width and bf16 LM head as decode, charged once
    # per prefill forward pass (not per layer).
    assert context_shape == generation_shape
    assert context_scale == 1.0
    context_logits = next(op for op in model.context_ops if op._name == "context_logits_gemm")
    generation_logits = next(op for op in model.generation_ops if op._name == "generation_logits_gemm")
    assert context_logits._seq_split == parallel.get("cp_size", 1)
    assert generation_logits._seq_split == 1


@pytest.mark.parametrize("backend", ["trtllm", "vllm", "sglang"])
def test_moe_resident_weights_include_the_bf16_lm_head_shard(backend):
    model = get_model(
        "Qwen/Qwen3-235B-A22B",
        config.ModelConfig(tp_size=2, moe_tp_size=1, moe_ep_size=2, **_BF16),
        backend,
    )

    logits = next(op for op in model.context_ops if op._name == "context_logits_gemm")
    expected_head_bytes = (151936 // 2) * 4096 * 2
    other_weights = sum(op.get_weights() for op in model.context_ops if op is not logits)

    assert logits.get_weights() == expected_head_bytes
    # This inventory is the resident-weight input to KV-capacity accounting.
    assert model.get_resident_weights_bytes() == other_weights + expected_head_bytes


@pytest.mark.parametrize("boundary_on_attn", [True, False])
def test_moe_context_logits_stays_on_the_afd_boundary(boundary_on_attn):
    model = get_model(
        "Qwen/Qwen3-30B-A3B",
        config.ModelConfig(tp_size=2, moe_tp_size=1, moe_ep_size=2, **_BF16),
        "sglang",
    )
    logits = next(op for op in model.context_ops if op._name == "context_logits_gemm")

    partition = build_afd_ops_partition(model, phase="context", boundary_on_attn=boundary_on_attn)

    assert logits in partition.boundary_ops
    selected_ops = partition.attn_ops if boundary_on_attn else partition.ffn_ops
    other_ops = partition.ffn_ops if boundary_on_attn else partition.attn_ops
    assert sum(op is logits for op in selected_ops) == 1
    assert logits not in other_ops
    assert logits not in partition.skipped_ops


@pytest.fixture(scope="module")
def afd_database():
    return get_database("b200_sxm", "vllm", "current")


@pytest.mark.parametrize("use_op_list", [True, False], ids=["compiled-list", "single-op-fallback"])
@pytest.mark.parametrize("boundary_on_attn", [True, False], ids=["a-boundary", "f-boundary"])
@pytest.mark.parametrize(
    ("is_context", "seq_len", "prefix"),
    [(True, 1, 0), (True, 1024, 0), (True, 512, 512), (False, 1024, 512)],
    ids=["prefill-one-token", "prefill-1024", "prefill-cached-prefix", "decode"],
)
def test_moe_afd_logits_latency_uses_one_token_per_request(
    monkeypatch, afd_database, use_op_list, boundary_on_attn, is_context, seq_len, prefix
):
    from aisimulate.sdk.inference_session import AFDInferenceSession

    model_config = config.ModelConfig(tp_size=4, moe_tp_size=1, moe_ep_size=4, **_BF16)
    model = get_model("Qwen/Qwen3-30B-A3B", model_config, "vllm")
    session = AFDInferenceSession(
        model.model_path,
        model_config,
        model_config,
        afd_database,
        get_backend("vllm"),
        config.AFDConfig(gpus_per_node=8),
    )
    if not use_op_list:
        # Exercise the per-op path used when an op is outside the compiled list.
        monkeypatch.setattr(session, "_sum_latency_with_rust", lambda *args, **kwargs: None)

    phase = "context" if is_context else "generation"
    partition = build_afd_ops_partition(model, phase=phase, boundary_on_attn=boundary_on_attn)
    runtime = config.RuntimeConfig(batch_size=4, isl=seq_len + prefix, osl=2, prefix=prefix)
    per_op = {}
    for worker_ops in (partition.attn_ops, partition.ffn_ops):
        # Use the real graph's GEMMs on both sides of the A/F boundary. The
        # QKV control must retain all uncached tokens while logits emits only
        # one token per request. Native tables supply all performance values.
        gemms = [op for op in worker_ops if op._name in {f"{phase}_qkv_gemm", f"{phase}_logits_gemm"}]
        total, values = session._sum_latency(
            gemms, batch_size=4, seq_len=seq_len, model=model, runtime_config=runtime, is_context=is_context
        )
        assert total == pytest.approx(sum(values.values()))
        per_op.update(values)

    phase_ops = model.context_ops if is_context else model.generation_ops
    logits = next(op for op in phase_ops if op._name == f"{phase}_logits_gemm")
    qkv = next(op for op in phase_ops if op._name == f"{phase}_qkv_gemm")
    # Independent shape anchors: four requests project four final hidden
    # states, while QKV processes 4*1024 tokens for the uncached prefill case.
    expected_logits = float(logits._engine_query(afd_database, x=4))
    expected_qkv = float(qkv._engine_query(afd_database, x=4 * seq_len if is_context else 4))
    assert per_op == {
        logits._name: pytest.approx(expected_logits),
        qkv._name: pytest.approx(expected_qkv),
    }
    if is_context and seq_len > 1:
        assert float(logits._engine_query(afd_database, x=4 * seq_len)) > expected_logits
