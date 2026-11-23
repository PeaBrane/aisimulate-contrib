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

from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, config
from aisim_core.sdk.models import get_model
from aisim_core.sdk.models.afd import build_afd_model
from aisim_core.sdk.perf_database import get_database, get_database_view

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


@pytest.mark.parametrize("role", ["a", "f"])
def test_moe_context_logits_stays_on_the_afd_boundary(role):
    model = build_afd_model(
        "Qwen/Qwen3-30B-A3B",
        config.ModelConfig(tp_size=2, moe_tp_size=1, moe_ep_size=2, **_BF16),
        "sglang",
        role,
    )
    for phase in ("context", "generation"):
        ops = getattr(model, f"{phase}_ops")
        heads = [i for i, op in enumerate(ops) if op._name == f"{phase}_logits_gemm"]
        # Native AFD owns boundary work on A. Its compatibility insertion for
        # a missing prefill head must not duplicate the newly complete graph.
        if role == "a":
            assert len(heads) == 1
            assert model.afd_plan[phase]["suffix"] == heads
            assert all(heads[0] not in t["ops"] + t["rank_ops"] for t in model.afd_plan[phase]["templates"])
        else:
            assert heads == model.afd_plan[phase]["suffix"] == []


@pytest.fixture(scope="module")
def afd_database():
    # Keep this regression on its original measurement grid as query slots move.
    return get_database("h200_sxm", "sglang", "0.5.14", allow_unlisted_version=True)


@pytest.mark.usefixtures("raw_data_coordinates")
@pytest.mark.parametrize("afd_pool", [False, True], ids=["ordinary", "afd-a-pool"])
@pytest.mark.parametrize(
    ("is_context", "seq_len", "prefix"),
    [(True, 1, 0), (True, 1024, 0), (True, 512, 512), (False, 1024, 512)],
    ids=["prefill-one-token", "prefill-1024", "prefill-cached-prefix", "decode"],
)
def test_moe_logits_latency_uses_one_token_per_request(afd_database, afd_pool, is_context, seq_len, prefix):
    from aisim_core.sdk.engine import EngineHandle, compile_engine

    model_config = config.ModelConfig(tp_size=4, moe_tp_size=1, moe_ep_size=4, **_BF16)
    model_path = "Qwen/Qwen3-30B-A3B"
    model = (
        build_afd_model(model_path, model_config, "sglang", "a")
        if afd_pool
        else get_model(model_path, model_config, "sglang")
    )
    phase = "context" if is_context else "generation"
    phase_ops = getattr(model, f"{phase}_ops")
    logits = next(op for op in phase_ops if op._name == f"{phase}_logits_gemm")
    qkv = next(op for op in phase_ops if op._name == f"{phase}_qkv_gemm")
    indices = [phase_ops.index(op) for op in (logits, qkv)]
    handle = EngineHandle(
        compile_engine(
            model_path,
            "h200_sxm",
            "sglang",
            "0.5.14",
            afd_pool_role="a" if afd_pool else None,
            tp_size=4,
            moe_tp_size=1,
            moe_ep_size=4,
            **{key: value.name for key, value in _BF16.items()},
        )
    )
    evaluate = handle.evaluate_context_ops if is_context else handle.evaluate_generation_ops
    entries = evaluate(indices, batch_size=4, s=seq_len, prefix=prefix)
    per_op = {name: latency for name, latency, _energy, _source in entries}

    # Independent shape anchors: four requests project four final hidden
    # states, while QKV processes 4*1024 tokens for the uncached prefill case.
    # Main removed the Python AFD session and its fallback; exercise the
    # supported native op-list surface and real AFD pool graph instead.
    expected_logits = float(logits._engine_query(afd_database, x=4))
    expected_qkv = float(qkv._engine_query(afd_database, x=4 * seq_len if is_context else 4))
    assert per_op == {
        logits._name: pytest.approx(expected_logits),
        qkv._name: pytest.approx(expected_qkv),
    }
    if is_context and seq_len > 1:
        assert float(logits._engine_query(afd_database, x=4 * seq_len)) > expected_logits


@pytest.mark.usefixtures("raw_data_coordinates")
@pytest.mark.parametrize("model_path", ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-R1"])
@pytest.mark.parametrize("prefill,decode", [(1, 0), (3, 5), (128, 256), (0, 5)])
def test_scheduled_iteration_logits_project_each_request_once(model_path, prefill, decode):
    # Exercise the public telemetry consumer of get_mix_step_ops, including
    # existing dense/DeepSeek heads. Formula-only SOL keeps the oracle's shape
    # independent of measurement-grid coverage. The large batch crosses the
    # LM-head memory/compute boundary and distinguishes request counts from 1.
    database = get_database_view("h200_sxm", "sglang", "0.5.14", database_mode="SOL")
    graph = get_model(model_path, config.ModelConfig(moe_tp_size=1, moe_ep_size=1, **_BF16), "sglang")
    metrics = [
        {
            "scheduled_requests": {
                "num_prefill_requests": prefill,
                "sum_prefill_tokens": prefill * 32,
                "sum_prefill_kv_tokens": prefill * 512,
                "num_decode_requests": decode,
                "sum_decode_kv_tokens": decode * 4096,
            }
        }
    ]
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=model_path,
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            worker_type="aggregated",
            estimation_mode="op_level",
            database_mode="SOL",
            moe_tp_size=1,
            moe_ep_size=1,
            **{key: value.name for key, value in _BF16.items()},
        )
    )
    try:
        detail = model.estimate_forward_pass_detailed(metrics)
        assert detail["latency_ms"] == model.estimate_forward_pass_time_ms(metrics)
        operations = detail["ranks"][0]["operations"]
        heads = [op for op in operations if op["name"].endswith("logits_gemm")]
        phase = "context" if prefill else "generation"
        logits = next(op for op in getattr(graph, f"{phase}_ops") if op._name.endswith("logits_gemm"))
        # There are prefill+decode final hidden states, even though the shared
        # transformer pass processes prefill*32+decode tokens. Decode attention
        # is a separate pass; generation_logits_gemm must not run there too.
        expected = float(logits._engine_query(database, x=prefill + decode))
        assert len(heads) == 1
        assert heads[0]["name"] == f"{phase}_logits_gemm"
        assert heads[0]["latency_ms"] == pytest.approx(expected, rel=1e-12)
        assert sum(op["latency_ms"] for op in operations) == pytest.approx(detail["latency_ms"], rel=1e-12)
        if prefill and decode and prefill + decode > 100:
            assert expected > float(logits._engine_query(database, x=1))
        if not prefill:
            static = model.static_phase_diagnostics(batch_size=decode, context_length=4095, prefill=False)
            assert operations == static
    finally:
        model.close()
