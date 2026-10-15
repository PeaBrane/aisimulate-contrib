# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/tests/unit/sdk/speculation/test_dspark_scheme.py

"""DSpark scheme tests (DeepSeek-V4 first target).

Ground facts pinned from deepseek-ai/DeepSeek-V4-Flash-DSpark @62af8fff
(safetensors index) and vLLM models/deepseek_v4/nvidia/dspark.py:
3 full-width draft blocks (mtp.{0,1,2}); main_proj = 3h -> h; markov head
rank 256; draft attention window-capped (SWA 128); one parallel draft
forward of N query tokens per round; verify width N + 1.
"""

from __future__ import annotations

import json

import pytest

from aisimulate_core.sdk import common, models
from aisimulate_core.sdk import config as sdk_config
from aisimulate_core.sdk.memory import KVCacheEstimator
from aisimulate_core.sdk.memory.state import estimate_state_cache
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aisimulate_core.sdk.speculation import SpeculationConfig
from aisimulate_core.sdk.speculation.dspark import DSparkScheme, draft_transfer_geometry
from aisimulate_core.sdk.utils import _load_model_config_from_model_path

pytestmark = pytest.mark.unit

# Real values from the V4-Flash DSpark draft config.json.
DRAFT_CONFIG = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
}


def _spec_config(**params) -> SpeculationConfig:
    return SpeculationConfig(kind="dspark", params=params, draft_config=DRAFT_CONFIG)


def test_v4_config_requires_target_layer_ids():
    # An absent field must not silently price main_proj at width h.
    partial = {k: v for k, v in DRAFT_CONFIG.items() if k != "dspark_target_layer_ids"}
    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        DSparkScheme.from_configs(None, SpeculationConfig(kind="dspark", params={}, draft_config=partial))


def _model_config(speculation: SpeculationConfig | None = None) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=8,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        speculation=speculation,
    )


def _v4_model_with_dspark(**params):
    # Inspect the scheme's structural model without admitting its unsupported
    # KV query override to the compiled execution path.
    model = models.get_model("deepseek-ai/DeepSeek-V4-Flash", _model_config(), "sglang")
    model.spec_scheme = DSparkScheme.from_configs(model.config, _spec_config(**params))
    model.spec_scheme.validate(model, "sglang")
    return model


class TestParamResolution:
    def test_defaults_from_draft_config(self):
        scheme = DSparkScheme.from_configs(_model_config(), _spec_config())
        assert scheme.num_draft_tokens == 5  # dspark_block_size
        assert scheme.num_draft_layers == 3
        assert scheme.markov_rank == 256
        assert scheme.verify_width() == 6

    def test_user_num_draft_tokens_overrides_block_size(self):
        scheme = DSparkScheme.from_configs(_model_config(), _spec_config(num_draft_tokens=7))
        assert scheme.num_draft_tokens == 7
        assert scheme.verify_width() == 8

    def test_missing_draft_config_raises(self):
        with pytest.raises(ValueError):
            DSparkScheme.from_configs(_model_config(), SpeculationConfig(kind="dspark", params={}))


class TestValidation:
    def test_materializes_v4_window_cap_at_model_boundary(self):
        cfg = _model_config(speculation=_spec_config(num_draft_tokens=7))
        model = models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "sglang")
        attention = next(op for op in model.generation_ops if op._name == "draft_dspark_attention")
        assert attention._draft_max_sequence_length == 135
        assert attention._draft_token_width == (7, 8)

    def test_rejects_non_v4_family(self):
        cfg = sdk_config.ModelConfig(
            tp_size=1,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            speculation=_spec_config(),
        )
        with pytest.raises(ValueError):
            models.get_model("Qwen/Qwen3-32B", cfg, "trtllm")


class TestDraftOpGraph:
    def test_generation_op_inventory(self):
        model = _v4_model_with_dspark(num_draft_tokens=7)
        scheme = model.spec_scheme
        specs = scheme.build_draft_generation_ops(model)
        names = [s.op._name for s in specs]

        assert "dspark_main_proj" in names
        assert "dspark_attention" in names
        assert "dspark_head_gemm" in names
        assert "dspark_markov_bias_gemm" in names

        by_name = {s.op._name: s for s in specs}
        # Backbone forward runs at the draft block width N.
        assert by_name["dspark_attention"].tokens_per_request == 7
        # Injection projection runs once per request per round.
        assert by_name["dspark_main_proj"].tokens_per_request == 8
        # Draft attention KV is window-capped: pinned s, independent of target isl.
        overrides = by_name["dspark_attention"].query_overrides
        assert overrides is not None and overrides["s"] <= 128 + 7
        # Markov chain: N sequential steps folded as count=N on one op.
        assert by_name["dspark_markov_bias_gemm"].op._scale_factor == 7

    def test_context_precompute_ops_exist(self):
        model = _v4_model_with_dspark()
        specs = model.spec_scheme.build_draft_context_ops(model)
        assert specs, "DSpark needs a context-KV precompute pass"
        names = [s.op._name for s in specs]
        assert "dspark_context_kv_proj" in names
        assert "dspark_context_kv_store" in names
        assert not any("attention" in n or "moe" in n for n in names)


class TestAccounting:
    def test_draft_weights_cross_check_against_index_diff(self):
        """Unique draft bytes ~= 10.9 GB (3 x 3.635 GB from the safetensors
        index diff at revision 62af8fff).

        draft_weights_bytes() returns PER-GPU resident bytes: sharded ops
        (attention/MoE/shared/router, tp8/ep8) contribute 1/8 of their
        unique bytes; replicated ops (main_proj, markov head) contribute in
        full on every GPU. Reconstruct the unique total accordingly.
        """
        model = _v4_model_with_dspark()
        per_gpu = model.spec_scheme.draft_weights_bytes(model)

        h = model._hidden_size
        vocab = model._vocab_size
        rank = model.spec_scheme.markov_rank
        # main_proj: fp8 (1 B/param) 3h -> h; markov: two vocab x rank bf16.
        replicated = h * (3 * h) * 1.0 + 2 * vocab * rank * 2.0
        unique_total = (per_gpu - replicated) * 8 + replicated
        assert unique_total == pytest.approx(10.9e9, rel=0.15)

    def test_draft_kv_is_window_capped(self):
        model = _v4_model_with_dspark()
        scheme = model.spec_scheme
        short = scheme.draft_kv_bytes_per_sequence(model, 64)
        capped_a = scheme.draft_kv_bytes_per_sequence(model, 10_000)
        capped_b = scheme.draft_kv_bytes_per_sequence(model, 1_000_000)
        assert 0 < short < capped_a
        assert capped_a == capped_b  # window cap
        # 3 layers x 128 window x head_dim 512 x 1 byte (fp8) minimum
        assert capped_a >= 3 * 128 * 512 * 1


# Minimal physical facts, independently expressed from Inferact's published
# cf6b8244620e7ea4b0651d214f28e89eac75bed6 checkpoint metadata.
KIMI_DRAFT = {
    "model_type": "k3_dspark",
    "hidden_size": 7168,
    "target_hidden_size": 7168,
    "intermediate_size": 14336,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 163840,
    "target_layer_ids": [2, 23, 47, 71, 89],
    "markov_rank": 256,
}


def _kimi_model(depth=None):
    cfg = _model_config(
        None
        if depth is None
        else SpeculationConfig(kind="dspark", params={"num_draft_tokens": depth}, draft_config=KIMI_DRAFT)
    )
    cfg.language_only = True
    return models.get_model("moonshotai/Kimi-K3", cfg, "vllm")


@pytest.mark.parametrize("depth", [2, 3])
def test_kimi_canonical_draft_has_distinct_width_and_target_only_kv(depth):
    target, model = _kimi_model(), _kimi_model(depth)
    assert model.verify_width == depth + 1
    assert model.config.nextn == 0
    verify = next(op for op in model.generation_ops if op._name == "generation_kda_verify")
    assert json.loads(verify._spec_json())["Kda"]["draft_tokens"] == depth + 1
    names = [op._name for op in model.generation_ops]
    assert "draft_dspark_context_kv_proj" in names
    assert "draft_attention" not in names  # legacy draft was not also injected
    assert model.get_kvcache_elements_per_token() == target.get_kvcache_elements_per_token()
    assert model.get_resident_weights_bytes() == target.get_resident_weights_bytes()
    assert model.spec_scheme.draft_kv_bytes_per_sequence(model, 1024) == 5 * 576 * 1024
    assert 0 < model.spec_scheme.draft_weights_bytes(model) < 4e9
    context = {op._name for op in model.context_ops if op._name.startswith("draft_dspark_")}
    assert context == {
        "draft_dspark_main_proj",
        "draft_dspark_main_norm",
        "draft_dspark_context_kv_proj",
        "draft_dspark_context_kv_store",
    }
    proposals = [
        json.loads(op._spec_json())["TokenScale"] for op in model.context_ops if op._name.startswith("draft_proposal_")
    ]
    assert proposals
    assert all(op["sequence_basis"] and op["denominator"] == 1 for op in proposals)
    assert not any("context_kv" in json.dumps(op) or "main_proj" in json.dumps(op) for op in proposals)


def test_kimi_draft_identity_rejects_wrong_geometry_and_quantization():
    for override in (
        {"kv_lora_rank": 256},
        {"target_hidden_size": 4096},
        {"torch_dtype": "float16"},
        {"quantization_config": {"quant_method": "fp8"}},
    ):
        with pytest.raises(ValueError):
            DSparkScheme.from_configs(
                None,
                SpeculationConfig(kind="dspark", params={"num_draft_tokens": 3}, draft_config=KIMI_DRAFT | override),
            )


@pytest.mark.parametrize("depth", [3, 5])
def test_v41_nested_metadata_keeps_draft_moe_separate(depth):
    raw = _load_model_config_from_model_path("deepseek-ai/DeepSeek-V4.1-Flash")
    cfg = _model_config(SpeculationConfig(kind="dspark", params={"num_draft_tokens": depth}, draft_config=raw))
    cfg.tp_size = cfg.moe_ep_size = 4
    cfg.language_only = True
    # SGLang's already-supported physical layout exercises the new draft graph
    # without depending on a newly-built native vLLM storage-layout variant.
    model = models.get_model("deepseek-ai/DeepSeek-V4.1-Flash", cfg, "sglang")
    assert model.verify_width == depth + 1
    draft_moe = [
        json.loads(op._spec_json())["Moe"]
        for op in model.generation_ops
        if op._name.startswith("draft_dspark_moe_") and "dispatch" not in op._name
    ]
    assert len(draft_moe) == 3
    assert {(op["num_experts"], op["topk"]) for op in draft_moe} == {(128, 3)}
    assert model.spec_scheme.draft_kv_bytes_per_sequence(model, 8192) == 3 * 128 * 584
    assert all("engram" not in op._name for op in model.generation_ops if op._name.startswith("draft_"))
    attention = [
        json.loads(op._spec_json())["TokenScale"]
        for op in model.generation_ops
        if op._name.startswith("draft_dspark_attention_") and "_attention_ar_" not in op._name
    ]
    assert len(attention) == 3
    for wrapper in attention:
        assert (wrapper["numerator"], wrapper["denominator"], wrapper["max_sequence_length"]) == (
            depth,
            depth + 1,
            128 + depth,
        )
        assert wrapper["op"]["Dsv41Attention"]["role"] == "swa"
        assert wrapper["op"]["Dsv41Attention"]["compress_ratio"] == 0


@pytest.mark.parametrize(
    "model_name,depth",
    [
        ("moonshotai/Kimi-K3", 2),
        ("moonshotai/Kimi-K3", 3),
        ("deepseek-ai/DeepSeek-V4.1-Flash", 3),
        ("deepseek-ai/DeepSeek-V4.1-Flash", 5),
    ],
)
def test_hybrid_canonical_constructor_and_memory(model_name, depth, monkeypatch):
    draft = KIMI_DRAFT if "Kimi" in model_name else _load_model_config_from_model_path(model_name)
    spec = {"kind": "dspark", "params": {"num_speculative_tokens": depth, "draft_config": draft}}
    config = {
        "model": model_name,
        "system": "b200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "aggregated",
        "tp": 8,
        "moe_tp_size": 8,
        "moe_ep_size": 1,
        "estimation_mode": "op_level",
        "database_mode": "SOL",
        "speculation": spec,
        "kv_cache_dtype": "fp8",
        "decoder_replay": "V4.1" in model_name,
    }
    model = RustForwardPassPerfModel.best_available(config)
    assert model.predict_decode_latency_total(1, 1024) > 0
    assert model.predict_prefill_latency(1, 1024) > 0
    memory_request = dict(
        max_num_tokens=128,
        max_batch_size=4,
        tp_size=8,
        moe_tp_size=8,
        moe_ep_size=1,
        kvcache_quant_mode="fp8",
        speculation=spec,
    )
    memory = KVCacheEstimator.from_request(
        model_name,
        "b200_sxm",
        "vllm",
        "0.24.0",
        **memory_request,
    ).breakdown
    assert memory["weights_bytes"] > 0
    assert memory["tokens_from_kv_bytes"](2e9) > 0
    # Adding a second execution phase must not allocate a second set of the
    # same draft weights. Compare the canonical capacity calculation with
    # the proposal phase omitted, leaving the draft geometry unchanged.
    monkeypatch.setattr(DSparkScheme, "build_draft_prefill_ops", lambda self, model: [])
    without_proposal = KVCacheEstimator.from_request(
        model_name, "b200_sxm", "vllm", "0.24.0", **memory_request
    ).breakdown
    assert memory["weights_bytes"] == without_proposal["weights_bytes"]
    provenance = model.diagnostics()["provenance"]
    assert provenance["modeling_assumptions"]
    # Native restore must retain the distinct draft identity, including the
    # nested V4.1 text config or Kimi artifact geometry.
    normalized = RustForwardPassPerfModel.normalize_config(config)
    assert normalized["speculation"]["params"]["draft_config"] == draft


@pytest.mark.parametrize("depth", [2, 3])
def test_kimi_rejection_window_is_reserved_without_multiplying_ssm(depth):
    base = estimate_state_cache("moonshotai/Kimi-K3", tp_size=8)
    spec = estimate_state_cache("moonshotai/Kimi-K3", tp_size=8, num_speculative_tokens=depth)
    assert spec["raw_bytes_per_request"] - base["raw_bytes_per_request"] == 69 * 3 * 12 * 128 * 2 * depth


@pytest.mark.parametrize(
    "name,draft,expected,window",
    [
        ("moonshotai/Kimi-K3", KIMI_DRAFT, 5 * 576, None),
        ("deepseek-ai/DeepSeek-V4.1-Flash", None, 3 * 528, 128),
    ],
)
def test_prepared_draft_transfer_uses_family_physical_geometry(name, draft, expected, window):
    draft = draft or _load_model_config_from_model_path(name)
    result = draft_transfer_geometry(
        name, {"params": {"draft_config": draft}}, "fp8", 8, backend="vllm", system="b200_sxm"
    )
    assert result == {"kv_transfer_draft_bytes_per_token": expected, "kv_transfer_draft_window": window}
