# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: independently expressed CPU-only operation and memory descriptions.
# Serving contract: vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516,
# vllm/models/kimi_k3/nvidia/dspark_mla.py and model_executor/models/utils.py.
# https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516

"""Artifact-derived Inferact MLA draft geometry, distinct from dense GQA drafts."""

from dataclasses import dataclass, fields

from aisimulate_core.sdk import common
from aisimulate_core.sdk import operations as ops
from aisimulate_core.sdk.speculation.base import positive_integer


@dataclass(frozen=True)
class KimiDraftGeometry:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    vocab_size: int

    @classmethod
    def from_config(cls, config: dict):
        if config.get("model_type") != "k3_dspark":
            raise ValueError("Kimi DSpark requires an Inferact k3_dspark MLA draft")
        if config.get("quantization_config"):
            raise ValueError("Kimi DSpark currently requires the unquantized BF16 draft checkpoint")
        if config.get("torch_dtype", config.get("dtype", "bfloat16")) != "bfloat16":
            raise ValueError("Kimi DSpark currently requires BF16 draft weights")
        if config.get("mla_use_output_gate") or config.get("mla_use_nope"):
            raise ValueError("Kimi DSpark requires RoPE and no draft output gate")
        result = cls(**{f.name: positive_integer(config.get(f.name), f.name) for f in fields(cls)})
        if config.get("target_hidden_size") != result.hidden_size:
            raise ValueError("Kimi DSpark target_hidden_size must match the shared embedding width")
        if config.get("draft_vocab_size", result.vocab_size) != result.vocab_size:
            raise ValueError("Kimi DSpark draft vocabulary remapping is unsupported")
        if (
            result.q_lora_rank,
            result.kv_lora_rank,
            result.qk_nope_head_dim,
            result.qk_rope_head_dim,
            result.v_head_dim,
        ) != (1536, 512, 128, 64, 128):
            raise ValueError("Kimi DSpark MLA operators require the published Inferact latent geometry")
        return result

    @property
    def latent_width(self):
        return self.kv_lora_rank + self.qk_rope_head_dim

    def context_ops(self):
        # The runtime duplicates each layer's latent projection into one fused,
        # replicated context projection. Its allocation is counted separately.
        return [
            ops.ElementWise("dspark_main_norm", 1, self.hidden_size, self.hidden_size, 0.8),
            ops.GEMM(
                "dspark_context_kv_proj",
                1,
                self.num_hidden_layers * self.latent_width,
                self.hidden_size,
                common.GEMMQuantMode.bfloat16,
            ),
            ops.ElementWise(
                "dspark_context_kv_store", self.num_hidden_layers, self.latent_width, self.latent_width, 0.8
            ),
        ]

    def block_ops(self, model):
        h, n, tp = self.hidden_size, self.num_hidden_layers, model.config.tp_size
        if self.num_attention_heads % tp or self.intermediate_size % tp:
            raise ValueError("Kimi draft attention heads and intermediate width must divide TP")
        q = common.GEMMQuantMode.bfloat16
        heads = self.num_attention_heads // tp
        return [
            ops.ElementWise("dspark_attn_norm", n, h, h, 0.8),
            ops.GEMM("dspark_downscale", n, self.q_lora_rank + self.latent_width, h, q),
            ops.GEMM("dspark_q_b", n, heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), self.q_lora_rank, q),
            ops.MLABmm("dspark_bmm_pre", n, heads, q, if_pre=True),
            # The generic MLA table lacks a noncausal query-width axis: retain
            # token-basis pricing and report that measurement gap explicitly.
            ops.GenerationMLA("dspark_attention", n, heads, model.config.kvcache_quant_mode),
            ops.MLABmm("dspark_bmm_post", n, heads, q, if_pre=False),
            ops.GEMM("dspark_output", n, h, heads * self.v_head_dim, q),
            ops.CustomAllReduce("dspark_attention_ar", n, h, tp),
            ops.ElementWise("dspark_ffn_norm", n, h, h, 0.8),
            ops.GEMM("dspark_gate_up", n, 2 * self.intermediate_size // tp, h, q),
            ops.ElementWise("dspark_act_gate", n, 2 * self.intermediate_size // tp, self.intermediate_size // tp, 0.8),
            ops.GEMM("dspark_down", n, h, self.intermediate_size // tp, q),
            ops.CustomAllReduce("dspark_ffn_ar", n, h, tp),
        ]

    def resident_block_bytes(self, tp: int):
        # Unlike a GQA surrogate, actual MLA also owns the kv_b projection.
        h, qk = self.hidden_size, self.qk_nope_head_dim + self.qk_rope_head_dim
        elements = h * (self.q_lora_rank + self.latent_width)
        elements += self.q_lora_rank * self.num_attention_heads * qk / tp
        elements += self.kv_lora_rank * self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim) / tp
        elements += h * self.num_attention_heads * self.v_head_dim / tp
        elements += 3 * h * self.intermediate_size / tp
        elements += 2 * h + self.q_lora_rank + self.kv_lora_rank
        return 2 * self.num_hidden_layers * elements

    def kv_bytes(self, model, length: int):
        # Latent cache is replicated across TP, unlike the RadixArk GQA draft.
        return float(
            self.num_hidden_layers * max(0, length) * self.latent_width * model.config.kvcache_quant_mode.value.memory
        )
