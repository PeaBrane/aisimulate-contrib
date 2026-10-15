# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe actual ac9126 Kimi MLA forwards with their serving metadata.

Original adapter for vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516,
vllm/models/kimi_k3/nvidia/mla.py:611-727 (Apache-2.0).
"""

import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from collector.vllm.kimi_native_contract import PREFILL_BACKEND, physical_identity


class KimiObserverWorker:
    def kimi_observer_install(self, output, source_hashes):
        if hasattr(self, "_kimi_original"):
            raise ValueError("native Kimi observer already installed")
        modules = {}
        for name, digest in source_hashes.items():
            module = importlib.import_module(name)
            if hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest() != digest:
                raise ValueError(f"native Kimi runtime source mismatch: {name}")
            modules[name] = module
        spec = self.vllm_config.speculative_config
        parallel = self.vllm_config.parallel_config
        if (
            not self.vllm_config.model_config.enforce_eager
            or spec is None
            or not spec.use_dspark()
            or spec.num_speculative_tokens != 2
            or spec.enable_adaptive_verification
            or parallel.tensor_parallel_size != 8
            or parallel.enable_expert_parallel
            or not self.vllm_config.attention_config.use_prefill_query_quantization
            or getattr(self.vllm_config.attention_config.mla_prefill_backend, "name", None) != PREFILL_BACKEND
        ):
            raise ValueError("native Kimi observer requires eager TP8 EP1 fixed DSpark K2 and FP8 prefill queries")
        torch = importlib.import_module("torch")
        context = importlib.import_module("vllm.forward_context")
        distributed = importlib.import_module("vllm.distributed")
        holder = modules["vllm.models.kimi_k3.nvidia.mla"].MultiHeadLatentAttention
        original = holder.forward
        self._kimi_original = (holder, original)
        self._kimi_output = Path(output)
        self._kimi_rank = distributed.get_tensor_model_parallel_rank()
        self._kimi_events, self._kimi_failures = [], []
        self._kimi_sample = None

        @functools.wraps(original)
        def observed(layer, positions, hidden_states):
            if self._kimi_sample is None:
                return original(layer, positions, hidden_states)
            try:
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("native MLA observer requires eager forwards")
                metadata = context.get_forward_context().attn_metadata[layer.layer_name]
                draft = bool(layer.non_causal_multi_token_decode)
                # Quantization ignores self_attn.* in the pinned target and
                # draft is BF16. Confirm the actual modules before labeling.
                projections = [m for m in layer.modules() if hasattr(m, "quant_method") and hasattr(m, "weight")]
                if not projections or any(
                    type(m.quant_method).__name__ != "UnquantizedLinearMethod" or m.weight.dtype != torch.bfloat16
                    for m in projections
                ):
                    raise ValueError("native MLA projection precision differs from BF16 contract")
                if (
                    layer.q_lora_rank,
                    layer.kv_lora_rank,
                    layer.qk_nope_head_dim,
                    layer.qk_rope_head_dim,
                    layer.v_head_dim,
                ) != (1536, 512, 128, 64, 128):
                    raise ValueError("native MLA latent geometry differs from pinned contract")
                kv_dtype = layer.kv_cache_dtype
                if kv_dtype not in ("fp8", "fp8_e4m3"):
                    raise ValueError(f"unsupported native MLA cache dtype: {kv_dtype}")
                backend = layer.attn_backend.get_name()
                if backend != "FLASHINFER_MLA" or layer.prefill_backend.get_name() != PREFILL_BACKEND:
                    raise ValueError("native Kimi actual attention backend differs from the pinned contract")
                if metadata.num_prefills:
                    if metadata.prefill.q_data_type != torch.float8_e4m3fn:
                        raise ValueError("native Kimi prefill metadata did not select FP8 queries")
                    prefill = layer.prefill_backend
                    backend += "/" + type(prefill).__module__ + "." + type(prefill).__qualname__
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(layer, positions, hidden_states)
                end.record()
                row = {
                    "sample_id": self._kimi_sample,
                    "invocation_id": len(self._kimi_events),
                    "draft": draft,
                    "causal": bool(metadata.causal),
                    "num_prefills": int(metadata.num_prefills),
                    "num_decodes": int(metadata.num_decodes),
                    "num_heads": int(layer.num_local_heads),
                    "tp_size": 8,
                    "output_reduce_results": bool(layer.o_proj.reduce_results),
                    "gemm_quant_mode": "bfloat16",
                    "kv_cache_dtype": "fp8",
                    "prefill_query_quantization": True,
                    "prefill_backend": layer.prefill_backend.get_name(),
                    "kernel_source": backend,
                    "layer_name": layer.layer_name,
                    "gemm_reduce_scatter": layer.gemm_rs_ar is not None,
                }
                self._kimi_events.append(
                    (
                        row,
                        start,
                        end,
                        positions[: metadata.num_actual_tokens].detach().clone(),
                        metadata.query_start_loc[: metadata.num_reqs + 1].detach().clone(),
                    )
                )
                return result
            except Exception as error:
                self._kimi_failures.append(f"{type(error).__name__}: {error}")
                raise

        holder.forward = observed
        return {"rank": self._kimi_rank, "source_hashes": source_hashes}

    def kimi_observer_begin(self, sample):
        if self._kimi_sample is not None:
            raise ValueError("overlapping native observation")
        self._kimi_sample = sample

    def kimi_observer_finish(self):
        self._kimi_sample = None
        torch = importlib.import_module("torch")
        torch.cuda.synchronize()
        rows = []
        for row, start, end, positions, starts in self._kimi_events:
            row = dict(
                row,
                positions=positions.cpu().tolist(),
                query_starts=starts.cpu().tolist(),
                latency_ms=float(start.elapsed_time(end)),
            )
            try:
                physical_identity(row)
            except ValueError as error:
                self._kimi_failures.append(str(error))
            rows.append(row)
        payload = {"rank": self._kimi_rank, "rows": rows, "failures": self._kimi_failures}
        self._kimi_output.mkdir(parents=True, exist_ok=True)
        path = self._kimi_output / f"rank-{self._kimi_rank}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return {"rank": self._kimi_rank, "path": str(path), "rows": len(rows), "failures": len(self._kimi_failures)}

    def kimi_observer_restore(self):
        self._kimi_sample = None
        holder, original = self._kimi_original
        holder.forward = original
        del self._kimi_original
