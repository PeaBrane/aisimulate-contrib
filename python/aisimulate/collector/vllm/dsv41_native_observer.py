# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe real eager V4.1 serving calls without constructing framework metadata.

Original integration with vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516
(Apache-2.0). See README.dsv41-native.md for source anchors and timing ownership.
"""

import functools
import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path

from collector.vllm.dsv41_native_contract import coverage_summary, query_identity


class NativeObserver:
    """One selected site class per pass prevents nested CUDA interval contamination."""

    def __init__(self, torch, get_context, output, rank):
        self.torch = torch
        self.get_context = get_context
        self.output = Path(output)
        self.rank = rank
        self.active = False
        self.sample = None
        self.site_kind = None
        self.frames = []
        self.events = []
        self.failures = []
        self.originals = []
        self.roots = []
        self.last_target = None
        self.preparation = None

    def _metadata(self, attention):
        metadata = self.get_context().attn_metadata
        swa = metadata.get(attention.swa_cache_layer.prefix) if isinstance(metadata, dict) else None
        if swa is None or swa.query_start_loc_cpu is None:
            raise RuntimeError("real SWA request metadata is required; profile/dummy calls cannot be measured")
        starts = swa.query_start_loc_cpu.tolist()
        counts = {"num_decodes": int(swa.num_decodes), "num_prefills": int(swa.num_prefills)}
        if len(starts) != sum(counts.values()) + 1:
            raise RuntimeError("native SWA metadata contains padded/unknown request boundaries")
        return starts, counts, swa

    @staticmethod
    def _cache_identity(attention):
        cache = attention.swa_cache_layer
        tensor = getattr(cache, "kv_cache", None)
        return {
            "kv_mxfp8": getattr(attention, "kv_mxfp8", None),
            "swa_bytes_per_token": getattr(attention, "swa_bytes_per_token", None),
            "compressed_bytes_per_token": getattr(attention, "compressed_bytes_per_token", None),
            "native_cache_dtype": getattr(attention, "kv_cache_dtype", None),
            "tensor_dtype": str(tensor.dtype) if tensor is not None else None,
            "record_width": int(tensor.shape[-1]) if tensor is not None and len(tensor.shape) else None,
        }

    def _boundary_row(self, frame, site, callable_name, start, end, positions):
        row = {
            "invocation_id": len(self.events),
            "parent_invocation_id": None,
            "sample_id": self.sample,
            "tp_rank": self.rank,
            "site": site,
            "callable": callable_name,
            "layer_id": frame["layer_id"],
            "owner": frame["owner"],
            "module_prefix": frame["prefix"],
            "measurement_scope": "inclusive_native_site",
            "used_cuda_graph": False,
            "input_token_rows": int(positions.shape[0]),
            "query_starts": frame["query_starts"],
            "phase_counts": frame["phase_counts"],
            "layer_execution": frame["layer_execution"],
            "decoder_replay": frame.get("decoder_replay"),
            "cache_identity": frame["cache_identity"],
        }
        self.events.append((row, start, end, positions.detach().clone()))

    def wrap_root(self, holder, name, owner):
        """Measure only the tail after the final decoder layer returns.

        The tail includes native post/reduction, bounded replay scatter, final
        collapse and target norm. Layer intervals never overlap this interval.
        Root metadata is retained because replay scatters back to full rows.
        """
        original = getattr(holder, name)

        @functools.wraps(original)
        def observed(model, *args, **kwargs):
            if not self.active:
                return original(model, *args, **kwargs)
            positions = args[1] if len(args) > 1 else kwargs["positions"]
            layers = [layer for layer in model.layers if hasattr(layer, "attn")]
            starts, counts, _ = self._metadata(layers[0].attn)
            frame = {
                "owner": owner,
                "layer_id": 39 if owner == "target" else 42,
                "prefix": layers[-1].attn.prefix,
                "first_prefix": layers[0].attn.prefix,
                "first_layer_id": 0 if owner == "target" else 40,
                "positions": positions,
                "query_starts": starts,
                "phase_counts": counts,
                "cache_identity": self._cache_identity(layers[0].attn),
                "layer_execution": {
                    "fuse_mhc_all_reduce": bool(getattr(model, "fuse_mhc_all_reduce", layers[0].fuse_mhc_all_reduce)),
                    "use_sequence_parallel": bool(model.use_sequence_parallel),
                },
            }
            frame["decoder_replay"] = (
                model.decoder_replay_layers is not None
                if owner == "target"
                else self.last_target["decoder_replay"]
                if self.last_target is not None
                else None
            )
            self.roots.append(frame)
            try:
                if self.site_kind == "input_preparation":
                    event = self.torch.cuda.Event(enable_timing=True)
                    event.record()
                    frame["input_start"] = event
                result = original(model, *args, **kwargs)
                if self.site_kind == "input_preparation" and "input_start" in frame:
                    raise RuntimeError("native input preparation did not reach the first layer")
                if self.site_kind == "finalize":
                    if "tail_start" not in frame:
                        raise RuntimeError("native finalization has no observed final layer boundary")
                    end = self.torch.cuda.Event(enable_timing=True)
                    end.record()
                    self._boundary_row(
                        frame,
                        "finalize",
                        f"{original.__module__}.{original.__qualname__}:tail",
                        frame["tail_start"],
                        end,
                        positions,
                    )
                if owner == "target":
                    self.last_target = frame
                return result
            except Exception as error:
                self.failures.append({"sample_id": self.sample, "site": "finalize", "error": str(error)})
                raise
            finally:
                self.roots.pop()

        setattr(holder, name, observed)
        self.originals.append((holder, name, original))

    def wrap_preparation(self, holder):
        """Bracket actual projection through KV insertion, preserving producer state."""
        combine = holder.combine_hidden_states
        insert = holder.precompute_and_store_context_kv

        @functools.wraps(combine)
        def start(model, *args, **kwargs):
            if self.active and self.site_kind == "context_preparation":
                if self.preparation is not None or self.last_target is None:
                    raise RuntimeError("native context preparation requires one preceding target forward")
                event = self.torch.cuda.Event(enable_timing=True)
                event.record()
                self.preparation = event
            return combine(model, *args, **kwargs)

        @functools.wraps(insert)
        def finish(model, *args, **kwargs):
            result = insert(model, *args, **kwargs)
            if self.active and self.site_kind == "context_preparation":
                if self.preparation is None:
                    raise RuntimeError("native context insertion has no projection boundary")
                end = self.torch.cuda.Event(enable_timing=True)
                end.record()
                positions = args[1] if len(args) > 1 else kwargs["context_positions"]
                frame = {
                    **self.last_target,
                    "owner": "draft",
                    "layer_id": 40,
                    "prefix": model.layers[0].attn.prefix,
                    "cache_identity": self._cache_identity(model.layers[0].attn),
                }
                self._boundary_row(
                    frame,
                    "context_preparation",
                    f"{combine.__module__}.{combine.__qualname__}:through_context_kv",
                    self.preparation,
                    end,
                    positions,
                )
                self.preparation = None
            return result

        holder.combine_hidden_states, holder.precompute_and_store_context_kv = start, finish
        self.originals.extend(
            [(holder, "combine_hidden_states", combine), (holder, "precompute_and_store_context_kv", insert)]
        )

    def wrap(self, holder, name, kind):
        original = getattr(holder, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            frame = self.frames[-1] if self.frames else None
            own_frame = kind == "layer"
            if own_frame:
                layer = args[0]
                # Native layer.forward(self, x, positions, ...), model.py:452.
                positions = args[2] if len(args) > 2 else kwargs["positions"]
                prefix = layer.attn.prefix
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
                if match is None:
                    raise RuntimeError(f"unrecognized native layer owner: {prefix}")
                layer_id = int(match.group(1))
                frame = {
                    "layer_id": layer_id,
                    "owner": "target" if layer_id < 40 else "draft",
                    "prefix": prefix,
                    "positions": positions,
                    "attention": layer.attn,
                    "fuse_mhc_all_reduce": bool(getattr(layer, "fuse_mhc_all_reduce", False)),
                    "use_sequence_parallel": bool(getattr(layer, "use_sequence_parallel", False)),
                    "has_engram": getattr(layer, "engram", None) is not None,
                    "capture_previous_aux": bool(kwargs.get("capture_previous_aux", False)),
                }
                self.frames.append(frame)
            try:
                if own_frame and self.site_kind == "input_preparation" and self.roots:
                    root = self.roots[-1]
                    if frame["layer_id"] == root["first_layer_id"]:
                        end = self.torch.cuda.Event(enable_timing=True)
                        end.record()
                        source = {**root, "layer_id": root["first_layer_id"], "prefix": root["first_prefix"]}
                        self._boundary_row(
                            source,
                            "input_preparation",
                            f"{original.__module__}:before_first_layer",
                            root.pop("input_start"),
                            end,
                            root["positions"],
                        )
                if kind != self.site_kind or frame is None:
                    result = original(*args, **kwargs)
                    if (
                        kind == "layer"
                        and self.site_kind == "finalize"
                        and self.roots
                        and frame["layer_id"] == self.roots[-1]["layer_id"]
                    ):
                        start = self.torch.cuda.Event(enable_timing=True)
                        start.record()
                        self.roots[-1]["tail_start"] = start
                    return result
                if self.torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("native observer requires eager execution; graph capture is not a measurement")
                starts, counts, swa = self._metadata(frame["attention"])
                start = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(*args, **kwargs)
                end.record()
                # Clone only AFTER the timed interval; native buffers are reused.
                positions = frame["positions"][: starts[-1]].detach().clone()
                row = {
                    "invocation_id": len(self.events),
                    "parent_invocation_id": None,
                    "sample_id": self.sample,
                    "tp_rank": self.rank,
                    "site": kind,
                    "callable": f"{original.__module__}.{original.__qualname__}",
                    "layer_id": frame["layer_id"],
                    "owner": frame["owner"],
                    "module_prefix": frame["prefix"],
                    "measurement_scope": "inclusive_native_site",
                    "used_cuda_graph": False,
                    "input_token_rows": int(frame["positions"].shape[0]),
                    "query_starts": starts,
                    "phase_counts": counts,
                    "cache_identity": self._cache_identity(frame["attention"]),
                    "decoder_replay": self.roots[-1]["decoder_replay"] if self.roots else None,
                    "decode_swa_width": getattr(swa, "decode_swa_width", None),
                    "max_decode_query_len": getattr(swa, "max_decode_query_len", None),
                    "includes_fused_all_reduce": bool(kwargs.get("reduce_results", False)),
                    "capture_aux": bool(kwargs.get("capture_aux", False)),
                    "layer_execution": {
                        key: frame[key]
                        for key in (
                            "fuse_mhc_all_reduce",
                            "use_sequence_parallel",
                            "has_engram",
                            "capture_previous_aux",
                        )
                    },
                }
                if kind in ("layer", "attention"):
                    attn = frame["attention"]
                    row["attention_identity"] = {
                        "module": f"{type(attn).__module__}.{type(attn).__qualname__}",
                        "backend": attn.backend_cls.get_name(),
                        "local_heads": int(attn.n_local_heads),
                        "head_dim": int(attn.head_dim),
                        "compress_ratio": int(attn.compress_ratio),
                        "window_size": int(attn.window_size),
                        "output_reduce_results": bool(attn.wo_b.reduce_results),
                        "gemm_reduce_scatter": attn.gemm_rs is not None,
                    }
                self.events.append((row, start, end, positions))
                return result
            except Exception as error:
                self.failures.append(
                    {"sample_id": self.sample, "site": kind, "error": f"{type(error).__name__}: {error}"}
                )
                raise
            finally:
                if own_frame:
                    self.frames.pop()

        setattr(holder, name, observed)
        self.originals.append((holder, name, original))

    def begin(self, sample, kind):
        if self.active or kind not in (
            "layer",
            "attention",
            "mhc",
            "engram",
            "finalize",
            "context_preparation",
            "input_preparation",
        ):
            raise ValueError("invalid or overlapping native observation pass")
        self.sample, self.site_kind, self.active = sample, kind, True
        self.last_target = None
        self.preparation = None

    def finish(self):
        self.active = False
        if self.preparation is not None:
            self.failures.append({"sample_id": self.sample, "error": "unfinished native context preparation"})
        self.torch.cuda.synchronize()
        rows = []
        for row, start, end, positions in self.events:
            row["latency_ms"] = float(start.elapsed_time(end))
            row["positions"] = positions.cpu().tolist()
            try:
                row["query_identity"] = query_identity(
                    row["query_starts"],
                    row["positions"],
                    owner=row["owner"],
                    phase_counts=row["phase_counts"],
                    causal=row["owner"] == "target",
                )
            except ValueError as error:
                row["query_identity"] = None
                self.failures.append({"invocation_id": row["invocation_id"], "error": str(error)})
            rows.append(row)
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"rank-{self.rank}.json"
        payload = {"rows": rows, "failures": self.failures, "coverage": coverage_summary(rows, self.failures)}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        return {"rank": self.rank, "path": str(path), "coverage": payload["coverage"]}

    def restore(self):
        self.active = False
        for holder, name, original in reversed(self.originals):
            setattr(holder, name, original)
        self.originals.clear()


class Dsv41ObserverWorker:
    """vLLM public worker extension; imports framework only in the GPU worker."""

    def dsv41_observer_install(self, output, source_hashes):
        if hasattr(self, "_dsv41_observer"):
            raise RuntimeError("observer already installed")
        modules = {}
        for name, expected in source_hashes.items():
            module = importlib.import_module(name)
            path = Path(inspect.getsourcefile(module))
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"runtime source differs from pinned observer contract: {name}")
            modules[name] = module
        config = self.vllm_config
        spec = config.speculative_config
        if not config.model_config.enforce_eager or spec is None or not spec.use_dspark():
            raise ValueError("observer requires real eager DSpark serving")
        if spec.num_speculative_tokens != 5 or spec.enable_adaptive_verification:
            raise ValueError("initial V4.1 observer requires fixed K5 and adaptive verification disabled")
        if config.parallel_config.tensor_parallel_size != 4 or config.parallel_config.enable_expert_parallel:
            raise ValueError("initial observer requires TP4 EP1")
        torch = importlib.import_module("torch")
        context = importlib.import_module("vllm.forward_context")
        distributed = importlib.import_module("vllm.distributed")
        observer = NativeObserver(
            torch, context.get_forward_context, output, distributed.get_tensor_model_parallel_rank()
        )
        model = modules["vllm.models.deepseek_v41.nvidia.model"]
        draft = modules["vllm.models.deepseek_v41.nvidia.dspark"]
        attention = modules["vllm.models.deepseek_v41.attention"]
        observer.wrap_root(model.DeepseekV4Model, "forward", "target")
        observer.wrap_root(draft.DSparkDeepseekV4Model, "forward", "draft")
        observer.wrap_preparation(draft.DSparkDeepseekV4Model)
        observer.wrap(model.DeepseekV4DecoderLayer, "forward", "layer")
        observer.wrap(attention.DeepseekV4Attention, "forward", "attention")
        for name in ("mhc_pre_delayed_tilelang", "mhc_shifted_post_pre", "mhc_post_tilelang"):
            observer.wrap(model, name, "mhc")
        observer.wrap(modules["vllm.models.deepseek_v41.common.engram"].Engram, "forward", "engram")
        self._dsv41_observer = observer
        return {"installed": True, "rank": observer.rank, "source_hashes": source_hashes}

    def dsv41_observer_begin(self, sample, kind):
        self._dsv41_observer.begin(sample, kind)

    def dsv41_observer_finish(self):
        return self._dsv41_observer.finish()

    def dsv41_observer_restore(self):
        self._dsv41_observer.restore()
