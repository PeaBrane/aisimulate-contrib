# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile measured Pro0813 aggregated or P/D pools with Replay cache geometry."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def reconcile(receipt, *, k, concurrency, input_length, output_length):
    """Preserve pool block count; expose rounding and steady-residency estimates.

    The pinned KVCacheTensor contract overlays groups on one allocation. This
    adapter checks that contract; it never sums aliased tensor-view sizes.
    """
    ranks = receipt["results"]
    if len(ranks) != 8 or {r["rank"] for r in ranks} != set(range(8)):
        raise ValueError("requires all eight unique TP ranks")
    return _reconcile_pool(
        ranks, replicas=8, k=k, concurrency=concurrency, input_length=input_length, output_length=output_length
    )


def _reconcile_pool(ranks, *, replicas, k, concurrency, input_length, output_length):
    """Validate one independent pool or a TP-replicated set of identical pools."""
    if k not in (3, 6) or min(concurrency, input_length, output_length) <= 0:
        raise ValueError("requires approved Pro K and positive workload dimensions")
    config = ranks[0]["kv_cache_config"]
    if any(r["kv_cache_config"] != config for r in ranks):
        raise ValueError("rank cache configurations differ")
    groups, tensors = config["kv_cache_groups"], config["kv_cache_tensors"]
    if config["kv_cache_layout"] != "BLHNC" or config["kv_tp_replicas"] != replicas:
        raise ValueError("requires replicated Pro MLA with BLHNC layout")
    if any(g["host_resident"] for g in groups) or config["hisparse_host_num_blocks"] is not None:
        raise ValueError("external cache is outside this campaign")
    strides = {t["block_stride"] for t in tensors}
    if len(strides) != 1:
        raise ValueError("requires one shared physical pool stride")
    stride = strides.pop()
    blocks = config["num_blocks"]
    if stride <= 0 or blocks <= 0 or any(t["size"] != blocks * stride or t["host_resident"] for t in tensors):
        raise ValueError("tensor allocation differs from shared pool capacity")
    if any(r["available_kv_cache_memory_bytes"] < blocks * stride for r in ranks):
        raise ValueError("allocation exceeds runtime memory budget")
    layers, shape, bounded_blocks, full_groups = {}, [], 0, 0
    peak_blocks = 0
    # ac9126 config/vllm.py:max_concurrent_batches permits at most two
    # concurrent batches at PP1. Keep both complete K+1 verification rounds
    # beyond the output frontier, then K proposal lookahead slots. The pinned
    # scheduler intentionally does not clip every final draft round to OSL.
    max_concurrent_batches = 2
    transient_tokens = input_length + output_length + max_concurrent_batches * (k + 1) + k
    for group in groups:
        spec = group["kv_cache_spec"]
        members = spec["kv_cache_specs"]
        if set(members) != set(group["layer_names"]) or set(layers) & set(members):
            raise ValueError("cache layer ownership is missing or duplicated")
        layers.update(members)
        types = {(m["type"], m["block_size"], m.get("sliding_window")) for m in members.values()}
        if len(types) != 1:
            raise ValueError("heterogeneous cache group")
        kind, block_size, window = types.pop()
        shape.append((kind, block_size, window, len(members)))
        if window is None:
            if kind != "MLAAttentionSpec" or block_size != 256:
                raise ValueError("unsupported full-context group")
            full_groups += 1
        else:
            bounded_blocks += ceildiv(window + k, block_size) + 1
        # Deliberately loose: retain all tokens even in sliding groups. This
        # covers prefill transients without claiming exact window allocation.
        peak_blocks += ceildiv(transient_tokens, block_size) + 1
    expected = [
        ("SlidingWindowMLASpec", 64, 128, 32),
        ("SlidingWindowMLASpec", 64, 128, 32),
        ("MLAAttentionSpec", 256, None, 91),
        ("SlidingWindowMLASpec", 8, 128, 31),
        ("SlidingWindowMLASpec", 4, 8, 60),
    ]
    if shape != expected or full_groups != 1:
        raise ValueError("cache groups differ from qualified Pro0813 target plus three draft layers")
    seen = Counter()
    for tensor in tensors:
        for index, name in enumerate(tensor["layers"]):
            seen[name] += 1
            page = layers[name]["page_size_bytes"]
            end = tensor["offset"] + index * tensor["layer_stride"] + page
            if tensor["offset"] < 0 or end > stride:
                raise ValueError("layer page exceeds shared block stride")
    if seen != Counter(dict.fromkeys(layers, 1)):
        raise ValueError("physical tensor views omit or duplicate cache layers")
    logical_block = 256
    token_bytes = ceildiv(stride, logical_block)
    represented_stride = logical_block * token_bytes
    return {
        "schema": "dspark-pro-cache-reconciliation-v1",
        "physical_pool_blocks": blocks,
        "physical_bytes_per_block": stride,
        "physical_pool_bytes_per_rank": blocks * stride,
        "tensor_view_count": len(tensors),
        "cache_groups": shape,
        "target_layers": 61,
        "draft_layers": 3,
        "approximate_steady_state_blocks_per_request": bounded_blocks,
        "representation_rounding_bytes_per_block": represented_stride - stride,
        "conservative_peak_blocks": concurrency * peak_blocks + 1,
        "conservative_retained_tokens_per_group": transient_tokens,
        "max_concurrent_batches_bound": max_concurrent_batches,
        "capacity_proven_nonbinding": concurrency * peak_blocks + 1 < blocks,
        "workload": {"k": k, "concurrency": concurrency, "input_length": input_length, "output_length": output_length},
        "replay_kv_cache": {
            "prefix_caching": False,
            "block_size": logical_block,
            "bytes_per_token": token_bytes,
            "capacity": {"type": "fixed", "blocks": blocks},
            "state_cache": {"bytes_per_request": bounded_blocks * represented_stride},
        },
        "approximations": [
            "Sliding-group residency is a fixed steady-state estimate, not a native active-step upper bound.",
            "Temporary prefill residency is not simulated; a full-context retention bound checks nonbinding capacity.",
            "The PP1 retention bound includes two complete unsettled verify rounds plus draft proposal lookahead.",
            "Integer token-byte rounding changes reported bytes slightly; physical pool block count is unchanged.",
            "This fixed-length prefix-disabled mapping does not qualify AgentX prefix reuse or P/D transfer.",
        ],
    }


def reconcile_pd(receipt, *, k, concurrency, input_length, output_length):
    """P TP1/DP8/EP8 has independent pools; D TP8/EP1 replicates one request set.

    Capacity is checked per physical pool, without dividing concurrency by DP.
    This covers uneven routing and source blocks held until transfer completes.
    It does not estimate NIXL bytes, bandwidth or transfer completion latency.
    """
    if set(receipt) != {"prefill", "decode"}:
        raise ValueError("P/D receipt requires separate prefill and decode roles")
    results = {}
    for role, dp, tp in (("prefill", 8, 1), ("decode", 1, 8)):
        ranks = receipt[role]
        expected = {(d, t) for d in range(dp) for t in range(tp)}
        keys = [(r["data_parallel_index"], r["tensor_parallel_rank"]) for r in ranks]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"{role} requires every unique DP/TP worker")
        for rank in ranks:
            if rank["schema"] != "dspark-native-cache-v1" or rank["parallelism"] != {
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
                "pipeline_parallel_size": 1,
                "enable_expert_parallel": role == "prefill",
            }:
                raise ValueError(f"{role} differs from pinned Pro P/D topology")
            groups = rank["kv_cache_config"]["kv_cache_groups"]
            if rank["transfer_group_ids"] != list(range(len(groups))):
                raise ValueError(f"{role} must expose all target and draft transfer groups")
            swa = {name for group in groups[:2] for name in group["layer_names"]}
            if swa != {f"model.layers.{i}.attn.swa_cache" for i in range(64)}:
                raise ValueError(f"{role} requires 61 target and three draft SWA layers")
            if not swa <= set(rank["layer_specs"]):
                raise ValueError(f"{role} runtime layer specs omit target or draft SWA")
        # Source serves only a hidden one-token response, then retains its
        # prompt blocks for the consumer; D owns the requested output window.
        role_output = 1 if role == "prefill" else output_length
        pools = []
        for d in range(dp):
            members = sorted(
                (r for r in ranks if r["data_parallel_index"] == d), key=lambda r: r["tensor_parallel_rank"]
            )
            pool = _reconcile_pool(
                members,
                replicas=tp,
                k=k,
                concurrency=concurrency,
                input_length=input_length,
                output_length=role_output,
            )
            pool["data_parallel_index"] = d
            pool["tensor_parallel_ranks"] = list(range(tp))
            pool["approximations"][-1] = "Role-specific prefix-disabled cache mapping; transport is not qualified here."
            pools.append(pool)
        geometry = {(p["physical_bytes_per_block"], p["approximate_steady_state_blocks_per_request"]) for p in pools}
        if len(geometry) != 1:
            raise ValueError(f"{role} pool geometries differ; one Replay role template cannot represent them")
        limiting = min(pools, key=lambda p: p["physical_pool_blocks"])
        results[role] = {
            "independent_request_pools": dp,
            "physical_tp_copies_per_pool": tp,
            "rank_pools": pools,
            "physical_pool_bytes_across_role": sum(p["physical_pool_bytes_per_rank"] * tp for p in pools),
            "capacity_proven_nonbinding": all(p["capacity_proven_nonbinding"] for p in pools),
            "concurrency_bound_per_pool": concurrency,
            "replay_capacity_policy": "minimum measured capacity across independent DP pools; no capacity pooling",
            "replay_kv_cache": limiting["replay_kv_cache"],
        }
    return {
        "schema": "dspark-pro-pd-cache-reconciliation-v1",
        "roles": results,
        "capacity_proven_nonbinding": all(r["capacity_proven_nonbinding"] for r in results.values()),
        "workload": {"k": k, "concurrency": concurrency, "input_length": input_length, "output_length": output_length},
        "source_prefill_backoff_tokens": 0,
        "decode_recompute_tokens": 1,
        "transport": {
            "logical_source_engines_per_request": 1,
            "destination_tp_readers": 8,
            "source_release_notifications": 8,
            "request_wire_bytes": None,
            "bandwidth_qualified": False,
            "scope": "Logical per-rank state, replicated destination reads, and physical wire bytes are distinct.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--input-length", required=True, type=int)
    parser.add_argument("--output-length", required=True, type=int)
    parser.add_argument("--deployment", choices=("aggregated", "disaggregated"), default="aggregated")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    result = (reconcile_pd if args.deployment == "disaggregated" else reconcile)(
        json.loads(raw),
        k=args.k,
        concurrency=args.concurrency,
        input_length=args.input_length,
        output_length=args.output_length,
    )
    result["runtime_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    result["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
