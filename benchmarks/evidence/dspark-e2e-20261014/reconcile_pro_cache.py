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


def validate_pool_observations(evidence, receipt):
    """Bind complete physical-pool observations to one measured serving window."""
    if evidence.get("schema") != "dspark-native-pool-evidence-v1":
        raise ValueError("unsupported native pool evidence schema")
    run_id, window = evidence.get("serving_run_id"), evidence.get("window", {})
    start, end = window.get("start_unix_ns"), window.get("end_unix_ns")
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or type(start) is not int
        or type(end) is not int
        or not 0 < start < end
        or not isinstance(window.get("source"), str)
        or not window["source"].strip()
        or not isinstance(window.get("sha256"), str)
        or len(window["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in window["sha256"])
    ):
        raise ValueError("pool evidence needs a serving identity and hashed absolute measurement window")
    root = Path(__file__).parent
    measurement_hash = evidence.get("measurement_identity_sha256")
    if (
        not isinstance(measurement_hash, str)
        or len(measurement_hash) != 64
        or any(c not in "0123456789abcdef" for c in measurement_hash)
    ):
        raise ValueError("pool evidence needs the frozen measurement identity hash")
    sources = json.loads((root / "source_pool_sources.json").read_text())
    observer_hash = hashlib.sha256((root / "source_pool_observer.py").read_bytes()).hexdigest()
    roles = evidence.get("roles", {})
    if set(roles) != {"prefill", "decode"}:
        raise ValueError("pool observations require both roles")
    result = {}
    for role, dp, tp in (("prefill", 8, 1), ("decode", 1, 8)):
        observations = roles[role]
        if (
            not isinstance(observations, list)
            or len(observations) != dp
            or any(type(r.get("data_parallel_index")) is not int for r in observations)
            or {r.get("data_parallel_index") for r in observations} != set(range(dp))
        ):
            raise ValueError(f"{role} pool observations require every unique engine")
        for row in observations:
            rank = row["data_parallel_index"]
            capacities = {r["kv_cache_config"]["num_blocks"] for r in receipt[role] if r["data_parallel_index"] == rank}
            if (
                row.get("schema") != "dspark-native-pool-observation-v1"
                or row.get("run_id") != run_id
                or row.get("role") != role
                or row.get("tensor_parallel_size") != tp
                or row.get("data_parallel_size") != dp
                or row.get("async_scheduler") is not True
                or row.get("prefix_caching") is not False
                or row.get("initial_null_block_included") is not True
                or row.get("runtime_revision") != "ac9126e58aa7bbab1856ba6593ba4d5003fea516"
                or row.get("source_hashes") != sources
                or row.get("observer_sha256") != observer_hash
                or capacities != {row.get("num_blocks")}
            ):
                raise ValueError(f"{role} pool observation differs from pinned runtime, observer or capacity")
            if (
                row.get("complete") is not True
                or row.get("reason") != "shutdown"
                or type(row.get("started_ns")) is not int
                or type(row.get("updated_ns")) is not int
                or not row["started_ns"] <= start < end <= row["updated_ns"]
            ):
                raise ValueError("pool observation does not cover the complete measurement window")
            counts = (
                "peak_used_blocks",
                "current_used_blocks",
                "pool_mutation_calls",
                "allocation_attempts",
                "allocation_denials",
                "observation_or_allocation_errors",
                "preemptions",
            )
            if any(type(row.get(key)) is not int or row[key] < 0 for key in counts):
                raise ValueError("pool observation counters must be nonnegative integers")
            if not 1 <= row["current_used_blocks"] <= row["peak_used_blocks"] <= row["num_blocks"]:
                raise ValueError("pool observation has invalid physical occupancy")
            if row["allocation_denials"] > row["allocation_attempts"]:
                raise ValueError("pool observation has more denials than allocation attempts")
            result[(role, rank)] = row
        if not any(row["allocation_attempts"] for row in observations):
            raise ValueError(f"{role} pool observations contain no allocation attempts")
    return result


def read_pool_evidence(path, runtime_receipt_hash):
    evidence = json.loads(path.read_text())
    if evidence.get("runtime_cache_receipt_sha256") != runtime_receipt_hash:
        raise ValueError("pool evidence runtime cache receipt hash differs")
    window = evidence["window"]
    raw = (path.parent / window["source"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != window["sha256"]:
        raise ValueError("pool evidence measurement window hash differs")
    captured = json.loads(raw)
    if (
        captured.get("serving_run_id") != evidence["serving_run_id"]
        or captured.get("identity_sha256") != evidence["measurement_identity_sha256"]
    ) or any(captured.get(key) != window[key] for key in ("start_unix_ns", "end_unix_ns")):
        raise ValueError("pool evidence measurement window identity differs")
    return evidence


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


def reconcile_pd(
    receipt,
    *,
    k,
    concurrency,
    input_length,
    output_length,
    bandwidth_evidence=None,
    source_residency_evidence=None,
    pool_observation_evidence=None,
):
    """P TP1/DP8/EP8 has independent pools; D TP8/EP1 replicates one request set.

    Capacity is checked per physical pool, without dividing concurrency by DP.
    This covers uneven routing and source blocks held until transfer completes.
    Transfer bounds include packed registration padding, not protocol wire overhead.
    """
    if set(receipt) != {"prefill", "decode"}:
        raise ValueError("P/D receipt requires separate prefill and decode roles")
    if source_residency_evidence is not None and pool_observation_evidence is not None:
        raise ValueError("choose source residency bounds or native pool observations, not both")
    observations = (
        validate_pool_observations(pool_observation_evidence, receipt) if pool_observation_evidence is not None else {}
    )
    source_resident_bounds = [None] * 8
    if source_residency_evidence is not None:
        if (
            source_residency_evidence.get("kind") not in ("measured", "conservative")
            or not isinstance(source_residency_evidence.get("source"), str)
            or not source_residency_evidence["source"].strip()
            or source_residency_evidence.get("includes_active_and_finished_holds") is not True
        ):
            raise ValueError("source residency evidence must bound active and finished held requests per pool")
        scalar = source_residency_evidence.get("max_resident_requests_per_pool")
        by_rank = source_residency_evidence.get("max_resident_requests_by_dp_rank")
        if (scalar is None) == (by_rank is None):
            raise ValueError("source residency requires exactly one scalar or complete DP-rank bound")
        if scalar is not None:
            source_resident_bounds = [scalar] * 8
        elif isinstance(by_rank, dict) and set(by_rank) == {str(i) for i in range(8)}:
            source_resident_bounds = [by_rank[str(i)] for i in range(8)]
        else:
            raise ValueError("source residency requires all eight DP-rank bounds")
        if any(type(bound) is not int or bound < 0 for bound in source_resident_bounds):
            raise ValueError("source residency bounds must be nonnegative integers")
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
            source_resident_bound = source_resident_bounds[d]
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
            # Reserve every sliding group for the entire workload up front.
            # This avoids pretending rolling windows are recurrent state whose
            # position cannot be rewound under the pinned B0/R1 contract.
            resident_bound = max(concurrency, source_resident_bound or 0) if role == "prefill" else concurrency
            reserve = resident_bound * sum(
                ceildiv(pool["conservative_retained_tokens_per_group"], block) + 1
                for _, block, window, _ in pool["cache_groups"]
                if window is not None
            )
            pool["static_rolling_reserve_blocks"] = reserve
            pool["static_resident_requests_bound"] = resident_bound
            pool["remaining_full_context_blocks"] = pool["physical_pool_blocks"] - reserve
            full_blocks_per_request = ceildiv(pool["conservative_retained_tokens_per_group"], 256) + 1
            # C bounds Replay's atomically released source copies. Native P
            # cleanup is asynchronous and requires a separate resident bound.
            pool["replay_peak_with_static_reserve_blocks"] = reserve + concurrency * full_blocks_per_request + 1
            pool["replay_capacity_proven_nonbinding"] = (
                pool["replay_peak_with_static_reserve_blocks"] < pool["physical_pool_blocks"]
            )
            native_bound = source_resident_bound if role == "prefill" else concurrency
            pool["native_conservative_peak_blocks"] = (
                reserve + native_bound * full_blocks_per_request + 1 if native_bound is not None else None
            )
            pool["native_resident_requests_bound"] = native_bound
            pool["native_capacity_proven_nonbinding"] = (
                native_bound is not None and pool["native_conservative_peak_blocks"] < pool["physical_pool_blocks"]
            )
            if observations:
                observation = observations[(role, d)]
                pool["native_observed_peak_blocks"] = observation["peak_used_blocks"]
                pool["native_capacity_proven_nonbinding"] = (
                    observation["peak_used_blocks"] < pool["physical_pool_blocks"]
                    and observation["allocation_denials"] == 0
                    and observation["observation_or_allocation_errors"] == 0
                    and observation["preemptions"] == 0
                )
            cache = pool["replay_kv_cache"]
            del cache["state_cache"]
            cache["capacity"]["blocks"] = pool["remaining_full_context_blocks"]
            if not pool["replay_capacity_proven_nonbinding"]:
                pool["replay_kv_cache"] = None
            pool["approximations"][-1] = "Role-specific prefix-disabled cache mapping; transport is not qualified here."
            pools.append(pool)
        geometry = {(p["physical_bytes_per_block"], p["approximate_steady_state_blocks_per_request"]) for p in pools}
        if len(geometry) != 1:
            raise ValueError(f"{role} pool geometries differ; one Replay role template cannot represent them")
        limiting = min(pools, key=lambda p: p["remaining_full_context_blocks"])
        results[role] = {
            "independent_request_pools": dp,
            "physical_tp_copies_per_pool": tp,
            "rank_pools": pools,
            "physical_pool_bytes_across_role": sum(p["physical_pool_bytes_per_rank"] * tp for p in pools),
            "capacity_proven_nonbinding": all(p["capacity_proven_nonbinding"] for p in pools),
            "replay_capacity_proven_nonbinding": all(p["replay_capacity_proven_nonbinding"] for p in pools),
            "native_capacity_proven_nonbinding": all(p["native_capacity_proven_nonbinding"] for p in pools),
            "concurrency_bound_per_pool": concurrency,
            "replay_capacity_policy": "minimum remaining capacity after static rolling reserve; no capacity pooling",
            "replay_kv_cache": limiting["replay_kv_cache"],
        }
    # ac9126 nixl/base_scheduler.py:135-147,252-295 clips each sliding
    # transfer group to ceil(window / block_size) + 1 blocks. Use the same
    # complete async allocation bound as P's independent capacity proof.
    transfer_groups = []
    source_token_bound = results["prefill"]["rank_pools"][0]["conservative_retained_tokens_per_group"]
    for group in receipt["prefill"][0]["kv_cache_config"]["kv_cache_groups"]:
        specs = group["kv_cache_spec"]["kv_cache_specs"]
        first = next(iter(specs.values()))
        block, window = first["block_size"], first.get("sliding_window")
        count = ceildiv(source_token_bound, block)
        if window is not None:
            count = min(count, ceildiv(window, block) + 1)
        transfer_groups.append(
            {
                "layers": list(specs),
                "block_count_bound": count,
                "page_bytes_per_block": sum(s["page_size_bytes"] for s in specs.values()),
            }
        )
    # All DP engines must use the same per-layer physical geometry. Capacity
    # may differ after runtime reservation; storage shape may not.
    reference_groups = receipt["prefill"][0]["kv_cache_config"]["kv_cache_groups"]
    if any(r["kv_cache_config"]["kv_cache_groups"] != reference_groups for rows in receipt.values() for r in rows):
        raise ValueError("P/D layer storage geometries differ")
    payload_bound = sum(g["block_count_bound"] * g["page_bytes_per_block"] for g in transfer_groups)
    # Packed MLA aliases register one whole storage-row region, not only the
    # layer pages: pinned nixl/base_worker.py:1664-1674,1699-1714. Group block
    # IDs address that shared region once per group block (:234-259).
    stride = results["prefill"]["rank_pools"][0]["physical_bytes_per_block"]
    registered_bound = sum(g["block_count_bound"] for g in transfer_groups) * stride
    bandwidth = None
    if bandwidth_evidence is not None:
        if (
            bandwidth_evidence.get("kind") not in ("measured", "assumed")
            or not isinstance(bandwidth_evidence.get("source"), str)
            or not bandwidth_evidence["source"].strip()
            or type(bandwidth_evidence.get("gb_per_second")) not in (int, float)
            or not 0 < bandwidth_evidence["gb_per_second"] < float("inf")
        ):
            raise ValueError("bandwidth evidence requires kind, positive finite gb_per_second, and source")
        bandwidth = bandwidth_evidence["gb_per_second"]
    return {
        "schema": "dspark-pro-pd-cache-reconciliation-v1",
        "roles": results,
        "capacity_proven_nonbinding": all(r["capacity_proven_nonbinding"] for r in results.values()),
        "capacity_proof_scope": "capacity_proven_nonbinding retains the original C-request Replay bound only.",
        "replay_capacity_proven_nonbinding": all(r["replay_capacity_proven_nonbinding"] for r in results.values()),
        "native_capacity_proven_nonbinding": all(r["native_capacity_proven_nonbinding"] for r in results.values()),
        "source_residency": {
            "evidence": source_residency_evidence,
            "native_bound_established": all(bound is not None for bound in source_resident_bounds),
            "scope": "Active prefill plus completed requests holding cache for connector cleanup, per DP pool.",
            "limitation": "Global client concurrency and max_num_seqs do not bound delayed native source holds.",
            "source_contract": "vllm ac9126 scheduler.py:893,2756-2764,3154-3157; holds are outside running queues.",
        },
        "native_pool_observations": pool_observation_evidence,
        "workload": {"k": k, "concurrency": concurrency, "input_length": input_length, "output_length": output_length},
        "source_prefill_backoff_tokens": 0,
        "decode_recompute_tokens": 1,
        "scope": "Fixed-length, prefix-disabled, closed-loop workload with proven nonbinding capacity only.",
        "replay_kv_transfer": {
            "bytes_per_token": ceildiv(registered_bound, input_length),
            "timing_mode": "full_prompt",
            "prefill_backoff_tokens": 0,
            "decode_recompute_tokens": 1,
            "bandwidth_gb_per_second": bandwidth,
        },
        "transport": {
            "logical_source_engines_per_request": 1,
            "destination_tp_readers": 8,
            "source_release_notifications": 8,
            "request_wire_bytes": None,
            "bandwidth_qualified": False,
            "bandwidth_evidence": bandwidth_evidence,
            "timing_status": "unmodeled" if bandwidth is None else "explicit bandwidth assumption with bounded payload",
            "per_reader_page_payload_bound_bytes": payload_bound,
            "per_reader_registered_region_bound_bytes": registered_bound,
            "registered_region_bytes_per_block": stride,
            "source_allocated_tokens_bound": source_token_bound,
            "scalar_rounding_bytes": ceildiv(registered_bound, input_length) * input_length - registered_bound,
            "groups": transfer_groups,
            "source_contract": "vllm ac9126e58aa7bbab1856ba6593ba4d5003fea516 nixl/base_scheduler.py:135-147,252-295",
            "packed_region_contract": "vllm ac9126 nixl/base_worker.py:234-259,1664-1674,1699-1714",
            "scope": "Logical content, padded registered payload per reader, and aggregate wire bytes are distinct.",
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
    parser.add_argument("--bandwidth-receipt", type=Path, help="Explicit measured/assumed bandwidth evidence for P/D")
    parser.add_argument("--source-residency-receipt", type=Path, help="Bound all active and finished held P requests")
    parser.add_argument("--pool-observation-receipt", type=Path, help="Complete native pool observations and window")
    args = parser.parse_args()
    if (
        args.bandwidth_receipt or args.source_residency_receipt or args.pool_observation_receipt
    ) and args.deployment != "disaggregated":
        parser.error("P/D evidence receipts require --deployment disaggregated")
    raw = args.receipt.read_bytes()
    runtime_hash = hashlib.sha256(raw).hexdigest()
    result = (reconcile_pd if args.deployment == "disaggregated" else reconcile)(
        json.loads(raw),
        k=args.k,
        concurrency=args.concurrency,
        input_length=args.input_length,
        output_length=args.output_length,
        **({"bandwidth_evidence": json.loads(args.bandwidth_receipt.read_text())} if args.bandwidth_receipt else {}),
        **(
            {"source_residency_evidence": json.loads(args.source_residency_receipt.read_text())}
            if args.source_residency_receipt
            else {}
        ),
        **(
            {"pool_observation_evidence": read_pool_evidence(args.pool_observation_receipt, runtime_hash)}
            if args.pool_observation_receipt
            else {}
        ),
    )
    result["runtime_receipt_sha256"] = runtime_hash
    result["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
