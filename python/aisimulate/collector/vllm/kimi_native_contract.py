# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact native Kimi MLA table identities, independent of GPU imports.

Original adapter for vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516,
vllm/models/kimi_k3/nvidia/{mla.py,model.py,dspark_mla.py} (Apache-2.0).
"""

import json
import math
from collections import defaultdict
from itertools import pairwise
from statistics import median

RUNTIME_REVISION = "ac9126e58aa7bbab1856ba6593ba4d5003fea516"
TARGET_CONFIG_SHA256 = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
DRAFT_CONFIG_SHA256 = "5a3c2f4f91c965ed93b14de5f12a4e9c17fd98d8c99916ed2deb26ce8702f970"
PREFILL_BACKEND = "TRTLLM_RAGGED"
PREFILL_KERNEL = "FLASHINFER_MLA/vllm.v1.attention.backends.mla.prefill.trtllm_ragged.TrtllmRaggedPrefillBackend"


def physical_identity(row):
    """Keep query width separate from sequences; refuse heterogeneous batches."""
    starts, positions = row["query_starts"], row["positions"]
    if (
        len(starts) < 2
        or any(type(n) is not int or n < 0 for n in starts + positions)
        or starts[0] != 0
        or starts[-1] != len(positions)
        or any(left >= right for left, right in pairwise(starts))
    ):
        raise ValueError("native MLA request boundaries do not cover positions")
    requests = []
    for left, right in pairwise(starts):
        chunk = positions[left:right]
        if not chunk or chunk != list(range(chunk[0], chunk[0] + len(chunk))):
            raise ValueError("native MLA requires consecutive query positions")
        requests.append((len(chunk), chunk[-1] + 1))
    if len(set(requests)) != 1:
        raise ValueError("heterogeneous native MLA batches require separate collected keys")
    width, total = requests[0]
    draft = row["draft"]
    if type(draft) is not bool or type(row["causal"]) is not bool or row["causal"] != (not draft):
        raise ValueError("native Kimi target is causal; DSpark draft is noncausal")
    prefill, decode = row["num_prefills"], row["num_decodes"]
    if (
        any(type(n) is not int or n < 0 for n in (prefill, decode))
        or (prefill > 0) == (decode > 0)
        or prefill + decode != len(requests)
        or (draft and prefill)
    ):
        raise ValueError("mixed or padded native phase metadata cannot be scored")
    if type(row["output_reduce_results"]) is not bool or row["output_reduce_results"] != (not draft):
        raise ValueError("native MLA collective ownership does not match the modeled scope")
    if row["gemm_quant_mode"] != "bfloat16" or row["kv_cache_dtype"] != "fp8":
        raise ValueError("initial native Kimi collector requires actual BF16 projections and FP8 cache")
    if row["prefill_query_quantization"] is not True:
        raise ValueError("native Kimi FP8 cache requires FP8 prefill queries after BF16 projections")
    if row.get("prefill_backend") != PREFILL_BACKEND:
        raise ValueError("native Kimi requires the explicit TRTLLM_RAGGED prefill backend")
    if row["tp_size"] != 8 or row["num_heads"] != (8 if draft else 12):
        raise ValueError("native Kimi MLA head geometry mismatch")
    return {
        "draft": draft,
        "causal": not draft,
        "is_context": prefill > 0,
        "query_tokens": width,
        "sequence_count": len(requests),
        "num_heads": row["num_heads"],
        "tensor_parallel_size": row["tp_size"],
        "kv_cache_dtype": row["kv_cache_dtype"],
        "gemm_quant_mode": row["gemm_quant_mode"],
    }, total


def table_rows(rank_payloads, sequence_counts=(1,)):
    """Use critical-rank time per invocation, then median repeated observations.

    Every native MLA layer has the same physical contract within its owner.
    Layer count belongs to the consumer; timings remain per layer.
    """
    if len(rank_payloads) != 8 or {p["rank"] for p in rank_payloads} != set(range(8)):
        raise ValueError("all eight TP ranks are required")
    invocations = defaultdict(dict)
    for payload in rank_payloads:
        if payload["failures"]:
            raise ValueError("native observation failures prevent table publication")
        for row in payload["rows"]:
            identity, total = physical_identity(row)
            latency = row["latency_ms"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("native MLA elapsed time must be positive and finite")
            if not isinstance(row["kernel_source"], str) or not row["kernel_source"]:
                raise ValueError("native MLA kernel identity is required")
            if type(row.get("gemm_reduce_scatter")) is not bool:
                raise ValueError("native MLA output-projection dispatch identity is required")
            kernel = json.dumps(
                {"attention": row["kernel_source"], "gemm_reduce_scatter": row["gemm_reduce_scatter"]},
                sort_keys=True,
            )
            key = (row["sample_id"], row["invocation_id"])
            if payload["rank"] in invocations[key]:
                raise ValueError("duplicate rank observation")
            invocations[key][payload["rank"]] = (identity, total, kernel, latency)
    physical = defaultdict(list)
    kernels = {}
    for ranks in invocations.values():
        if set(ranks) != set(range(8)):
            raise ValueError("native invocation is missing a TP rank")
        identities = {(json.dumps(v[0], sort_keys=True), v[1], v[2]) for v in ranks.values()}
        if len(identities) != 1:
            raise ValueError("TP ranks disagree on native execution identity")
        identity, total, kernel = next(iter(identities))
        key = (identity, total)
        if key in kernels and kernels[key] != kernel:
            raise ValueError("native MLA kernel identity changes within a physical timing key")
        kernels[key] = kernel
        physical[(identity, total, kernel)].append(max(v[3] for v in ranks.values()))
    for identity, _, kernel in physical:
        expected = PREFILL_KERNEL if json.loads(identity)["is_context"] else "FLASHINFER_MLA"
        if json.loads(kernel)["attention"] != expected:
            raise ValueError("native MLA actual kernel differs from the pinned attention backend")
    rows = [
        {
            "identity": identity,
            "total_kv_tokens": total,
            "kernel_source": kernel,
            "latency": median(values),
            "sample_count": len(values),
            "rank_count": 8,
            "use_cuda_graph": False,
            "prefill_query_quantization": True,
            "prefill_backend": PREFILL_BACKEND,
            "runtime_revision": RUNTIME_REVISION,
            "measurement_scope": "kimi_native_mla_inclusive",
            "target_config_sha256": TARGET_CONFIG_SHA256,
            "draft_config_sha256": DRAFT_CONFIG_SHA256,
        }
        for (identity, total, kernel), values in sorted(physical.items())
    ]
    identities = [json.loads(r["identity"]) for r in rows]
    shapes = {(r["draft"], r["query_tokens"], r["sequence_count"]) for r in identities}
    required = {
        (draft, width, count) for draft, width in ((False, 1024), (False, 3), (True, 2)) for count in sequence_counts
    }
    if not required <= shapes:
        raise ValueError("missing target prefill1024, verify3 or draft2 sequence-count observations")
    return rows
