# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict observation identities; raw native sites are not imported timing rows.

Serving API contract: vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516,
vllm/models/deepseek_v41/{attention.py,nvidia/model.py,nvidia/dspark.py},
and vllm/v1/attention/backends/mla/sparse_swa.py (Apache-2.0).
Original adapter; no upstream implementation is copied.
"""

import hashlib
import json
import math
from itertools import pairwise

RUNTIME_REVISION = "ac9126e58aa7bbab1856ba6593ba4d5003fea516"
MODEL_REVISION = "5254448103b6254fc646dacbca1cbbf20e9216c1"
MODEL_CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def query_identity(query_starts, positions, *, owner, phase_counts, causal):
    """Read native request boundaries and positions without inventing KV state.

    positions are absolute token positions, not the number of resident SWA rows.
    Preserve every request separately; heterogeneous or bounded-tail metadata is
    never averaged into a homogeneous physical key.
    """
    if owner not in ("target", "draft") or type(causal) is not bool:
        raise ValueError("explicit target/draft owner and causal flag required")
    if causal != (owner == "target"):
        raise ValueError("pinned text DSpark target is causal and draft is noncausal")
    if not isinstance(query_starts, list) or len(query_starts) < 2:
        raise ValueError("native query starts require at least one request")
    if any(type(value) is not int or value < 0 for value in query_starts + positions):
        raise ValueError("native positions and request boundaries must be nonnegative integers")
    if query_starts[0] != 0 or query_starts[-1] != len(positions):
        raise ValueError("native query boundaries do not cover the observed token positions")
    if set(phase_counts) != {"num_decodes", "num_prefills"}:
        raise ValueError("native prefill/decode counts required")
    if any(type(v) is not int or v < 0 for v in phase_counts.values()):
        raise ValueError("native phase counts must be nonnegative integers")
    if sum(phase_counts.values()) != len(query_starts) - 1:
        raise ValueError("phase counts disagree with native request boundaries")
    requests = []
    for index, (start, end) in enumerate(pairwise(query_starts)):
        if end <= start:
            raise ValueError("empty or decreasing native request boundary")
        values = positions[start:end]
        if any(right != left + 1 for left, right in pairwise(values)):
            raise ValueError("nonconsecutive native query positions need a separate measurement contract")
        requests.append(
            {
                "query_width": end - start,
                "past_kv_tokens": values[0],
                "inclusive_kv_end": values[-1] + 1,
                "phase": "decode" if index < phase_counts["num_decodes"] else "prefill",
            }
        )
    return {"owner": owner, "causal": causal, "sequence_count": len(requests), "requests": requests}


def validate_observations(rows):
    """Validate nested intervals without summing inclusive parent/child costs."""
    by_id = {}
    for row in rows:
        if row["invocation_id"] in by_id:
            raise ValueError("duplicate native invocation id")
        if not isinstance(row["latency_ms"], (float, int)) or isinstance(row["latency_ms"], bool):
            raise ValueError("native elapsed time must be numeric")
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0:
            raise ValueError("native elapsed time must be positive finite milliseconds")
        if row.get("measurement_scope") != "inclusive_native_site":
            raise ValueError("raw observations must retain inclusive ownership")
        parent = row["parent_invocation_id"]
        if parent is not None and parent not in by_id:
            raise ValueError("native parent must precede its child")
        if parent is not None and by_id[parent]["sample_id"] != row["sample_id"]:
            raise ValueError("parent and child belong to different samples")
        by_id[row["invocation_id"]] = row
    return {"invocations": len(rows), "roots": sum(r["parent_invocation_id"] is None for r in rows)}


def coverage_summary(rows, failures):
    """Coverage is observational; no timing row is implicitly qualified."""
    shapes = {}
    for row in rows:
        if row.get("query_identity") is None:
            continue
        key = canonical({"site": row["site"], "query": row["query_identity"]})
        shapes[key] = shapes.get(key, 0) + 1
    required = {("target", "prefill", 1024), ("target", "decode", 6), ("draft", "decode", 5)}
    observed = {
        (row["query_identity"]["owner"], request["phase"], request["query_width"])
        for row in rows
        if row.get("query_identity") is not None
        for request in row["query_identity"]["requests"]
    }
    return {
        **validate_observations(rows),
        "shape_counts": [{"identity": json.loads(key), "samples": count} for key, count in sorted(shapes.items())],
        "failure_count": len(failures),
        "observed_site_kinds": sorted({row["site"] for row in rows}),
        "missing_initial_shapes": [list(shape) for shape in sorted(required - observed)],
        "import_eligible": False,
        "remaining_gate": "Map fused execution sites to the vLLM consumer graph before producing timing rows.",
    }
