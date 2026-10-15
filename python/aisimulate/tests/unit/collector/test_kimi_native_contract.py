# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
from collector.vllm.kimi_native_contract import PREFILL_KERNEL, physical_identity, table_rows

pytestmark = pytest.mark.unit


def row(draft=False, width=3, prefill=False):
    return {
        "draft": draft,
        "causal": not draft,
        "query_starts": [0, width],
        "positions": list(range(0 if prefill else 1024, width if prefill else 1024 + width)),
        "num_prefills": int(prefill),
        "num_decodes": int(not prefill),
        "num_heads": 8 if draft else 12,
        "tp_size": 8,
        "output_reduce_results": not draft,
        "gemm_quant_mode": "bfloat16",
        "kv_cache_dtype": "fp8",
        "prefill_query_quantization": True,
        "prefill_backend": "TRTLLM_RAGGED",
        "sample_id": "0",
        "invocation_id": 0,
        "kernel_source": PREFILL_KERNEL if prefill else "FLASHINFER_MLA",
        "gemm_reduce_scatter": not draft,
        "latency_ms": 1.0,
    }


def test_width_sequence_causal_and_kv_are_distinct():
    identity, total = physical_identity(row(draft=True, width=2))
    assert identity["sequence_count"] == 1
    assert identity["query_tokens"] == 2
    assert identity["causal"] is False
    assert total == 1026
    identity, total = physical_identity(row(width=3))
    assert identity["causal"] is True
    assert total == 1027


@pytest.mark.parametrize(
    "change",
    [
        {"causal": False},
        {"output_reduce_results": False},
        {"gemm_quant_mode": "fp8"},
        {"num_heads": 8},
        {"num_prefills": 1},
        {"query_starts": [0, 2]},
        {"query_starts": []},
        {"query_starts": [0, 3, 3]},
        {"positions": [-1, 0, 1]},
        {"causal": 1},
        {"num_prefills": -1, "num_decodes": 2},
        {"prefill_query_quantization": False},
        {"prefill_backend": "FLASH_ATTN"},
        {"prefill_backend": None},
    ],
)
def test_wrong_native_contract_fails(change):
    with pytest.raises(ValueError):
        physical_identity(row() | change)


def test_rank_aggregation_does_not_sum_layers_or_ranks():
    payloads = []
    for rank in range(8):
        rows = [row(width=1024, prefill=True), row(), row(draft=True, width=2)]
        for index, value in enumerate(rows):
            value.update(invocation_id=index, latency_ms=1.0 + rank)
        payloads.append({"rank": rank, "rows": rows, "failures": []})
    result = table_rows(payloads)
    assert len(result) == 3
    assert all(r["latency"] == 8 for r in result)
    assert {json.loads(r["identity"])["query_tokens"] for r in result} == {2, 3, 1024}
    broken = copy.deepcopy(payloads)
    broken[0]["rows"].pop()
    with pytest.raises(ValueError, match="missing a TP rank"):
        table_rows(broken)
    broken = copy.deepcopy(payloads)
    broken[0]["rows"][0]["kernel_source"] = "other"
    with pytest.raises(ValueError, match="disagree"):
        table_rows(broken)
    broken = copy.deepcopy(payloads)
    broken[0]["rows"][0]["gemm_reduce_scatter"] = False
    with pytest.raises(ValueError, match="disagree"):
        table_rows(broken)
    broken = copy.deepcopy(payloads)
    for payload in broken:
        payload["rows"].append(dict(payload["rows"][0], invocation_id=3, gemm_reduce_scatter=False))
    with pytest.raises(ValueError, match="changes within"):
        table_rows(broken)
    with pytest.raises(ValueError, match="all eight"):
        table_rows(payloads + [payloads[0]])
    broken = copy.deepcopy(payloads)
    for payload in broken:
        extra = dict(payload["rows"][0], invocation_id=3, kernel_source="other")
        payload["rows"].append(extra)
    with pytest.raises(ValueError, match="changes within"):
        table_rows(broken)
    with pytest.raises(ValueError, match="sequence-count"):
        table_rows(payloads, sequence_counts=(1, 4))
    broken = copy.deepcopy(payloads)
    for payload in broken:
        payload["rows"][0]["kernel_source"] = "FLASHINFER_MLA/FlashAttention"
    with pytest.raises(ValueError, match="actual kernel"):
        table_rows(broken)
