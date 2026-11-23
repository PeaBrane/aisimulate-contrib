# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2-Flash on SGLang reads the fa4 lane where serving runs fa4.

SGLang 0.5.23 puts MiMo-V2 on ``fa4`` on SM100/103 (``model_overrides/mimo_v2.py``), and the
collector times the model there from that release on. The earlier releases' rows for the same
geometry are triton rows that stay in the merged table, and the global SM100/103 default lane
is triton, so without a per-architecture default the older kernel's rows answer the query.
"""

import pytest

from aisim.sdk import common
from aisim.sdk.attention_lanes import (
    ATTENTION_BACKEND_CHOICES,
    resolve_attention_lane_order,
    resolve_attention_override_lanes,
)
from aisim.sdk.errors import UnsupportedAttentionBackendError

pytestmark = pytest.mark.unit

_ARCH = "MiMoV2FlashForCausalLM"


@pytest.mark.parametrize("sm_version", [100, 103])
def test_the_packaged_default_pins_fa4_on_sm100_from_sglang_0_5_23(sm_version):
    order = resolve_attention_lane_order("sglang", "0.5.23", sm_version, None, None, _ARCH)
    assert order[0] == "fa4" and order.pinned_count == 1
    assert order.count("fa4") == 1 and order[-1] == "default"


@pytest.mark.parametrize(
    ("version", "sm_version"),
    [("0.5.21", 100), ("0.5.21", 103), ("0.5.23", 90), ("0.5.23", 120)],
)
def test_earlier_releases_and_other_sms_keep_the_global_default(version, sm_version):
    with_arch = resolve_attention_lane_order("sglang", version, sm_version, None, None, _ARCH)
    without = resolve_attention_lane_order("sglang", version, sm_version, None, None, None)
    assert tuple(with_arch) == tuple(without) and with_arch[0] != "fa4"


def test_fa4_is_an_override_on_sglang_only():
    assert "fa4" in ATTENTION_BACKEND_CHOICES and common.AttentionBackend("fa4") is common.AttentionBackend.fa4
    assert resolve_attention_override_lanes("sglang", "fa4") == ("fa4",)
    order = resolve_attention_lane_order("sglang", "0.5.23", 90, "fa4", None, None)
    assert order[0] == "fa4"
    with pytest.raises(UnsupportedAttentionBackendError):
        resolve_attention_override_lanes("vllm", "fa4")


@pytest.mark.parametrize("system", ["b200_sxm", "b300_sxm", "gb200", "gb300"])
def test_an_older_triton_row_does_not_answer_for_the_fa4_measurement(system):
    """Shipped data: the default query of a MiMo-V2-Flash layer returns the fa4 row, not the
    triton row an earlier release left at the same shape."""
    from aisim_core.sdk.engine import _evaluate_single_op
    from aisim_core.sdk.operations.attention import ContextAttention, resolved_lane_order_for_op
    from aisim_core.sdk.perf_database import get_database

    db = get_database(system, "sglang", "current")
    assert db.version == "0.5.23"
    order = resolved_lane_order_for_op(db, "_context_attention_data", architecture=_ARCH)
    assert order[0] == "fa4"

    def latency(lanes):
        op = ContextAttention(
            "mimo_probe",
            1.0,
            8,
            1,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
            head_size=192,
            window_size=0,
        )
        op._lane_order = lanes
        return float(_evaluate_single_op(db, op, is_context=True, batch_size=1, s=8192, prefix=0))

    default, fa4, triton = latency(order), latency(["fa4"]), latency(["triton"])
    assert default == fa4
    assert fa4 != triton
