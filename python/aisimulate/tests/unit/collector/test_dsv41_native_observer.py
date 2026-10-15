# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from collector.vllm.dsv41_native_contract import coverage_summary, query_identity, validate_observations
from collector.vllm.dsv41_native_observer import NativeObserver

pytestmark = pytest.mark.unit


def test_native_query_identity_preserves_sequences_width_and_causality():
    result = query_identity(
        [0, 6, 12],
        list(range(1024, 1030)) + list(range(2048, 2054)),
        owner="target",
        phase_counts={"num_decodes": 2, "num_prefills": 0},
        causal=True,
    )
    assert result["sequence_count"] == 2
    assert result["requests"] == [
        {"query_width": 6, "past_kv_tokens": 1024, "inclusive_kv_end": 1030, "phase": "decode"},
        {"query_width": 6, "past_kv_tokens": 2048, "inclusive_kv_end": 2054, "phase": "decode"},
    ]
    draft = query_identity(
        [0, 5], list(range(1023, 1028)), owner="draft", phase_counts={"num_decodes": 1, "num_prefills": 0}, causal=False
    )
    assert draft["causal"] is False
    assert draft["requests"][0]["past_kv_tokens"] == 1023


def test_bounded_native_tail_retains_absolute_positions():
    result = query_identity(
        [0, 128],
        list(range(896, 1024)),
        owner="target",
        phase_counts={"num_decodes": 0, "num_prefills": 1},
        causal=True,
    )
    assert result["requests"] == [
        {"query_width": 128, "past_kv_tokens": 896, "inclusive_kv_end": 1024, "phase": "prefill"},
    ]


@pytest.mark.parametrize(
    "starts,positions", [([0, 2], [1]), ([1, 2], [1, 2]), ([0, 0], []), ([0, 2], [1, 3]), ([0, 1], [True])]
)
def test_invalid_native_shape_never_becomes_homogeneous_key(starts, positions):
    with pytest.raises(ValueError):
        query_identity(
            starts, positions, owner="target", phase_counts={"num_decodes": 1, "num_prefills": 0}, causal=True
        )


def test_wrong_causal_identity_is_rejected():
    with pytest.raises(ValueError, match="noncausal"):
        query_identity(
            [0, 5], list(range(5)), owner="draft", phase_counts={"num_decodes": 1, "num_prefills": 0}, causal=True
        )


class Tensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(values),)

    def __getitem__(self, key):
        return Tensor(self.values[key])

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class Event:
    def __init__(self, *, enable_timing):
        assert enable_timing

    def record(self):
        pass

    def elapsed_time(self, other):
        return 2.0  # Independent fake CUDA timer; not a model timing oracle.


def test_observer_keeps_native_call_and_query_state_and_restores(tmp_path):
    cuda = SimpleNamespace(Event=Event, is_current_stream_capturing=lambda: False, synchronize=lambda: None)
    attn = SimpleNamespace(
        prefix="model.layers.40.attn",
        swa_cache_layer=SimpleNamespace(prefix="swa"),
        backend_cls=SimpleNamespace(get_name=lambda: "observed_native_backend"),
        n_local_heads=16,
        head_dim=512,
        compress_ratio=0,
        window_size=128,
        wo_b=SimpleNamespace(reduce_results=False),
        gemm_rs=None,
    )
    metadata = SimpleNamespace(query_start_loc_cpu=Tensor([0, 5]), num_decodes=1, num_prefills=0)
    observer = NativeObserver(
        SimpleNamespace(cuda=cuda), lambda: SimpleNamespace(attn_metadata={"swa": metadata}), tmp_path, 0
    )

    class Layer:
        def __init__(self):
            self.attn = attn
            self.calls = 0

        def forward(self, x, positions):
            self.calls += 1
            return x

    original = Layer.forward
    layer = Layer()
    observer.wrap(Layer, "forward", "layer")
    positions = Tensor(range(1023, 1028))
    value = object()
    assert layer.forward(value, positions) is value  # warmup has no record
    observer.begin("layer-0", "layer")
    assert layer.forward(value, positions) is value
    positions.values[:] = [0] * 5  # native buffer reuse must not rewrite evidence
    receipt = observer.finish()
    assert receipt["coverage"]["invocations"] == 1
    assert receipt["coverage"]["import_eligible"] is False
    assert receipt["coverage"]["shape_counts"][0]["identity"]["query"]["requests"][0]["past_kv_tokens"] == 1023
    assert layer.calls == 2
    observer.restore()
    assert Layer.forward is original


def test_partial_metadata_is_preserved_as_failure(tmp_path):
    observer = NativeObserver(
        SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)),
        lambda: SimpleNamespace(attn_metadata=None),
        tmp_path,
        0,
    )

    class Layer:
        attn = SimpleNamespace(prefix="model.layers.0.attn", swa_cache_layer=SimpleNamespace(prefix="swa"))

        def forward(self, x, positions):
            raise AssertionError("dummy call must fail before measurement")

    observer.wrap(Layer, "forward", "layer")
    observer.begin("layer-0", "layer")
    try:
        with pytest.raises(RuntimeError, match="real SWA"):
            Layer().forward(None, Tensor([0]))
        assert observer.frames == []
        assert len(observer.failures) == 1
    finally:
        observer.restore()


def test_nested_records_never_summed_or_imported():
    root = {
        "invocation_id": 0,
        "parent_invocation_id": None,
        "sample_id": "0",
        "site": "layer",
        "latency_ms": 4.0,
        "measurement_scope": "inclusive_native_site",
    }
    child = {**root, "invocation_id": 1, "parent_invocation_id": 0, "site": "attention", "latency_ms": 2.0}
    result = coverage_summary([root, child], [])
    assert result["roots"] == 1
    assert result["import_eligible"] is False
    with pytest.raises(ValueError, match="duplicate"):
        validate_observations([root, root])
