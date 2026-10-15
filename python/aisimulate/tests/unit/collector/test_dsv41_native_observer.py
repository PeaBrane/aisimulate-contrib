# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collector.vllm.dsv41_native_contract import (
    config_identity_digest,
    coverage_summary,
    query_identity,
    stage_table_rows,
    validate_observations,
)
from collector.vllm.dsv41_native_observer import NativeObserver

from aisimulate_core.sdk import common, rust_engine_step
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.perf_database import PerfDatabase
from aisimulate_core.sdk.speculation import SpeculationConfig
from aisimulate_core.sdk.utils import _load_model_config_from_model_path

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


def test_final_tail_keeps_full_request_identity_after_bounded_decoder_replay(tmp_path):
    cuda = SimpleNamespace(Event=Event, is_current_stream_capturing=lambda: False, synchronize=lambda: None)
    full = SimpleNamespace(query_start_loc_cpu=Tensor([0, 1024]), num_decodes=0, num_prefills=1)
    tail = SimpleNamespace(query_start_loc_cpu=Tensor([0, 128]), num_decodes=0, num_prefills=1)
    context = SimpleNamespace(attn_metadata={"swa": full})
    attn = SimpleNamespace(prefix="model.layers.39.attn", swa_cache_layer=SimpleNamespace(prefix="swa"))
    observer = NativeObserver(SimpleNamespace(cuda=cuda), lambda: context, tmp_path, 0)

    class Layer:
        fuse_mhc_all_reduce = False

        def __init__(self):
            self.attn = attn

        def forward(self, x, positions):
            assert positions.tolist() == list(range(896, 1024))
            return x

    class Model:
        use_sequence_parallel = False
        decoder_replay_layers = object()

        def __init__(self):
            self.layers = [Layer()]

        def forward(self, ids, positions):
            context.attn_metadata = {"swa": tail}
            self.layers[0].forward(None, positions[896:])
            context.attn_metadata = {"swa": full}
            return "native-full-row-result"

    observer.wrap_root(Model, "forward", "target")
    observer.wrap(Layer, "forward", "layer")
    observer.begin("tail-0", "finalize")
    try:
        assert Model().forward(None, Tensor(range(1024))) == "native-full-row-result"
        observer.finish()
        rows = json.loads((tmp_path / "rank-0.json").read_text())["rows"]
        assert len(rows) == 1  # No nested layer timing is added to the tail.
        assert rows[0]["site"] == "finalize"
        assert rows[0]["query_identity"]["requests"] == [
            {"query_width": 1024, "past_kv_tokens": 0, "inclusive_kv_end": 1024, "phase": "prefill"}
        ]
        assert observer.frames == observer.roots == []
    finally:
        observer.restore()


def native_stage_payloads():
    cache = {"kv_mxfp8": True, "swa_bytes_per_token": 528, "record_width": 528, "tensor_dtype": "torch.uint8"}
    payloads = []
    sites = [("layer", layer) for layer in range(43)] + [
        ("finalize", 39),
        ("finalize", 42),
        ("context_preparation", 40),
        ("input_preparation", 0),
        ("input_preparation", 40),
    ]
    for rank in range(4):
        rows = []
        for sample in range(2):
            for site, layer in sites:
                owner = "draft" if layer >= 40 else "target"
                width = 5 if owner == "draft" else 6
                rows.append(
                    {
                        "invocation_id": len(rows),
                        "parent_invocation_id": None,
                        "sample_id": f"{site}-{sample}",
                        "tp_rank": rank,
                        "site": site,
                        "layer_id": layer,
                        "owner": owner,
                        "latency_ms": float((rank + 1) * (sample + 1)),
                        "used_cuda_graph": False,
                        "measurement_scope": "inclusive_native_site",
                        "decoder_replay": True,
                        "cache_identity": cache,
                        "callable": f"observed.{site}",
                        "layer_execution": {},
                        "query_identity": query_identity(
                            [0, width],
                            list(range(1024, 1024 + width)),
                            owner=owner,
                            phase_counts={"num_decodes": 1, "num_prefills": 0},
                            causal=owner == "target",
                        ),
                    }
                )
        payloads.append({"rank": rank, "rows": rows, "failures": []})
    return payloads


def test_native_stage_aggregation_uses_slowest_rank_then_repeat_median():
    rows = stage_table_rows(native_stage_payloads(), config_digest="a" * 64, image_digest="sha256:" + "b" * 64)
    assert len(rows) == 48
    assert {row["latency"] for row in rows} == {6.0}
    assert {row["sample_count"] for row in rows} == {2}
    assert {row["rank_count"] for row in rows} == {4}
    assert all(json.loads(row["identity"])["query_width"] in (5, 6) for row in rows)


@pytest.mark.parametrize(
    "fault", ["missing_rank", "shape_mismatch", "cache_layout", "missing_finalize", "failed_invocation"]
)
def test_native_stage_partial_or_conflicting_observations_cannot_publish(fault):
    payloads = copy.deepcopy(native_stage_payloads())
    if fault == "missing_rank":
        payloads.pop()
    elif fault == "shape_mismatch":
        payloads[0]["rows"][0]["query_identity"]["requests"][0]["past_kv_tokens"] += 1
    elif fault == "cache_layout":
        payloads[0]["rows"][0]["cache_identity"]["record_width"] = 584
    elif fault == "missing_finalize":
        for payload in payloads:
            payload["rows"] = [row for row in payload["rows"] if row["site"] != "finalize"]
    else:
        payloads[0]["failures"].append({"error": "observed native failure"})
    with pytest.raises(ValueError):
        stage_table_rows(payloads, config_digest="a" * 64, image_digest="sha256:" + "b" * 64)


def test_config_digest_ignores_only_declared_sdk_aliases_and_preserves_geometry_quantization():
    raw = {"text_config": {"hidden_size": 5120}, "quantization_config": {"quant_method": "fp8"}}
    sdk = {**raw, "quant_algo": "FP8", "quant_dynamic": True, "kv_cache_quant_algo": "FP8"}
    assert config_identity_digest(raw) == config_identity_digest(sdk)
    assert config_identity_digest({**sdk, "text_config": {"hidden_size": 4096}}) != config_identity_digest(raw)
    assert config_identity_digest({**sdk, "quantization_config": {"quant_method": "other"}}) != config_identity_digest(
        raw
    )


def test_native_producer_rows_reach_real_model_stage_without_summing_children(tmp_path):
    # CPU selection witnesses only: these timers are not silicon evidence.
    model_name = "deepseek-ai/DeepSeek-V4.1-Flash"
    raw = _load_model_config_from_model_path(model_name)
    config = ModelConfig(
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        language_only=True,
        system="b200_sxm",
        decoder_replay=True,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        speculation=SpeculationConfig(kind="dspark", params={"num_draft_tokens": 5}, draft_config=raw),
    )
    model = get_model(model_name, config, "vllm")
    layer = model.generation_ops[1]
    scope = json.loads(layer._spec_json())["Dsv41Stage"]["native_scope"]
    pristine = {
        key: value for key, value in raw.items() if key not in ("quant_algo", "quant_dynamic", "kv_cache_quant_algo")
    }
    assert scope["config_sha256"] == config_identity_digest(pristine)
    assert scope["query_tokens"] == 6
    source = Path(__file__).resolve().parents[3] / "src/aisimulate_core/systems"
    shutil.copy(source / "b200_sxm.yaml", tmp_path / "b200_sxm.yaml")
    data = tmp_path / "data/b200_sxm/vllm/devac9126"
    data.mkdir(parents=True)
    rows = stage_table_rows(
        native_stage_payloads(), config_digest=config_identity_digest(pristine), image_digest="sha256:" + "b" * 64
    )
    pq.write_table(pa.Table.from_pylist(rows), data / "dsv41_native_stage_perf.parquet")
    db = PerfDatabase(
        "b200_sxm",
        "vllm",
        "devac9126",
        str(tmp_path),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=False,
    )
    # Isolate the real model stage at the private operator test boundary. The
    # canonical constructor and full graph are covered by test_dspark_scheme.
    model.context_ops = []
    model.generation_ops = [layer]
    handle = rust_engine_step._cached_engine_handle(model, db)
    assert handle.predict_decode_latency(1, 1024) == pytest.approx(6.0)
    with pytest.raises(PerfDataNotAvailableError, match="missing exact SILICON query"):
        handle.predict_decode_latency(1, 1025)
