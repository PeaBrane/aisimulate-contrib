# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aisim-e2e/v1 evolution contract: unknown fields survive, old and new readers interoperate, names are canonical."""

import warnings

import pyarrow as pa
import pytest

from aisim_core.data import names
from aisim_core.data.e2e import (
    METRICS_SCHEMA,
    MINOR_METADATA_KEY,
    OBSERVATIONS_SCHEMA,
    REQUESTS_SCHEMA,
    SCHEMA_MINOR,
    E2eConfigurationData,
    E2eManifestBody,
    MetricRow,
    ObservationRow,
    RequestRow,
    SeriesRow,
    Workload,
    evaluation_points,
    load_e2e,
    make_metric,
    name_findings_for,
    name_findings_for_dataset,
    rows_to_table,
    table_to_rows,
    write_configuration,
)
from aisim_core.data.names import UnknownNameError

pytestmark = pytest.mark.unit


def _body(**over):
    fields = dict(
        origin="self-collected",
        model_id="Qwen/Qwen3.5-397B-A17B",
        deployment="aggregated",
        hardware="b300_sxm",
        framework="vllm",
        framework_version="0.31.0",
        parallelism={"aggregated": {"tp": 8}},
        parallelism_complete=True,
        precision={"weights": "bfloat16"},
        source_kind="developer",
    )
    fields.update(over)
    return E2eManifestBody(**fields)


def _obs(body, **over):
    fields = dict(
        observation_id="dev:1",
        configuration_id=body.configuration_id(),
        record_sha256="0" * 64,
        source_kind="developer",
        outcome="complete",
    )
    fields.update(over)
    return ObservationRow(**fields)


def _config(body=None, **extra):
    body = body or _body()
    metrics = [
        make_metric("dev:1", "ttft", "median", 120.0, "ms"),
        make_metric("dev:1", "ttft", "mean", 130.0, "ms"),
    ]
    return E2eConfigurationData(body, [_obs(body)], metrics, **extra)


# ------------------------------------------------------------------ unknown fields survive


def test_unknown_row_field_is_kept_through_a_table_round_trip():
    row = RequestRow(observation_id="dev:1", request_id="r0", future_field="x", future_number=3)
    table = rows_to_table(REQUESTS_SCHEMA, [row])
    assert {"future_field", "future_number"} <= set(table.schema.names)
    back = table_to_rows(REQUESTS_SCHEMA, table)[0]
    assert back.model_extra == {"future_field": "x", "future_number": 3}
    assert rows_to_table(REQUESTS_SCHEMA, [back]).equals(table)


def test_unknown_manifest_body_and_row_fields_survive_write_and_load(tmp_path):
    body = _body(brand_new_setting={"a": 1})
    data = _config(body)
    data.observations[0] = _obs(body, brand_new_column="kept")
    path = write_configuration(tmp_path, data)
    dataset = load_e2e(tmp_path, path)
    assert dataset.body.model_extra == {"brand_new_setting": {"a": 1}}
    assert dataset.observations[0].model_extra == {"brand_new_column": "kept"}


def test_extension_namespace_round_trips_and_is_not_identity(tmp_path):
    plain = _body()
    ext = _body(ext={"agentx/1": {"repeat": 0, "features": ["agentx.live_warmup"]}})
    assert plain.configuration_id() == ext.configuration_id()
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, _config(ext)))
    assert dataset.body.ext["agentx/1"]["repeat"] == 0


# ------------------------------------------------------------------ old and new readers


def test_missing_optional_columns_read_as_null():
    obs = _obs(_body())
    table = rows_to_table(OBSERVATIONS_SCHEMA, [obs])
    kept = [n for n in table.schema.names if n not in ("observer", "attrs", "valid_for_submission", "time_origin")]
    old = table.select(kept)  # a file written before those columns existed
    row = table_to_rows(OBSERVATIONS_SCHEMA, old)[0]
    assert row.observer is None and row.attrs is None and row.valid_for_submission is None


def test_missing_required_column_is_an_error():
    table = rows_to_table(METRICS_SCHEMA, [make_metric("dev:1", "ttft", "mean", 1.0, "ms")])
    with pytest.raises(ValueError, match="missing required columns"):
        table_to_rows(METRICS_SCHEMA, table.drop_columns(["unit"]))


def test_column_order_does_not_matter():
    table = rows_to_table(METRICS_SCHEMA, [make_metric("dev:1", "ttft", "mean", 1.0, "ms")])
    shuffled = table.select(list(reversed(table.schema.names)))
    assert table_to_rows(METRICS_SCHEMA, shuffled)[0].metric == "ttft"


def test_a_newer_minor_loads_with_a_warning():
    table = rows_to_table(METRICS_SCHEMA, [make_metric("dev:1", "ttft", "mean", 1.0, "ms")])
    newer = table.replace_schema_metadata({MINOR_METADATA_KEY: str(SCHEMA_MINOR + 1).encode()})
    with pytest.warns(UserWarning, match="newer than this reader"):
        assert table_to_rows(METRICS_SCHEMA, newer)
    current = table.replace_schema_metadata({MINOR_METADATA_KEY: str(SCHEMA_MINOR).encode()})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table_to_rows(METRICS_SCHEMA, current)


def test_must_have_core_is_small():
    # A source that only knows the identity core and one metric still produces a valid configuration.
    body = _body()
    assert body.workload is None and body.ext == {} and body.implementation == {} and body.source_ids == {}
    RequestRow(observation_id="dev:1", request_id="r0")  # only the key and status default
    SeriesRow(observation_id="dev:1", metric="running_requests", start_ms=0.0, end_ms=1000.0)


def test_series_and_requests_are_written_and_loaded(tmp_path):
    body = _body(workload=Workload(kind="agentic_trace", concurrency=4, boundary="successful_profile_span"))
    data = _config(
        body,
        requests=[
            RequestRow(
                observation_id="dev:1", request_id="r0", phase="profile", start_ms=0.0, end_ms=5.0, cached_tokens=64
            )
        ],
        series=[SeriesRow(observation_id="dev:1", metric="running_requests", start_ms=0.0, end_ms=1000.0, value=4.0)],
    )
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, data))
    assert dataset.requests[0].cached_tokens == 64 and dataset.series[0].value == 4.0
    assert dataset.body.workload.kind == "agentic_trace"
    assert dataset.body.schema_minor == SCHEMA_MINOR


# ------------------------------------------------------------------ canonical names


def test_alias_resolves_to_the_canonical_name_in_converters_only():
    assert names.normalize("hardware", "B300") == "b300_sxm"
    assert names.normalize("statistic", "p50") == "median"
    assert names.normalize("phase", "preparation") == "warmup"
    assert names.normalize("request_status", "ok") == "completed"
    assert names.split_framework("dynamo-sglang") == {"framework": "sglang", "orchestrator": "dynamo"}
    assert names.split_framework("trt") == {"framework": "trtllm"}
    assert names.split_framework("mori-sglang")["settings"] == {"kv_transfer_backend": "mori"}


def test_closed_kind_rejects_unknown_but_open_kind_accepts_it():
    with pytest.raises(UnknownNameError):
        names.normalize("hardware", "not-a-gpu")
    assert names.normalize("model", "some-org/Some-New-Model") == "some-org/Some-New-Model"
    assert names.normalize("model", "dsr1") == "deepseek-ai/DeepSeek-R1-0528"


def test_every_registered_alias_is_unambiguous():
    for kind in names.kinds():
        seen: dict[str, str] = {}
        for canonical in names.canonical_names(kind):
            for spelling in (canonical, *names.aliases(kind, canonical)):
                key = names._key(spelling)
                assert seen.setdefault(key, canonical) == canonical, f"{kind}: {spelling!r} maps to two names"


def test_validator_flags_non_canonical_names_with_the_canonical_spelling():
    body = _body(hardware="B300", framework="trt", workload=Workload(kind="agentic_traces"))
    metrics = [make_metric("dev:1", "ttft", "p50", 1.0, "ms"), make_metric("dev:1", "ttft", "median", 1.0, "ms")]
    requests = [RequestRow(observation_id="dev:1", request_id="r0", status="completed", phase="profiling")]
    found = {(f.kind, f.value): f.canonical for f in name_findings_for(body, metrics=metrics, requests=requests)}
    assert found[("hardware", "B300")] == "b300_sxm"
    assert found[("framework", "trt")] == "trtllm"
    assert found[("workload_kind", "agentic_traces")] == "agentic_trace"
    assert found[("statistic", "p50")] == "median"
    assert found[("phase", "profiling")] == "profile"
    assert ("statistic", "median") not in found
    assert "use 'b300_sxm'" in str(next(f for f in name_findings_for(body) if f.kind == "hardware"))


def test_validator_is_quiet_for_canonical_data(tmp_path):
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, _config()))
    assert name_findings_for_dataset(dataset) == []


def test_unknown_metric_is_accepted_with_its_unit_but_unknown_hardware_is_flagged():
    row = make_metric("dev:1", "brand_new_metric", "mean", 3.0, "tokens")
    assert isinstance(row, MetricRow) and row.unit == "tokens" and row.value == 3.0
    assert name_findings_for(_body(), metrics=[row]) == []
    assert [f.kind for f in name_findings_for(_body(hardware="mystery-gpu"))] == ["hardware"]


def test_consumers_ignore_non_canonical_metric_names(tmp_path):
    body = _body()
    data = E2eConfigurationData(
        body,
        [_obs(body)],
        [make_metric("dev:1", "ttft", "p50", 1.0, "ms"), make_metric("dev:1", "ttft", "median", 2.0, "ms")],
    )
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, data))
    (point,) = evaluation_points(dataset)
    assert point.metrics == {("ttft", "median"): 2.0}


def test_arrow_types_follow_the_annotations():
    from aisim_core.data.e2e import OBSERVATIONS_ARROW, REQUESTS_ARROW

    assert OBSERVATIONS_ARROW.field("valid_for_submission").type == pa.bool_()
    assert REQUESTS_ARROW.field("cached_tokens").type == pa.int64()
    assert REQUESTS_ARROW.field("start_ms").type == pa.float64()
    assert REQUESTS_ARROW.field("status").type == pa.string()
