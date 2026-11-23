# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aisim-e2e/v1: InferenceX and developer fixtures share one evaluator loader."""

import gzip
import json

import pyarrow.parquet as pq
import pytest

from aisim_core.data import check_manifest
from aisim_core.data.e2e import (
    LOG_CAP_BYTES,
    E2eManifestBody,
    MetricRow,
    ProvenanceFile,
    ProvenanceInfo,
    Setting,
    SnapshotAcquisition,
    SnapshotMember,
    aiperf_to_configuration,
    evaluation_points,
    inferencex_to_configurations,
    load_e2e,
    make_metric,
    public_release_problems,
    read_provenance,
    read_raw_record,
    read_snapshot,
    redact,
    store_provenance,
    throughput_per_device,
    to_canonical,
    write_configuration,
    write_snapshot,
)
from aisim_core.data.loaders import LoaderError
from aisim_core.data.parquet import read_parquet_schema_id

pytestmark = pytest.mark.unit


def _infx_record(**over):
    # Same shape as the InferenceX cache fixture in tests/test_e2e_hf_measurements.py.
    rec = dict(
        model="test-model",
        hardware="h100_sxm",
        framework="vllm",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=1,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=1,
        num_prefill_gpu=0,
        num_decode_gpu=8,
        id=123,
        date="2026-10-17",
        isl=1024,
        osl=128,
        conc=32,
        metrics={
            "mean_ttft": 0.1,
            "median_ttft": 0.09,
            "p99_ttft": 0.3,
            "mean_tpot": 0.01,
            "median_tpot": 0.0,
            "mean_e2el": 1.5,
            "tput_per_gpu": 1200.0,
            "output_tput_per_gpu": 150.0,
            "mean_intvty": 100.0,
            "weird_metric": 7,
        },
        image="test-image",
        benchmark_type="single_turn",
        run_url="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987/attempts/2",
    )
    rec.update(over)
    return rec


def _dev_body():
    return E2eManifestBody(
        origin="self-collected",
        model_id="test-model",
        deployment="aggregated",
        hardware="h100_sxm",
        framework="vllm",
        framework_version="0.11.0",
        parallelism={"aggregated": {"tp": 8, "ep": 1}},
        parallelism_complete=True,
        precision={"weights": "fp8"},
        settings={"max_num_seqs": Setting(value=256, authority="observed")},
        source_kind="developer",
    )


def _aiperf_summary(**over):
    s = {
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "time_to_first_token": {"unit": "ms", "avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 300.0},
        "inter_token_latency": {"unit": "ms", "avg": 10.0, "p50": 9.0, "p90": 14.0},
        "request_latency": {"unit": "seconds", "avg": 1.5, "p50": 1.4, "p90": 2.0},
        "output_token_throughput": {"unit": "tokens/second", "avg": 9600.0},
    }
    s.update(over)
    return s


def _aiperf_records():
    return [
        {
            "request_id": i,
            "error": None,
            "metrics": {
                "input_sequence_length": {"value": 1024, "unit": "tokens"},
                "output_sequence_length": {"value": 128, "unit": "tokens"},
                "time_to_first_token": {"value": 0.1, "unit": "seconds"},
                "inter_token_latency": {"value": 10, "unit": "ms"},
            },
        }
        for i in range(2)
    ]


def _dev_config(**kw):
    return aiperf_to_configuration(
        _aiperf_summary(), _aiperf_records(), _dev_body(), observation_id="dev:1", device_count=8, **kw
    )


# ---------------------------------------------------------------- metrics


def test_missing_is_never_zero():
    assert make_metric("o", "ttft", "mean", None, "s").status == "missing"
    zero = make_metric("o", "ttft", "mean", 0, "s")
    assert zero.status == "valid" and zero.value == 0.0
    for bad in (float("nan"), float("inf"), -1.0, True, "1"):
        row = make_metric("o", "ttft", "mean", bad, "s")
        assert row.status == "invalid" and row.value is None
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="missing", value=0.0)
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="valid")


def test_unit_conversions():
    assert to_canonical(0.25, "s", "ms") == 250.0
    assert to_canonical(250, "milliseconds", "ms") == 250
    assert to_canonical(500, "us", "ms") == pytest.approx(0.5)
    assert to_canonical(3, "tok/s", "tokens/s") == 3
    with pytest.raises(ValueError):
        to_canonical(1, "furlongs", "ms")
    with pytest.raises(ValueError):
        to_canonical(1, "tokens/s", "ms")
    row = make_metric("o", "ttft", "p90", 0.2, "s")
    assert (row.value, row.unit, row.source_value, row.source_unit) == (200.0, "ms", 0.2, "s")


def test_per_device_throughput_is_explicit():
    per = make_metric("o", "output_throughput", "value", 150.0, "tokens/s", normalization="per_device", device_count=8)
    total = make_metric("o", "output_throughput", "value", 1200.0, "tokens/s", normalization="total", device_count=8)
    assert throughput_per_device(per) == 150.0 and throughput_per_device(total) == 150.0
    unknown = total.model_copy(update={"device_count": None})
    with pytest.raises(ValueError, match="device_count"):
        throughput_per_device(unknown)
    with pytest.raises(ValueError):
        throughput_per_device(make_metric("o", "ttft", "mean", 1, "ms"))


# ---------------------------------------------------------------- round trip through one loader


def test_inferencex_and_developer_share_the_evaluator_loader(tmp_path):
    (infx,) = inferencex_to_configurations([_infx_record()])
    dev = _dev_config(log=b"server log\n" * 10, log_redacted=True)
    points = {}
    for name, cfg in (("infx", infx), ("dev", dev)):
        path = write_configuration(tmp_path, cfg)
        check = load_e2e(tmp_path, path, config=None)
        assert check.manifest.category == "e2e" and check.manifest.configuration_id == cfg.body.configuration_id()
        (point,) = evaluation_points(check)
        points[name] = point
        for artifact in check.manifest.artifacts:
            if artifact.format == "parquet":
                assert read_parquet_schema_id(tmp_path / artifact.path) == artifact.artifact_schema
                assert pq.ParquetFile(tmp_path / artifact.path).metadata.row_group(0).column(0).compression == "ZSTD"
    i, d = points["infx"], points["dev"]
    # Canonical units regardless of source: seconds and ms both become ms.
    assert i.metrics[("ttft", "mean")] == pytest.approx(100.0) and d.metrics[("ttft", "mean")] == pytest.approx(100.0)
    assert i.metrics[("e2el", "mean")] == pytest.approx(1500.0) and d.metrics[("e2el", "mean")] == pytest.approx(1500.0)
    assert d.metrics[("ttft", "p90")] == 150.0
    assert ("ttft", "p90") in i.unavailable  # InferenceX did not report P90
    assert ("itl", "mean") in d.unavailable  # AIPerf has no separate ITL
    assert (
        i.per_device_throughput["output_throughput"] == 150.0 and d.per_device_throughput["output_throughput"] == 1200.0
    )
    assert i.per_device_throughput["total_throughput"] == 1200.0
    assert set(i.__dataclass_fields__) == set(d.__dataclass_fields__)


def test_inferencex_metrics_every_statistic_zero_vs_missing(tmp_path):
    (cfg,) = inferencex_to_configurations([_infx_record()])
    path = write_configuration(tmp_path, cfg)
    ds = load_e2e(tmp_path, path)
    by_key = {(m.metric, m.statistic): m for m in ds.metrics}
    assert by_key[("ttft", "p99")].value == pytest.approx(300.0)  # extra statistic stored
    assert by_key[("tpot", "median")].status == "valid" and by_key[("tpot", "median")].value == 0.0
    assert by_key[("itl", "mean")].status == "missing" and by_key[("itl", "mean")].value is None
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90"):
            assert (metric, stat) in by_key
    assert by_key[("output_throughput", "value")].normalization == "per_device"
    assert by_key[("output_throughput", "value")].device_count == 8
    assert by_key[("interactivity", "mean")].normalization == "per_user"
    # Unmapped metrics are not invented; they stay in the raw artifact and are noted verbatim.
    assert ds.body.source_fields["unmapped_metrics"] == ["weird_metric"]
    raw = read_raw_record(tmp_path, ds, ds.observations[0].observation_id)
    assert b"weird_metric" in raw
    assert json.loads(ds.observations[0].unmapped_metrics_json) == {"weird_metric": 7}  # typed, verbatim
    assert ds.body.dp_attention == {"decode": False, "prefill": False}  # booleans never become degrees


def test_inferencex_disaggregated_groups_and_identity():
    configs = inferencex_to_configurations(
        [
            _infx_record(),
            _infx_record(id=124, conc=64),
            _infx_record(id=125, disagg=True, prefill_tp=4, prefill_ep=2, prefill_num_workers=1, num_prefill_gpu=4),
        ]
    )
    assert sorted(len(c.observations) for c in configs) == [1, 2]
    dis = next(c for c in configs if c.body.deployment == "disaggregated")
    assert dis.body.parallelism == {"decode": {"tp": 8}, "prefill": {"tp": 4, "ep": 2}}
    assert dis.observations[0].device_count == 12
    assert dis.body.configuration_id() != next(c for c in configs if c is not dis).body.configuration_id()


def test_open_settings_do_not_change_identity():
    base = _dev_body()
    more = base.model_copy(update={"settings": {**base.settings, "new_flag": Setting(value=True, authority="unknown")}})
    assert base.configuration_id() == more.configuration_id()
    other = base.model_copy(update={"parallelism": {"aggregated": {"tp": 4}}})
    assert base.configuration_id() != other.configuration_id()


def test_requests_table_and_developer_observation(tmp_path):
    cfg = _dev_config()
    ds = load_e2e(tmp_path, write_configuration(tmp_path, cfg))
    assert len(ds.requests) == 2 and ds.requests[0].ttft_ms == pytest.approx(100.0)
    assert ds.requests[0].itl_ms is None  # not measured is null, not zero
    obs = ds.observations[0]
    assert (obs.input_tokens, obs.output_tokens, obs.requests_total, obs.outcome) == (1024, 128, 2, "complete")
    assert check_manifest is not None


def test_aiperf_per_user_throughput_unit_spelling_converts():
    summary = _aiperf_summary(output_token_throughput_per_user={"unit": "tokens/sec/user", "avg": 42.0})
    cfg = aiperf_to_configuration(summary, _aiperf_records(), _dev_body(), observation_id="dev:1", device_count=8)
    row = next(m for m in cfg.metrics if m.metric == "interactivity" and m.statistic == "mean")
    assert (row.status, row.value, row.unit) == ("valid", 42.0, "tokens/s/user")


def test_loader_rejects_tamper_and_wrong_schema(tmp_path):
    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    table = next(tmp_path.rglob("metrics.parquet"))
    table.write_bytes(table.read_bytes() + b"x")
    with pytest.raises(ValueError):
        load_e2e(tmp_path, path)


# ---------------------------------------------------------------- provenance


def test_provenance_cap_redaction_and_hash_link(tmp_path):
    big = b"A" * 100 + b"x" * (LOG_CAP_BYTES + 10) + b"Z" * 100
    artifact, info = store_provenance(tmp_path, "d", ProvenanceFile("log", big, "o"))
    assert info.truncated and info.tier == 3 and info.full_bytes == len(big)
    assert info.full_sha256 != artifact.sha256
    kept = read_provenance(tmp_path, artifact)
    assert kept.startswith(b"AAAA") and kept.endswith(b"ZZZZ") and len(kept) < len(big)
    small, small_info = store_provenance(tmp_path, "d", ProvenanceFile("log", b"hello", "o", name="s"))
    assert not small_info.truncated and gzip.decompress((tmp_path / small.path).read_bytes()) == b"hello"
    with pytest.raises(ValueError, match="never truncated"):
        store_provenance(tmp_path, "d", ProvenanceFile("raw-output", b"x" * 11, "o", cap_bytes=10))
    assert b"hf_" not in redact(b"token=hf_" + b"a" * 30)
    body = _dev_body().model_copy(update={"provenance_artifacts": [info]})
    assert public_release_problems(body) == [f"unredacted tier-3 artifact: {info.artifact_id}"]


def test_provenance_manifest_mismatch_detected(tmp_path):
    cfg = _dev_config(log=b"log")
    path = write_configuration(tmp_path, cfg)
    ds = load_e2e(tmp_path, path)
    assert ds.observations[0].log_artifact_id in {p.artifact_id for p in ds.body.provenance_artifacts}


# ---------------------------------------------------------------- snapshot


def test_snapshot_round_trip_and_tamper(tmp_path):
    members = [
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o2", record_sha256="b" * 64, manifest_path="m"
        ),
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o1", record_sha256="c" * 64, manifest_path="m"
        ),
    ]
    snap = write_snapshot(
        tmp_path,
        "snap.json",
        members,
        source_scope={"kind": "api-current-model-view"},
        selection_policy="p",
        completeness="complete",
        acquisition=[SnapshotAcquisition(observation_id="o1", source_uri="https://x")],
    )
    got, rows = read_snapshot(tmp_path, "snap.json")
    assert got == snap and [r.observation_id for r in rows] == ["o1", "o2"]
    data = (tmp_path / "snap.json").read_text().replace('"complete"', '"partial"')
    (tmp_path / "snap.json").write_text(data)
    assert read_snapshot(tmp_path, "snap.json")[0].completeness == "partial"
    (tmp_path / "snap.json").write_text(data.replace(snap.snapshot_id, "0" * 64))
    with pytest.raises(ValueError, match="snapshot ID"):
        read_snapshot(tmp_path, "snap.json")


def test_loader_refuses_foreign_table_schema(tmp_path):
    from aisim_core.data.loaders import E2eLoader

    path = write_configuration(tmp_path, _dev_config())
    with pytest.raises(LoaderError, match="unsupported table schema"):
        E2eLoader(accepted_schemas=("aisim-e2e-requests/v1",)).load(tmp_path, path)


# ---------------------------------------------------------------- review regressions


def _snapshot_args(observation):
    member = SnapshotMember(
        configuration_id="cfg-" + "a" * 64, observation_id=observation, record_sha256="b" * 64, manifest_path="m"
    )
    return [member], {"source_scope": {"kind": "k"}, "selection_policy": "p", "completeness": "complete"}


def test_sibling_snapshots_in_one_directory_stay_readable(tmp_path):
    (tmp_path / "snapshots").mkdir()
    for name, observation in (("a", "o1"), ("b", "o2")):
        members, kwargs = _snapshot_args(observation)
        write_snapshot(
            tmp_path,
            f"snapshots/{name}.json",
            members,
            acquisition=[SnapshotAcquisition(observation_id=observation, source_uri="https://x")],
            **kwargs,
        )
    assert [m.observation_id for m in read_snapshot(tmp_path, "snapshots/a.json")[1]] == ["o1"]
    assert [m.observation_id for m in read_snapshot(tmp_path, "snapshots/b.json")[1]] == ["o2"]


def test_snapshot_configuration_count_is_verified(tmp_path):
    members, kwargs = _snapshot_args("o1")
    write_snapshot(tmp_path, "snap.json", members, **kwargs)
    document = json.loads((tmp_path / "snap.json").read_text())
    document["counts"]["configurations"] = 2
    (tmp_path / "snap.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="configuration count"):
        read_snapshot(tmp_path, "snap.json")


def test_a_failed_replacement_leaves_the_existing_configuration_readable(tmp_path):
    import dataclasses

    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = [cfg.metrics[0].model_copy(update={"value": 12345.0}), *cfg.metrics[1:]]
    oversize = ProvenanceFile("raw-output", b"x" * 11, "dev:1", cap_bytes=10)
    with pytest.raises(ValueError, match="never truncated"):
        write_configuration(tmp_path, dataclasses.replace(cfg, metrics=changed, provenance=[oversize]))
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before  # nothing staged is left either
    assert len(load_e2e(tmp_path, path).metrics) == len(cfg.metrics)


def test_two_tables_for_one_schema_are_rejected(tmp_path):
    path = write_configuration(tmp_path, _dev_config())
    manifest = json.loads((tmp_path / path).read_text())
    table = next(a for a in manifest["artifacts"] if a["schema"] == "aisim-e2e-metrics/v1")
    copy = {**table, "artifact_id": "metrics-2", "path": table["path"].replace("metrics.parquet", "metrics-2.parquet")}
    (tmp_path / copy["path"]).write_bytes((tmp_path / table["path"]).read_bytes())
    manifest["artifacts"].append(copy)
    (tmp_path / path).write_text(json.dumps(manifest))
    with pytest.raises(LoaderError, match="one table per schema"):
        load_e2e(tmp_path, path)


@pytest.mark.parametrize("client_first", [True, False])
def test_client_and_server_timings_never_mix_in_an_evaluation_point(tmp_path, client_first):
    import dataclasses

    ds = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    rows = [m for m in ds.metrics if (m.metric, m.statistic) != ("ttft", "mean")]
    pair = [
        make_metric("dev:1", "ttft", "mean", value, "ms", observer=who)
        for who, value in (("client", 200.0), ("server", 20.0))
    ]
    both = dataclasses.replace(ds, metrics=[*rows, *(pair if client_first else pair[::-1])])
    with pytest.raises(ValueError, match="select one with observer"):
        evaluation_points(both)
    for who, expected in (("client", 200.0), ("server", 20.0)):
        (point,) = evaluation_points(both, observer=who)
        assert point.metrics[("ttft", "mean")] == expected and point.observers[("ttft", "mean")] == who


def test_series_phase_spelling_is_checked(tmp_path):
    from aisim_core.data.e2e import SeriesRow, name_findings_for

    ds = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    series = [SeriesRow(observation_id="dev:1", metric="running_requests", start_ms=0.0, end_ms=1.0, phase="profiling")]
    findings = name_findings_for(ds.body, ds.observations, ds.metrics, ds.requests, series)
    assert [(f.kind, f.location, f.canonical) for f in findings] == [("phase", "series[0].phase", "profile")]


def test_aiperf_request_count_is_the_successful_requests(tmp_path):
    summary = _aiperf_summary(request_count={"avg": 8}, error_request_count={"avg": 2})
    cfg = aiperf_to_configuration(summary, _aiperf_records(), _dev_body(), observation_id="dev:1", device_count=8)
    ds = load_e2e(tmp_path, write_configuration(tmp_path, cfg))
    obs = ds.observations[0]
    assert (obs.requests_total, obs.requests_completed, obs.requests_failed) == (10, 8, 2)


def test_a_failure_while_publishing_restores_the_files_already_replaced(tmp_path, monkeypatch):
    import dataclasses

    from aisim_core.data import e2e as e2e_module

    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = [cfg.metrics[0].model_copy(update={"value": 12345.0}), *cfg.metrics[1:]]
    real_replace = e2e_module.os.replace

    def fail_on_the_second_publish(src, dst):
        if str(dst).endswith("metrics.parquet") and ".previous" not in str(src):
            raise OSError("disk full")  # observations.parquet was already replaced when this fires
        return real_replace(src, dst)

    monkeypatch.setattr(e2e_module.os, "replace", fail_on_the_second_publish)
    new_observation = cfg.observations[0].model_copy(update={"input_tokens": 999})
    with pytest.raises(OSError, match="disk full"):
        write_configuration(tmp_path, dataclasses.replace(cfg, observations=[new_observation], metrics=changed))
    monkeypatch.setattr(e2e_module.os, "replace", real_replace)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert load_e2e(tmp_path, path).observations[0].input_tokens == cfg.observations[0].input_tokens


def test_a_rejected_snapshot_replacement_leaves_the_existing_snapshot_readable(tmp_path):
    members, kwargs = _snapshot_args("o1")
    write_snapshot(tmp_path, "baseline.json", members, **kwargs)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    other, kwargs = _snapshot_args("o2")
    with pytest.raises(ValueError):
        write_snapshot(tmp_path, "baseline.json", other, **{**kwargs, "selection_policy": ""})
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert [m.observation_id for m in read_snapshot(tmp_path, "baseline.json")[1]] == ["o1"]


def test_a_registered_metric_must_be_stored_in_its_canonical_unit():
    with pytest.raises(ValueError, match="must be stored in 'ms'"):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="s", status="valid", value=1.0)
    row = MetricRow(observation_id="o", metric="x_custom", statistic="mean", unit="whatever", status="valid", value=1.0)
    assert row.unit == "whatever"  # extension metrics keep their own unit


def test_the_inferencex_transfer_backend_is_part_of_the_configuration_identity():
    configs = inferencex_to_configurations(
        [
            _infx_record(id=1, framework="sglang"),
            _infx_record(id=2, framework="mori-sglang"),
            _infx_record(id=3, framework="mori-sglang", conc=64),  # run telemetry only: merges with id 2
        ]
    )
    assert sorted(len(c.observations) for c in configs) == [1, 2]
    plain = next(c for c in configs if len(c.observations) == 1)
    mori = next(c for c in configs if len(c.observations) == 2)
    assert plain.body.framework == mori.body.framework == "sglang"
    assert mori.body.identity_settings["kv_transfer_backend"] == "mori"
    assert plain.body.configuration_id() != mori.body.configuration_id()


def test_rejected_replacements_leave_the_previous_configuration_loadable(tmp_path):
    import dataclasses

    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    changed = [cfg.metrics[0].model_copy(update={"value": 12345.0}), *cfg.metrics[1:]]
    observation = cfg.observations[0]
    duplicated = dataclasses.replace(cfg, metrics=changed, observations=[observation, observation])
    with pytest.raises(ValueError, match="duplicate observation_id"):
        write_configuration(tmp_path, duplicated)
    dangling = observation.model_copy(update={"raw_artifact_id": "raw-output:missing"})
    with pytest.raises(ValueError, match="references unknown artifact raw-output:missing"):
        write_configuration(tmp_path, dataclasses.replace(cfg, metrics=changed, observations=[dangling]))
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert len(load_e2e(tmp_path, path).metrics) == len(cfg.metrics)


@pytest.mark.parametrize("client_first", [True, False])
def test_rows_that_differ_only_in_normalization_collide_on_the_returned_key(tmp_path, client_first):
    import dataclasses

    ds = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    per_user = make_metric(
        "dev:1", "interactivity", "mean", 40.0, "tokens/s/user", normalization="per_user", observer="client"
    )
    plain = make_metric(
        "dev:1", "interactivity", "mean", 90.0, "tokens/s/user", normalization="none", observer="server"
    )
    both = dataclasses.replace(ds, metrics=[*ds.metrics, *([per_user, plain] if client_first else [plain, per_user])])
    with pytest.raises(ValueError, match="select one with observer"):
        evaluation_points(both)
    (client,) = evaluation_points(both, observer="client")
    (server,) = evaluation_points(both, observer="server")
    assert client.metrics[("interactivity", "mean")] == 40.0 and client.observers[("interactivity", "mean")] == "client"
    assert server.metrics[("interactivity", "mean")] == 90.0


def test_a_retained_provenance_entry_without_its_artifact_is_rejected_before_publishing(tmp_path):
    import dataclasses

    cfg = _dev_config()
    path = write_configuration(tmp_path, cfg)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    stale = ProvenanceInfo(
        artifact_id="raw-output:retired", observation_id="dev:1", kind="raw-output", redacted=True, truncated=False,
        full_sha256="0" * 64, full_bytes=1, stored_bytes=1, cap_bytes=1,
    )  # fmt: skip
    body = cfg.body.model_copy(update={"provenance_artifacts": [stale]})
    observation = cfg.observations[0].model_copy(
        update={"raw_artifact_id": None, "log_artifact_id": None, "evidence_refs": []}
    )
    replacement = dataclasses.replace(cfg, body=body, observations=[observation], provenance=[])
    with pytest.raises(ValueError, match="provenance artifacts and manifest body disagree"):
        write_configuration(tmp_path, replacement)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert len(load_e2e(tmp_path, path).metrics) == len(cfg.metrics)
