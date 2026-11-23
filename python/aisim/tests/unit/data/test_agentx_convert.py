# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX bundles convert to aisim-e2e/v1 with canonical names and unchanged silicon statistics."""

import json

import pytest

from aisim_core.data.agentx_convert import (
    AgentxBundleError,
    agentx_bundle_to_configuration,
    agentx_bundles_to_configurations,
)
from aisim_core.data.e2e import evaluation_points, load_e2e, name_findings_for_dataset, write_configuration

pytestmark = pytest.mark.unit


def _row(key, phase, start, end, out, ttft, tpot, status="completed", cached=None):
    return {
        "key": key,
        "identity_valid": True,
        "phase": phase,
        "status": status,
        "failure_reason": None,
        "start_ms": start,
        "end_ms": end,
        "input_length": 1000,
        "output_length": out,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": end - start,
        "cached_tokens": cached,
    }


def _bundle(root, name, *, category="self_collected", lanes=4, with_keys=True, model="Qwen/Qwen3.5-397B-A17B"):
    path = root / name
    path.mkdir()
    manifest = {
        "schema": "aisimulate.agentx.benchmark.v1",
        "id": name,
        "category": category,
        "model": model,
        "hardware": "B300",
        "backend": "vllm",
        "measured_backend_version": "0.31.0",
        "tensor_parallel": 8,
        "lanes": lanes,
        "corpus_id": "weka-062126-context-262144",
        "measured_requests": 3,
    }
    rows = [
        _row("k0", "preparation", 0.0, 100.0, 1, 100.0, None),
        _row("k1", "profile", 1000.0, 2000.0, 11, 200.0, 10.0, cached=64),
        _row("k2", "profile", 1500.0, 3000.0, 21, 400.0, 20.0),
        _row("k3", "profile", 2000.0, 2500.0, 5, 0.0, 0.0, status="failed"),
    ]
    configuration = {
        "schema": "aisimulate.agentx.silicon-config.v1",
        "identity": {},
        "evidence_gaps": [],
        "implementation_identity": {"image_sha256": "a" * 64},
        "effective": [{"rank": 0, "scheduler": {"max_num_seqs": 64}, "cache_config": {"block_size": 528}}],
    }
    reference = [
        {"key": "k1", "conversation_id": "c1", "source_trace_id": "t1", "source_outer_idx": 3, "agent_depth": 0}
    ]
    if not with_keys:
        reference = [{"request_id": "r1", "conversation_id": "c1"}]
    workload = {
        "variant": {"lanes": lanes, "profile_seconds": 3600},
        "silicon_reference": {
            "reference": reference,
            "silicon": {
                "metric_definitions": {"running": {"kind": "gauge", "unit": "requests"}},
                "phases": {
                    "profile": {"series": {"running": [{"start": 0.0, "end": 1.0, "value": 2.0, "coverage": 1.0}]}}
                },
                "provenance": {"aiperf_version": "1.0.12"},
            },
        },
        "other_silicon_settings_and_observations": [
            {
                "id": "agentx.weka",
                "parameters": [{"name": "context_limit", "value": 262144}, {"name": "seed", "value": 42}],
            }
        ],
    }
    for file, value in (
        ("manifest.json", manifest),
        ("configuration.json", configuration),
        ("workload.json", workload),
        ("observations.json", rows),
        ("features.json", {"features": [{"id": "speculation", "state": "disabled", "parameters": []}]}),
    ):
        (path / file).write_text(json.dumps(value))
    return path


def test_bundle_converts_with_canonical_names_and_lineage(tmp_path):
    data = agentx_bundle_to_configuration(_bundle(tmp_path, "qwen-weka-l4"))
    body = data.body
    assert (body.origin, body.hardware, body.framework, body.framework_version) == (
        "self-collected",
        "b300_sxm",
        "vllm",
        "0.31.0",
    )
    assert body.workload.kind == "agentic_trace" and body.workload.seed == 42
    assert body.settings["scheduler.max_num_seqs"].authority == "observed"
    assert body.source_ids["agentx"]["hardware"] == "B300"
    obs = data.observations[0]
    assert obs.concurrency == 4 and obs.observer == "client" and obs.valid_for_submission is False
    assert (obs.requests_total, obs.requests_completed, obs.requests_failed) == (3, 2, 1)
    phases = {r.request_id: r.phase for r in data.requests}
    assert phases == {"k0": "warmup", "k1": "profile", "k2": "profile", "k3": "profile"}
    k1 = next(r for r in data.requests if r.request_id == "k1")
    assert k1.cached_tokens == 64 and k1.source_trace_id == "t1" and k1.source_outer_idx == 3
    assert data.series[0].metric == "running_requests" and data.series[0].end_ms == 1000.0


def test_silicon_statistics_match_the_agentx_cohort_definition(tmp_path):
    data = agentx_bundle_to_configuration(_bundle(tmp_path, "qwen-weka-l4"))
    out = tmp_path / "out"
    out.mkdir()
    dataset = load_e2e(out, write_configuration(out, data))
    (point,) = evaluation_points(dataset)
    # successful profile requests k1 and k2: mean ttft 300, mean tpot 15, tokens 32 over the 2000 ms span
    assert point.metrics[("ttft", "mean")] == 300.0
    assert point.metrics[("tpot", "mean")] == 15.0
    assert point.metrics[("ttft", "median")] == 300.0
    assert point.metrics[("output_throughput:total", "value")] == pytest.approx(32 * 1000 / 2000)
    assert name_findings_for_dataset(dataset) == []


def test_bundles_with_one_serving_identity_merge_into_one_configuration(tmp_path):
    configs = agentx_bundles_to_configurations([_bundle(tmp_path, "l1", lanes=1), _bundle(tmp_path, "l4", lanes=4)])
    assert len(configs) == 1
    assert [o.observation_id for o in configs[0].observations] == ["l1", "l4"]
    assert {o.concurrency for o in configs[0].observations} == {1, 4}


def test_official_bundle_without_reference_keys_records_a_lineage_gap(tmp_path):
    data = agentx_bundle_to_configuration(_bundle(tmp_path, "agentx-1", category="agentx", with_keys=False))
    assert data.body.origin == "official" and data.observations[0].valid_for_submission is None
    assert any(g["field"] == "requests.lineage" for g in data.body.evidence_gaps)
    assert all(r.source_trace_id is None for r in data.requests)


def test_unsupported_bundle_schema_is_rejected(tmp_path):
    path = _bundle(tmp_path, "bad")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema"] = "something-else/v9"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AgentxBundleError):
        agentx_bundle_to_configuration(path)


def _set_effective_parallelism(path, **parallelism):
    configuration = json.loads((path / "configuration.json").read_text())
    configuration["effective"][0]["parallelism"] = parallelism
    (path / "configuration.json").write_text(json.dumps(configuration))


def test_device_count_is_the_observed_gpu_count_not_the_tensor_parallel_degree(tmp_path):
    path = _bundle(tmp_path, "dp2")
    _set_effective_parallelism(
        path,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=2,
        prefill_context_parallel_size=1,
    )
    data = agentx_bundle_to_configuration(path)
    assert data.observations[0].device_count == 16
    throughput = next(m for m in data.metrics if m.metric == "output_throughput")
    assert throughput.device_count == 16
    assert not any(g["field"] == "device_count" for g in data.body.evidence_gaps)


@pytest.mark.parametrize("parallelism", [None, {"tensor_parallel_size": 8}])
def test_unrecorded_gpu_count_is_null_with_an_evidence_gap(tmp_path, parallelism):
    path = _bundle(tmp_path, "unknown")
    if parallelism is not None:
        _set_effective_parallelism(path, **parallelism)
    data = agentx_bundle_to_configuration(path)
    assert data.observations[0].device_count is None
    assert next(m for m in data.metrics if m.metric == "output_throughput").device_count is None
    assert any(g["field"] == "device_count" for g in data.body.evidence_gaps)


_TP8 = {
    "tensor_parallel_size": 8,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "prefill_context_parallel_size": 1,
}


def test_observed_degrees_enter_the_configuration_identity(tmp_path):
    dp1, dp2, pp2 = (_bundle(tmp_path, name) for name in ("dp1", "dp2", "pp2"))
    _set_effective_parallelism(dp1, **_TP8)
    _set_effective_parallelism(dp2, **{**_TP8, "data_parallel_size": 2})
    _set_effective_parallelism(pp2, **{**_TP8, "pipeline_parallel_size": 2})
    configs = {n: agentx_bundle_to_configuration(p) for n, p in (("dp1", dp1), ("dp2", dp2), ("pp2", pp2))}
    assert configs["dp2"].body.parallelism == {"aggregated": {"tp": 8, "attention_dp": 2}}
    assert configs["pp2"].body.parallelism == {"aggregated": {"tp": 8, "pp": 2}}
    assert all(c.body.parallelism_complete for c in configs.values())
    assert len({c.body.configuration_id() for c in configs.values()}) == 3
    # TP8/DP1 and TP8/DP2 bundles are different deployments and must not merge.
    assert len(agentx_bundles_to_configurations([dp1, dp2])) == 2


def test_unobserved_degrees_are_not_declared_complete(tmp_path):
    plain = agentx_bundle_to_configuration(_bundle(tmp_path, "plain"))  # effective settings, no parallelism section
    assert plain.body.parallelism == {"aggregated": {"tp": 8}} and plain.body.parallelism_complete is False
    partial_path = _bundle(tmp_path, "partial")
    _set_effective_parallelism(partial_path, tensor_parallel_size=8)
    assert agentx_bundle_to_configuration(partial_path).body.parallelism_complete is False


def test_alias_spellings_give_the_same_profile_counts_and_metrics(tmp_path):
    canonical = agentx_bundle_to_configuration(_bundle(tmp_path, "canonical"))
    aliased_path = _bundle(tmp_path, "aliased")
    rows = json.loads((aliased_path / "observations.json").read_text())
    for row in rows:
        row["phase"] = {"profile": "profiling"}.get(row["phase"], row["phase"])
        row["status"] = {"completed": "success"}.get(row["status"], row["status"])
    (aliased_path / "observations.json").write_text(json.dumps(rows))
    aliased = agentx_bundle_to_configuration(aliased_path)

    def summary(d):
        obs = d.observations[0]
        return (
            (obs.requests_total, obs.requests_completed, obs.requests_failed),
            [(m.metric, m.statistic, m.value) for m in d.metrics],
            [(r.phase, r.status) for r in d.requests],
        )

    assert summary(aliased) == summary(canonical)
    assert canonical.observations[0].requests_total == 3 and canonical.observations[0].requests_completed == 2


def test_a_run_url_is_not_stored_in_the_producer_identity(tmp_path):
    path = _bundle(tmp_path, "run")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["source_run_url"] = "https://github.com/org/repo/actions/runs/123/attempts/2"
    (path / "manifest.json").write_text(json.dumps(manifest))
    obs = agentx_bundle_to_configuration(path).observations[0]
    assert (obs.producer_kind, obs.producer_id) == ("github-actions", "org/repo#123")
    manifest["source_run_url"] = "https://ci.internal.example/job/9"
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert agentx_bundle_to_configuration(path).observations[0].producer_id is None


def test_merged_bundles_keep_each_bundles_lineage_gap_with_its_observation(tmp_path):
    keyed = _bundle(tmp_path, "keyed", lanes=1)
    unkeyed = _bundle(tmp_path, "unkeyed", lanes=4, with_keys=False)
    (merged,) = agentx_bundles_to_configurations([keyed, unkeyed])
    gaps = [g for g in merged.body.evidence_gaps if g["field"] == "requests.lineage"]
    assert [g["observation_id"] for g in gaps] == ["unkeyed"]
    (merged,) = agentx_bundles_to_configurations([unkeyed, keyed])
    assert [g["observation_id"] for g in merged.body.evidence_gaps if g["field"] == "requests.lineage"] == ["unkeyed"]


def test_a_reported_degree_separates_configurations_even_when_the_gpu_count_is_unknown(tmp_path):
    pp1, pp2 = _bundle(tmp_path, "pp1"), _bundle(tmp_path, "pp2")
    _set_effective_parallelism(pp1, tensor_parallel_size=8, pipeline_parallel_size=1)
    _set_effective_parallelism(pp2, tensor_parallel_size=8, pipeline_parallel_size=2)
    first, second = agentx_bundle_to_configuration(pp1), agentx_bundle_to_configuration(pp2)
    assert second.body.parallelism == {"aggregated": {"tp": 8, "pp": 2}}
    assert first.body.parallelism_complete is False and first.observations[0].device_count is None
    assert first.body.configuration_id() != second.body.configuration_id()
    assert len(agentx_bundles_to_configurations([pp1, pp2])) == 2


def _set_speculation(path, draft_length):
    features = {
        "features": [
            {
                "id": "speculation",
                "state": "enabled",
                "parameters": [
                    {"name": "method", "value": "eagle"},
                    {"name": "num_speculative_tokens", "value": draft_length},
                ],
            }
        ]
    }
    (path / "features.json").write_text(json.dumps(features))


def test_speculation_parameters_are_part_of_the_serving_identity(tmp_path):
    short, long_, other_lanes = (_bundle(tmp_path, n, lanes=lanes) for n, lanes in (("s", 4), ("l", 4), ("o", 8)))
    _set_speculation(short, 2)
    _set_speculation(long_, 3)
    _set_speculation(other_lanes, 2)
    ids = {
        n: agentx_bundle_to_configuration(p).body.configuration_id()
        for n, p in (("s", short), ("l", long_), ("o", other_lanes))
    }
    assert ids["s"] != ids["l"]  # a different draft length is another configuration
    assert ids["s"] == ids["o"]  # only the lane count differs: still one configuration
    assert len(agentx_bundles_to_configurations([short, long_, other_lanes])) == 2


def test_a_lineage_correction_changes_the_observation_revision_and_snapshot_id(tmp_path):
    from aisim_core.data.e2e import SnapshotMember, snapshot_id

    path = _bundle(tmp_path, "lineage")
    before = agentx_bundle_to_configuration(path).observations[0]
    workload = json.loads((path / "workload.json").read_text())
    workload["silicon_reference"]["reference"][0]["conversation_id"] = "c-corrected"
    (path / "workload.json").write_text(json.dumps(workload))
    after = agentx_bundle_to_configuration(path).observations[0]
    assert after.observation_id == before.observation_id and after.configuration_id == before.configuration_id
    assert after.record_sha256 != before.record_sha256

    def snapshot(observation):
        member = SnapshotMember(
            configuration_id=observation.configuration_id,
            observation_id=observation.observation_id,
            record_sha256=observation.record_sha256,
            manifest_path="m",
        )
        return snapshot_id([member], {"kind": "k"}, "p")

    assert snapshot(after) != snapshot(before)
