# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Weka -> public API -> native G2 acceptance, using original fixtures."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aisim import _runtime
from aisim import main as cli
from aisim.compiler import prediction_to_replay_spec
from aisim.config.cli import CorePredictionConfig
from aisim.runner import (
    EngineReplayRunnerFactory,
    _materialize_engine_execution_spec,
)
from aisim.sweeper.replay import AdapterReplaySpec, ReplayOutputRequirements, RuntimeHookSpec
from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    config = yaml.safe_load((ROOT / "examples/cli" / name).read_text())
    config["traffic"]["source"]["paths"] = [
        str(ROOT / "tests/e2e/configs/unified_cli/fixtures/traces/weka-g2-restore.jsonl")
    ]
    config["engine"]["model"] = str(ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model")
    return config


def _run(config: dict) -> dict:
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    result = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    )
    return result.metadata["native_report"]


def _restored(report: dict) -> dict:
    # The original fixture's third outer event resumes the parent after the
    # child evicts its prefix. Weka currently encodes that event as :outer:2;
    # keep this selector tied to that fixture if request-ID encoding changes.
    return next(row for row in report["per_request"] if row["request_id"].endswith(":outer:2"))


def _evidence(report: dict) -> dict:
    fields = (
        "first_admission_g1_reused_input_tokens",
        "first_admission_host_reused_input_tokens",
        "first_admit_ms",
        "ttft_ms",
        "admission_history",
    )
    return {
        "completed_requests": report["completed_requests"],
        "committed_prefill_tokens": report["committed_prefill_tokens"],
        "g2_domains": report.get("g2_domains", []),
        "qualification": report["agentic_qualification"],
        "restore": {key: _restored(report)[key] for key in fields},
    }


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_weka_g2_cli_python_and_native_agree(name, tmp_path, capsys) -> None:
    config = _config(name)
    python_report = _run(config)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    native_payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    native_report = json.loads(_runtime.run_replay_json(json.dumps(native_payload)))
    path = tmp_path / "agentx.yaml"
    path.write_text(yaml.safe_dump(config))
    assert (
        cli.main(
            [
                "predict",
                "--stack",
                "engine",
                "--config",
                str(path),
                "--capture-per-request",
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert _evidence(cli_report) == _evidence(python_report) == _evidence(native_report)
    restored = _restored(python_report)
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] == 8
    assert python_report["agentic_qualification"] == "functional_only"
    assert python_report["completed_requests"] == 4
    assert sum(row["first_admission_host_reused_input_tokens"] for row in python_report["per_request"]) == 8
    if config["engine"]["mode"] == "disaggregated":
        assert {row["pool"] for row in restored["admission_history"]} == {
            "prefill",
            "decode",
        }
        assert len(python_report["g2_domains"]) == 1
        assert python_report["g2_domains"][0]["capacity_blocks"] == 8


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_weka_g2_restore_has_causal_bandwidth_and_capacity_controls(name) -> None:
    config = _config(name)
    fast = _run(config)
    slow_config, small_config, hbm_config = (copy.deepcopy(config) for _ in range(3))
    # YAML aliases can share objects; assign constants rather than relative edits.
    for worker in slow_config["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["h2d_bandwidth_gbps"] = 0.01
    for worker in small_config["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["num_host_blocks"] = 1
    for worker in hbm_config["engine"]["workers"].values():
        worker["kv_cache"].pop("host_offload", None)
    slow, small, hbm = (_run(candidate) for candidate in (slow_config, small_config, hbm_config))
    assert _restored(slow)["first_admit_ms"] >= _restored(fast)["first_admit_ms"] + 100
    assert _restored(slow)["ttft_ms"] >= _restored(fast)["ttft_ms"] + 100
    assert _restored(small)["first_admission_host_reused_input_tokens"] == 0
    assert _restored(hbm)["reused_input_tokens"] == 0
    assert fast["committed_prefill_tokens"] < hbm["committed_prefill_tokens"]


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_weka_g2_runs_on_multiple_replicas(name) -> None:
    config = _config(name)
    single = _run(config)
    for worker in config["engine"]["workers"].values():
        worker["parallelism"] = {**worker["parallelism"], "replicas": 2}
    report = _run(config)
    assert report["completed_requests"] == single["completed_requests"]
    pools = ("prefill", "decode") if config["engine"]["mode"] == "disaggregated" else ("agg",)
    for pool in pools:
        assert {
            route["logical_worker_id"]
            for row in report["per_request"]
            for route in row["routing_history"]
            if route["pool"] == pool
        } == {0, 1}
    # One cluster-shared pool, counted once whatever the replica count.
    assert report.get("g2_domains", []) == single.get("g2_domains", [])


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_multi_replica_g2_restores_evicted_prefixes(name, tmp_path) -> None:
    config = _multi_replica_pressure_config(name, tmp_path)
    report = _run(config)
    restored = next(row for row in report["per_request"] if row["request_id"].endswith(":outer:4"))
    assert report["completed_requests"] == 5
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] == 8

    for worker in config["engine"]["workers"].values():
        worker["kv_cache"].pop("host_offload", None)
    hbm = _run(config)
    assert all("first_admission_host_reused_input_tokens" not in row for row in hbm["per_request"])
    hbm_restored = next(row for row in hbm["per_request"] if row["request_id"].endswith(":outer:4"))
    assert hbm_restored["reused_input_tokens"] == 0
    assert report["committed_prefill_tokens"] < hbm["committed_prefill_tokens"]


def _multi_replica_pressure_config(name: str, tmp_path: Path) -> dict:
    config = _config(name)
    trace = json.loads(Path(config["traffic"]["source"]["paths"][0]).read_text())
    # Round robin sends 0, 2, 4 to worker 0. Request 2 evicts request 0's
    # entire GPU prefix, so request 4 must restore it from G2 in either scope.
    template = trace["requests"][0]
    trace["requests"] = [
        {**template, "t": index * 0.1, "hash_ids": hashes}
        for index, hashes in enumerate(([1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33], [1, 2, 3]))
    ]
    path = tmp_path / "multi-replica-pressure.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    config["traffic"]["source"]["paths"] = [str(path)]
    for worker in config["engine"]["workers"].values():
        worker["parallelism"] = {**worker["parallelism"], "replicas": 2}
        # Retain every source prefix across several plays; a cold next play
        # must follow from identity isolation, not host-capacity eviction.
        worker["kv_cache"]["host_offload"]["num_host_blocks"] = 128
    return config


@pytest.mark.parametrize("name", ["agentx-g2-local.yaml", "agentx-g2-shared-pd.yaml"])
def test_multi_replica_g2_isolates_recycled_profile_plays(name, tmp_path) -> None:
    config = _multi_replica_pressure_config(name, tmp_path)
    load = config["traffic"]["load"]
    load["agentic_snapshot"] = {"seed": 42}
    sampled = _run(config)["agentic_snapshots"][0]
    # Start the first play at turn zero so it fills G2 with every source prefix
    # before any recycled play can start at a sampled cut.
    load["agentic_snapshot"].update(
        initial_selections=[{"source_play_id": sampled["source_play_id"], "t_star_ms": 0}],
        corpus_cursor=1,
    )
    load["agentic_profile"] = {"duration_seconds": 2, "response_grace_seconds": 1}
    report = _run(config)
    snapshots = report["agentic_snapshots"]
    assert report["agentic_profile"]["plays_started"] >= 2
    assert len({snapshot["cache_id"] for snapshot in snapshots}) == len(snapshots)
    plays = {}
    for row in report["per_request"]:
        plays.setdefault(row["play_id"], []).append(row)
    assert len(plays) >= 2
    first_play = min(plays.values(), key=lambda rows: min(row["first_admit_ms"] for row in rows))
    assert any(row["first_admission_host_reused_input_tokens"] == 8 for row in first_play)
    for rows in plays.values():
        first = min(rows, key=lambda row: row["first_admit_ms"])
        assert first["first_admission_g1_reused_input_tokens"] == 0
        assert first["first_admission_host_reused_input_tokens"] == 0


def test_weka_shared_g2_contract_across_replicas() -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    for worker in config["engine"]["workers"].values():
        worker["parallelism"] = {**worker["parallelism"], "replicas": 2}
    shared = _run(config)
    # Four participants join one deployment-owned pool, counted once.
    assert [domain["capacity_blocks"] for domain in shared["g2_domains"]] == [8]
    assert shared["completed_requests"] == _run(_config("agentx-g2-shared-pd.yaml"))["completed_requests"]

    disabled, private = copy.deepcopy(config), copy.deepcopy(config)
    for worker in disabled["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"] = {**worker["kv_cache"]["host_offload"], "scope": "dp_rank_local"}
    for worker in private["engine"]["workers"].values():
        offload = dict(worker["kv_cache"]["host_offload"], scope="dp_rank_local")
        for field in ("shared_d2h_bandwidth_gbps", "shared_h2d_bandwidth_gbps"):
            offload.pop(field, None)
        worker["kv_cache"]["host_offload"] = offload
    disabled_report, private_report = _run(disabled), _run(private)
    # Disabling sharing reproduces private G2 even with shared-only fields left set.
    assert "g2_domains" not in disabled_report or not disabled_report["g2_domains"]
    assert disabled_report["per_request"] == private_report["per_request"]
    assert disabled_report["committed_prefill_tokens"] == private_report["committed_prefill_tokens"]


def test_shared_g2_capacity_and_cross_role_restores(tmp_path) -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    a, f, b, c, d, e = ([base, base + 1, base + 2] for base in (1, 11, 21, 31, 41, 51))
    # Independent AgentX requests let prefill run ahead of slower decode.
    # Both prefill replicas store six two-block prefixes in one eight-block
    # G2, evicting A/F. Their queued returns must miss the host cache.
    # Decode later republishes B/C after prefill pressure evicts them, so C
    # can restore on P0 from D1, and B on P1 from D0. Neither reader has the
    # prefix in G1. Disabling decode offload must remove both host hits.
    waves = [(0, a), (1, f), (10, b), (11, c), (20, d), (21, e), (110, a), (111, f), (500, c), (501, b)]
    header = {
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        "block_size": 4,
        "hash_id_scope": "local",
        "source": {"format": "synthetic-test", "digest": "cross-role-pressure"},
    }
    requests = [
        {
            "request_id": str(index),
            "play_id": "pressure",
            "session_id": str(index),
            "model": "synthetic-model",
            "input_length": 9,
            "output_length": 2,
            "hash_ids": hashes,
            "not_before_ms": at_ms,
            "dependencies": [],
        }
        for index, (at_ms, hashes) in enumerate(waves)
    ]
    path = tmp_path / "cross-role-pressure.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [header, *requests]) + "\n")
    config["traffic"]["source"].update(format="agentic_mooncake", paths=[str(path)])
    config["traffic"]["source"].pop("nested_timestamp_basis")
    workers = config["engine"]["workers"]
    # Detach the example's YAML alias before setting role-specific controls.
    workers["decode"] = copy.deepcopy(workers["decode"])
    for role, worker in workers.items():
        worker["parallelism"]["replicas"] = 2
        worker["kv_cache"]["capacity"]["blocks"] = 7 if role == "prefill" else 3
        worker["timing"]["decode_ms"] = 100 if role == "decode" else 1

    def run(candidate):
        report = _run(candidate)
        assert report["completed_requests"] == len(waves)
        return report, {int(row["request_id"]): row for row in report["per_request"]}

    shared, rows = run(config)
    assert [domain["capacity_blocks"] for domain in shared["g2_domains"]] == [8]
    for index, replica in ((2, 0), (3, 1), (8, 0), (9, 1)):
        assert [(route["pool"], route["logical_worker_id"]) for route in rows[index]["routing_history"]] == [
            ("prefill", replica),
            ("decode", replica),
        ]
    for index in (6, 7):
        assert rows[index]["first_admission_g1_reused_input_tokens"] == 4
        assert rows[index]["first_admission_host_reused_input_tokens"] == 0
    for index in (8, 9):
        assert rows[index]["first_admission_g1_reused_input_tokens"] == 0
        assert rows[index]["first_admission_host_reused_input_tokens"] == 8

    no_decode = copy.deepcopy(config)
    no_decode["engine"]["workers"]["decode"]["kv_cache"].pop("host_offload")
    _, no_decode_rows = run(no_decode)
    for index in (8, 9):
        assert no_decode_rows[index]["first_admission_g1_reused_input_tokens"] == 0
        assert no_decode_rows[index]["first_admission_host_reused_input_tokens"] == 0

    larger, private = copy.deepcopy(config), copy.deepcopy(config)
    for worker in larger["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["num_host_blocks"] = 16
    for worker in private["engine"]["workers"].values():
        worker["kv_cache"]["host_offload"]["scope"] = "dp_rank_local"
    _, larger_rows = run(larger)
    _, private_rows = run(private)
    for control_rows in (larger_rows, private_rows):
        # Extra capacity, including accidentally private eight-block pools,
        # retains A/F: this must differ from the shared pool's earlier misses.
        for index in (6, 7):
            assert control_rows[index]["first_admission_g1_reused_input_tokens"] == 4
            assert control_rows[index]["first_admission_host_reused_input_tokens"] == 4
    for index in (8, 9):
        assert private_rows[index]["first_admission_g1_reused_input_tokens"] == 0
        assert private_rows[index]["first_admission_host_reused_input_tokens"] == 0


@pytest.mark.parametrize("field,value", [("num_host_blocks", 16), ("shared_h2d_bandwidth_gbps", 2)])
def test_weka_g2_rejects_inconsistent_shared_pool(field, value) -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    config["engine"]["workers"]["decode"] = copy.deepcopy(config["engine"]["workers"]["decode"])
    config["engine"]["workers"]["decode"]["kv_cache"]["host_offload"][field] = value
    with pytest.raises(RuntimeError, match="cluster_shared host_offload participants are incompatible"):
        _run(config)


def test_weka_g2_rejects_incompatible_shared_layout() -> None:
    config = _config("agentx-g2-shared-pd.yaml")
    config["engine"]["workers"]["decode"] = copy.deepcopy(config["engine"]["workers"]["decode"])
    config["engine"]["workers"]["decode"]["parallelism"]["tensor"] = 2
    with pytest.raises(RuntimeError, match="cluster_shared host_offload participants are incompatible"):
        _run(config)


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    "fault,message",
    [
        ("backend", "agentic host offload requires backend=vllm on every role"),
        ("aic_nextn", "requires canonical mtp speculation"),
        ("nextn", "requires canonical mtp speculation"),
        ("speculation", "requires canonical mtp speculation"),
        ("g3", "agentic replay does not support G3 offload"),
        ("scope", "dp_rank_local"),
        ("scaling", "agentic host offload requires static worker pools without a scaling policy"),
    ],
)
def test_agentic_invalid_configuration_matrix_rejects_python_and_native(role, fault, message) -> None:
    """The same semantic failures must fail both API boundaries.

    Start from one valid lowered payload, then express each fault in the public
    and native schemas. Python aliases map to native aic_nextn; scaling hooks
    map to the native scaling provider. Native scope remains a typed enum, so
    its invalid-variant diagnostic can differ from Python's early scope gate.
    """
    aggregated = role == "aggregated"
    config = _config("agentx-g2-local.yaml" if aggregated else "agentx-g2-shared-pd.yaml")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    args_field = "agg_engine_args" if aggregated else f"{role}_engine_args"
    args = getattr(spec.backend_deployment, args_field)
    engine = payload["spec"]["engine"] if aggregated else payload["spec"]["engine"][role]
    rank = engine["rank"]
    if fault == "backend":
        args["engine_type"] = "sglang"
        rank["backend"] = "sglang"
    elif fault in {"aic_nextn", "nextn", "speculation"}:
        args[fault] = {"num_speculative_tokens": 1} if fault == "speculation" else 1
        rank["aic_nextn"] = 1
    elif fault == "g3":
        args["g3_offload"] = rank["g3_offload"] = {"scope": "worker_local", "num_g3_blocks": 8}
    elif fault == "scope":
        args["native_host_offload"]["scope"] = "unsupported_scope"
        rank["native_host_offload"]["scope"] = "unsupported_scope"
    elif fault == "scaling":
        hook = RuntimeHookSpec(provider="test.planner", kind="scaling_policy", api_version=1)
        spec = replace(spec, adapters={"planner": AdapterReplaySpec(runtime_hooks=(hook,))})
        payload["spec"]["adapters"]["scaling"]["provider"] = hook.provider
    else:
        raise AssertionError(f"unhandled semantic fault: {fault}")
    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    native_message = "requires.*canonical" if fault in {"aic_nextn", "nextn", "speculation"} else message
    with pytest.raises(RuntimeError, match=native_message):
        _runtime.run_replay_json(json.dumps(payload))


def _mtp_with_drop_off(engine: dict, depth: int) -> dict:
    """Canonical MTP with the EAGLE block drop off; G2 requires canonical speculation and default timing."""
    engine.update(
        model="meta-llama/Meta-Llama-3.1-8B",
        backend_version="0.24.0",
        estimation_mode="op_level",
        speculation={
            "kind": "mtp",
            "params": {"depth": depth},
            "expected_accepted_tokens": float(depth),
            "disable_eagle_block_drop": True,
        },
    )
    worker = engine["workers"]["aggregated"]
    worker["timing"] = {"type": "default"}
    return worker


@pytest.mark.parametrize("window", [False, True])
def test_agentic_g2_accepts_mtp_for_a_state_cache(window: bool) -> None:
    """Both API boundaries accept MTP with G2: a recurrent state with the block drop off, a sliding window with it."""
    config = _config("agentx-g2-local.yaml")
    kv_cache = _mtp_with_drop_off(config["engine"], 1)["kv_cache"]
    kv_cache["capacity"]["blocks"] = 16
    if window:
        del config["engine"]["speculation"]["disable_eagle_block_drop"]
        kv_cache["state_cache"] = {"bytes_per_request": 1_000_000, "sliding_window_lookahead_tokens": 0}
    else:
        kv_cache.update(state_cache={"bytes_per_request": 1_000_000}, prefix_match_unit=4)
        kv_cache["host_offload"]["connector"] = "simple_cpu_v0_32"
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    # A sliding window runs MTP with G2 synchronously, as a token cache does.
    assert payload["spec"]["engine"]["rank"].get("async_scheduling") == (False if window else None)
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    _runtime.run_replay_json(json.dumps(payload))


def test_hybrid_g2_forward_waits_for_stores_of_a_draft_slot_preemption() -> None:
    """A request preempted for MTP draft slots makes the forward wait for its slow stores.

    Before the forward read that wait, this run stopped with an effect-free zero-duration pass.
    """
    config = {
        "engine": {
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 256,
            "workers": {
                "aggregated": {
                    "scheduler": {"max_batched_tokens": 16, "max_sequences": 16},
                    "kv_cache": {
                        "block_size": 4,
                        "bytes_per_token": 16,
                        "prefix_match_unit": 4,
                        "capacity": {"type": "fixed", "blocks": 12},
                        "state_cache": {"bytes_per_request": 64, "num_speculative_states": 3},
                        "host_offload": {
                            "num_host_blocks": 32,
                            "connector": "offloading_v0_32",
                            "d2h_bandwidth_gbps": 1e-5,
                            "h2d_bandwidth_gbps": 1e-5,
                        },
                    },
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }
    _mtp_with_drop_off(config["engine"], 3)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    payload.pop("traffic")
    payload["spec"].update(max_sim_time_ms=5000, max_in_flight=8)
    assert payload["spec"]["engine"]["rank"]["aic_nextn_accept_rates"] == "1,1,1"
    turns = [(1, 0, 16, 4), (1, 0, 8, 4), (1, 1000, 32, 1), (100, 0, 22, 8), (200, 0, 8, 20)]
    payload["spec"]["requests"] = [
        {
            "id": str(index),
            "arrival_time_ms": arrival,
            "input_tokens": length,
            "input_token_ids": list(range(first, first + length)),
            "output_tokens": output,
        }
        for index, (arrival, first, length, output) in enumerate(turns)
    ]
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["num_requests"] == len(turns)


def test_agentic_g2_runs_multi_worker_attention_dp_pd_with_mtp_on_one_shared_pool(tmp_path) -> None:
    """Two DP2 prefill workers and one DP2 decode worker share one G2 pool under MTP.

    The resumed parent is the third request, so round-robin places it on a prefill
    rank that never held its prefix: it restores from the shared pool. Six output
    tokens per request make every decode run draft rounds.
    """
    config = _canonical_mtp(_config("agentx-g2-shared-pd.yaml"))
    source = config["traffic"]["source"]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(Path(source["paths"][0]).read_text().replace('"out":1', '"out":6'))
    source["paths"] = [str(trace)]
    prefill = config["engine"]["workers"]["prefill"]
    prefill["kv_cache"]["capacity"]["blocks"] = 4  # 9 input and 6 output tokens
    decode = config["engine"]["workers"]["decode"] = copy.deepcopy(prefill)
    prefill["parallelism"].update(replicas=2, attention_data=2)
    decode["parallelism"].update(attention_data=2)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    lowered = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)["spec"]
    assert lowered["topology"]["prefill"]["initial_workers"] == 2
    for role in ("prefill", "decode"):
        assert lowered["engine"][role]["dp_size"] == 2
        assert lowered["engine"][role]["rank"]["aic_nextn"] == 2
    report = _run(config)
    assert report["completed_requests"] == 4
    assert report["total_output_tokens"] == 24
    assert [domain["capacity_blocks"] for domain in report["g2_domains"]] == [8]
    restored = _restored(report)
    assert restored["first_admission_g1_reused_input_tokens"] == 0
    assert restored["first_admission_host_reused_input_tokens"] > 0


def test_agentic_g2_runs_on_several_aggregated_workers_with_private_pools() -> None:
    """Several aggregated workers with a private G2 each run through both boundaries."""
    config = _config("agentx-g2-local.yaml")
    config["engine"]["workers"]["aggregated"]["parallelism"]["replicas"] = 2
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    assert payload["spec"]["topology"]["workers"]["initial_workers"] == 2
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    assert report["completed_requests"] == 4
    assert report["agentic_qualification"] == "functional_only"
    workers = {entry["logical_worker_id"] for row in report["per_request"] for entry in row["routing_history"]}
    assert workers == {0, 1}


@pytest.mark.parametrize("scope", ["cluster_shared", "dp_rank_local"])
@pytest.mark.parametrize("prefill,decode", [(1, 3), (3, 1), (2, 3)])
def test_agentic_g2_runs_on_several_prefill_and_decode_workers(scope, prefill, decode) -> None:
    """P/D accepts several workers per role; a private pool restores only on the worker that stored the prefix."""
    config = _config("agentx-g2-shared-pd.yaml")
    workers = config["engine"]["workers"]
    workers["decode"] = copy.deepcopy(workers["prefill"])
    for role, replicas in (("prefill", prefill), ("decode", decode)):
        workers[role]["parallelism"]["replicas"] = replicas
        host = workers[role]["kv_cache"]["host_offload"]
        host["scope"] = scope
        if scope == "dp_rank_local":
            del host["shared_d2h_bandwidth_gbps"], host["shared_h2d_bandwidth_gbps"]
    single = copy.deepcopy(config)
    for role in ("prefill", "decode"):
        single["engine"]["workers"][role]["parallelism"]["replicas"] = 1
    report, baseline = _run(config), _run(single)
    assert report["completed_requests"] == baseline["completed_requests"] == 4
    assert report["agentic_qualification"] == "functional_only"
    for role, replicas in (("prefill", prefill), ("decode", decode)):
        used = {
            entry["logical_worker_id"]
            for row in report["per_request"]
            for entry in row["routing_history"]
            if entry["pool"] == role
        }
        assert used == set(range(replicas))
    restored = _restored(report)["first_admission_host_reused_input_tokens"]
    if prefill == 2:
        # The child runs on the other prefill worker, so the parent's prefix stays in GPU memory.
        assert restored == 0
        assert _restored(report)["first_admission_g1_reused_input_tokens"] == 8
    elif scope == "cluster_shared" or prefill == 1:
        # One pool, or one prefill worker: the parent finds its prefix as with 1P1D.
        assert restored == _restored(baseline)["first_admission_host_reused_input_tokens"] == 8
        assert report["committed_prefill_tokens"] == baseline["committed_prefill_tokens"]
    else:
        # Round robin sends the resumed parent to a prefill worker that never stored it.
        assert restored == 0
        assert report["committed_prefill_tokens"] > baseline["committed_prefill_tokens"]


def _planner_pass(prefills: int, prefill_tokens: int, decodes: int, kv: int) -> list[dict]:
    ms = 2.0 + 0.004 * prefill_tokens + 0.3 * decodes + 2e-5 * kv
    counters = {"num_prefill_requests": prefills, "sum_prefill_tokens": prefill_tokens}
    counters |= {"num_decode_requests": decodes, "sum_decode_kv_tokens": kv}
    return [{"version": 1, "wall_time": ms / 1000, "scheduled_requests": counters}]


def _planner_snapshot(path: Path, worker_type: str, kinds: tuple[str, ...]) -> str:
    """A planner fits its regression for an unknown deployment, so the snapshot records attention DP as unknown."""
    unknown = dict.fromkeys(("system", "backend_version", "tp", "pp", "attention_dp"), "unknown")
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=str(ROOT / "tests/e2e/configs/unified_cli/fixtures/tiny-model"),
            backend="vllm",
            worker_type=worker_type,
            kv_block_size=4,
            estimation_mode="fpm_regression",
            **unknown,
        )
    )
    passes = []
    for i in range(48):
        decodes, kv = 1 + i % 8, (1 + i % 8) * (128 + 37 * (i % 11))
        shapes = {
            "prefill": (1 + i % 3, 128 * (1 + i % 5), 0, 0),
            "decode": (0, 0, decodes, kv),
            "mixed": (1, 32 * (1 + i % 6), decodes, kv),
        }
        passes += [_planner_pass(*shapes[kind]) for kind in kinds]
    model.tune_with_fpms(passes)
    model.save_state_snapshot(str(path))
    return str(path)


@pytest.mark.parametrize(
    "name,kinds",
    [
        ("agentx-g2-local.yaml", {"aggregated": ("prefill", "decode", "mixed")}),
        ("agentx-g2-shared-pd.yaml", {"prefill": ("prefill",), "decode": ("decode",)}),
    ],
)
def test_agentic_g2_replays_planner_snapshots_with_an_unknown_attention_dp(name, kinds, tmp_path) -> None:
    config = _config(name)
    workers = config["engine"]["workers"]
    for role, role_kinds in kinds.items():
        snapshot = _planner_snapshot(tmp_path / role, role, role_kinds)
        timing = {"estimation_mode": "fpm_regression", "estimator_config": {"state_snapshot": {"path": snapshot}}}
        workers[role] = {**workers[role], "timing": timing}  # The P/D example aliases one worker block.
    report = _run(config)
    assert report["completed_requests"] == 4
    # The snapshots, not the example's fixed timing, price the passes.
    fixed = _run(_config(name))
    assert [row["ttft_ms"] for row in report["per_request"]] != [row["ttft_ms"] for row in fixed["per_request"]]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("with_g2", [False, True])
def test_agentic_g3_rejection_matches_without_requiring_g2(backend, with_g2) -> None:
    config = _config("agentx-g2-local.yaml")
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    payload = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)
    spec = replace(spec, backend_deployment=replace(spec.backend_deployment, backend=backend))
    args = spec.backend_deployment.agg_engine_args
    rank = payload["spec"]["engine"]["rank"]
    args["engine_type"] = rank["backend"] = backend
    args["g3_offload"] = rank["g3_offload"] = {"scope": "worker_local", "num_g3_blocks": 8}
    if not with_g2:
        args.pop("native_host_offload")
        rank.pop("native_host_offload")
    message = "agentic replay does not support G3 offload"
    with pytest.raises(ValueError, match=message):
        EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    with pytest.raises(RuntimeError, match=message):
        _runtime.run_replay_json(json.dumps(payload))


def _canonical_mtp(config: dict) -> dict:
    config["engine"].update(
        model="meta-llama/Meta-Llama-3.1-8B",
        backend_version="0.24.0",
        estimation_mode="op_level",
        speculation={
            "kind": "mtp",
            "params": {"depth": 2},
            "expected_accepted_tokens": 1.5,
            "seed": 17,
        },
    )
    for worker in config["engine"]["workers"].values():
        worker["timing"] = {"type": "default"}
    return config


def test_agentic_canonical_mtp_executes_with_local_g2() -> None:
    config = _canonical_mtp(_config("agentx-g2-local.yaml"))
    config["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = 4
    report = _run(config)
    assert report["completed_requests"] == 4
    assert report["total_output_tokens"] == 4
    assert report["agentic_qualification"] == "functional_only"
    assert _restored(report)["first_admission_host_reused_input_tokens"] > 0


@pytest.mark.parametrize("replicas,attention_dp", [(1, 2), (2, 1), (2, 2)])
def test_agentic_canonical_mtp_executes_on_static_local_g2_fleets(replicas, attention_dp, tmp_path) -> None:
    config = _canonical_mtp(_config("agentx-g2-local.yaml"))
    source = config["traffic"]["source"]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(Path(source["paths"][0]).read_text().replace('"out":1', '"out":6'))
    source["paths"] = [str(trace)]
    worker = config["engine"]["workers"]["aggregated"]
    # The last draft burst reserves 9 prompt + 5 emitted + 3 lookahead tokens:
    # ceil(17 / 4) = 5 blocks, even though only one output token remains.
    worker["kv_cache"]["capacity"]["blocks"] = 5
    worker["parallelism"].update(replicas=replicas, attention_data=attention_dp)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    lowered = _materialize_engine_execution_spec(spec, trace_block_size=4, record_per_request=True)["spec"]
    assert lowered["topology"]["workers"]["initial_workers"] == replicas
    assert lowered["engine"]["dp_size"] == attention_dp
    assert lowered["engine"]["rank"]["aic_nextn"] == 2
    report = _run(config)
    assert report["completed_requests"] == 4
    assert report["total_output_tokens"] == 24
    assert report["agentic_qualification"] == "functional_only"
