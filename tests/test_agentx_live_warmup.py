# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aisim.config.traffic import (
    TrafficPredictionConfig,
    TrafficRecommendationConfig,
)
from aisim.resources import ResourceLimitError
from aisim.runner import EngineReplayRunnerFactory
from aisim.sweeper import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)
from aisim.sweeper.config import Workload


@pytest.mark.parametrize("backend,tp,dp", [("vllm", 8, 4), ("vllm", 16, 2), ("sglang", 8, 1)])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("policy", ["repeated_prefixes", "live_trajectory_v1"])
def test_native_all_phase_telemetry_preserves_replay(backend, tp, dp, mode, policy):
    """Self-authored fixed-time workload exercises the supported serving shapes."""
    args = {
        "engine_type": backend,
        "aic_backend": backend,
        "aic_model_path": "telemetry-fixture",
        "aic_system": "h200_sxm",
        "aic_tp_size": tp,
        "aic_attention_dp_size": dp,
        "block_size": 4,
        "num_gpu_blocks": 64,
        # These toy pools cannot hold SGLang's default 512-token decode reserve.
        **({"sglang": {"num_reserved_decode_tokens": 0}} if backend == "sglang" else {}),
        "enable_prefix_caching": True,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
    }
    if mode == "agg":
        deployment = BackendDeploymentSpec(
            deployment_mode=mode,
            backend=backend,
            backend_version="test",
            agg_engine_args={**args, "worker_type": "aggregated"},
            num_workers=1,
        )
    else:
        deployment = BackendDeploymentSpec(
            deployment_mode=mode,
            backend=backend,
            backend_version="test",
            prefill_engine_args={**args, "worker_type": "prefill"},
            decode_engine_args={**args, "worker_type": "decode"},
            num_prefill_workers=1,
            num_decode_workers=1,
        )
    trace = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-two-plays.jsonl"
    spec = ReplaySpec(
        backend_deployment=deployment,
        workload={
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_path": str(trace.resolve()),
            "trace_format": "weka",
            "trace_block_size": 4,
            "agentic_lanes": 4,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": True,
            "agentic_warmup_policy": policy,
            "agentic_profile": {
                "duration_seconds": 0.05,
                "response_grace_seconds": 0.01,
            },
        },
        goal={"target": "throughput"},
    )
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        baseline = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
        captured = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
                capture_telemetry=True,
                telemetry_sample_interval_ms=2.5,
            ),
        )
    finally:
        runner.close()
    # These measure simulator execution speed, not the virtual replay outcomes.
    wall_clock_metrics = {
        "wall_time_ms",
        "processed_output_tokens_per_s",
        "processed_tokens_per_s",
    }
    assert {k: v for k, v in captured.metrics.items() if k not in wall_clock_metrics} == {
        k: v for k, v in baseline.metrics.items() if k not in wall_clock_metrics
    }
    plain = baseline.metadata["native_report"]
    report = captured.metadata["native_report"]
    assert report["per_request"] == plain["per_request"]
    assert "telemetry" not in plain
    telemetry = report["telemetry"]
    assert telemetry["clock"] == "absolute_virtual_ms"
    assert telemetry["profile_start_ms"] > 0
    assert {row["phase"] for row in telemetry["samples"]} == {"preparation", "profile"}
    assert "warmup" in {row["phase"] for row in telemetry["requests"]}
    profile_rows = [row for row in telemetry["requests"] if row["phase"] == "profile"]
    assert len(profile_rows) == len(report["per_request"])
    assert len({row["agentic"]["play_id"] for row in profile_rows}) > 4
    origin = telemetry["profile_start_ms"]
    measured = {row["uuid"]: row for row in report["per_request"]}
    for row in profile_rows:
        assert row["arrival_time_ms"] == pytest.approx(measured[row["uuid"]]["arrival_time_ms"] + origin)
        assert row["terminal_time_ms"] == pytest.approx(measured[row["uuid"]]["terminal_time_ms"] + origin)
    for row in telemetry["requests"]:
        assert row["arrival_time_ms"] <= row["terminal_time_ms"]
        if row["output_length"] < 2:
            assert row["itl_ms"] is None
    gauges = [
        gauge
        for sample in telemetry["samples"]
        for role in ("prefill_scheduler_metrics", "decode_scheduler_metrics")
        for gauge in sample[role]
    ]
    assert {gauge["dp_rank"] for gauge in gauges} == set(range(dp))
    assert telemetry["lifecycle"]
    requests_by_uuid = {row["uuid"]: row for row in telemetry["requests"]}
    for event in telemetry["lifecycle"]:
        if event["event"] in {"dispatch", "causal_terminal", "quiescent"}:
            request = requests_by_uuid[event["request_uuid"]]
            assert event["phase"] == request["phase"]
            assert event["agentic"] == request["agentic"]
            assert request["arrival_time_ms"] <= event["at_ms"]
            assert event["at_ms"] <= request["terminal_time_ms"]


def traffic():
    return {
        "source": {"type": "trace", "paths": ["corpus"], "format": "weka"},
        "load": {
            "type": "trace_timestamps",
            "agentic_lanes": 2,
            "agentic_snapshot": {"seed": 42},
            "agentic_warmup": True,
            "agentic_warmup_policy": "live_trajectory_v1",
            "agentic_profile": {},
        },
    }


@pytest.mark.parametrize("schema", [TrafficPredictionConfig, TrafficRecommendationConfig])
def test_live_warmup_policy_roundtrip_and_legacy_default_omission(schema):
    source = traffic()
    parsed = schema.model_validate(source)
    assert parsed.load.agentic_warmup_policy == "live_trajectory_v1"
    assert parsed.model_dump()["load"]["agentic_warmup_policy"] == "live_trajectory_v1"
    source["load"].pop("agentic_warmup_policy")
    assert "agentic_warmup_policy" not in schema.model_validate(source).model_dump()["load"]


@pytest.mark.parametrize("missing", ["agentic_warmup", "agentic_profile"])
def test_live_warmup_requires_preparation_and_continuous_profile(missing):
    source = traffic()
    source["load"].pop(missing)
    with pytest.raises(ValidationError, match="live_trajectory_v1 requires"):
        TrafficPredictionConfig.model_validate(source)


def test_primers_only_requires_preparation_but_not_a_profile():
    source = traffic()
    source["load"]["agentic_warmup_policy"] = "primers_only"
    source["load"].pop("agentic_profile")
    assert TrafficPredictionConfig.model_validate(source).load.agentic_warmup_policy == "primers_only"
    source["load"].pop("agentic_warmup")
    with pytest.raises(ValidationError, match="primers_only requires agentic_warmup"):
        TrafficPredictionConfig.model_validate(source)
    workload = {
        "source_type": "trace",
        "load_type": "trace_timestamps",
        "trace_path": "corpus",
        "trace_format": "weka",
        "agentic_lanes": 2,
        "agentic_snapshot": {"seed": 42},
        "agentic_warmup_policy": "primers_only",
    }
    with pytest.raises(ValidationError, match="primers_only requires agentic_warmup"):
        Workload.model_validate(workload)


@pytest.mark.parametrize("policy", [True, "live", {}, None])
def test_live_warmup_rejects_unknown_policy(policy):
    source = traffic()
    source["load"]["agentic_warmup_policy"] = policy
    with pytest.raises(ValidationError):
        TrafficPredictionConfig.model_validate(source)


def test_live_warmup_workload_serialization_keeps_the_policy_explicit():
    data = {
        "source_type": "trace",
        "load_type": "trace_timestamps",
        "trace_path": "corpus",
        "trace_format": "weka",
        "agentic_lanes": 2,
        "agentic_snapshot": {"seed": 42},
        "agentic_warmup": True,
        "agentic_profile": {},
    }
    assert "agentic_warmup_policy" not in Workload.model_validate(data).model_dump()
    data["agentic_warmup_policy"] = "live_trajectory_v1"
    assert Workload.model_validate(data).model_dump()["agentic_warmup_policy"] == "live_trajectory_v1"


class ReachedExecution(Exception):
    pass


def supervise_warmup_prediction(tmp_path, monkeypatch, policy, *extra, probe=None, stack="legacy"):
    from aisim import cli_args, supervision

    raw = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "workers": {"aggregated": {}},
        },
        "traffic": traffic(),
    }
    raw["traffic"]["load"]["agentic_warmup_policy"] = policy
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)
    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["legacy"])

    def observe_run(command, **kwargs):
        if "warmup-support" not in command:
            raise ReachedExecution
        if isinstance(probe, Exception):
            raise probe
        if callable(probe):
            return probe(command, **kwargs)
        return probe or {"status": "completed", "exit_code": 0}

    monkeypatch.setattr(supervision, "run_process", observe_run)
    return supervision.main(
        [
            "predict",
            "--config",
            "unused.yaml",
            "--stack",
            stack,
            "--output-dir",
            str(tmp_path),
            "--overwrite",
            *extra,
        ]
    )


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
def test_supervisor_leaves_warmup_support_to_the_installed_stack(tmp_path, monkeypatch, policy):
    with pytest.raises(ReachedExecution):
        supervise_warmup_prediction(tmp_path, monkeypatch, policy)


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("stack", ["legacy", "engine"])
def test_supervisor_rejects_online_warmup_before_overwrite(tmp_path, monkeypatch, capsys, policy, stack):
    prior = tmp_path / "prediction.json"
    prior.write_bytes(b"previous valid results\n")
    with pytest.raises(SystemExit) as exc:
        supervise_warmup_prediction(tmp_path, monkeypatch, policy, "--online", stack=stack)
    assert exc.value.code == 2
    assert f"{policy} requires offline execution" in capsys.readouterr().err
    assert prior.read_bytes() == b"previous valid results\n"


@pytest.mark.parametrize(
    "probe,code,reason",
    [
        ({"status": "cancelled", "reason": "execution cancelled"}, 130, "execution cancelled"),
        ({"status": "failed", "exit_code": -signal.SIGINT}, 130, "warmup support check cancelled"),
        ({"status": "failed", "exit_code": -signal.SIGTERM}, 130, "warmup support check cancelled"),
        ({"status": "resource_limited", "reason": "memory budget"}, 3, "memory budget"),
        ({"status": "resource_limited", "exit_code": -signal.SIGTERM, "reason": "memory budget"}, 3, "memory budget"),
        ({"status": "timed_out", "reason": "initialization timed out"}, 124, "initialization timed out"),
        (
            {"status": "timed_out", "exit_code": -signal.SIGTERM, "reason": "initialization timed out"},
            124,
            "initialization timed out",
        ),
        (ResourceLimitError("no host memory remains"), 3, "no host memory remains"),
    ],
)
def test_supervisor_keeps_outputs_when_the_warmup_probe_cannot_answer(
    tmp_path, monkeypatch, capsys, probe, code, reason
):
    prior = tmp_path / "prediction.json"
    prior.write_bytes(b"previous valid results\n")
    assert supervise_warmup_prediction(tmp_path, monkeypatch, "live_trajectory_v1", probe=probe) == code
    assert list(tmp_path.iterdir()) == [prior]
    assert f"aisim: {reason}" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX child signal return codes")
def test_supervisor_keeps_outputs_when_only_the_probe_receives_sigint(tmp_path, monkeypatch, capsys):
    from aisim import supervision

    run_process = supervision.run_process

    def interrupt_probe(command, **kwargs):
        return run_process([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGINT)"], **kwargs)

    prior = tmp_path / "prediction.json"
    prior.write_text("previous valid results")
    assert supervise_warmup_prediction(tmp_path, monkeypatch, "live_trajectory_v1", probe=interrupt_probe) == 130
    assert list(tmp_path.iterdir()) == [prior]
    assert prior.read_text() == "previous valid results"
    assert "warmup support check cancelled" in capsys.readouterr().err


@pytest.mark.parametrize("diagnostic", ["missing", "unreadable"])
def test_supervisor_keeps_outputs_when_policy_refusal_has_no_readable_diagnostic(
    tmp_path, monkeypatch, capsys, diagnostic
):
    def refuse_probe(command, **kwargs):
        if diagnostic == "unreadable":
            Path(command[command.index("warmup-support") + 1]).mkdir()
        return {"status": "failed", "exit_code": 2}

    prior = tmp_path / "prediction.json"
    prior.write_text("previous valid results")
    with pytest.raises(SystemExit) as exc:
        supervise_warmup_prediction(tmp_path, monkeypatch, "live_trajectory_v1", probe=refuse_probe)
    assert exc.value.code == 2
    assert "does not advertise this agentic warmup policy" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [prior]
    assert prior.read_text() == "previous valid results"


@pytest.mark.parametrize("field,value", [("agentic_warmup_policy", "primers_only"), ("agentic_warmup", 1)])
def test_supervisor_preserves_outputs_when_config_changes_during_probe(tmp_path, monkeypatch, capsys, field, value):
    from aisim import supervision

    config = warmup_cli_config(tmp_path, "predict", "live_trajectory_v1")
    output = tmp_path / "output"
    output.mkdir()

    def change_config(command, **kwargs):
        assert "warmup-support" in command, "execution must not start"
        raw = yaml.safe_load(config.read_text())
        raw["traffic"]["load"][field] = value
        config.write_text(yaml.safe_dump(raw))
        return {"status": "completed", "exit_code": 0}

    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["advertising"])
    monkeypatch.setattr(supervision, "run_process", change_config)
    prior = output / "prediction.json"
    prior.write_text("previous valid results")
    with pytest.raises(SystemExit) as exc:
        supervision.main(
            ["predict", "--config", str(config), "--stack", "advertising", "--output-dir", str(output), "--overwrite"]
        )
    assert exc.value.code == 2
    assert "configuration changed during the warmup support check" in capsys.readouterr().err
    assert list(output.iterdir()) == [prior]
    assert prior.read_text() == "previous valid results"


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_supervisor_preserves_outputs_when_adapter_order_changes_during_probe(tmp_path, monkeypatch, capsys, command):
    from aisim import supervision

    config = warmup_cli_config(tmp_path, command, "live_trajectory_v1")
    raw = yaml.safe_load(config.read_text())
    raw.update(alpha={}, beta={})
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    output = tmp_path / "output"
    output.mkdir()
    prior = output / "prediction.json"
    prior.write_text("previous valid results")

    def reorder_adapters(child_command, **kwargs):
        assert "warmup-support" in child_command, "execution must not start"
        current = yaml.safe_load(config.read_text())
        current["alpha"] = current.pop("alpha")
        assert current == raw  # Mapping equality alone misses this runtime-hook order change.
        config.write_text(yaml.safe_dump(current, sort_keys=False))
        return {"status": "completed", "exit_code": 0}

    monkeypatch.setattr(
        supervision, "_installed_names", lambda _: ["advertising", "advertising.alpha", "advertising.beta"]
    )
    monkeypatch.setattr(supervision, "run_process", reorder_adapters)
    with pytest.raises(SystemExit) as exc:
        supervision.main(
            [command, "--config", str(config), "--stack", "advertising", "--output-dir", str(output), "--overwrite"]
        )
    assert exc.value.code == 2
    assert "configuration changed during the warmup support check" in capsys.readouterr().err
    assert list(output.iterdir()) == [prior]
    assert prior.read_text() == "previous valid results"


@pytest.mark.parametrize("probe_status,probe_code", [("completed", 0), ("completed", 1), ("failed", 1)])
def test_supervisor_snapshot_survives_later_file_edits_and_cleans_up(tmp_path, monkeypatch, probe_status, probe_code):
    from aisim import cli_args, supervision

    config = warmup_cli_config(tmp_path, "predict", "live_trajectory_v1")
    original = yaml.safe_load(config.read_text())
    output = tmp_path / "output"
    output.mkdir()
    prior = output / "prediction.json"
    prior.write_text("previous valid results")
    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["advertising"])
    snapshots = []

    def observe_run(command, **kwargs):
        if "warmup-support" in command:
            return {"status": probe_status, "exit_code": probe_code}
        assert not prior.exists()
        snapshot = Path(command[command.index("cli-prepared") + 1])
        snapshots.append(snapshot)
        config.write_text("invalid: changed after cleanup")
        with snapshot.open("rb") as source:
            prepared = pickle.load(source)
        assert prepared == original
        assert cli_args._load_mapping(str(config)) != prepared
        return {"status": "completed", "exit_code": 0}

    monkeypatch.setattr(supervision, "run_process", observe_run)
    assert (
        supervision.main(
            ["predict", "--config", str(config), "--stack", "advertising", "--output-dir", str(output), "--overwrite"]
        )
        == 0
    )
    assert len(snapshots) == 1
    assert not snapshots[0].exists()


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_overwrite_executes_the_checked_mapping_after_a_later_config_edit(tmp_path, monkeypatch, command):
    from aisim import supervision

    config = warmup_cli_config(tmp_path, command, "live_trajectory_v1")
    (tmp_path / "logging_stack.py").write_text(
        "from pathlib import Path\n"
        "from aisim.runner import EngineReplayRunnerFactory\n\n"
        "class LoggingRunner:\n"
        "    def __init__(self, runner):\n"
        "        self.runner = runner\n"
        "    def run(self, spec, **kwargs):\n"
        "        with Path(__file__).with_name('policy.log').open('a') as log:\n"
        "            log.write(spec.workload['agentic_warmup_policy'] + '\\n')\n"
        "        return self.runner.run(spec, **kwargs)\n"
        "    def close(self):\n"
        "        self.runner.close()\n\n"
        "class LoggingStack(EngineReplayRunnerFactory):\n"
        "    def create(self, worker_id):\n"
        "        return LoggingRunner(super().create(worker_id))\n"
    )
    metadata = tmp_path / "warmup_stack-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: warmup-stack\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[aisim.runner_factories]\nadvertising = logging_stack:LoggingStack\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))))
    monkeypatch.setattr(supervision, "_installed_names", lambda _: ["advertising"])
    run_process = supervision.run_process

    def edit_before_execution(child_command, **kwargs):
        if "warmup-support" not in child_command:
            config.write_text("invalid: changed after cleanup")
        return run_process(child_command, **kwargs)

    monkeypatch.setattr(supervision, "run_process", edit_before_execution)
    output = tmp_path / "output"
    output.mkdir()
    (output / "prediction.json").write_text("previous valid results")
    assert (
        supervision.main(
            [
                command,
                "--config",
                str(config),
                "--stack",
                "advertising",
                "--output-dir",
                str(output),
                "--overwrite",
                "--format",
                "json",
                "--set",
                "traffic.load.agentic_warmup_policy=primers_only",
            ]
        )
        == 0
    )
    result_name = "prediction.json" if command == "predict" else "recommendation.json"
    assert json.loads((output / result_name).read_text())
    assert json.loads((output / "resource-runtime.json").read_text())["status"] == "completed"
    assert set((tmp_path / "policy.log").read_text().splitlines()) == {"primers_only"}


@pytest.fixture
def warmup_worker(monkeypatch, tmp_path):
    from aisim import resource_worker

    monkeypatch.setattr(resource_worker, "runtime_budget", lambda: {"cpu_limit": 1})
    if hasattr(os, "sched_setaffinity"):
        monkeypatch.setattr(os, "sched_setaffinity", lambda *args: None)
    if hasattr(os, "nice"):
        monkeypatch.setattr(os, "nice", lambda *args: None)
    monkeypatch.setattr(
        sys, "argv", ["resource_worker", "warmup-support", str(tmp_path / "refusal"), "silent", "live_trajectory_v1"]
    )
    return resource_worker


@pytest.mark.parametrize("failure_at", ["resolution", "wrapped_resolution", "capabilities"])
@pytest.mark.parametrize("resource_refusal", [False, True])
def test_warmup_worker_distinguishes_resource_refusals_from_plugin_errors(
    warmup_worker, tmp_path, monkeypatch, capsys, failure_at, resource_refusal
):
    from aisim import stack

    def fail(*args):
        raise ResourceLimitError("stack memory budget") if resource_refusal else RuntimeError("broken plugin")

    class Factory:
        capabilities = fail

    def resolve(_):
        if failure_at == "capabilities":
            return Factory()
        if failure_at == "wrapped_resolution":
            try:
                fail()
            except Exception as exc:
                raise stack.StackLoadError("cannot load plugin") from exc
        fail()

    monkeypatch.setattr(stack, "resolve_runner_factory", resolve)
    assert warmup_worker.main() == (3 if resource_refusal else 1)
    assert not (tmp_path / "refusal").exists()
    if resource_refusal:
        assert "stack memory budget" in capsys.readouterr().err


def test_warmup_worker_returns_policy_refusal_when_diagnostic_write_fails(warmup_worker, tmp_path, monkeypatch):
    from aisim import stack

    monkeypatch.setattr(stack, "resolve_runner_factory", lambda _: WarmupStack(False, tmp_path / "runner.log"))
    (tmp_path / "refusal").mkdir()
    assert warmup_worker.main() == 2
    assert not (tmp_path / "runner.log").exists()


@pytest.mark.parametrize("policy", ["repeated_prefixes", "live_trajectory_v1", "primers_only", "live_trajectory_v2"])
@pytest.mark.parametrize("live,primers", [(False, False), (False, True), (True, False), (True, True)])
def test_public_warmup_capability_check(policy, live, primers):
    from aisim.sweeper.replay import RunnerCapabilities

    capabilities = RunnerCapabilities(supports_agentic_live_warmup=live, supports_agentic_primers_only=primers)
    if policy == "live_trajectory_v2":
        with pytest.raises(ValueError, match="unknown agentic_warmup_policy"):
            capabilities.require_agentic_warmup_policy(policy)
    elif (policy == "live_trajectory_v1" and not live) or (policy == "primers_only" and not primers):
        with pytest.raises(ValueError, match=f"before selecting {policy}"):
            capabilities.require_agentic_warmup_policy(policy)
    else:
        capabilities.require_agentic_warmup_policy(policy)


@pytest.mark.parametrize(
    "policy,capability",
    [("live_trajectory_v1", "supports_agentic_live_warmup"), ("primers_only", "supports_agentic_primers_only")],
)
def test_recommendation_refuses_unsupported_warmup_before_any_candidate(policy, capability):
    from aisim.config.cli import CoreRecommendationConfig
    from aisim.recommend import recommendation_to_sweeper
    from aisim.sweeper.search import Sweeper

    class Factory:
        def capabilities(self):
            return replace(EngineReplayRunnerFactory().capabilities(), **{capability: False})

        def create(self, worker_id):
            pytest.fail("a candidate ran")

    raw = {
        "engine": {
            "model": str(Path(__file__).parent / "e2e/configs/unified_cli/fixtures/tiny-model"),
            "hardware": "h200_sxm",
            "mode": "aggregated",
            "workers": {"aggregated": {}},
        },
        "traffic": traffic(),
        "optimization": {"target": "throughput_per_gpu", "constraints": {"max_candidate_gpus": 1}},
        "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
    }
    raw["traffic"]["load"]["agentic_warmup_policy"] = policy
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    with pytest.raises(ValueError, match=f"before selecting {policy}"):
        Sweeper(runner_factory=Factory(), show_progress=False).run(smart)


class WarmupStack:
    """Engine stack that may hide its warmup support; its runners log each replayed policy to ``log``."""

    def __init__(self, advertise, log):
        self.advertise = advertise
        self.log = log
        self.engine = EngineReplayRunnerFactory()

    def capabilities(self):
        capabilities = self.engine.capabilities()
        if self.advertise:
            return capabilities
        return replace(capabilities, supports_agentic_live_warmup=False, supports_agentic_primers_only=False)

    def create(self, worker_id):
        return LoggingRunner(self.engine.create(worker_id), self.log)


class LoggingRunner:
    def __init__(self, runner, log):
        self.runner = runner
        self.log = log

    def run(self, spec, **kwargs):
        with self.log.open("a") as log:
            log.write(spec.workload.get("agentic_warmup_policy", "repeated_prefixes") + "\n")
        return self.runner.run(spec, **kwargs)

    def close(self):
        self.runner.close()


def warmup_cli_config(tmp_path, command, policy):
    fixtures = Path(__file__).parent / "e2e/configs/unified_cli/fixtures"
    raw = {
        "traffic": traffic(),
        "engine": {
            "mode": "aggregated",
            "model": str(fixtures / "tiny-model"),
            "hardware": "h200_sxm",
            "backend": "vllm",
            "context_length": 2048,
            "workers": {
                "aggregated": {
                    "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 256}},
                    "timing": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
                }
            },
        },
    }
    raw["traffic"]["source"]["paths"] = [str(fixtures / "traces/weka-two-plays.jsonl")]
    raw["traffic"]["load"].update(
        agentic_warmup_policy=policy,
        agentic_profile={"duration_seconds": 0.05, "response_grace_seconds": 0.01},
    )
    if command == "recommend":
        raw["optimization"] = {"target": "throughput", "constraints": {"max_candidate_gpus": 1}}
        raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1, "seed": 11}
    config = tmp_path / f"{command}.yaml"
    config.write_text(yaml.safe_dump(raw))
    return config


def run_warmup_cli(tmp_path, monkeypatch, command, stack, policy):
    from aisim import main as cli

    config = warmup_cli_config(tmp_path, command, policy)
    log = tmp_path / f"{stack}.log"
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda name: WarmupStack(name == "advertising", log))
    argv = [command, "--stack", stack, "--config", str(config), "--output-dir", str(tmp_path / stack)]
    return cli.main([*argv, "--format", "json"]), log


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_cli_runs_warmup_only_on_a_stack_that_advertises_it(tmp_path, monkeypatch, capsys, command, policy):
    # Call the CLI as the supervisor's child: agentic profiles have no static memory bound.
    monkeypatch.setenv(
        "_AISIM_SUPERVISED_BUDGET",
        json.dumps(
            {
                "supervisor_pid": os.getpid(),
                "memory_limit_bytes": 4_000_000_000,
                "cpu_limit": 1,
                "reserved_host_memory_bytes": 1_000_000_000,
                "events_path": str(tmp_path / "events.jsonl"),
                "ready_path": str(tmp_path / "ready"),
            }
        ),
    )
    status, log = run_warmup_cli(tmp_path, monkeypatch, command, "advertising", policy)
    assert status == 0
    assert set(log.read_text().split()) == {policy}

    with pytest.raises(SystemExit) as exc:
        run_warmup_cli(tmp_path, monkeypatch, command, "silent", policy)
    assert exc.value.code == 2
    assert f"before selecting {policy}" in capsys.readouterr().err


def run_installed_stack(tmp_path, command, stack, policy, *, overwrite=True, leftovers=None, output_adapter=False):
    (tmp_path / "warmup_stacks.py").write_text(
        "from dataclasses import replace\n\n"
        "from aisim.runner import EngineReplayRunnerFactory\n\n\n"
        "class Silent(EngineReplayRunnerFactory):\n"
        "    def capabilities(self):\n"
        "        flags = {'supports_agentic_live_warmup': False, 'supports_agentic_primers_only': False}\n"
        "        return replace(super().capabilities(), **flags)\n"
        "\nclass ResourceRefusal(EngineReplayRunnerFactory):\n"
        "    def capabilities(self):\n"
        "        from pathlib import Path\n"
        "        from aisim.resources import ResourceLimitError\n"
        "        with Path(__file__).with_name('capabilities.log').open('a') as log:\n"
        "            log.write('called\\n')\n"
        "        raise ResourceLimitError('stack memory budget')\n"
        "\nclass ConstructorResourceRefusal(EngineReplayRunnerFactory):\n"
        "    def __init__(self):\n"
        "        from pathlib import Path\n"
        "        from aisim.resources import ResourceLimitError\n"
        "        with Path(__file__).with_name('constructor.log').open('a') as log:\n"
        "            log.write('called\\n')\n"
        "        raise ResourceLimitError('stack constructor memory budget')\n"
        "\nclass Interrupted(EngineReplayRunnerFactory):\n"
        "    def capabilities(self):\n"
        "        import os, signal\n"
        "        from pathlib import Path\n"
        "        with Path(__file__).with_name('capabilities.log').open('a') as log:\n"
        "            log.write('called\\n')\n"
        "        os.kill(os.getpid(), signal.SIGTERM)\n"
        "\nclass Broken(EngineReplayRunnerFactory):\n"
        "    def capabilities(self):\n"
        "        from pathlib import Path\n"
        "        with Path(__file__).with_name('capabilities.log').open('a') as log:\n"
        "            log.write('called\\n')\n"
        "        raise RuntimeError('broken plugin')\n"
        "\nclass LoggingOutputAdapter:\n"
        "    name = 'warmup_log'\n"
        "    api_version = 1\n"
        "    def record(self, event):\n"
        "        from pathlib import Path\n"
        "        with Path(__file__).with_name('adapter.log').open('a') as log:\n"
        "            log.write(event + '\\n')\n"
        "    def subscribe(self, config):\n"
        "        from aisim.output_adapter import RecommendationOutputCallbacks\n"
        "        self.record('subscribe')\n"
        "        return RecommendationOutputCallbacks(\n"
        "            on_complete=lambda: self.record('complete'),\n"
        "            on_failure=lambda error: self.record('failure: ' + str(error)),\n"
        "        )\n"
        "    def write(self, config, *, result, output_dir):\n"
        "        self.record('write')\n"
        "        return []\n"
    )
    metadata = tmp_path / "warmup_stacks-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: warmup-stacks\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(
        "[aisim.runner_factories]\n"
        "advertising = aisim.runner:EngineReplayRunnerFactory\n"
        "silent = warmup_stacks:Silent\n"
        "resource_refusal = warmup_stacks:ResourceRefusal\n"
        "constructor_resource_refusal = warmup_stacks:ConstructorResourceRefusal\n"
        "interrupted = warmup_stacks:Interrupted\n"
        "broken = warmup_stacks:Broken\n"
        "\n[aisim.output_adapters]\n"
        "warmup_log = warmup_stacks:LoggingOutputAdapter\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    if overwrite:
        (output / "recommendations").mkdir()
    original = (
        {
            "prediction.json": "old result",
            "recommendation.json": "old result",
            "recommendations/0001.yaml": "old config",
            "resource-runtime.json": "old diagnostics",
            "execution-events.jsonl": "old events\n",
        }
        if overwrite
        else dict(leftovers or {})
    )
    for name, contents in original.items():
        (output / name).write_text(contents)
    config = warmup_cli_config(tmp_path, command, policy)
    if output_adapter:
        raw = yaml.safe_load(config.read_text())
        raw["warmup_log"] = {}
        config.write_text(yaml.safe_dump(raw))
    path = os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "aisim", command, "--config", str(config), "--stack", stack]
        + ["--output-dir", str(output), "--format", "json"]
        + (["--overwrite"] if overwrite else [])
        + (["--output", "warmup_log"] if output_adapter else []),
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=120,
    )
    outputs = {str(path.relative_to(output)): path.read_text() for path in output.rglob("*") if path.is_file()}
    return result, original, outputs


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_overwrite_keeps_outputs_when_the_installed_stack_lacks_the_policy(tmp_path, command, policy):
    result, original, outputs = run_installed_stack(tmp_path, command, "silent", policy)
    assert result.returncode == 2, result.stderr
    assert f"before selecting {policy}" in result.stderr
    assert outputs == original


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_cli_without_overwrite_refuses_an_unsupported_policy_with_exit_2(tmp_path, command, policy):
    result, _, outputs = run_installed_stack(tmp_path, command, "silent", policy, overwrite=False)
    assert result.returncode == 2, result.stderr
    assert f"before selecting {policy}" in result.stderr
    # Refused before resource admission and output adapters: the run writes nothing.
    assert outputs == {}


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("stack", ["silent", "advertising"])
def test_recommendation_cli_checks_policy_before_output_adapter_callbacks(tmp_path, policy, stack):
    result, _, outputs = run_installed_stack(tmp_path, "recommend", stack, policy, overwrite=False, output_adapter=True)
    log = tmp_path / "adapter.log"
    if stack == "silent":
        assert result.returncode == 2, result.stderr
        assert f"before selecting {policy}" in result.stderr
        assert not log.exists(), "a refused policy must neither subscribe nor call on_failure"
        assert outputs == {}
    else:
        assert result.returncode == 0, result.stderr
        assert log.read_text().splitlines() == ["subscribe", "complete", "write"]
        assert json.loads(outputs["recommendation.json"])["candidates"]


FAILED_RUN = {
    "fpm-coverage.json": "{}",
    "resource-runtime.json": json.dumps({"schema_version": 1, "status": "failed"}),
}


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_retry_keeps_failed_run_diagnostics_when_the_stack_lacks_the_policy(tmp_path, command):
    result, original, outputs = run_installed_stack(
        tmp_path, command, "silent", "primers_only", overwrite=False, leftovers=FAILED_RUN
    )
    assert result.returncode == 2, result.stderr
    assert "before selecting primers_only" in result.stderr
    assert outputs == original == FAILED_RUN


def test_retry_reuses_failed_run_diagnostics_when_the_stack_advertises_the_policy(tmp_path):
    result, _, outputs = run_installed_stack(
        tmp_path, "predict", "advertising", "primers_only", overwrite=False, leftovers=FAILED_RUN
    )
    assert result.returncode == 0, result.stderr
    assert "removed diagnostics left by a failed run" in result.stderr
    assert "prediction.json" in outputs
    assert json.loads(outputs["resource-runtime.json"])["status"] == "completed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX child signal return codes")
@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_overwrite_keeps_outputs_when_installed_support_probe_receives_sigterm(tmp_path, command, policy):
    result, original, outputs = run_installed_stack(tmp_path, command, "interrupted", policy)
    assert result.returncode == 130, result.stderr
    assert "warmup support check cancelled" in result.stderr
    assert outputs == original
    assert (tmp_path / "capabilities.log").read_text() == "called\n"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_installed_plugin_failure_keeps_exit_1(tmp_path, command, overwrite):
    result, _, outputs = run_installed_stack(tmp_path, command, "broken", "live_trajectory_v1", overwrite=overwrite)
    assert result.returncode == 1, result.stderr
    assert "broken plugin" in result.stderr
    assert json.loads(outputs["resource-runtime.json"])["exit_code"] == 1
    # A positive-exit probe failure still reaches the execution child for normal error reporting,
    # so the plugin is asked twice, with or without --overwrite.
    assert (tmp_path / "capabilities.log").read_text() == "called\n" * 2


def test_overwrite_replaces_outputs_on_an_installed_stack_with_the_policy(tmp_path):
    result, original, outputs = run_installed_stack(tmp_path, "predict", "advertising", "live_trajectory_v1")
    assert result.returncode == 0, result.stderr
    assert outputs["prediction.json"] != original["prediction.json"]
    assert json.loads(outputs["resource-runtime.json"])["status"] == "completed"


@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_overwrite_keeps_outputs_when_installed_capabilities_refuse_resources(tmp_path, command):
    result, original, outputs = run_installed_stack(tmp_path, command, "resource_refusal", "live_trajectory_v1")
    assert result.returncode == 3, result.stderr
    assert "stack memory budget" in result.stderr
    assert outputs == original
    assert (tmp_path / "capabilities.log").read_text() == "called\n"


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
@pytest.mark.parametrize("command", ["predict", "recommend"])
def test_overwrite_keeps_outputs_when_installed_constructor_refuses_resources(tmp_path, command, policy):
    result, original, outputs = run_installed_stack(tmp_path, command, "constructor_resource_refusal", policy)
    assert result.returncode == 3, result.stderr
    assert "stack constructor memory budget" in result.stderr
    assert outputs == original
    # A retry in an execution child would invoke the constructor a second time.
    assert (tmp_path / "constructor.log").read_text() == "called\n"


@pytest.mark.parametrize("policy", ["live_trajectory_v1", "primers_only"])
def test_unsupervised_recommendation_wraps_unsupported_warmup_as_runtime_error(tmp_path, monkeypatch, policy):
    from aisim.config.cli import CoreRecommendationConfig
    from aisim.recommend import run_recommendation

    monkeypatch.delenv("_AISIM_SUPERVISED_BUDGET", raising=False)
    path = warmup_cli_config(tmp_path, "recommend", policy)
    config = CoreRecommendationConfig.model_validate(yaml.safe_load(path.read_text()))
    with pytest.raises(RuntimeError, match=f"before selecting {policy}"):
        run_recommendation(
            config, stack="silent", runner_factory=WarmupStack(False, tmp_path / "runner.log"), show_progress=False
        )
    assert not (tmp_path / "runner.log").exists()


def test_supervisor_default_recommendation_without_traffic_keeps_existing_path(tmp_path, monkeypatch):
    from aisim import cli_args, supervision

    raw = {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "mode": "aggregated",
            "workers": {"aggregated": {}},
        },
        "optimization": {
            "target": "throughput_per_gpu",
            "constraints": {"max_candidate_gpus": 1},
        },
    }
    monkeypatch.setattr(cli_args, "_load_mapping", lambda _: raw)

    class ReachedExecution(Exception):
        pass

    def observe_run(*args, **kwargs):
        raise ReachedExecution

    monkeypatch.setattr(supervision, "run_process", observe_run)
    output = tmp_path / "output"
    with pytest.raises(ReachedExecution):
        supervision.main(
            [
                "recommend",
                "--config",
                "unused.yaml",
                "--stack",
                "engine",
                "--output-dir",
                str(output),
            ]
        )
    assert output.is_dir()
