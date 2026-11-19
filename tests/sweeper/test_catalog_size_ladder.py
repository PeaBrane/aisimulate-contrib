# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local CLI regressions; prediction and recommendation subprocesses are fake runners."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from aisim.config.cli import CoreRecommendationConfig
from aisim.sweeper.parallel_enum import enumerate_parallel_configs

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ladder():
    path = ROOT / "skills/aisim-gpu-selection-and-sizing/scripts/size_ladder.py"
    spec = importlib.util.spec_from_file_location("size_ladder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sizing_config(tmp_path):
    cfg = {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 2048, "output_tokens": 256},
            "load": {"type": "poisson", "requests_per_second": 20},
            "stop": {"requests": 6000},
        },
        "engine": {
            "mode": "aggregated",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "hardware": "b200_sxm",
            "backend": {"choices": ["vllm", "sglang"]},
            "workers": {"aggregated": {}},
        },
        "evaluation": {"sla": {"ttft_ms": 500, "itl_ms": 50}},
        "optimization": {
            "target": "min_gpus",
            "constraints": {"max_candidate_gpus": 16, "min_goodput_rps": 18},
        },
    }
    CoreRecommendationConfig.model_validate(cfg)
    path = tmp_path / "sizing.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def invoke(ladder, monkeypatch, sizing_config, tmp_path, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        ["size_ladder.py", str(sizing_config), "--hardware", "b200_sxm", "--out-root", str(tmp_path / "out"), *args],
    )
    return ladder.main()


@pytest.mark.parametrize(
    "override",
    [
        "traffic.load.requests_per_second=0",
        "traffic.load.requests_per_second=-1",
        "traffic.load.requests_per_second=.nan",
        "traffic.load.requests_per_second=fast",
        "optimizer.max_trials=0",
        "optimization.constraints.min_candidate_gpus=2",
        "traffic.load.requests_per_second.bad=1",
    ],
)
def test_invalid_effective_config_stops_before_execution(
    ladder, monkeypatch, sizing_config, tmp_path, capsys, override
):
    monkeypatch.setattr(ladder, "run", lambda *args: pytest.fail("invalid config must not start AISim"))
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path, "--set", override) == 2
    output = capsys.readouterr()
    assert "invalid recommendation config" in output.err
    assert "Traceback" not in output.err
    assert not (tmp_path / "out/probe").exists()


def test_zero_rate_without_schema_stops_before_arithmetic(ladder, monkeypatch, sizing_config, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "aisim.config.cli", None)
    monkeypatch.setattr(ladder, "run", lambda *args: pytest.fail("invalid rate must not start AISim"))
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path, "--set", "traffic.load.requests_per_second=0") == 2
    assert "positive and finite" in capsys.readouterr().err


@pytest.mark.parametrize("timeout_at", [1, 2])
def test_probe_timeout_stops_before_next_probe_or_recommend(
    ladder, monkeypatch, sizing_config, tmp_path, capsys, timeout_at
):
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        assert cmd[1] == "predict"
        assert len(calls) <= timeout_at
        if len(calls) == timeout_at:
            return None, "", "timed out after 1800s"
        return 0, "", ""

    monkeypatch.setattr(ladder, "run", fake_run)
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path) == 3
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == timeout_at
    assert list(report["probe"]) == ["vllm", "sglang"][:timeout_at]
    assert report["probe"][["vllm", "sglang"][timeout_at - 1]]["state"] == "timeout"
    assert report["steps"] == []
    assert report["answer"] is None
    assert report["stopped"] == "timeout during the coverage probe"


def test_valid_override_and_one_gpu_fit_failure_can_proceed(ladder, monkeypatch, sizing_config, tmp_path, capsys):
    # An override may repair the file's rate; validation must use the effective mapping.
    cfg = yaml.safe_load(sizing_config.read_text())
    cfg["traffic"]["load"]["requests_per_second"] = 0
    sizing_config.write_text(yaml.safe_dump(cfg))
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        if cmd[1] == "predict":
            return 1, "", "model does not fit one GPU"
        assert "traffic.load.requests_per_second=20" in cmd
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir()
        (out / "recommendation.json").write_text(
            json.dumps({"views": {"top_n": ["pick"]}, "candidates": [{"candidate_id": "pick", "used_gpus": 1}]})
        )
        return 0, "", ""

    monkeypatch.setattr(ladder, "run", fake_run)
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path, "--set", "traffic.load.requests_per_second=20") == 0
    report = json.loads(capsys.readouterr().out)
    assert [cmd[1] for cmd in calls] == ["predict", "predict", "recommend"]
    assert report["offered_rps"] == 20
    assert report["arrival_window_s"] == 300
    assert report["answer"]["used_gpus"] == 1


@pytest.mark.parametrize("file_max_gpus", [2, 16])
def test_retry_above_rejected_two_gpu_answer(ladder, monkeypatch, sizing_config, tmp_path, capsys, file_max_gpus):
    cfg = yaml.safe_load(sizing_config.read_text())
    cfg["optimization"]["constraints"]["max_candidate_gpus"] = file_max_gpus
    sizing_config.write_text(yaml.safe_dump(cfg))
    budgets = []

    def fake_run(cmd, timeout):
        if cmd[1] == "predict":
            return 0, "", ""
        overrides = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--set"]
        effective = ladder.apply_sets(yaml.safe_load(sizing_config.read_text()), overrides)
        constraints = CoreRecommendationConfig.model_validate(effective).optimization.constraints
        assert constraints.min_candidate_gpus == 3
        assert constraints.max_candidate_gpus == 3
        budgets.append(constraints.max_candidate_gpus)
        allowed = enumerate_parallel_configs(
            is_moe=True,
            backend="vllm",
            gpu_budget=constraints.max_candidate_gpus,
            min_gpu_budget=constraints.min_candidate_gpus,
        )
        assert allowed and all(candidate.total_gpus > 2 for candidate in allowed)
        used_gpus = min(candidate.total_gpus for candidate in allowed)
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir()
        (out / "recommendation.json").write_text(
            json.dumps(
                {
                    "views": {"top_n": ["replacement"]},
                    "candidates": [{"candidate_id": "replacement", "used_gpus": used_gpus}],
                }
            )
        )
        return 0, "", ""

    monkeypatch.setattr(ladder, "run", fake_run)
    assert (
        invoke(
            ladder,
            monkeypatch,
            sizing_config,
            tmp_path,
            "--budgets",
            "3,4,8,16",
            "--no-refine",
            "--set",
            "optimization.constraints.min_candidate_gpus=3",
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert budgets == [3]
    assert report["answer"]["used_gpus"] == 3


@pytest.mark.parametrize(("goodput_share", "expected"), [(18 / 20, "pass"), (0.95, "fail")])
def test_headroom_uses_agreed_throughput_floor(monkeypatch, sizing_config, tmp_path, capsys, goodput_share, expected):
    path = ROOT / "skills/aisim-gpu-selection-and-sizing/scripts/headroom_scan.py"
    spec = importlib.util.spec_from_file_location("headroom_scan", path)
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    cfg = yaml.safe_load(sizing_config.read_text())
    cfg.pop("optimization")
    cfg["engine"]["backend"] = "vllm"
    sizing_config.write_text(yaml.safe_dump(cfg))
    metrics = {
        "completed_requests": 6000,
        "num_requests": 6000,
        "goodput_completed_requests": 6000,
        "goodput_request_throughput_rps": 18.8,
        "mean_ttft_ms": 100,
        "num_ttft_samples": 6000,
        "mean_tpot_ms": 10,
        "num_tpot_samples": 6000,
    }

    def fake_predict(cmd, **kwargs):
        assert cmd[1] == "predict"
        assert "traffic.load.requests_per_second=20.0" in cmd
        return scan.subprocess.CompletedProcess(cmd, 0, json.dumps(metrics), "")

    monkeypatch.setattr(scan.subprocess, "run", fake_predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headroom_scan.py",
            str(sizing_config),
            "--min-compliant-share",
            "0.99",
            "--min-goodput-share",
            str(goodput_share),
            "--steps",
            "1.0",
            "--workdir",
            str(tmp_path / "headroom"),
        ],
    )
    assert scan.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows"][0]["status"] == expected
    assert report["rows"][0]["goodput_rps"] == 18.8


@pytest.mark.parametrize(
    "override",
    [
        "engine.hardware=h200_sxm",
        "engine={hardware: h200_sxm}",
        "optimization.constraints.max_candidate_gpus=64",
        "optimization.constraints={max_candidate_gpus: 64, min_goodput_rps: 18}",
        "optimization={target: min_gpus, constraints: {max_candidate_gpus: 64, min_goodput_rps: 18}}",
    ],
)
def test_ladder_managed_overrides_stop_before_execution(ladder, monkeypatch, sizing_config, tmp_path, capsys, override):
    monkeypatch.setattr(ladder, "run", lambda *args: pytest.fail("conflicting override must not start AISim"))
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path, "--set", override) == 2
    assert "managed by --hardware or --budgets" in capsys.readouterr().err
    assert not (tmp_path / "out/probe").exists()


@pytest.fixture
def headroom():
    path = ROOT / "skills/aisim-gpu-selection-and-sizing/scripts/headroom_scan.py"
    spec = importlib.util.spec_from_file_location("headroom_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prediction_config(sizing_config):
    cfg = yaml.safe_load(sizing_config.read_text())
    cfg.pop("optimization")
    cfg["engine"]["backend"] = "vllm"
    sizing_config.write_text(yaml.safe_dump(cfg))
    return sizing_config


def invoke_headroom(headroom, monkeypatch, prediction_config, tmp_path, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headroom_scan.py",
            str(prediction_config),
            "--min-compliant-share",
            "0.99",
            "--min-goodput-share",
            "0.90",
            "--steps",
            "1.0,1.25",
            "--workdir",
            str(tmp_path / "headroom"),
            *args,
        ],
    )
    return headroom.main()


@pytest.mark.parametrize(
    "override",
    [
        "traffic.load.requests_per_second=40",
        "traffic.load={type: poisson, requests_per_second: 40}",
        "traffic.stop.requests=12000",
        "traffic.stop.requests_per_load_unit=1",
        "traffic.stop={requests: 12000}",
        "traffic={load: {type: poisson, requests_per_second: 40}}",
        "evaluation.sla.ttft_ms=50",
        "evaluation.sla={ttft_ms: 50}",
        "evaluation={sla: {ttft_ms: 50}}",
    ],
)
def test_headroom_managed_overrides_stop_before_execution(
    headroom, monkeypatch, prediction_config, tmp_path, capsys, override
):
    monkeypatch.setattr(
        headroom.subprocess, "run", lambda *args, **kwargs: pytest.fail("conflicting override must not start AISim")
    )
    assert invoke_headroom(headroom, monkeypatch, prediction_config, tmp_path, "--set", override) == 2
    assert "set traffic.load, traffic.stop and evaluation.sla in the input YAML" in capsys.readouterr().err
    assert not (tmp_path / "headroom").exists()


def test_headroom_resource_override_keeps_report_and_prediction_inputs_aligned(
    headroom, monkeypatch, prediction_config, tmp_path, capsys
):
    from aisim.cli_args import _apply_overrides
    from aisim.config.cli import CorePredictionConfig

    predictions = []

    def fake_predict(cmd, **kwargs):
        effective = yaml.safe_load(prediction_config.read_text())
        overrides = [cmd[index + 1] for index, value in enumerate(cmd[:-1]) if value == "--set"]
        _apply_overrides(effective, overrides, command="predict")
        parsed = CorePredictionConfig.model_validate(effective)
        rate = parsed.traffic.load.requests_per_second
        total = parsed.traffic.stop.requests
        assert effective["execution"]["resources"]["memory_limit_gb"] == 4
        assert effective["evaluation"]["sla"] == {"ttft_ms": 500, "itl_ms": 50}
        predictions.append((rate, total))
        metrics = {
            "num_requests": total,
            "completed_requests": total,
            "goodput_completed_requests": total,
            "goodput_request_throughput_rps": rate * 0.99,
            "mean_ttft_ms": 100,
            "num_ttft_samples": total,
            "mean_tpot_ms": 10,
            "num_tpot_samples": total,
        }
        return headroom.subprocess.CompletedProcess(cmd, 0, json.dumps(metrics), "")

    monkeypatch.setattr(headroom.subprocess, "run", fake_predict)
    assert (
        invoke_headroom(
            headroom, monkeypatch, prediction_config, tmp_path, "--set", "execution.resources.memory_limit_gb=4"
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert predictions == [(20, 6000), (25, 7500)]
    assert [row["offered_rps"] for row in report["rows"]] == [rate for rate, _ in predictions]
    assert [row["num_requests"] for row in report["rows"]] == [total for _, total in predictions]
    assert [row["status"] for row in report["rows"]] == ["pass", "pass"]


FALLBACK_LINE = "Loading low-fidelity fallback rows for gemm (b200_sxm, vllm 0.33.0)"


def test_ladder_keeps_fallback_warnings_from_a_successful_search(ladder, monkeypatch, sizing_config, tmp_path, capsys):
    def fake_run(cmd, timeout):
        if cmd[1] == "predict":
            return 0, "", ""
        out = Path(cmd[cmd.index("--output-dir") + 1])
        out.mkdir()
        (out / "recommendation.json").write_text(
            json.dumps({"views": {"top_n": ["pick"]}, "candidates": [{"candidate_id": "pick", "used_gpus": 1}]})
        )
        return 0, "searched\n", f"INFO start\nWARNING {FALLBACK_LINE}\n"

    monkeypatch.setattr(ladder, "run", fake_run)
    assert invoke(ladder, monkeypatch, sizing_config, tmp_path) == 0
    report = json.loads(capsys.readouterr().out)
    step = report["steps"][0]
    assert step["status"] == "qualified"
    assert step["fallback_warnings"] == [f"WARNING {FALLBACK_LINE}"]
    log = Path(step["log_file"])
    assert log.parent == Path(step["output_dir"]).parent
    assert FALLBACK_LINE in log.read_text() and "searched" in log.read_text()


def test_headroom_keeps_fallback_warnings_from_a_successful_prediction(
    headroom, monkeypatch, prediction_config, tmp_path, capsys
):
    def fake_predict(cmd, **kwargs):
        total = int(next(v.split("=", 1)[1] for v in cmd if v.startswith("traffic.stop.requests=")))
        rate = float(next(v.split("=", 1)[1] for v in cmd if v.startswith("traffic.load.requests_per_second=")))
        metrics = {
            "num_requests": total,
            "completed_requests": total,
            "goodput_completed_requests": total,
            "goodput_request_throughput_rps": rate * 0.99,
            "mean_ttft_ms": 100,
            "num_ttft_samples": total,
            "mean_tpot_ms": 10,
            "num_tpot_samples": total,
        }
        return headroom.subprocess.CompletedProcess(cmd, 0, json.dumps(metrics), f"WARNING {FALLBACK_LINE}\n")

    monkeypatch.setattr(headroom.subprocess, "run", fake_predict)
    assert invoke_headroom(headroom, monkeypatch, prediction_config, tmp_path) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [row["status"] for row in rows] == ["pass", "pass"]
    for row in rows:
        assert row["fallback_warnings"] == [f"WARNING {FALLBACK_LINE}"]
        assert FALLBACK_LINE in Path(row["log_file"]).read_text()


@pytest.fixture
def consumer_fixture():
    return json.loads((ROOT / "benchmarks/evidence/skills-catalog/consumer-fixture.json").read_text())


def test_ladder_classifies_saved_consumer_fixture(ladder, consumer_fixture, tmp_path):
    (tmp_path / "recommendation.json").write_text(json.dumps(consumer_fixture["recommendation"]))
    status, _, report = ladder.classify(0, "", tmp_path)
    assert status == consumer_fixture["expected"]["ladder_status"]
    assert ladder.best_selected(report)["used_gpus"] == 2
    assert ladder.classify(3, "memory budget", tmp_path)[0] == "qualified_partial"
    report["candidates"] = [report["candidates"][1]]
    report["views"]["top_n"] = []
    (tmp_path / "recommendation.json").write_text(json.dumps(report))
    assert ladder.classify(1, "no feasible candidates", tmp_path)[0] == "infeasible"


def test_headroom_parses_saved_consumer_fixture(
    headroom, consumer_fixture, monkeypatch, prediction_config, tmp_path, capsys
):
    def predict(cmd, **kwargs):
        return headroom.subprocess.CompletedProcess(cmd, 0, json.dumps(consumer_fixture["prediction_report"]), "")

    monkeypatch.setattr(headroom.subprocess, "run", predict)
    assert invoke_headroom(headroom, monkeypatch, prediction_config, tmp_path, "--steps", "1.0") == 0
    row = json.loads(capsys.readouterr().out)["rows"][0]
    expected = consumer_fixture["expected"]
    assert row["status"] == expected["headroom_status"]
    assert row["compliant_share"] == expected["headroom_compliant_share"]
    assert row["goodput_share"] == pytest.approx(expected["headroom_goodput_share"])


@pytest.mark.parametrize(
    ("changes", "passes"),
    [
        ({}, True),
        ({"mean_ttft_ms": 500}, True),
        ({"mean_ttft_ms": 500.01}, False),
        ({"mean_tpot_ms": 50}, True),
        ({"mean_tpot_ms": 50.01}, False),
        ({"mean_e2e_latency_ms": 3000}, True),
        ({"mean_e2e_latency_ms": 3000.01}, False),
        ({"completed_requests": 0}, False),
        ({"completed_requests": None}, False),
        ({"num_ttft_samples": 0}, False),
        ({"num_tpot_samples": None}, False),
        ({"num_e2e_latency_samples": float("nan")}, False),
        ({"mean_ttft_ms": None}, False),
        ({"mean_tpot_ms": float("inf")}, False),
        ({"mean_e2e_latency_ms": float("nan")}, False),
    ],
)
def test_headroom_fallback_against_aisim_oracle(headroom, consumer_fixture, monkeypatch, changes, passes):
    import builtins

    from aisim.sweeper.config import SLATarget
    from aisim.sweeper.score import aggregate_sla_violations

    original_import = builtins.__import__

    def without_score(name, *args, **kwargs):
        if name == "aisim.sweeper.score":
            raise ImportError("exercise standalone skill fallback")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", without_score)
        fallback, checker = headroom.sla_checker()
    assert checker == "built-in fallback (same rules)"
    report = {**consumer_fixture["prediction_report"], **changes}
    sla = {"e2e_ms": 3000} if any("e2e" in key for key in changes) else {"ttft_ms": 500, "itl_ms": 50}
    # Independent expected verdicts include equality, just-over-bound, and unqualified samples.
    assert (not fallback(report, sla)) is passes
    assert (not aggregate_sla_violations(report, SLATarget(**sla))) is passes


@pytest.mark.parametrize(("base_requests", "window"), [(1000, 300), (6000, 300), (12000, 600)])
def test_headroom_preserves_arrival_window_at_double_rate(
    headroom, consumer_fixture, monkeypatch, prediction_config, tmp_path, capsys, base_requests, window
):
    cfg = yaml.safe_load(prediction_config.read_text())
    cfg["traffic"]["stop"]["requests"] = base_requests
    prediction_config.write_text(yaml.safe_dump(cfg))

    def predict(cmd, **kwargs):
        rate = float(next(v.split("=", 1)[1] for v in cmd if v.startswith("traffic.load.requests_per_second=")))
        total = next((int(v.split("=", 1)[1]) for v in cmd if v.startswith("traffic.stop.requests=")), base_requests)
        report = {
            **consumer_fixture["prediction_report"],
            "num_requests": total,
            "completed_requests": total,
            "goodput_completed_requests": total,
            "goodput_request_throughput_rps": rate * 0.98,
        }
        return headroom.subprocess.CompletedProcess(cmd, 0, json.dumps(report), "")

    monkeypatch.setattr(headroom.subprocess, "run", predict)
    assert invoke_headroom(headroom, monkeypatch, prediction_config, tmp_path, "--steps", "1.0,2.0") == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert [row["arrival_window_s"] for row in rows] == [window, window]
    assert [row["num_requests"] for row in rows] == [20 * window, 40 * window]
