# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer contracts over synthetic and recorded native artifacts; no runtime build."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from aisim.sweeper.config import OptimizationTarget
from aisim.sweeper.score import make_candidate

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/evidence/skills-catalog"
SKILLS = ("aisim-deployment-search", "aisim-gpu-selection-and-sizing")


@pytest.fixture(params=SKILLS)
def summarizer(request):
    path = ROOT / "skills" / request.param / "scripts/summarize_recommendation.py"
    spec = importlib.util.spec_from_file_location("summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact():
    return json.loads((EVIDENCE / "consumer-fixture.json").read_text())


def invoke(module, monkeypatch, tmp_path, data, *args):
    if data is not None:
        (tmp_path / "recommendation.json").write_text(json.dumps(data))
    monkeypatch.setattr(sys, "argv", ["summarize_recommendation.py", str(tmp_path), *args])
    return module.main()


def test_copies_and_release_metric_contract():
    sources = [(ROOT / "skills" / skill / "scripts/summarize_recommendation.py").read_bytes() for skill in SKILLS]
    assert sources[0] == sources[1]
    contract = json.loads((EVIDENCE / "producer-contract.json").read_text())
    # Verify every literal metric read against the inspected, immutable 0.14.0 producer.
    reads = {
        call.args[0].value
        for call in ast.walk(ast.parse(sources[0]))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "m"
        and call.func.attr == "get"
    }
    assert reads <= set(contract["metric_keys"])


@pytest.mark.parametrize("json_mode", [False, True])
def test_summary_consumes_producer_metrics(summarizer, artifact, monkeypatch, tmp_path, capsys, json_mode):
    ledger = artifact["recommendation"]
    # Exercise the actual checked-out producer, including its metric whitelist.
    produced = make_candidate(
        ledger["candidates"][0]["config"], artifact["prediction_report"], OptimizationTarget.THROUGHPUT_PER_GPU
    )
    ledger["candidates"][0].update(produced.model_dump())
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, *(["--json"] if json_mode else [])) == 0
    out = capsys.readouterr().out
    if json_mode:
        result = json.loads(out)
        row = result["selected"][0]
        for field in ("completed_requests", "sla_compliant_requests", "output_tok_s_per_gpu"):
            assert row[field] == artifact["expected"][field]
        assert result["summary"]["selected_total"] == 1
        assert result["summary"]["not_feasible_total"] == 1
        assert not {"num_requests", "p90_tpot_ms", "p90_ttft_ms", "mean_itl_ms"} & row.keys()
    else:
        assert "completed=6000 sla_ok=5940" in out
        assert "(1000.0/gpu)" in out
        assert "completed=6000/" not in out
        assert "candidate-000002 infeasible sla_constraint" in out


@pytest.mark.parametrize("json_mode", [False, True])
def test_valid_empty_run(summarizer, monkeypatch, tmp_path, capsys, json_mode):
    assert (
        invoke(summarizer, monkeypatch, tmp_path, {"candidates": [], "views": {}}, *(["--json"] if json_mode else []))
        == 0
    )
    if json_mode:
        assert json.loads(capsys.readouterr().out)["selected"] == []


def test_missing_file(summarizer, monkeypatch, tmp_path, capsys):
    assert invoke(summarizer, monkeypatch, tmp_path, None) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [{}, [], {"candidates": [], "views": []}, {"candidates": [{"candidate_id": "candidate-000001"}], "views": {}}],
)
def test_malformed_envelope(summarizer, monkeypatch, tmp_path, capsys, data):
    assert invoke(summarizer, monkeypatch, tmp_path, data, "--json") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", None),
        ("status", "unknown"),
        ("status", []),
        ("candidate_id", []),
        ("config", []),
        ("metrics", []),
        ("metrics", {"mean_ttft_ms": "fast"}),
        ("metrics", {"mean_ttft_ms": float("nan")}),
        ("score", "high"),
        ("score", None),
        ("used_gpus", True),
        ("used_gpus", None),
        ("used_gpus", -1),
        ("prediction_config", []),
        ("reason", "unexpected rejection"),
        ("config", {"deployment_mode": []}),
    ],
)
def test_invalid_candidate(summarizer, artifact, monkeypatch, tmp_path, capsys, field, value):
    ledger = artifact["recommendation"]
    ledger["candidates"][0][field] = value
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json", "--best-per-mode") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


@pytest.mark.parametrize(
    "view", [["missing"], ["candidate-000002"], ["candidate-000001", "candidate-000001"], [1], None, "candidate-000001"]
)
def test_invalid_view(summarizer, artifact, monkeypatch, tmp_path, capsys, view):
    ledger = artifact["recommendation"]
    ledger["views"]["top_n"] = view
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


def test_duplicate_ids(summarizer, artifact, monkeypatch, tmp_path):
    ledger = artifact["recommendation"]
    ledger["candidates"].append(copy.deepcopy(ledger["candidates"][0]))
    assert invoke(summarizer, monkeypatch, tmp_path, ledger) == 2


@pytest.mark.parametrize("candidate", ["missing", "candidate-000002"])
def test_ineligible_export(summarizer, artifact, monkeypatch, tmp_path, candidate):
    dest = tmp_path / "export.yaml"
    assert (
        invoke(summarizer, monkeypatch, tmp_path, artifact["recommendation"], "--export", candidate, "--to", str(dest))
        == 2
    )
    assert not dest.exists()


def test_export_requires_destination(summarizer, artifact, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        invoke(summarizer, monkeypatch, tmp_path, artifact["recommendation"], "--export", "candidate-000001")
    assert exc.value.code == 2


def test_export_and_best_per_mode(summarizer, artifact, monkeypatch, tmp_path, capsys):
    ledger = artifact["recommendation"]
    ledger["views"]["top_n"] = []
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json", "--best-per-mode") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["best_per_mode"]["agg"]["candidate_id"] == "candidate-000001"
    assert result["best_per_mode"]["agg"]["config_file"] is None
    dest = tmp_path / "export.yaml"
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--export", "candidate-000001", "--to", str(dest)) == 0
    assert yaml.safe_load(dest.read_text()) == ledger["candidates"][0]["prediction_config"]


@pytest.mark.parametrize("json_mode", [False, True])
def test_summary_reads_recorded_native_0140(summarizer, monkeypatch, tmp_path, capsys, json_mode):
    # Replay producer bytes from the pinned native wheel, without running a model in CI.
    native = EVIDENCE / "native-0.14.0"
    ledger = json.loads((native / "recommendation/recommendation.json").read_text())
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, *(["--json"] if json_mode else [])) == 0
    output = capsys.readouterr().out
    selected_ids = ledger["views"]["top_n"][:5]
    if json_mode:
        summary = json.loads(output)
        assert [row["candidate_id"] for row in summary["selected"]] == selected_ids
        by_id = {candidate["candidate_id"]: candidate for candidate in ledger["candidates"]}
        for row in summary["selected"]:
            metrics = by_id[row["candidate_id"]]["metrics"]
            assert row["completed_requests"] == metrics["completed_requests"]
            assert row["mean_ttft_ms"] == metrics["mean_ttft_ms"]
            assert row["output_tok_s_per_gpu"] == pytest.approx(
                metrics["output_throughput_tok_s"] / by_id[row["candidate_id"]]["used_gpus"]
            )
    else:
        assert output.count("completed=100") == len(selected_ids)
        assert "ttft=" in output
        assert "ttft=-" not in output
        assert "sla_ok=-" in output


def test_ladder_classifies_recorded_native_0140():
    spec = importlib.util.spec_from_file_location(
        "native_ladder", ROOT / "skills/aisim-gpu-selection-and-sizing/scripts/size_ladder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status, detail, result = module.classify(0, "", EVIDENCE / "native-0.14.0/recommendation")
    assert status == "qualified"
    assert not detail
    assert result["views"]["top_n"]
    assert result["counts"]["evaluated"] == len(result["candidates"])


def test_headroom_parses_recorded_native_0140(monkeypatch, tmp_path, capsys):
    import subprocess

    native = EVIDENCE / "native-0.14.0"
    spec = importlib.util.spec_from_file_location(
        "native_headroom", ROOT / "skills/aisim-gpu-selection-and-sizing/scripts/headroom_scan.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((native / "headroom-input.yaml").read_text())
    # Input schema validation and native execution are recorded separately. This
    # replay test exercises the consumer of the actual stdout metric envelope.
    monkeypatch.setattr(module, "load_config", lambda path: (config, "recorded-native-input"))

    def replay(cmd, **kwargs):
        out = Path(cmd[cmd.index("--output-dir") + 1])
        metrics = (native / "headroom" / out.name / "prediction.json").read_text()
        return subprocess.CompletedProcess(cmd, 0, stdout=metrics, stderr="")

    monkeypatch.setattr(module.subprocess, "run", replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headroom_scan.py",
            str(native / "headroom-input.yaml"),
            "--min-goodput-share",
            "0.9",
            "--min-compliant-share",
            "0.95",
            "--steps",
            "1.0,2.0",
            "--workdir",
            str(tmp_path / "scan"),
        ],
    )
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    captured = json.loads((native / "headroom.json").read_text())
    for row, expected in zip(result["rows"], captured["rows"], strict=True):
        assert row["status"] == expected["status"]
        for key in ("num_requests", "sla_compliant_requests", "goodput_rps", "arrival_window_s", "mean_ttft_ms"):
            assert row[key] == pytest.approx(expected[key])
