# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "dspark_compare", ROOT / "benchmarks/evidence/dspark-e2e-20261014/compare.py"
)
compare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_module)


def fixture_runs():
    cell = {"cell_id": "pro-pd-k6-c1", "expected_requests": 128, "total_gpus": 16, "repetitions": 3}
    run = {
        "cell_id": cell["cell_id"],
        "repetition": 1,
        "run_id": "measure-1",
        "identity_sha256": "a" * 64,
        "status": "success",
        "qualification": {"eligible": True},
        "metrics_contract": compare_module.CONTRACT,
        "expected_requests": 128,
        "total_gpus": 16,
        "metrics": {
            "mean_ttft_ms": 10.0,
            "mean_tpot_ms": 2.0,
            "output_throughput_tok_s": 100.0,
            "completed_requests": 128,
            "failed_requests": 0,
        },
    }
    prediction = copy.deepcopy(run)
    prediction["predictor"] = "aisimulate.engine_replay"
    prediction["run_id"] = "predict-1"
    return cell, run, prediction


def test_missing_points_are_not_zero_error_and_gpu_throughput_is_separate():
    cell, run, prediction = fixture_runs()
    prediction["metrics"]["mean_ttft_ms"] = 12.0
    rows, report = compare_module.compare([cell], [run], [prediction])
    assert report["runs"] == {"planned": 3, "paired": 1, "missing_measurement": 2}
    assert report["requests"]["paired"] == 128
    assert report["aggregates"]["mean_ttft_ms"] == {
        "scored_runs": 1,
        "unscored_runs": 2,
        "mape_percent": 20.0,
        "signed_bias_percent": 20.0,
    }
    assert rows[2]["measured_per_gpu"] == 6.25
    assert rows[3]["ape_percent"] is None
    assert all(item["measured_sample_stddev"] is None for item in report["variability"])


def test_conditional_and_mean_acceptance_errors_remain_separate():
    cell, run, prediction = fixture_runs()
    cell.update(model_key="pro", deployment="pd", num_speculative_tokens=6, acceptance="real")
    other, predicted_other = copy.deepcopy(run), copy.deepcopy(prediction)
    other.update(repetition=2, run_id="measure-2")
    predicted_other.update(repetition=2, run_id="predict-2")
    prediction["acceptance"] = {"sampling_model": "conditional_prefix_rates"}
    predicted_other["acceptance"] = {"sampling_model": "adjacent_integer_mean_only"}
    prediction["metrics"]["mean_ttft_ms"] = 12
    predicted_other["metrics"]["mean_ttft_ms"] = 9
    _, report = compare_module.compare([cell], [run, other], [prediction, predicted_other])
    groups = {row["sampling_model"]: row for row in report["stratified"] if row["metric"] == "mean_ttft_ms"}
    assert groups["conditional_prefix_rates"]["signed_bias_percent"] == 20
    assert groups["adjacent_integer_mean_only"]["signed_bias_percent"] == -10
    assert groups[None]["mape_percent"] is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("identity_sha256", "b" * 64, "identity_mismatch"),
        ("metrics_contract", "streaming_chunk_gaps", "metric_contract_mismatch"),
        ("predictor", "analytical_estimate", "unsupported_predictor"),
        ("qualification", {"eligible": False, "reasons": ["draft transfer unknown"]}, "prediction_unqualified"),
        ("status", "failed", "prediction_failed"),
    ],
)
def test_mismatched_or_unqualified_predictions_are_not_scored(field, value, reason):
    cell, run, prediction = fixture_runs()
    prediction[field] = value
    rows, report = compare_module.compare([cell], [run], [prediction])
    assert rows[0]["status"] == reason
    assert report["aggregates"]["mean_ttft_ms"]["mape_percent"] is None
    assert report["status"] == "pending evidence"


def test_failed_requests_and_zero_latencies_are_visible():
    cell, run, prediction = fixture_runs()
    run["metrics"]["failed_requests"] = 1
    rows, _ = compare_module.compare([cell], [run], [prediction])
    assert rows[0]["status"] == "incomplete_requests"
    run["metrics"]["failed_requests"] = 0
    run["metrics"]["mean_ttft_ms"] = 0
    rows, report = compare_module.compare([cell], [run], [prediction])
    assert rows[0]["status"] == "invalid_metric"
    assert report["aggregates"]["mean_ttft_ms"]["mape_percent"] is None


def test_mape_is_mean_of_point_errors_not_error_of_means():
    cell, run, prediction = fixture_runs()
    second, predicted_second = copy.deepcopy(run), copy.deepcopy(prediction)
    second.update(repetition=2, run_id="measure-2")
    predicted_second.update(repetition=2, run_id="predict-2")
    prediction["metrics"]["mean_ttft_ms"] = 20.0  # +100%
    second["metrics"]["mean_ttft_ms"] = 100.0
    predicted_second["metrics"]["mean_ttft_ms"] = 90.0  # -10%
    _, report = compare_module.compare([cell], [run, second], [prediction, predicted_second])
    assert report["aggregates"]["mean_ttft_ms"]["mape_percent"] == 55
    assert report["aggregates"]["mean_ttft_ms"]["signed_bias_percent"] == 45


def test_duplicate_runs_are_rejected_instead_of_selecting_best_attempt():
    cell, run, prediction = fixture_runs()
    with pytest.raises(ValueError, match="duplicate run key"):
        compare_module.compare([cell], [run, run], [prediction])


def test_no_records_preserves_complete_coverage():
    cell, _, _ = fixture_runs()
    rows, report = compare_module.compare([cell], [], [])
    assert len(rows) == 9
    assert report["requests"] == {"planned": 384}
    assert report["configurations"]["with_qualified_pairs"] == 0
    assert all(metric["mape_percent"] is None for metric in report["aggregates"].values())
