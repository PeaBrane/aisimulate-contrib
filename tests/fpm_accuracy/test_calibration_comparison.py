# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest

from scripts.fpm_accuracy.compare_calibration import compare_events, fixed_cohort


def event(time_ms, worker="a", target=0.01, prefill=1):
    return dict(
        time_ms=time_ms,
        worker_id=worker,
        metrics=[
            dict(scheduled_requests=dict(sum_prefill_tokens=prefill), wall_time=target)
        ],
    )


class Fake:
    def __init__(self, method, log, prediction=10):
        self.method, self.log, self.prediction = method, log, prediction
        self.updates = 0

    def estimate_forward_pass_time_ms(self, query):
        assert all("wall_time" not in rank for rank in query)
        self.log.append(("predict", self.method, self.updates))
        return self.prediction + self.updates if self.prediction is not None else None

    def tune_with_fpms(self, iterations):
        self.log.append(("update", self.method, iterations[0][0]["wall_time"]))
        self.updates += 1


def test_all_predictions_precede_feedback_and_gaps_do_not_catch_up():
    log = []
    models = []

    def factory(method):
        model = Fake(method, log)
        models.append(model)
        return model

    result = compare_events([event(t) for t in (0, 5000, 5000, 19000, 20000)], factory)
    assert result["scheduled_events"] == result["feedback_events"] == 3
    assert [m.updates for m in models] == [0, 3, 3]
    assert log[3:8] == [
        ("predict", "raw", 0),
        ("predict", "fixed", 0),
        ("predict", "dynamic", 0),
        ("update", "fixed", 0.01),
        ("update", "dynamic", 0.01),
    ]
    assert log[8:11] == [
        ("predict", "raw", 0),
        ("predict", "fixed", 1),
        ("predict", "dynamic", 1),
    ]


def test_worker_state_is_independent_and_clock_is_shared():
    models = []

    def factory(method):
        model = Fake(method, [])
        models.append(model)
        return model

    result = compare_events(
        [event(0), event(5000), event(6000, "b"), event(10000, "b")], factory
    )
    assert result["workers"] == 2 and result["feedback_events"] == 2
    assert [m.updates for m in models] == [0, 1, 1, 0, 1, 1]


@pytest.mark.parametrize("prediction", [None, 0, -1, float("nan"), float("inf")])
def test_unavailable_or_invalid_predictions_have_no_paired_coverage(prediction):
    result = compare_events(
        [event(0), event(5000)],
        lambda m: Fake(m, [], prediction if m == "dynamic" else 10),
    )
    assert result["available"] == {"raw": 2, "fixed": 2, "dynamic": 0}
    assert result["paired_predictions"] == result["feedback_events"] == 0
    assert result["cohorts"]["all"]["coverage"] == 0
    assert result["cohorts"]["all"]["methods"]["dynamic"]["mape_pct"] is None


def test_targets_never_reach_prediction_and_future_labels_cannot_change_prior_predictions():
    rows = [event(0), event(5000, target=0.02)]
    before = copy.deepcopy(rows)
    log = []
    compare_events(rows, lambda m: Fake(m, log))
    changed = copy.deepcopy(rows)
    changed[1]["metrics"][0]["wall_time"] = 1000
    second = []
    compare_events(changed, lambda m: Fake(m, second))
    assert [v for v in log if v[0] == "predict"] == [
        v for v in second if v[0] == "predict"
    ]
    assert rows == before


@pytest.mark.parametrize("times", [[None], [float("nan")], [float("inf")], [1000, 999]])
def test_missing_nonfinite_and_decreasing_timestamps_are_errors(times):
    with pytest.raises(ValueError, match="time_ms"):
        compare_events([event(t) for t in times], lambda m: Fake(m, []))


def test_invalid_targets_are_reported_and_maximum_positive_rank_is_label():
    row = event(5000, target=0.02)
    row["metrics"].append(
        dict(scheduled_requests=dict(sum_prefill_tokens=1), wall_time=0.04)
    )
    result = compare_events([event(0, target=-1), row], lambda m: Fake(m, []))
    assert result["invalid_targets"] == 1 and result["valid_targets"] == 1
    assert result["cohorts"]["all"]["methods"]["fixed"]["mape_pct"] == 75


@pytest.mark.parametrize(
    "invalid", [None, "0.01", True, -1, 0, float("nan"), float("inf")]
)
def test_mixed_rank_invalid_labels_are_normalized_only_in_feedback_copy(invalid):
    feedback = []

    class RecordingFake(Fake):
        def tune_with_fpms(self, iterations):
            feedback.append(copy.deepcopy(iterations))
            super().tune_with_fpms(iterations)

    row = event(5000, target=invalid)
    row["metrics"].append(
        dict(scheduled_requests=dict(sum_prefill_tokens=2), wall_time=0.02)
    )
    result = compare_events([event(0), row], lambda m: RecordingFake(m, []))
    assert result["feedback_events"] == 1
    assert len(feedback) == 2
    for update in feedback:
        assert [rank["wall_time"] for rank in update[0]] == [0.0, 0.02]
        assert [rank["scheduled_requests"] for rank in update[0]] == [
            rank["scheduled_requests"] for rank in row["metrics"]
        ]
    assert row["metrics"][0]["wall_time"] is invalid
    assert row["metrics"][1]["wall_time"] == 0.02
    assert result["cohorts"]["all"]["methods"]["fixed"]["mape_pct"] == 25


def test_boundary_cohorts_are_inclusive_and_choose_last_critical_rank_tie():
    inside = event(0, prefill=8192)["metrics"]
    outside = event(0, prefill=8193)["metrics"]
    decode = [
        dict(scheduled_requests=dict(num_decode_requests=1, sum_decode_kv_tokens=8192))
    ]
    assert fixed_cohort(inside) == "inside_fixed_grid"
    assert fixed_cohort(outside) == "outside_fixed_grid"
    assert fixed_cohort(outside + decode) == "inside_fixed_grid"
    assert fixed_cohort(decode + outside) == "outside_fixed_grid"
    rows = [event(0, prefill=8192), event(1, target=0.02, prefill=8193)]
    result = compare_events(rows, lambda m: Fake(m, []))
    assert result["cohorts"]["inside_fixed_grid"]["methods"]["fixed"]["mape_pct"] == 0
    assert result["cohorts"]["outside_fixed_grid"]["methods"]["fixed"]["mape_pct"] == 50


def test_rank_worker_must_match_wrapper_and_interval_must_be_positive():
    row = event(0)
    row["metrics"][0]["worker_id"] = "other"
    with pytest.raises(ValueError, match="worker_id"):
        compare_events([row], lambda m: Fake(m, []))
    for interval in (0, -1, math.nan, math.inf):
        with pytest.raises(ValueError, match="interval_ms"):
            compare_events([], lambda m: Fake(m, []), interval)
