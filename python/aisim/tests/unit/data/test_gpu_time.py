# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aisim_core.data.gpu_time import (
    GpuTime,
    SlurmParseError,
    check_gpu_time,
    gpu_time_from_sacct,
    gpu_time_ok,
    parse_sacct,
)

pytestmark = pytest.mark.unit

HEADER = "JobID|JobName|State|Submit|Start|End|ElapsedRaw|Elapsed|NNodes|AllocTRES"


def _sacct(*rows):
    return "\n".join([HEADER, *rows]) + "\n"


MAIN = (
    "100|bench|COMPLETED|2026-01-01T00:00:00|2026-01-01T01:00:00|2026-01-01T02:00:00"
    "|3600|01:00:00|2|billing=16,gres/gpu:h100=16,node=2"
)


def test_missing_is_not_zero():
    zero = GpuTime(authority="measured", gpu_type="h100", gpu_count=8, wall_seconds=0)
    assert zero.gpu_seconds == 0 and zero.is_recorded
    assert check_gpu_time(zero, "qualification") == []
    for missing in (None, GpuTime.unknown()):
        assert check_gpu_time(missing, "archival")[0].severity == "warn"
        assert check_gpu_time(missing, "qualification")[0].severity == "error"
        assert gpu_time_ok(check_gpu_time(missing, "archival"))
        assert not gpu_time_ok(check_gpu_time(missing, "qualification"))


def test_unknown_cannot_carry_zero():
    with pytest.raises(ValidationError):
        GpuTime(authority="unknown", wall_seconds=0)
    with pytest.raises(ValidationError):
        GpuTime(authority="measured", gpu_count=8)


def test_gpu_seconds_derived_and_checked():
    t = GpuTime(authority="declared", gpu_type="b200", gpu_count=4, wall_seconds=10)
    assert t.gpu_seconds == 40
    with pytest.raises(ValidationError):
        GpuTime(authority="declared", gpu_count=4, wall_seconds=10, gpu_seconds=39)


def test_declared_warns_at_qualification_only():
    t = GpuTime(authority="declared", gpu_type="b200", gpu_count=1, wall_seconds=5)
    assert [f.code for f in check_gpu_time(t, "qualification")] == ["gpu_time.declared"]
    assert check_gpu_time(t, "archival") == []


def test_json_round_trip():
    t = gpu_time_from_sacct(_sacct(MAIN))
    assert GpuTime.model_validate_json(t.model_dump_json(by_alias=True)) == t


def test_slurm_job_record_populates_gpu_time():
    t = gpu_time_from_sacct(_sacct(MAIN, "100.batch|batch|COMPLETED||||3600|01:00:00|2|gres/gpu=16"))
    assert t.authority == "measured"
    assert (t.gpu_type, t.gpu_count, t.wall_seconds) == ("h100", 16, 3600)
    assert t.gpu_seconds == 3600 * 16
    assert t.node_hours == 2.0
    assert t.failed is None and t.retried is None
    assert t.source["job_ids"] == "100"


def test_queue_wait_excluded():
    # Submitted 1 day before start; only elapsed counts.
    row = "7|b|COMPLETED|2025-12-30T00:00:00|2026-01-01T00:00:00|2026-01-01T00:10:00|600|00:10:00|1|gres/gpu=8"
    t = gpu_time_from_sacct(_sacct(row), gpu_type="b200")
    assert t.wall_seconds == 600 and t.gpu_type == "b200"


def test_failed_and_retried_kept_separate():
    rows = [
        "1|b|FAILED||||100|00:01:40|1|gres/gpu=8",
        "2|b|REQUEUED||||50|00:00:50|1|gres/gpu=8",
        "3|b|COMPLETED||||200|00:03:20|1|gres/gpu=8",
    ]
    t = gpu_time_from_sacct(_sacct(*rows))
    assert t.wall_seconds == 200 and t.gpu_seconds == 1600
    assert (t.failed.count, t.failed.gpu_seconds) == (1, 800)
    assert (t.retried.count, t.retried.gpu_seconds) == (1, 400)


def test_elapsed_fallback_and_days():
    text = "JobID|State|Elapsed|NNodes|AllocTRES\n9|COMPLETED|1-02:00:30|1|gres/gpu=1\n"
    assert parse_sacct(text)[0].elapsed_seconds == 26 * 3600 + 30


def test_no_gpu_in_tres_is_zero_but_empty_tres_is_error():
    t = gpu_time_from_sacct(_sacct("5|b|COMPLETED||||60|00:01:00|1|cpu=4,node=1"))
    assert t.gpu_count == 0 and t.gpu_seconds == 0
    with pytest.raises(SlurmParseError, match="AllocTRES"):
        parse_sacct(_sacct("5|b|COMPLETED||||60|00:01:00|1|"))


@pytest.mark.parametrize("state", ["RUNNING", "PENDING"])
def test_unfinished_job_rejected(state):
    with pytest.raises(SlurmParseError, match=state):
        parse_sacct(_sacct(f"5|b|{state}||||60|00:01:00|1|gres/gpu=1"))


def test_only_failed_jobs_rejected():
    with pytest.raises(SlurmParseError, match="COMPLETED"):
        gpu_time_from_sacct(_sacct("5|b|FAILED||||60|00:01:00|1|gres/gpu=1"))


def test_headerless_needs_fields():
    row = "5|COMPLETED|60|1|gres/gpu=2"
    with pytest.raises(SlurmParseError):
        parse_sacct(row)
    jobs = parse_sacct(row, fields=["JobID", "State", "ElapsedRaw", "NNodes", "AllocTRES"])
    assert jobs[0].gpu_count == 2


def test_tres_untyped_and_typed_not_double_counted():
    row = "5|b|COMPLETED||||60|00:01:00|1|gres/gpu=8,gres/gpu:b200=8,node=1"
    t = gpu_time_from_sacct(_sacct(row))
    assert (t.gpu_count, t.gpu_type) == (8, "b200")


# ------------------------------------------------------------------ Kubernetes


def _pod(name, phase, start, end, gpus="4"):
    status = {"phase": phase, "startTime": "2026-11-07T00:00:00Z"}
    if start:
        status["containerStatuses"] = [{"state": {"terminated": {"startedAt": start, "finishedAt": end}}}]
    return {
        "metadata": {"name": name},
        "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": gpus}}}]},
        "status": status,
    }


class _FakeKube:
    def __init__(self, job, pods):
        self.job, self.pods = job, pods

    def get_job(self, namespace, name):
        return self.job

    def list_job_pods(self, namespace, job_name):
        return self.pods


_JOB = {
    "metadata": {"name": "bench", "namespace": "ns"},
    "spec": {"template": {"spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]}}},
    "status": {"succeeded": 1, "startTime": "2026-11-07T00:10:00Z", "completionTime": "2026-11-07T00:20:00Z"},
}


def test_kubernetes_capture_excludes_queue_wait_and_splits_failed():
    from aisim_core.data.gpu_time import capture_gpu_time_kubernetes

    pods = [
        _pod("p0", "Failed", "2026-11-07T00:01:00Z", "2026-11-07T00:02:00Z"),
        _pod("p1", "Succeeded", "2026-11-07T00:10:00Z", "2026-11-07T00:20:00Z"),
    ]
    gt = capture_gpu_time_kubernetes(_FakeKube(_JOB, pods), "ns", "bench", gpu_type="h100")
    assert gt.authority == "measured" and gt.gpu_count == 4 and gt.gpu_type == "h100"
    assert gt.wall_seconds == 600 and gt.gpu_seconds == 2400
    assert gt.failed.count == 1 and gt.failed.gpu_seconds == 240
    assert gt.source["source"] == "kubernetes"


def test_kubernetes_takes_gpus_from_the_job_template_when_the_pod_spec_has_none():
    from aisim_core.data.gpu_time import gpu_time_from_kubernetes

    pod = {
        "metadata": {"name": "p"},
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {"state": {"terminated": {"startedAt": "2026-11-07T00:10:00Z", "finishedAt": "2026-11-07T00:20:00Z"}}}
            ],
        },
    }
    gt = gpu_time_from_kubernetes(_JOB, [pod])
    assert gt.wall_seconds == 600 and gt.gpu_count == 4


def test_kubernetes_does_not_report_job_wall_time_as_measured_gpu_time():
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    # Job startTime..completionTime includes queue wait; without container timestamps nothing is measured.
    pod = {"metadata": {"name": "p"}, "status": {"phase": "Succeeded"}}
    with pytest.raises(KubeParseError, match="no terminated container times"):
        gpu_time_from_kubernetes(_JOB, [pod])


@pytest.mark.parametrize(
    "job,pods",
    [
        ({**_JOB, "status": {}}, [_pod("p", "Succeeded", "2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z")]),
        (_JOB, [_pod("p", "Running", None, None)]),
        (_JOB, [_pod("p", "Failed", "2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z")]),
        (_JOB, [_pod("p", "Succeeded", "2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z", gpus="0")]),
    ],
)
def test_kubernetes_rejects_unfinished_or_gpu_less(job, pods):
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    if pods[0]["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "0":
        job = {**job, "spec": {}}
    with pytest.raises(KubeParseError):
        gpu_time_from_kubernetes(job, pods)


# ------------------------------------------------------------------ review regressions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wall_seconds": float("inf"), "gpu_count": 1, "gpu_seconds": 0.0},
        {"wall_seconds": float("nan"), "gpu_count": 1},
        {"wall_seconds": 1.0, "gpu_count": 1, "gpu_seconds": float("inf")},
        {"wall_seconds": 1.0, "gpu_count": 1, "node_hours": float("inf")},
        {"wall_seconds": 1e308, "gpu_count": 8},  # finite operands whose product overflows
    ],
)
def test_non_finite_gpu_time_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GpuTime(authority="measured", **kwargs)
    assert GpuTime(authority="measured", wall_seconds=0.0, gpu_count=0).gpu_seconds == 0.0  # finite zero stays valid


def test_non_finite_attempts_and_phases_are_rejected():
    from aisim_core.data.gpu_time import Attempts, Phases

    with pytest.raises(ValueError):
        Attempts(count=1, wall_seconds=float("inf"), gpu_seconds=1.0)
    with pytest.raises(ValueError):
        Phases(warmup=float("nan"))


def test_malformed_gpu_allocation_is_not_a_zero_gpu_job():
    from aisim_core.data.gpu_time import SlurmParseError, _gpus_from_tres

    with pytest.raises(SlurmParseError, match="bad GPU allocation"):
        _gpus_from_tres("cpu=8,gres/gpu=oops")
    with pytest.raises(SlurmParseError, match="bad GPU allocation"):
        _gpus_from_tres("gres/gpu:h100=")
    assert _gpus_from_tres("cpu=8,mem=4G") == (0, None)  # a CPU-only record keeps its intentional zero
    assert _gpus_from_tres("cpu=8,gres/gpumem=80G,gres/gpu=2")[0] == 2  # other gres/gpu* resources are not GPUs


def _failed_pod(container_status):
    return {
        "metadata": {"name": "bad"},
        "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "4"}}}]},
        "status": {"phase": "Failed", "containerStatuses": container_status},
    }


@pytest.mark.parametrize(
    "state",
    [
        {"terminated": {"startedAt": "2026-11-07T00:01:00Z"}},  # started, never finished
        {"terminated": {"startedAt": "2026-11-07T00:01:00Z", "finishedAt": "garbage"}},
        {"running": {"startedAt": "2026-11-07T00:01:00Z"}},
    ],
)
def test_incomplete_failed_pod_execution_records_are_rejected(state):
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    good = _pod("ok", "Succeeded", "2026-11-07T00:10:00Z", "2026-11-07T00:20:00Z")
    with pytest.raises(KubeParseError):
        gpu_time_from_kubernetes(_JOB, [_failed_pod([{"state": state}]), good])


def test_a_failed_pod_that_never_started_costs_nothing():
    from aisim_core.data.gpu_time import gpu_time_from_kubernetes

    good = _pod("ok", "Succeeded", "2026-11-07T00:10:00Z", "2026-11-07T00:20:00Z")
    never = _failed_pod([{"state": {"waiting": {"reason": "ImagePullBackOff"}}}])
    assert gpu_time_from_kubernetes(_JOB, [never, good]).failed is None
    full = _pod("full", "Failed", "2026-11-07T00:01:00Z", "2026-11-07T00:02:00Z")
    assert gpu_time_from_kubernetes(_JOB, [full, good]).failed.gpu_seconds == 240


def _container(started, finished, restart_count=0, last=None):
    status = {"state": {"terminated": {"startedAt": started, "finishedAt": finished}}, "restartCount": restart_count}
    if last is not None:
        status["lastState"] = {"terminated": {"startedAt": last[0], "finishedAt": last[1]}}
    return status


def _pod_with(statuses, init=None, name="p"):
    pod = {
        "metadata": {"name": name},
        "spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}]},
        "status": {"phase": "Succeeded", "containerStatuses": statuses},
    }
    if init is not None:
        pod["status"]["initContainerStatuses"] = init
    return pod


def test_a_container_restarted_in_place_reports_its_earlier_run_as_failed_cost():
    from aisim_core.data.gpu_time import gpu_time_from_kubernetes

    job = {**_JOB, "spec": {}}  # the pod spec carries the one GPU
    # Failed after 60 s, restarted, then succeeded after 30 s.
    pod = _pod_with(
        [
            _container(
                "2026-11-07T00:02:00Z",
                "2026-11-07T00:02:30Z",
                restart_count=1,
                last=("2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z"),
            )
        ]
    )
    gt = gpu_time_from_kubernetes(job, [pod])
    assert gt.wall_seconds == 30 and gt.gpu_seconds == 30
    assert gt.failed.count == 1 and gt.failed.wall_seconds == 60 and gt.failed.gpu_seconds == 60


def test_a_container_restarted_more_than_once_has_an_incomplete_record():
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    pod = _pod_with(
        [
            _container(
                "2026-11-07T00:05:00Z",
                "2026-11-07T00:05:30Z",
                restart_count=2,
                last=("2026-11-07T00:02:00Z", "2026-11-07T00:03:00Z"),
            )
        ]
    )
    with pytest.raises(KubeParseError, match="restarted 2 times"):
        gpu_time_from_kubernetes({**_JOB, "spec": {}}, [pod])


def test_init_containers_count_toward_the_gpu_window_and_must_have_finished():
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    job = {**_JOB, "spec": {}}
    init = [_container("2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z")]  # 60 s before the app container
    app = [_container("2026-11-07T00:01:00Z", "2026-11-07T00:01:30Z")]
    assert gpu_time_from_kubernetes(job, [_pod_with(app, init=init)]).wall_seconds == 90
    unfinished = [{"state": {"terminated": {"startedAt": "2026-11-07T00:00:00Z"}}}]
    with pytest.raises(KubeParseError, match="no finish time"):
        gpu_time_from_kubernetes(job, [_pod_with(app, init=unfinished)])


def test_an_earlier_run_overlapping_the_final_window_is_not_charged_twice():
    from aisim_core.data.gpu_time import KubeParseError, gpu_time_from_kubernetes

    job = {**_JOB, "spec": {}}
    # The main container runs 0-60 s, restarts and runs 60-90 s; a helper container runs the whole 0-90 s.
    main = _container(
        "2026-11-07T00:01:00Z",
        "2026-11-07T00:01:30Z",
        restart_count=1,
        last=("2026-11-07T00:00:00Z", "2026-11-07T00:01:00Z"),
    )
    helper = _container("2026-11-07T00:00:00Z", "2026-11-07T00:01:30Z")
    with pytest.raises(KubeParseError, match="cannot be attributed"):
        gpu_time_from_kubernetes(job, [_pod_with([main, helper])])
    # Disjoint runs keep the existing split (earlier run failed cost, final run measured).
    gt = gpu_time_from_kubernetes(job, [_pod_with([main])])
    assert (gt.wall_seconds, gt.failed.wall_seconds) == (30, 60)
