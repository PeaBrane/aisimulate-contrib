# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise cancellation decisions without canceling real GitHub runs."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.ci import watch_full_ci as controller

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ai-dynamo/aisim"
SHA = "a" * 40


def run_record(**overrides):
    return {
        "id": 42,
        "run_attempt": 2,
        "workflow_id": 123,
        "path": controller.WORKFLOW,
        "name": "Full CI",
        "event": "push",
        "head_sha": SHA,
        "head_branch": "pull-request/100",
        "head_repository": {"full_name": REPOSITORY},
        "repository": {"full_name": REPOSITORY},
        "status": "in_progress",
        **overrides,
    }


def event(**overrides):
    return {
        "action": "in_progress",
        "repository": {"full_name": REPOSITORY},
        "workflow_run": run_record(**overrides),
    }


def job(name="Application Tests (amd64, unit, 1)", conclusion="failure", **overrides):
    return {
        "id": 7,
        "name": name,
        "run_id": 42,
        "run_attempt": 2,
        "head_sha": SHA,
        "status": "completed",
        "conclusion": conclusion,
        "completed_at": "2026-11-16T18:00:00Z",
        **overrides,
    }


class Server:
    def __init__(self, job_polls, runs=None):
        self.job_polls = job_polls
        self.runs = runs or [run_record()]
        self.calls = []
        self.cancellations = []
        self.now = 0

    def api(self, endpoint):
        self.calls.append(endpoint)
        if "/jobs?" in endpoint:
            assert endpoint.endswith("/attempts/2/jobs?per_page=100")
            result = self.job_polls.pop(0) if len(self.job_polls) > 1 else self.job_polls[0]
            return copy.deepcopy(result)
        assert endpoint == f"repos/{REPOSITORY}/actions/runs/42"
        return [copy.deepcopy(self.runs.pop(0) if len(self.runs) > 1 else self.runs[0])]

    def cancel(self, repository, run_id):
        self.cancellations.append((repository, run_id))

    def sleep(self, seconds):
        self.now += seconds

    def watch(self, **kwargs):
        return controller.watch(
            controller.Target.from_event(event(), REPOSITORY),
            api=self.api,
            cancel=self.cancel,
            clock=lambda: self.now,
            sleep=self.sleep,
            **kwargs,
        )


@pytest.mark.parametrize("conclusion", ["failure", "timed_out"])
def test_cancel_only_the_watched_run_after_blocking_job_finishes(conclusion):
    server = Server([[{"jobs": [job(conclusion=conclusion)]}]])
    result = server.watch()
    assert result["outcome"] == "cancellation-requested"
    assert result["failed_job"] == "Application Tests (amd64, unit, 1)"
    assert result["job_url"].endswith("/actions/runs/42/job/7")
    assert server.cancellations == [(REPOSITORY, 42)]
    assert server.calls[-1] == f"repos/{REPOSITORY}/actions/runs/42"


def test_failure_after_first_job_page_cancels_without_waiting_for_other_jobs():
    pages = [{"jobs": [job(conclusion="success", id=i) for i in range(100)]}, {"jobs": [job(id=101)]}]
    server = Server([pages])
    assert server.watch()["outcome"] == "cancellation-requested"
    assert server.cancellations == [(REPOSITORY, 42)]


def test_report_first_failure_and_ignore_advisory_failure():
    server = Server(
        [
            [
                {
                    "jobs": [
                        job(next(iter(controller.ADVISORY_JOBS)), id=1, completed_at="2026-11-16T17:00:00Z"),
                        job("Rust (amd64)", id=2, completed_at="2026-11-16T19:00:00Z"),
                        job("Engine Golden Regression", id=3),
                    ]
                }
            ]
        ]
    )
    assert server.watch()["failed_job"] == "Engine Golden Regression"


@pytest.mark.parametrize("conclusion", ["success", "skipped", "cancelled", None])
def test_non_failures_and_informational_failure_do_not_cancel(conclusion):
    server = Server([[{"jobs": [job(conclusion=conclusion), job(next(iter(controller.ADVISORY_JOBS)), id=9)]}]])
    with pytest.raises(controller.WatchError, match="timed out"):
        server.watch(timeout=0)
    assert not server.cancellations


def test_failure_of_a_new_required_job_is_not_silently_ignored():
    server = Server([[{"jobs": [job("New required validation")]}]])
    assert server.watch()["outcome"] == "cancellation-requested"


def test_running_test_failure_is_not_mistaken_for_completed_job_failure():
    server = Server([[{"jobs": [job(status="in_progress")]}]])
    with pytest.raises(controller.WatchError, match="timed out"):
        server.watch(timeout=0)
    assert not server.cancellations


def test_monitor_waits_through_empty_queue_then_cancels_failure():
    server = Server([[{"jobs": []}], [{"jobs": [job()]}]])
    assert server.watch()["outcome"] == "cancellation-requested"
    assert server.now == 60


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled"])
def test_finished_validation_never_cancels_publication(conclusion):
    server = Server(
        [[{"jobs": [job(controller.VALIDATION_GATE, conclusion, id=9), job("Stage Application Wheel (amd64)")]}]]
    )
    assert server.watch()["outcome"] == "validation-finished"
    assert not server.cancellations


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_publication_failure_never_cancels_when_gate_page_is_stale(arch):
    server = Server(
        [
            [
                {"jobs": [job(controller.VALIDATION_GATE, None, status="in_progress", id=9)]},
                {"jobs": [job(f"Stage Application Wheel ({arch})")]},
            ]
        ]
    )
    with pytest.raises(controller.WatchError, match="timed out"):
        server.watch(timeout=0)
    assert not server.cancellations


@pytest.mark.parametrize("changes", [{"run_attempt": 3}, {"status": "completed"}])
@pytest.mark.parametrize("when", ["startup", "before-cancel", "periodic"])
def test_old_or_finished_attempt_is_never_canceled(changes, when):
    runs = [run_record(**changes)] if when == "startup" else [run_record(), run_record(**changes)]
    server = Server([[{"jobs": [job()] if when != "periodic" else []}]], runs=runs)
    assert server.watch()["outcome"] == "finished-or-superseded"
    assert not server.cancellations
    if when == "periodic":
        assert server.now == 300


@pytest.mark.parametrize(
    "changes",
    [
        {"id": 43},
        {"workflow_id": 124},
        {"path": ".github/workflows/nightly-ci.yml"},
        {"name": "Other workflow"},
        {"event": "pull_request"},
        {"head_sha": "b" * 40},
        {"head_branch": "main"},
        {"head_repository": {"full_name": "someone/fork"}},
        {"repository": {"full_name": "someone/other"}},
        {"status": None},
    ],
)
def test_changed_run_identity_never_cancels(changes):
    server = Server([[{"jobs": [job()]}]], runs=[run_record(**changes)])
    with pytest.raises(controller.WatchError, match="identity changed"):
        server.watch()
    assert not server.cancellations


@pytest.mark.parametrize("changes", [{"head_sha": "b" * 40}, {"run_id": 43}, {"run_attempt": 1}])
def test_mismatched_job_evidence_never_cancels(changes):
    server = Server([[{"jobs": [job(**changes)]}]])
    with pytest.raises(controller.WatchError, match="Job evidence"):
        server.watch()
    assert not server.cancellations


@pytest.mark.parametrize("branch", ["main", "release/0.8.0", "pull-request/100"])
@pytest.mark.parametrize("kind", ["push", "workflow_dispatch"])
def test_supported_full_ci_sources(branch, kind):
    target = controller.Target.from_event(event(head_branch=branch, event=kind), REPOSITORY)
    assert target.verify(run_record(head_branch=branch, event=kind))


@pytest.mark.parametrize(
    "changes", [{"id": True}, {"run_attempt": 0}, {"head_sha": "short"}, {"head_branch": "feature/untrusted"}]
)
def test_invalid_event_target_is_rejected(changes):
    with pytest.raises(controller.WatchError):
        controller.Target.from_event(event(**changes), REPOSITORY)


def test_api_error_leaves_original_validation_running():
    server = Server([[{"jobs": []}]])

    def broken_api(_):
        raise controller.WatchError("unavailable")

    with pytest.raises(controller.WatchError, match="unavailable"):
        controller.watch(controller.Target.from_event(event(), REPOSITORY), api=broken_api, cancel=server.cancel)
    assert not server.cancellations


def test_cancellation_racing_with_completion_is_harmless():
    server = Server([[{"jobs": [job()]}]], runs=[run_record(), run_record(), run_record(status="completed")])

    def reject_cancel(*_):
        raise controller.WatchError("conflict")

    result = controller.watch(controller.Target.from_event(event(), REPOSITORY), api=server.api, cancel=reject_cancel)
    assert result["outcome"] == "finished-or-superseded"


def test_api_adapter_paginates_and_cancel_adapter_only_posts_cancel():
    with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '[{"jobs": []}]')) as call:
        assert controller.github_api("fixture") == [{"jobs": []}]
        assert call.call_args.args[0] == ["gh", "api", "--paginate", "--slurp", "fixture"]
    with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "")) as call:
        controller.cancel_workflow(REPOSITORY, 42)
        assert call.call_args.args[0] == ["gh", "api", "--method", "POST", f"repos/{REPOSITORY}/actions/runs/42/cancel"]


def load_workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_controller_executes_only_default_branch_code_with_scoped_permissions():
    workflow = load_workflow("full-ci-fail-fast.yml")
    assert workflow["on"] == {"workflow_run": {"workflows": ["Full CI"], "types": ["in_progress"]}}
    assert workflow["permissions"] == {}
    job_config = workflow["jobs"]["cancel-on-failure"]
    assert job_config["permissions"] == {"contents": "read", "actions": "write"}
    assert job_config["runs-on"] == "ubuntu-24.04"
    checkout, monitor = job_config["steps"]
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["sparse-checkout"] == "scripts/ci"
    assert monitor["run"] == "python3 scripts/ci/watch_full_ci.py"
    assert set(monitor["env"]) == {"GH_TOKEN"}
    assert "run_attempt" in workflow["concurrency"]["group"]
    assert "actions" not in load_workflow("ci.yml")["permissions"]


def test_advisory_exceptions_match_the_actual_full_ci_workflow_tree():
    advisory = set()

    def visit(name, prefix=""):
        for config in load_workflow(name)["jobs"].values():
            label = prefix + config.get("name", "")
            if config.get("continue-on-error", "false") != "false":
                assert config["continue-on-error"] == "true"
                advisory.add(label)
            if config.get("uses", "").startswith("./.github/workflows/"):
                visit(config["uses"].rsplit("/", 1)[1], label + " / ")

    visit("ci.yml")
    assert advisory == controller.ADVISORY_JOBS
    jobs = load_workflow("ci.yml")["jobs"]
    publication = jobs["stage-application-wheel"]
    assert publication["name"].startswith(controller.PUBLICATION_JOB_PREFIX)
    assert "readiness" in publication["needs"]
    assert "stage-application-wheel" not in jobs["readiness"]["needs"]


def test_main_writes_cancellation_summary_without_executing_watched_code(tmp_path, monkeypatch):
    payload = tmp_path / "event.json"
    payload.write_text(json.dumps(event()))
    summary = tmp_path / "summary.md"
    for key, value in {
        "GITHUB_EVENT_NAME": "workflow_run",
        "GITHUB_EVENT_PATH": str(payload),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(key, value)
    result = {"outcome": "cancellation-requested", "job_url": "https://github.com/fixture", "conclusion": "failure"}
    with patch.object(controller, "watch", return_value=result):
        controller.main()
    assert "attempt 2" in summary.read_text()
    assert "does not produce successful validation evidence" in summary.read_text()


def test_fast_ci_runs_controller_regression_tests():
    policy = load_workflow("fast-ci.yml")["jobs"]["policy"]
    commands = "\n".join(step.get("run", "") for step in policy["steps"])
    assert "tests/test_full_ci_fail_fast.py" in commands
