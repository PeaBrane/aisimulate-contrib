#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancel one Full CI attempt after a blocking failure, using trusted controller code."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

WORKFLOW = ".github/workflows/ci.yml"
ADVISORY_JOBS = {"Collector Data / Perf data sanity (informational)"}
BLOCKING_CONCLUSIONS = {"failure", "timed_out"}
VALIDATION_GATE = "Full CI Success"
PUBLICATION_JOB_PREFIX = "Stage Application Wheel ("


class WatchError(RuntimeError):
    """Monitoring failed; the original Full CI must continue unchanged."""


@dataclass(frozen=True)
class Target:
    repository: str
    run_id: int
    attempt: int
    sha: str
    branch: str
    workflow_id: int

    @classmethod
    def from_event(cls, event: dict, repository: str) -> Target:
        run = event["workflow_run"]
        target = cls(repository, run["id"], run["run_attempt"], run["head_sha"], run["head_branch"], run["workflow_id"])
        if (
            not re.fullmatch(r"[\w.-]+/[\w.-]+", repository)
            or event.get("action") != "in_progress"
            or event["repository"]["full_name"] != repository
            or not all(type(n) is int and n > 0 for n in (target.run_id, target.attempt, target.workflow_id))
            or not re.fullmatch(r"[0-9a-f]{40}", target.sha)
            or not re.fullmatch(r"main|release/.+|pull-request/[1-9][0-9]*", target.branch)
        ):
            raise WatchError("Expected an in-progress Full CI attempt on a trusted repository branch")
        return target

    def verify(self, run: dict) -> bool:
        """Reject another workflow/revision; return false when this attempt ended."""
        if (
            run.get("id") != self.run_id
            or run.get("workflow_id") != self.workflow_id
            or run.get("path") != WORKFLOW
            or run.get("name") != "Full CI"
            or run.get("event") not in {"push", "workflow_dispatch"}
            or run.get("head_sha") != self.sha
            or run.get("head_branch") != self.branch
            or (run.get("head_repository") or {}).get("full_name") != self.repository
            or (run.get("repository") or {}).get("full_name") != self.repository
            or run.get("status") not in {"queued", "requested", "waiting", "pending", "in_progress", "completed"}
        ):
            raise WatchError("Full CI run identity changed; refusing cancellation")
        return run.get("run_attempt") == self.attempt and run.get("status") != "completed"


def github_api(endpoint: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["gh", "api", "--paginate", "--slurp", endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        pages = json.loads(result.stdout)
        if not isinstance(pages, list) or not pages or not all(isinstance(page, dict) for page in pages):
            raise ValueError("expected GitHub object pages")
        return pages
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise WatchError("Cannot read Full CI metadata; leaving validation running") from error


def cancel_workflow(repository: str, run_id: int) -> None:
    try:
        subprocess.run(
            ["gh", "api", "--method", "POST", f"repos/{repository}/actions/runs/{run_id}/cancel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise WatchError("GitHub did not accept the Full CI cancellation request") from error


def watch(
    target: Target,
    *,
    api: Callable = github_api,
    cancel: Callable = cancel_workflow,
    timeout: float = 7200,
    interval: float = 60,
    clock: Callable = time.monotonic,
    sleep: Callable = time.sleep,
) -> dict:
    if not math.isfinite(timeout) or not math.isfinite(interval) or timeout < 0 or interval <= 0:
        raise WatchError("Invalid monitoring timeout or interval")
    run_endpoint = f"repos/{target.repository}/actions/runs/{target.run_id}"
    jobs_endpoint = f"{run_endpoint}/attempts/{target.attempt}/jobs?per_page=100"

    def active() -> bool:
        pages = api(run_endpoint)
        if len(pages) != 1:
            raise WatchError("Expected exactly one Full CI run")
        return target.verify(pages[0])

    if not active():
        return {"outcome": "finished-or-superseded"}
    deadline = clock() + timeout
    refresh_at = clock() + 300
    while True:
        # Attempt-specific pagination cannot mix a previous failure into a rerun.
        jobs = [job for page in api(jobs_endpoint) for job in page["jobs"]]
        if any(
            (job.get("run_id"), job.get("run_attempt"), job.get("head_sha"))
            != (target.run_id, target.attempt, target.sha)
            for job in jobs
        ):
            raise WatchError("Job evidence does not belong to the watched Full CI attempt")
        # Publication can outlive required validation. Never cancel staging after
        # the aggregate gate has completed, even if publication later fails.
        if any(job["name"] == VALIDATION_GATE and job["status"] == "completed" for job in jobs):
            return {"outcome": "validation-finished"}
        failures = [
            job
            for job in jobs
            if job["status"] == "completed"
            and job.get("conclusion") in BLOCKING_CONCLUSIONS
            and job["name"] not in ADVISORY_JOBS
            # Pages can observe different instants around gate completion.
            # Publication failures are never blocking-validation failures.
            and not job["name"].startswith(PUBLICATION_JOB_PREFIX)
        ]
        if failures:
            failure = min(failures, key=lambda job: (job["completed_at"], job["id"]))
            # Cancellation addresses a run ID, not an attempt. Refresh immediately
            # before writing so an old monitor cannot cancel an observed newer rerun.
            if not active():
                return {"outcome": "finished-or-superseded"}
            result = {
                "outcome": "cancellation-requested",
                "failed_job": failure["name"],
                "conclusion": failure["conclusion"],
                "job_url": f"https://github.com/{target.repository}/actions/runs/{target.run_id}/job/{failure['id']}",
            }
            print(json.dumps(result), flush=True)
            try:
                cancel(target.repository, target.run_id)
            except WatchError:
                # A run may finish between the last read and the cancellation.
                if not active():
                    return {"outcome": "finished-or-superseded"}
                raise
            return result
        if clock() >= refresh_at:
            if not active():
                return {"outcome": "finished-or-superseded"}
            refresh_at = clock() + 300
        if clock() >= deadline:
            raise WatchError("Full CI monitoring timed out; leaving validation running")
        sleep(min(interval, deadline - clock()))


def main() -> None:
    try:
        if os.environ.get("GITHUB_EVENT_NAME") != "workflow_run":
            raise WatchError("This controller only accepts workflow_run events")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        target = Target.from_event(event, os.environ["GITHUB_REPOSITORY"])
        result = watch(target)
    except (WatchError, OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"::error::{error}") from error
    summary = (
        f"### Full CI fail-fast\n\nRun: {target.run_id}, attempt {target.attempt}, commit `{target.sha}`.\n\n"
        f"Outcome: `{result['outcome']}`. This controller does not produce successful validation evidence.\n"
    )
    if url := result.get("job_url"):
        summary += f"\n[First blocking failure]({url}) (`{result['conclusion']}`).\n"
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(summary)


if __name__ == "__main__":
    main()
