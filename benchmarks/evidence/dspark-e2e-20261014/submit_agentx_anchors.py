# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit frozen native context anchors serially after an explicit platform job."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def submit(directory, platform, storage_root, after_job):
    if not after_job.isdecimal() or storage_root.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError("explicit numeric predecessor and personal benchmark storage outside home required")
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text())
    identity = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan.get("schema") != "dspark-agentx-native-anchor-jobs-v1" or plan.get("completed_run_exclusions") != []:
        raise ValueError("expected independent native anchor manifest without exclusions")
    state_path = directory / f"submitted-{platform}.json"
    with (directory / f"submit-{platform}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "schema": "dspark-agentx-anchor-submissions-v1",
                "platform": platform,
                "plan_sha256": identity,
                "initial_predecessor": after_job,
                "jobs": [],
            }
        )
        if state.get("pending_submission"):
            raise ValueError("uncertain previous submission: reconcile scheduler job name before retry")
        if (state["platform"], state["plan_sha256"], state["initial_predecessor"]) != (platform, identity, after_job):
            raise ValueError("resume identity differs from frozen plan/platform/predecessor")
        expected = [job for job in plan["jobs"] if job["platform"] == platform]
        if [job["key"] for job in state["jobs"]] != [job["key"] for job in expected[: len(state["jobs"])]]:
            raise ValueError("submitted jobs are not a prefix of the frozen platform plan")
        previous = state["jobs"][-1]["job_id"] if state["jobs"] else after_job
        bundle = storage_root / "jobs" / plan["campaign"] / "anchor-bundle"
        for name, digest in plan["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("staged runtime differs from the frozen anchor bundle")
        logs = storage_root / "logs" / plan["campaign"]
        logs.mkdir(parents=True, exist_ok=True)
        for job in expected[len(state["jobs"]) :]:
            script = directory / job["script"]
            if hashlib.sha256(script.read_bytes()).hexdigest() != job["script_sha256"]:
                raise ValueError("anchor script differs from frozen manifest")
            for query in (
                ["sacctmgr", "-nP", "show", "assoc", "where", f"user={os.environ['USER']}", "format=account,QOS"],
                ["sinfo", "-h", "-p", "batch", "-o", "%P|%G|%D|%t|%l"],
            ):
                subprocess.run(query, check=True, capture_output=True, text=True)
            queue = subprocess.run(
                ["squeue", "--me", "-h", "-o", "%i|%j"], check=True, capture_output=True, text=True
            ).stdout
            name = "dsa-" + job["key"]
            if any(line.partition("|")[2] == name for line in queue.splitlines()):
                raise ValueError("unrecorded duplicate anchor job: reconcile scheduler before retry")
            command = [
                "sbatch",
                "--parsable",
                "--account=coreai_comparch_inferencex",
                "--partition=batch",
                "--qos=normal",
                f"--nodes={job['nodes']}",
                f"--ntasks={job['nodes']}",
                "--ntasks-per-node=1",
                f"--gres=gpu:{job['gpus_per_node']}",
                f"--cpus-per-task={job['cpus_per_node']}",
                f"--mem={job['memory_gb_per_node']}G",
                f"--time={job['time_limit']}",
                f"--job-name={name}",
                f"--output={logs}/%x-%j.out",
                f"--error={logs}/%x-%j.err",
                f"--dependency=afterany:{previous}",
                f"--export=ALL,storage_root={storage_root},DSPARK_BUNDLE_DIR={bundle}",
                str(script),
            ]
            state["pending_submission"] = {"key": job["key"], "command": command}
            save(state_path, state)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                state.pop("pending_submission")
                state["submission_error"] = {"key": job["key"], "stderr": result.stderr}
                save(state_path, state)
                raise RuntimeError(result.stderr)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdecimal():
                raise RuntimeError("ambiguous scheduler response; pending submission retained for reconciliation")
            state["jobs"].append(
                {
                    "key": job["key"],
                    "job_id": job_id,
                    "dependency_afterany": previous,
                    "script_sha256": job["script_sha256"],
                    "submission_command": command,
                }
            )
            state.pop("pending_submission")
            state.pop("submission_error", None)
            save(state_path, state)
            previous = job_id
            print(f"{platform} {job_id} {job['key']}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--platform", choices=("b200", "b300"), required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--after-job", required=True)
    args = parser.parse_args()
    submit(args.directory, args.platform, args.storage_root, args.after_job)


if __name__ == "__main__":
    main()
