# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and submit the remaining fixed-length matrix on its two Slurm platforms."""

import argparse
import collections
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from campaign import MODELS

CAMPAIGN = "dspark-full-20261015"
ACCOUNT = "coreai_comparch_inferencex"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan(matrix, comparison):
    done = []
    for source in comparison["sources"]:
        path = Path(source["path"])
        if "qualified-summary" not in path.name:
            continue
        summary = read(path)
        if summary["status"] != "success" or not summary["qualification"]["eligible"]:
            raise ValueError("completed-run exclusion requires a qualified successful measurement")
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("completed measurement changed since comparison")
        done.append({"cell_id": summary["cell_id"], "repetition": summary["repetition"]})
    completed = {(row["cell_id"], row["repetition"]) for row in done}
    if len(completed) != len(done):
        raise ValueError("duplicate completed measurement")
    cell_ids = [cell["cell_id"] for cell in matrix["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("duplicate matrix cell")
    expected = {
        (cell["cell_id"], repetition) for cell in matrix["cells"] for repetition in range(1, cell["repetitions"] + 1)
    }
    if not completed <= expected:
        raise ValueError("completed measurement lies outside the matrix")
    groups = collections.defaultdict(list)
    for cell in matrix["cells"]:
        key = tuple(
            cell[name] for name in ("model_key", "deployment", "num_speculative_tokens", "acceptance", "input_length")
        )
        for repetition in range(1, cell["repetitions"] + 1):
            if (cell["cell_id"], repetition) not in completed:
                groups[key].append(
                    {"cell_id": cell["cell_id"], "repetition": repetition, "requests": cell["expected_requests"]}
                )
    jobs = []
    for (model, deployment, k, acceptance, isl), windows in groups.items():
        width = MODELS[model]["gpus_per_role"]
        nodes = 2 if deployment == "disagg" and width == 8 else 1
        gpus = width * (2 if deployment == "disagg" else 1)
        memory = 2048 if model == "kimi-k3" else (1200 if model == "v4-pro" else 768)
        if deployment == "disagg" and nodes == 1:
            memory *= 2
        jobs.append(
            {
                "key": f"{model}-{deployment}-k{k}-{acceptance}-i{isl}",
                "kind": "sweep",
                "model_key": model,
                "deployment": deployment,
                "k": k,
                "acceptance": acceptance,
                "input_length": isl,
                "platform": MODELS[model]["hardware"],
                "nodes": nodes,
                "gpus_per_node": gpus // nodes,
                "memory_gb_per_node": memory,
                "cpus_per_node": 56 if deployment == "disagg" and nodes == 1 else 32,
                "time_limit": "04:00:00",
                "windows": windows,
            }
        )
    # Native operator-table gaps do not block serving. Only the additional-K
    # synthetic controls require new real-acceptance pilots.
    for model in ("v41-flash", "kimi-k3"):
        jobs.append(
            {
                "key": f"{model}-agg-k3-pilot",
                "kind": "pilot",
                "model_key": model,
                "deployment": "agg",
                "k": 3,
                "acceptance": "real",
                "input_length": 1024,
                "platform": MODELS[model]["hardware"],
                "nodes": 1,
                "gpus_per_node": MODELS[model]["gpus_per_role"],
                "memory_gb_per_node": 2048 if model == "kimi-k3" else 768,
                "cpus_per_node": 32,
                "time_limit": "03:00:00",
                "windows": [],
            }
        )

    def priority(job):
        model = job["model_key"]
        return (
            job["platform"],
            {"v4-pro": 0, "v41-flash": 1, "kimi-k3": 0}[model],
            0 if job["kind"] == "pilot" else 1,
            0 if job["k"] == MODELS[model]["reference_k"] else 1,
            0 if job["deployment"] == "disagg" and model == "v4-pro" else 1,
            job["input_length"],
            job["acceptance"],
            job["key"],
        )

    jobs.sort(key=priority)
    return {
        "schema": "dspark-submission-plan-v1",
        "campaign": CAMPAIGN,
        "completed": done,
        "jobs": jobs,
        "counts": {
            "planned_configurations": len(matrix["cells"]),
            "completed_runs": len(done),
            "remaining_production_runs": sum(len(j["windows"]) for j in jobs),
            "remaining_production_requests": sum(w["requests"] for j in jobs for w in j["windows"]),
            "gpu_jobs": len(jobs),
            "new_acceptance_pilots": 2,
        },
        "concurrency_policy": (
            "One GPU job per platform, enforced with Slurm afterany dependencies. P/D roles share one allocation."
        ),
        "failure_policy": (
            "Each job performs its own native smoke before production. Missing pilot controls fail before serving "
            "startup; raw failures remain evidence. A failure does not block independent configurations."
        ),
        "phase_b": (
            "Separate AgentX readiness and submission ledger; this manifest covers all fixed-length matrix cells only."
        ),
    }


def script(job):
    model, k = job["model_key"], job["k"]
    exports = {
        "DSPARK_MODEL_KEY": model,
        "DSPARK_K": str(k),
        "DSPARK_RUN_MODE": job["kind"],
        "DSPARK_GRAPH_POLICY": "runtime-default",
        "DSPARK_STARTUP_TIMEOUT": "6000",
        "DSPARK_LOAD_STRATEGY": "eager",
        "DSPARK_INPUT_LENGTH": str(job["input_length"]),
    }
    if job["kind"] == "sweep":
        exports["DSPARK_CELL_IDS"] = " ".join(dict.fromkeys(w["cell_id"] for w in job["windows"]))
    if model == "v4-pro" and job["deployment"] == "disagg":
        exports["DSPARK_PREFILL_GPU_MEMORY_UTILIZATION"] = "0.92"
        exports["DSPARK_OBSERVE_SOURCE_POOL"] = "1"
    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        ': "${storage_root:?}"',
        ': "${DSPARK_BUNDLE_DIR:?}"',
        'cd "$storage_root"',
        'case "$PWD/" in "$HOME"/*) exit 2;; esac',
        'for setting in "${!DSPARK_@}"; do',
        '  [[ "$setting" == DSPARK_BUNDLE_DIR ]] || unset "$setting"',
        "done",
    ]
    if job["kind"] == "sweep":
        body += ['export DSPARK_COMPLETED_RUNS="/workspace/jobs/dspark-full-20261015/bundle/completed-runs.json"']
    body += [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
    if job["acceptance"] == "synthetic":
        if k == MODELS[model]["reference_k"]:
            body += [f"export DSPARK_SYNTHETIC_AL={MODELS[model]['reference_al']}"]
        else:
            control = f"synthetic-control-{model}-k{k}.json"
            body += [
                f'test -s "$storage_root/jobs/{CAMPAIGN}/controls/{control}" || '
                f'{{ echo "Missing frozen pilot control: {control}" >&2; exit 42; }}',
                f'export DSPARK_SYNTHETIC_CONTROL="/workspace/jobs/{CAMPAIGN}/controls/{control}"',
            ]
    launcher = "pd-smoke.sh" if job["deployment"] == "disagg" else "agg-smoke.sh"
    body += [f'bash "$DSPARK_BUNDLE_DIR/{launcher}"']
    if job["kind"] == "pilot":
        control = f"synthetic-control-{model}-k{k}.json"
        # The pilot output path is provided by the existing launcher contract.
        body += [
            "srun --nodes=1 --ntasks=1 --gres=gpu:0 --cpus-per-task=2 --mem=4G "
            '--container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh" '
            '--container-mounts="$storage_root:/workspace" '
            f"/usr/bin/python3 /workspace/jobs/{CAMPAIGN}/bundle/freeze_acceptance.py "
            f'--pilot "/workspace/results/dspark-e2e-20261014/pilot-{model}-k{k}-$SLURM_JOB_ID/pilot-c1-r1" '
            f'--output "/workspace/jobs/{CAMPAIGN}/controls/{control}"'
        ]
    return "\n".join(body) + "\n"


def prepare(root, output):
    if output.exists():
        raise ValueError("queue directory already exists; resume its frozen plan instead of replacing it")
    data = plan(read(root / "matrix.json"), read(root / "comparison/report.json"))
    output.mkdir(parents=True)
    save(output / "completed-runs.json", {"schema": "dspark-completed-runs-v1", "completed": data["completed"]})
    for job in data["jobs"]:
        name = job["key"] + ".sbatch"
        path = output / name
        path.write_text(script(job))
        job["script"] = name
        job["script_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(output / "plan.json", data)
    print(json.dumps(data["counts"], indent=2))


def submit(directory, platform, storage_root):
    if storage_root.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError("benchmark storage must be outside home")
    data = read(directory / "plan.json")
    state_path = directory / f"submitted-{platform}.json"
    with (directory / f"submit-{platform}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(state_path) if state_path.exists() else {"platform": platform, "jobs": []}
        if state.get("pending_submission"):
            raise ValueError("previous submission outcome is uncertain; reconcile its job name before resuming")
        by_key = {row["key"]: row for row in state["jobs"]}
        previous = state["jobs"][-1]["job_id"] if state["jobs"] else None
        bundle = storage_root / "jobs" / CAMPAIGN / "bundle"
        logs = storage_root / "logs" / CAMPAIGN
        logs.mkdir(parents=True, exist_ok=True)
        for job in data["jobs"]:
            if job["platform"] != platform or job["key"] in by_key:
                continue
            path = directory / job["script"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != job["script_sha256"]:
                raise ValueError("job script differs from frozen submission plan")
            # Inspect live access and the queue immediately before every submission.
            subprocess.run(
                ["sacctmgr", "-nP", "show", "assoc", "where", f"user={os.environ['USER']}", "format=account,QOS"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["sinfo", "-h", "-p", "batch", "-o", "%P|%G|%D|%t|%l"], check=True, capture_output=True, text=True
            )
            existing = subprocess.run(
                ["squeue", "--me", "-h", "-o", "%i|%j"], check=True, capture_output=True, text=True
            ).stdout
            name = "dsf-" + job["key"]
            if any(line.partition("|")[2] == name for line in existing.splitlines()):
                raise ValueError(f"unrecorded duplicate job name {name}; reconcile before resuming")
            command = [
                "sbatch",
                "--parsable",
                f"--account={ACCOUNT}",
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
                f"--export=ALL,storage_root={storage_root},DSPARK_BUNDLE_DIR={bundle}",
            ]
            if previous:
                command += [f"--dependency=afterany:{previous}"]
            command.append(str(path))
            state["pending_submission"] = {"key": job["key"], "command": command}
            save(state_path, state)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                state.pop("pending_submission", None)
                state["submission_error"] = {"key": job["key"], "stderr": result.stderr, "command": command}
                save(state_path, state)
                raise RuntimeError(result.stderr)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"ambiguous scheduler response: {result.stdout!r}")
            state["jobs"].append(
                {
                    "key": job["key"],
                    "job_id": job_id,
                    "dependency_afterany": previous,
                    "script_sha256": job["script_sha256"],
                    "submission_command": command,
                }
            )
            state.pop("pending_submission", None)
            state.pop("submission_error", None)
            save(state_path, state)
            previous = job_id
            print(f"{platform} {job_id} {job['key']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--platform", choices=["b200", "b300"])
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(Path(__file__).parent, args.directory)
    else:
        if args.platform is None or args.storage_root is None:
            parser.error("submission requires explicit platform and storage root")
        submit(args.directory, args.platform, args.storage_root)


if __name__ == "__main__":
    main()
