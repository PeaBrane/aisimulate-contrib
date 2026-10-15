# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze six independent 1M-context native anchor jobs; never submit them."""

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

from campaign import IMAGE, MODELS, RUNTIME_REVISION

CAMPAIGN = "dspark-agentx-20261015"
FILES = (
    "acceptance.py",
    "agg-smoke.sh",
    "campaign.py",
    "checkpoint-references.json",
    "collect_fixed.py",
    "inspect_runtime.py",
    "matrix.json",
    "pd-smoke.sh",
    "pd_driver.py",
    "runtime_receipt.py",
    "serve_role.py",
    "serve_smoke.py",
    "source_pool_observer.py",
    "source_pool_sources.json",
    "window_plan.py",
)


def jobs():
    result = []
    for key, model in MODELS.items():
        for deployment in ("agg", "disagg"):
            width = model["gpus_per_role"]
            nodes = 2 if deployment == "disagg" and width == 8 else 1
            gpus = width * (2 if deployment == "disagg" else 1)
            memory = 2048 if key == "kimi-k3" else 1200 if key == "v4-pro" else 768
            if deployment == "disagg" and nodes == 1:
                memory *= 2
            k = model["reference_k"]
            env = {
                "DSPARK_MODEL_KEY": key,
                "DSPARK_K": str(k),
                "DSPARK_RUN_MODE": "initial",
                "DSPARK_GRAPH_POLICY": "runtime-default",
                "DSPARK_STARTUP_TIMEOUT": "6000",
                "DSPARK_LOAD_STRATEGY": "eager",
                "DSPARK_CONTEXT_LENGTH": "1048576",
                "DSPARK_SYNTHETIC_AL": str(model["reference_al"]),
                "DSPARK_INPUT_LENGTH": "1024",
                "DSPARK_REPETITION": "1",
                "DSPARK_CELL_IDS": " ".join(f"{key}-{deployment}-k{k}-i1024-o1024-c{c}-synthetic" for c in (1, 4)),
            }
            if deployment == "disagg":
                env.update(
                    DSPARK_PREFILL_MAX_BATCHED_TOKENS="2048" if key == "v4-pro" else "8192",
                    DSPARK_DECODE_MAX_BATCHED_TOKENS="8192",
                )
                if key == "v4-pro":
                    env["DSPARK_PREFILL_GPU_MEMORY_UTILIZATION"] = "0.92"
            result.append(
                {
                    "key": f"{key}-{deployment}-context1m-anchor",
                    "model_key": key,
                    "deployment": deployment,
                    "platform": model["hardware"],
                    "nodes": nodes,
                    "gpus_per_node": gpus // nodes,
                    "memory_gb_per_node": memory,
                    "cpus_per_node": 56 if deployment == "disagg" and nodes == 1 else 32,
                    "time_limit": "04:00:00",
                    "environment": env,
                    "launcher": "pd-smoke.sh" if deployment == "disagg" else "agg-smoke.sh",
                    "windows": [{"concurrency": c, "repetition": 1, "requests": 128} for c in (1, 4)],
                    "result_pattern": (
                        f"results/dspark-e2e-20261014/{'pd-' if deployment == 'disagg' else ''}"
                        f"initial-{key}-k{k}-JOB_ID"
                    ),
                    "status": "prepared_not_submitted",
                }
            )
    return result


def script(job):
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
        '(cd "$DSPARK_BUNDLE_DIR" && sha256sum --check source.sha256)',
        *[f"export {key}={shlex.quote(value)}" for key, value in job["environment"].items()],
        f'bash "$DSPARK_BUNDLE_DIR/{job["launcher"]}"',
    ]
    return "\n".join(body) + "\n"


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    files = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in FILES}
    archive = output / "anchor-source.tar.gz"
    checksum = "".join(f"{value}  {name}\n" for name, value in files.items()).encode()
    with tarfile.open(archive, "w:gz") as tar:
        for name in FILES:
            tar.add(source / name, arcname=name)
        info = tarfile.TarInfo("source.sha256")
        info.size = len(checksum)
        tar.addfile(info, io.BytesIO(checksum))
    plan = {
        "schema": "dspark-agentx-native-anchor-jobs-v1",
        "campaign": CAMPAIGN,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_state": "Exact archived working-tree bytes; commit and source archive identities are separate.",
        "image": IMAGE,
        "runtime_revision": RUNTIME_REVISION,
        "files": files,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "jobs": jobs(),
        "completed_run_exclusions": [],
        "submitted_jobs": [],
        "submission_policy": (
            "Parent chains after Phase A per platform; recheck scheduler access and duplicate names before submission."
        ),
        "success_policy": (
            "Actual native C1/C4 completion and identity gates; no simulator qualification or numerical accuracy gate."
        ),
    }
    for job in plan["jobs"]:
        path = output / (job["key"] + ".sbatch")
        path.write_text(script(job))
        job.update(script=path.name, script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(Path(__file__).parent, args.output)
    print(
        json.dumps({"jobs": len(plan["jobs"]), "windows": 12, "submitted": 0, "archive_sha256": plan["archive_sha256"]})
    )


if __name__ == "__main__":
    main()
