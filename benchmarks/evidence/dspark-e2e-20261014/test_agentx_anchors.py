# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for independent native context-anchor job selection and freezing."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from campaign import MODELS
from prepare_agentx_anchors import FILES, jobs, prepare, script
from window_plan import select_runs


class AnchorTests(unittest.TestCase):
    def test_six_resources_and_exact_windows(self):
        planned = jobs()
        self.assertEqual(len(planned), 6)
        self.assertEqual(sum(len(j["windows"]) for j in planned), 12)
        for job in planned:
            model = MODELS[job["model_key"]]
            env = job["environment"]
            args = argparse.Namespace(
                model_key=job["model_key"],
                k=model["reference_k"],
                mode=env["DSPARK_RUN_MODE"],
                cell_id=env["DSPARK_CELL_IDS"].split(),
                window=None,
                concurrency=None,
                repetition=[int(env["DSPARK_REPETITION"])],
                input_length=[1024],
                completed_runs=None,
            )
            runs = select_runs(args, job["deployment"], "synthetic")
            self.assertEqual([(r["cell"]["concurrency"], r["repetition"]) for r in runs], [(1, 1), (4, 1)])
            self.assertEqual(
                job["nodes"] * job["gpus_per_node"],
                model["gpus_per_role"] * (2 if job["deployment"] == "disagg" else 1),
            )
            self.assertEqual(env["DSPARK_CONTEXT_LENGTH"], "1048576")
            self.assertNotIn("DSPARK_COMPLETED_RUNS", env)
            self.assertNotIn("DSPARK_OBSERVE_SOURCE_POOL", env)
        pro = next(j for j in planned if j["model_key"] == "v4-pro" and j["deployment"] == "disagg")
        self.assertEqual(pro["environment"]["DSPARK_PREFILL_MAX_BATCHED_TOKENS"], "2048")
        self.assertEqual(pro["environment"]["DSPARK_DECODE_MAX_BATCHED_TOKENS"], "8192")

    def test_job_clears_inherited_exclusions_and_keeps_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            launcher = bundle / "agg-smoke.sh"
            launcher.write_text('#!/bin/bash\nenv > "$storage_root/captured.env"\n')
            (bundle / "source.sha256").write_text(
                hashlib.sha256(launcher.read_bytes()).hexdigest() + "  agg-smoke.sh\n"
            )
            job = root / "job.sh"
            job.write_text(script(jobs()[0]))
            subprocess.run(
                ["bash", str(job)],
                env={
                    **os.environ,
                    "storage_root": str(root),
                    "DSPARK_BUNDLE_DIR": str(bundle),
                    "DSPARK_COMPLETED_RUNS": "unrelated.json",
                    "DSPARK_WINDOWS": "1:3",
                    "DSPARK_SYNTHETIC_CONTROL": "wrong.json",
                },
                check=True,
                capture_output=True,
            )
            env = dict(line.split("=", 1) for line in (root / "captured.env").read_text().splitlines() if "=" in line)
            self.assertNotIn("DSPARK_COMPLETED_RUNS", env)
            self.assertNotIn("DSPARK_WINDOWS", env)
            self.assertNotIn("DSPARK_SYNTHETIC_CONTROL", env)
            self.assertEqual(env["DSPARK_REPETITION"], "1")
            self.assertEqual(env["DSPARK_BUNDLE_DIR"], str(bundle))

    def test_frozen_archive_contains_only_declared_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            plan = prepare(Path(__file__).parent, root)
            archive = root / plan["archive"]
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), plan["archive_sha256"])
            with tarfile.open(archive) as tar:
                self.assertEqual(set(tar.getnames()), {*FILES, "source.sha256"})
                for name, digest in plan["files"].items():
                    self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(), digest)
            self.assertEqual(json.loads((root / "plan.json").read_text())["submitted_jobs"], [])
            for job in plan["jobs"]:
                subprocess.run(["bash", "-n", str(root / job["script"])], check=True)
            with self.assertRaises(FileExistsError):
                prepare(Path(__file__).parent, root)


if __name__ == "__main__":
    unittest.main()
