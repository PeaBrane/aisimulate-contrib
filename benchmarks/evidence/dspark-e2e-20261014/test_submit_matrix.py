# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for matrix coverage and resumable Slurm submission."""

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import submit_matrix as queue

ROOT = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_fixture(directory):
    """The thirteen scored production windows, without requiring ignored raw files."""
    keys = [("v4-pro-agg-k6-i1024-o1024-c1-synthetic", 1)]
    keys += [
        (f"v4-pro-disagg-k6-i1024-o1024-c{concurrency}-{acceptance}", repetition)
        for acceptance in ("real", "synthetic")
        for concurrency in (1, 4)
        for repetition in (1, 2, 3)
    ]
    sources = []
    for index, (cell, repetition) in enumerate(keys):
        path = directory / f"{index}-qualified-summary.json"
        queue.save(
            path,
            {
                "cell_id": cell,
                "repetition": repetition,
                "status": "success",
                "qualification": {"eligible": True},
            },
        )
        sources.append({"path": str(path), "sha256": digest(path)})
    return {"sources": sources}, set(keys)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.matrix = queue.read(ROOT / "matrix.json")
        self.comparison, self.completed = completed_fixture(self.root)

    def test_exact_remaining_matrix_and_pilot_order(self):
        data = queue.plan(self.matrix, self.comparison)
        self.assertEqual(
            data["counts"],
            {
                "planned_configurations": 196,
                "completed_runs": 13,
                "remaining_production_runs": 575,
                "remaining_production_requests": 151936,
                "gpu_jobs": 50,
                "new_acceptance_pilots": 2,
            },
        )
        sweeps = [j for j in data["jobs"] if j["kind"] == "sweep"]
        self.assertEqual(len(sweeps), 48)
        remaining = [(w["cell_id"], w["repetition"]) for j in sweeps for w in j["windows"]]
        expected = {
            (c["cell_id"], repetition) for c in self.matrix["cells"] for repetition in range(1, c["repetitions"] + 1)
        } - self.completed
        self.assertEqual(set(remaining), expected)
        self.assertEqual(len(remaining), len(set(remaining)))
        for model in ("v41-flash", "kimi-k3"):
            pilot = next(i for i, j in enumerate(data["jobs"]) if j["model_key"] == model and j["kind"] == "pilot")
            dependents = [
                i
                for i, j in enumerate(data["jobs"])
                if j["model_key"] == model and j["k"] == 3 and j["acceptance"] == "synthetic"
            ]
            self.assertTrue(all(pilot < i for i in dependents))

    def test_declared_gpu_cpu_memory_geometry(self):
        for job in queue.plan(self.matrix, self.comparison)["jobs"]:
            with self.subTest(job=job["key"]):
                width = queue.MODELS[job["model_key"]]["gpus_per_role"]
                pd = job["deployment"] == "disagg"
                self.assertEqual(job["nodes"] * job["gpus_per_node"], width * (2 if pd else 1))
                self.assertLessEqual(job["gpus_per_node"], 8)
                if pd and width == 8:
                    self.assertEqual((job["nodes"], job["cpus_per_node"]), (2, 32))
                if pd and width == 4:
                    self.assertEqual((job["nodes"], job["cpus_per_node"], job["memory_gb_per_node"]), (1, 56, 1536))
                if job["model_key"] == "kimi-k3":
                    self.assertEqual(job["memory_gb_per_node"], 2048)

    def test_pilots_do_not_receive_production_exclusions(self):
        for job in queue.plan(self.matrix, self.comparison)["jobs"]:
            script = queue.script(job)
            if job["kind"] == "pilot":
                self.assertNotIn("export DSPARK_COMPLETED_RUNS=", script)
                self.assertNotIn("export DSPARK_CELL_IDS=", script)
                self.assertIn("freeze_acceptance.py", script)
            else:
                self.assertIn("export DSPARK_COMPLETED_RUNS=", script)

    def test_completed_exclusions_reject_duplicates_mutation_and_unknown_runs(self):
        source = self.comparison["sources"][0]
        with self.assertRaises(ValueError):
            queue.plan(self.matrix, {"sources": [source, source]})
        path = Path(source["path"])
        original = queue.read(path)
        for update in ({"cell_id": "not-in-matrix"}, {"repetition": 4}, {"qualification": {"eligible": False}}):
            row = original | update
            queue.save(path, row)
            with self.subTest(update=update), self.assertRaises(ValueError):
                queue.plan(self.matrix, {"sources": [{"path": str(path), "sha256": digest(path)}]})
        with self.assertRaises(ValueError):
            queue.plan(self.matrix, {"sources": [source]})

    def make_submission(self):
        directory = self.root / "queue"
        directory.mkdir()
        data = queue.plan(self.matrix, self.comparison)
        jobs = []
        for platform in ("b200", "b300"):
            jobs.extend([j for j in data["jobs"] if j["platform"] == platform][:2])
        for job in jobs:
            path = directory / (job["key"] + ".sbatch")
            path.write_text(queue.script(job))
            job.update(script=path.name, script_sha256=digest(path))
        queue.save(directory / "plan.json", {"jobs": jobs})
        return directory, self.root / "storage", jobs

    def test_platform_chains_are_serial_and_resume_is_idempotent(self):
        directory, storage, jobs = self.make_submission()
        submitted = []

        def run(command, **kwargs):
            if command[0] == "sbatch":
                submitted.append(command)
                return subprocess.CompletedProcess(command, 0, f"{100 + len(submitted)};cluster\n", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(queue.subprocess, "run", side_effect=run) as invoked, patch.dict(os.environ, USER="test"):
            queue.submit(directory, "b200", storage)
            queue.submit(directory, "b300", storage)
            calls = invoked.call_count
            queue.submit(directory, "b200", storage)
            queue.submit(directory, "b300", storage)
            self.assertEqual(invoked.call_count, calls)
        self.assertEqual(len(submitted), 4)
        self.assertFalse(any(x.startswith("--dependency") for x in submitted[0]))
        self.assertIn("--dependency=afterany:101", submitted[1])
        self.assertFalse(any(x.startswith("--dependency") for x in submitted[2]))
        self.assertIn("--dependency=afterany:103", submitted[3])
        for command, job in zip(submitted, jobs, strict=True):
            self.assertIn(f"--nodes={job['nodes']}", command)
            self.assertIn(f"--ntasks={job['nodes']}", command)
            self.assertIn("--ntasks-per-node=1", command)
            self.assertIn(f"--mem={job['memory_gb_per_node']}G", command)

    def test_failed_submission_preserves_previous_dependency_for_retry(self):
        directory, storage, _ = self.make_submission()
        count = 0

        def run(command, **kwargs):
            nonlocal count
            if command[0] != "sbatch":
                return subprocess.CompletedProcess(command, 0, "", "")
            count += 1
            return subprocess.CompletedProcess(command, 1 if count == 2 else 0, "201\n", "denied" if count == 2 else "")

        with (
            patch.object(queue.subprocess, "run", side_effect=run),
            patch.dict(os.environ, USER="test"),
            self.assertRaises(RuntimeError),
        ):
            queue.submit(directory, "b200", storage)
        state = queue.read(directory / "submitted-b200.json")
        self.assertEqual(len(state["jobs"]), 1)
        self.assertIn("submission_error", state)
        commands = []

        def retry(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "202\n" if command[0] == "sbatch" else "", "")

        with patch.object(queue.subprocess, "run", side_effect=retry), patch.dict(os.environ, USER="test"):
            queue.submit(directory, "b200", storage)
        launches = [c for c in commands if c[0] == "sbatch"]
        self.assertEqual(len(launches), 1)
        self.assertIn("--dependency=afterany:201", launches[0])

    def test_live_unrecorded_duplicate_and_changed_script_never_submit(self):
        directory, storage, jobs = self.make_submission()

        def run(command, **kwargs):
            self.assertNotEqual(command[0], "sbatch")
            output = f"999|dsf-{jobs[0]['key']}\n" if command[0] == "squeue" else ""
            return subprocess.CompletedProcess(command, 0, output, "")

        with (
            patch.object(queue.subprocess, "run", side_effect=run),
            patch.dict(os.environ, USER="test"),
            self.assertRaises(ValueError),
        ):
            queue.submit(directory, "b200", storage)
        (directory / jobs[0]["script"]).write_text("changed")
        with patch.object(queue.subprocess, "run") as invoked, self.assertRaises(ValueError):
            queue.submit(directory, "b200", storage)
        invoked.assert_not_called()

    def test_uncertain_submission_intent_blocks_duplicate_after_queue_disappears(self):
        directory, storage, jobs = self.make_submission()

        def run(command, **kwargs):
            if command[0] == "sbatch":
                state = queue.read(directory / "submitted-b200.json")
                self.assertEqual(state["pending_submission"]["key"], jobs[0]["key"])
                return subprocess.CompletedProcess(command, 0, "unparseable scheduler response", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch.object(queue.subprocess, "run", side_effect=run),
            patch.dict(os.environ, USER="test"),
            self.assertRaises(RuntimeError),
        ):
            queue.submit(directory, "b200", storage)
        self.assertTrue(queue.read(directory / "submitted-b200.json")["pending_submission"])
        with patch.object(queue.subprocess, "run") as invoked, self.assertRaises(ValueError):
            queue.submit(directory, "b200", storage)
        invoked.assert_not_called()

    def test_real_job_clears_inherited_synthetic_and_observer_controls(self):
        job = next(
            j
            for j in queue.plan(self.matrix, self.comparison)["jobs"]
            if j["model_key"] == "v41-flash"
            and j["kind"] == "sweep"
            and j["acceptance"] == "real"
            and j["deployment"] == "agg"
        )
        storage = self.root / "storage"
        storage.mkdir()
        bundle = self.root / "bundle"
        bundle.mkdir()
        captured = self.root / "environment.txt"
        (bundle / "agg-smoke.sh").write_text('env > "$CAPTURE_ENV"\n')
        path = self.root / "real.sh"
        path.write_text(queue.script(job))
        env = os.environ | {
            "storage_root": str(storage),
            "DSPARK_BUNDLE_DIR": str(bundle),
            "CAPTURE_ENV": str(captured),
            "DSPARK_SYNTHETIC_AL": "3.77",
            "DSPARK_SYNTHETIC_CONTROL": "/wrong/control.json",
            "DSPARK_OBSERVE_SOURCE_POOL": "1",
            "HOME": str(self.root / "fake-home"),
        }
        subprocess.run(["bash", str(path)], env=env, check=True, capture_output=True)
        actual = captured.read_text().splitlines()
        for key in ("DSPARK_SYNTHETIC_AL", "DSPARK_SYNTHETIC_CONTROL", "DSPARK_OBSERVE_SOURCE_POOL"):
            self.assertFalse(any(line.startswith(key + "=") for line in actual), key)


if __name__ == "__main__":
    unittest.main()
