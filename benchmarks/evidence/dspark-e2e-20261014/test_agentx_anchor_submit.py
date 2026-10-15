# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only submission tests; scheduler calls are replaced with deterministic replies."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_agentx_anchors import FILES, prepare
from submit_agentx_anchors import submit


class AnchorSubmissionTests(unittest.TestCase):
    def setup_paths(self, root):
        directory = root / "anchors"
        prepare(Path(__file__).parent, directory)
        bundle = root / "jobs" / "dspark-agentx-20261015" / "anchor-bundle"
        bundle.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(Path(__file__).with_name(name), bundle / name)
        return directory

    def test_serial_submission_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.setup_paths(root)
            commands = []

            def run(command, **kwargs):
                if command[0] == "sbatch":
                    commands.append(command)
                    return subprocess.CompletedProcess(command, 0, str(1000 + len(commands)) + "\n", "")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("submit_agentx_anchors.subprocess.run", side_effect=run):
                state = submit(directory, "b300", root, "999")
                self.assertEqual([j["job_id"] for j in state["jobs"]], ["1001", "1002"])
                self.assertIn("--dependency=afterany:999", commands[0])
                self.assertIn("--dependency=afterany:1001", commands[1])
                submit(directory, "b300", root, "999")
                self.assertEqual(len(commands), 2)
                with self.assertRaisesRegex(ValueError, "resume identity"):
                    submit(directory, "b300", root, "998")
            self.assertEqual(json.loads((directory / "submitted-b300.json").read_text())["jobs"], state["jobs"])

    def test_ambiguous_submit_cannot_silently_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.setup_paths(root)
            with (
                patch(
                    "submit_agentx_anchors.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "unexpected", ""),
                ),
                self.assertRaisesRegex(RuntimeError, "ambiguous"),
            ):
                submit(directory, "b300", root, "999")
            with self.assertRaisesRegex(ValueError, "uncertain previous submission"):
                submit(directory, "b300", root, "999")


if __name__ == "__main__":
    unittest.main()
