# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for window orchestration, preserving actual native/client gates."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentx_run import client_command, client_windows, preflight


class AgentxRunTests(unittest.TestCase):
    def test_windows_cover_plan_and_accept_only_valid_subsets(self):
        self.assertEqual(client_windows("pilot"), [(1, 1), (4, 1)])
        self.assertEqual(len(client_windows("profile")), 9)
        self.assertEqual(client_windows("profile", [(16, 3)]), [(16, 3)])
        for windows in ([(16, 1)], [(1, 2)], [(1, 1), (1, 1)]):
            with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                client_windows("pilot", windows)

    def args(self, root, phase="pilot"):
        receipt = root / "context.json"
        receipt.write_text("{}")
        executable = root / "aiperf"
        executable.write_text("fixture")
        return argparse.Namespace(
            model_key="v4-pro",
            deployment="agg",
            phase=phase,
            window=[(1, 1)],
            fixed_measurement=[root / "anchor.json"],
            pilot_check=[],
            context_audit=receipt,
            aiperf=executable,
            producer_checkout=root / "producer",
            output=root / "run",
        )

    def test_client_keeps_real_gates_without_token_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            cmd = client_command(args, 1, 1, "http://127.0.0.1:8010", ["http://127.0.0.1:8010/metrics"])
            for flag in (
                "--execute",
                "--collection-only",
                "--context-audit",
                "--fixed-measurement",
                "--server-identity",
            ):
                self.assertIn(flag, cmd)
            self.assertEqual(cmd[cmd.index("--acceptance") + 1], "synthetic")
            self.assertNotIn("--diagnostic-token-echo", cmd)
            self.assertNotIn("--fixed-pair", cmd)

    def test_native_anchor_failure_prevents_client_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            with (
                patch("agentx_run.fixed_gate", side_effect=ValueError("incomplete native anchor")),
                self.assertRaisesRegex(ValueError, "incomplete native anchor"),
            ):
                preflight(args)
            with patch("agentx_run.fixed_gate") as fixed:
                preflight(args)
                self.assertEqual(fixed.call_args.kwargs["required_concurrencies"], (1,))
                self.assertTrue(fixed.call_args.kwargs["collection_only"])
            args.phase = "profile"
            with (
                patch("agentx_run.fixed_gate") as fixed,
                patch("agentx_run.profile_gate", side_effect=ValueError("pilot review missing")),
            ):
                with self.assertRaisesRegex(ValueError, "pilot review missing"):
                    preflight(args)
                self.assertEqual(fixed.call_args.kwargs["required_concurrencies"], (1, 4))


if __name__ == "__main__":
    unittest.main()
