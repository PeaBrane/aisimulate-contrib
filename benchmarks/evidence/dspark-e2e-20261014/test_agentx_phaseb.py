# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for Phase B gates and unchanged fixed-run defaults."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentx_context_audit import inspect_requests
from agentx_phaseb import CORPUS, context_gate, fixed_gate, producer_command
from agentx_server import validate_role
from campaign import IMAGE, MODELS, RUNTIME_REVISION, cells
from compare import CONTRACT, METRICS
from serve_smoke import server_command


class PhaseBTests(unittest.TestCase):
    def pair(self, root, concurrency=1):
        cell = next(
            c
            for c in cells()
            if c["model_key"] == "v4-pro"
            and c["deployment"] == "agg"
            and c["num_speculative_tokens"] == 6
            and c["input_length"] == 1024
            and c["concurrency"] == concurrency
            and c["acceptance"] == "real"
        )
        root.mkdir()
        identity = {
            "model": MODELS["v4-pro"],
            "image": IMAGE,
            "runtime_revision": RUNTIME_REVISION,
            "server_command": ["serve", "--max-model-len", "16384"],
            "phase": "initial",
        }
        (root / "identity.json").write_text(json.dumps(identity))
        (root / "requests.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "status": "success",
                        "prompt_token_ids": [1] * 1024,
                        "output_token_ids": [2] * 1024,
                    }
                )
                for _ in range(cell["expected_requests"])
            )
        )
        record = {
            "cell_id": cell["cell_id"],
            "repetition": 1,
            "status": "success",
            "qualification": {"eligible": True},
            "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "metrics_contract": CONTRACT,
            "expected_requests": cell["expected_requests"],
            "total_gpus": cell["total_gpus"],
            "request_records": "requests.jsonl",
            "metrics": {
                **dict.fromkeys(METRICS, 1.0),
                "completed_requests": cell["expected_requests"],
                "failed_requests": 0,
            },
        }
        measured, predicted = root / "measured.json", root / "predicted.json"
        measured.write_text(json.dumps(record))
        record["predictor"] = "aisimulate.engine_replay"
        record["metrics"].update(dict.fromkeys(METRICS, 1000.0))
        predicted.write_text(json.dumps(record))
        return measured, predicted

    def test_gate_accepts_large_error_and_per_concurrency_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = self.pair(Path(directory) / "c1")
            self.assertEqual(len(fixed_gate([pair], "v4-pro", "agg", 6, (1,))), 1)
            with self.assertRaisesRegex(ValueError, "concurrency gate"):
                fixed_gate([pair], "v4-pro", "agg", 6)
            pair4 = self.pair(Path(directory) / "c4", 4)
            self.assertEqual(len(fixed_gate([pair, pair4], "v4-pro", "agg", 6)), 2)
            record = json.loads(pair[1].read_text())
            record["qualification"]["eligible"] = False
            pair[1].write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "prediction_unqualified"):
                fixed_gate([pair], "v4-pro", "agg", 6, (1,))

    def test_source_context_counts_children_and_minimum_one_output(self):
        state = dict.fromkeys(
            (
                "subagent_groups",
                "requests",
                "input_tokens",
                "recorded_output_tokens",
                "max_input",
                "max_requested_output",
                "max_input_plus_requested_output",
                "over_16384",
                "over_1048576",
            ),
            0,
        )
        inspect_requests(
            [
                {"type": "n", "in": 16384, "out": 0},
                {"type": "subagent", "requests": [{"type": "s", "in": 900000, "out": 60000}]},
            ],
            state,
        )
        self.assertEqual(state["requests"], 2)
        self.assertEqual(state["subagent_groups"], 1)
        self.assertEqual(state["max_input_plus_requested_output"], 960000)
        self.assertEqual(state["over_16384"], 2)
        self.assertEqual(state["over_1048576"], 0)

    def test_native_runtime_policy_uses_real_source_and_formatter_witness(self):
        path = Path(__file__).with_name("agentx-context-preflight-pro.json")
        result = context_gate(
            path,
            {"max_model_len": 1048576, "prefix_cache": True},
            [{"max_model_len": 1048576, "prefix_cache": False}],
            "v4-pro",
        )
        self.assertEqual(result["validation_policy"], "native_runtime_per_request")
        with self.assertRaisesRegex(ValueError, "context limit"):
            context_gate(
                path,
                {"max_model_len": 16384, "prefix_cache": True},
                [{"max_model_len": 16384, "prefix_cache": False}],
                "v4-pro",
            )
        with self.assertRaisesRegex(ValueError, "do not match"):
            context_gate(
                path,
                {"max_model_len": 1048576, "prefix_cache": True},
                [{"max_model_len": 16384, "prefix_cache": False}],
                "v4-pro",
            )

    def test_exact_command_and_diagnostic_pilot(self):
        for phase, duration in [("pilot", "600"), ("profile", "3600")]:
            cmd = producer_command(
                Path("/venv/bin/aiperf"),
                "v4-pro",
                "http://localhost:8000",
                4,
                phase,
                Path("/tmp/run"),
                [],
            )
            self.assertEqual(cmd[cmd.index("--benchmark-duration") + 1], duration)
            self.assertEqual("--unsafe-override" in cmd, phase == "pilot")
            self.assertNotIn("--burst-phase-starts", cmd)
            self.assertEqual(cmd[cmd.index("--agentic-warmup-grace-period") + 1], "1800")
            self.assertNotIn("--synthesis-max-isl", cmd)
            self.assertNotIn("--synthesis-max-osl", cmd)

    def test_explicit_server_overrides_preserve_old_defaults(self):
        with patch("serve_smoke.inspect_checkpoint", return_value={"path": "/checkpoint"}):
            old, _, _ = server_command("v4-pro", 6, 8000)
            new, env, _ = server_command(
                "v4-pro",
                6,
                8000,
                prefix_cache=True,
                context_length=1048576,
                load_strategy="eager",
                startup_timeout_s=7200,
                graph_policy="runtime-default",
            )
        self.assertIn("--no-enable-prefix-caching", old)
        self.assertEqual(old[old.index("--max-model-len") + 1], "16384")
        self.assertIn("--enable-prefix-caching", new)
        self.assertEqual(new[new.index("--max-model-len") + 1], "1048576")
        self.assertEqual(new[new.index("--safetensors-load-strategy") + 1], "eager")
        self.assertEqual(env["VLLM_ENGINE_READY_TIMEOUT_S"], "7200")
        identity = {
            "command": new,
            "prefix_cache": True,
            "graph_policy": "runtime-default",
        }
        validate_role(identity, "v4-pro", 6, "agg")
        with self.assertRaises(ValueError):
            validate_role(identity, "v4-pro", 3, "agg")

    def test_context_requires_verified_full_payloads_and_matched_anchor(self):
        with self.assertRaisesRegex(ValueError, "context audit"):
            context_gate(None, {}, [], "v4-pro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payloads.json").write_text("[]")
            audit = {
                "corpus": CORPUS,
                "tokenizer": MODELS["v4-pro"]["model"],
                "tokenizer_revision": MODELS["v4-pro"]["revision"],
                "all_generated_payloads_verified": True,
                "truncation_or_filtering": False,
                "output_policy": "recorded_out_minimum_one_ignore_eos",
                "max_prompt_plus_requested_output": 900000,
                "evidence": [
                    {
                        "path": "payloads.json",
                        "sha256": hashlib.sha256(b"[]").hexdigest(),
                    }
                ],
            }
            path = root / "audit.json"
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "do not match"):
                context_gate(
                    path,
                    {"max_model_len": 16384, "prefix_cache": True},
                    [{"max_model_len": 16384, "prefix_cache": True}],
                    "v4-pro",
                )
            context_gate(
                path,
                {"max_model_len": 1048576, "prefix_cache": True},
                [{"max_model_len": 1048576, "prefix_cache": False}],
                "v4-pro",
            )
            with self.assertRaisesRegex(ValueError, "do not match"):
                context_gate(
                    path,
                    {"max_model_len": 1048576, "prefix_cache": True},
                    [{"max_model_len": 16384, "prefix_cache": True}],
                    "v4-pro",
                )


if __name__ == "__main__":
    unittest.main()
