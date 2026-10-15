# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original CPU fixtures: fork/join frontier, native token echo and failed records."""

import json
import tempfile
import unittest
from pathlib import Path

from agentx_capture import PRODUCER, audit, extract_request


def raw(request_id="r1", overflow=False):
    chunks = [
        {"prompt_token_ids": [1, 2], "choices": []},
        {"choices": [{"index": 0, "token_ids": [3, 4], "finish_reason": "length"}]},
        {"usage": {"prompt_tokens": 2, "completion_tokens": 2}, "choices": []},
    ]
    return {
        "metadata": {
            "x_request_id": request_id,
            "x_correlation_id": "child-session",
            "conversation_id": "child",
            "turn_index": 1,
            "benchmark_phase": "warmup" if overflow else "profiling",
            "context_overflow_skip": overflow,
            "parent_correlation_id": "root-session",
        },
        "payload": {
            "messages": [{"role": "user", "content": "original fixture"}],
            "max_tokens": 2,
            "ignore_eos": True,
        },
        "responses": [{"packets": [{"name": "data", "value": json.dumps(chunk)}]} for chunk in chunks],
    }


class CaptureTests(unittest.TestCase):
    def fixture(self, root):
        templates = {
            "root": {"turns": [{"delay_ms": 12}, {}]},
            "child": {"turns": [{}, {}]},
        }
        (root / "tree.json").write_text(json.dumps(templates))
        frontier = [
            {
                "snapshot": {
                    "states": [
                        {
                            "conversation_id": "root",
                            "next_turn_index": 1,
                            "waiting_on_children": True,
                        },
                        {
                            "conversation_id": "child",
                            "next_turn_index": 1,
                            "join_gate_memberships": [["join", 1]],
                        },
                    ]
                }
            }
        ]
        events = [
            dict(
                producer_revision=PRODUCER,
                phase=phase,
                event=event,
                trajectories=frontier,
                templates={"root": "tree.json"},
            )
            for phase, event in [("warmup", "finalize"), ("profiling", "setup")]
        ]
        (root / "events-1.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
        path = root / "raw.jsonl"
        path.write_text(json.dumps(raw()) + "\n" + json.dumps(raw("overflow", True)) + "\n")
        return path

    def test_preserves_tokens_stop_and_parent_without_arrivals(self):
        record = extract_request(raw())
        self.assertEqual(record["input_token_ids"], [1, 2])
        self.assertEqual(record["output_token_ids"], [3, 4])
        self.assertTrue(record["output_length_matches_request"])
        self.assertEqual(record["parent_session_id"], "root-session")
        self.assertNotIn("arrival_ns", record)

    def test_missing_tokens_and_overflow_are_not_synthesized_or_dropped(self):
        row = raw(overflow=True)
        row["responses"] = []
        record = extract_request(row)
        self.assertIsNone(record["input_token_ids"])
        self.assertFalse(record["native_tokens_complete"])
        self.assertTrue(record["context_overflow_skip"])
        self.assertTrue(record["failed_or_cancelled"])

    def test_invalid_native_token_id_rejected(self):
        row = raw()
        row["responses"][0]["packets"][0]["value"] = json.dumps({"prompt_token_ids": [True]})
        with self.assertRaisesRegex(ValueError, "uint32"):
            extract_request(row)

    def test_live_join_handoff_and_raw_overflow_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = audit(root, [self.fixture(root)], root / "out")
            self.assertEqual(result["counts"]["requests"], 2)
            self.assertEqual(result["counts"]["context_overflow_skip"], 1)
            self.assertFalse(result["scoring_eligible"])
            self.assertFalse(result["measured_arrivals_used_for_replay"])

    def test_duplicate_and_truncated_raw_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit(root, [path, path], root / "out")
            path.write_text('{"metadata":')
            with self.assertRaisesRegex(ValueError, "truncated"):
                audit(root, [path], root / "out")

    def test_duplicate_phase_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            events = root / "events-1.jsonl"
            rows = events.read_text().splitlines()
            events.write_text("\n".join([rows[0], *rows]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate phase boundaries"):
                audit(root, [path], root / "out")

    def test_changed_handoff_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            events = root / "events-1.jsonl"
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            rows[1]["trajectories"][0]["snapshot"]["states"][0]["next_turn_index"] = 0
            events.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "changed"):
                audit(root, [path], root / "out")


if __name__ == "__main__":
    unittest.main()
