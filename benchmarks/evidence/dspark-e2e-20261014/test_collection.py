# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for collected token bursts and acceptance-window accounting."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acceptance import measured_acceptance
from campaign import cells
from collect_fixed import one_request
from freeze_acceptance import freeze


def snapshot(rounds, proposed, accepted, positions):
    names = ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]
    text = "\n".join(
        f'vllm:spec_decode_{name}_total{{engine="0"}} {value}'
        for name, value in zip(names, [rounds, proposed, accepted], strict=True)
    )
    text += "\n" + "\n".join(
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}} {n}'
        for i, n in enumerate(positions)
    )
    return {"worker": text}


class CollectionTests(unittest.TestCase):
    def test_only_complete_separate_real_pilot_can_freeze_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = {
                "phase": "pilot",
                "model": {"revision": "test-revision"},
                "cell": {"acceptance": "real", "model_key": "v4-pro", "num_speculative_tokens": 3},
            }
            summary = {
                "status": "success",
                "expected_requests": 128,
                "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "metrics": {"completed_requests": 128, "failed_requests": 0},
            }
            (root / "identity.json").write_text(json.dumps(identity))
            (root / "summary.json").write_text(json.dumps(summary))
            (root / "acceptance.json").write_text(json.dumps({"accepted_tokens": 1.75}))
            result = freeze(root)
            self.assertEqual(result["synthetic_acceptance_length"], 2.75)
            self.assertEqual(result["accepted_tokens"], 1.75)
            self.assertTrue(result["frozen_before_measurement"])
            summary["metrics"]["completed_requests"] = 127
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "128 requests"):
                freeze(root)

    def test_bursts_use_token_count_not_number_of_chunks(self):
        records = [
            {"choices": [{"token_ids": [1, 2, 3]}]},
            {"choices": [{"token_ids": [4, 5]}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
        ]
        response = io.BytesIO(b"".join(b"data: " + json.dumps(row).encode() + b"\n\n" for row in records))
        with (
            patch("collect_fixed.fetch", return_value=response),
            patch(
                "collect_fixed.time.perf_counter_ns",
                side_effect=[0, 10_000_000, 30_000_000, 31_000_000],
            ),
        ):
            row = one_request("http://example", "model", [9, 8, 7], 5, "r", 0, 1.0)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["ttft_ms"], 10)
        self.assertEqual(row["tpot_ms"], 5)
        self.assertEqual(len(row["output_token_ids"]), 5)

    def test_acceptance_uses_deltas_and_converts_cumulative_to_conditional(self):
        before = snapshot(100, 300, 200, [100, 70, 30])
        after = snapshot(110, 330, 218, [108, 76, 34])
        measured = measured_acceptance(before, after, 3)
        self.assertEqual(measured["accepted_tokens"], 1.8)
        self.assertEqual(measured["acceptance_rates"], [0.8, 0.75, 2 / 3])
        self.assertEqual(measured["committed_tokens_per_round"], 2.8)

    def test_counter_reset_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError, "reset"):
            measured_acceptance(snapshot(10, 30, 10, [10, 0, 0]), snapshot(1, 3, 1, [1, 0, 0]), 3)

    def test_matrix_preserves_asymmetric_pd_and_mtp_anchor(self):
        rows = cells()
        self.assertEqual(len(rows), 196)
        self.assertEqual(sum(row["stage"] == "initial" for row in rows), 48)
        self.assertEqual(sum(row["expected_requests"] * 3 for row in rows), 153600)
        row = next(row for row in rows if row["cell_id"] == "v4-pro-disagg-k6-i8192-o1024-c8-synthetic")
        self.assertEqual(row["prefill"], {"tp": 1, "attention_dp": 8, "ep": 8})
        self.assertEqual(row["decode"], {"tp": 8, "attention_dp": 1, "ep": 1})


if __name__ == "__main__":
    unittest.main()
