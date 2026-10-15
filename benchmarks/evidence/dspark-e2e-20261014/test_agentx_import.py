# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original small CPU fixtures for exact prompts and authored dependency lowering."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from agentx_capture import PRODUCER
from agentx_import import bind_sampled, lower, lower_dependencies, materialize_raw_record
from test_agentx_capture import raw

from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper import BackendDeploymentSpec, ReplaySpec


def templates(mode="fork"):
    return {
        "root": {
            "replay_scope_id": "scope",
            "agent_depth": 0,
            "turns": [
                {"timestamp_ms": 0, "api_time_ms": 10, "branch_ids": ["b"]},
                {
                    "timestamp_ms": 20,
                    "api_time_ms": 5,
                    "delay_ms": 10,
                    "prerequisites": [{"kind": "spawn_join", "branch_id": "b"}],
                },
            ],
            "branches": [
                {
                    "branch_id": "b",
                    "mode": mode,
                    "dispatch_timing": "post",
                    "start_timestamp_ms": 2,
                    "child_conversation_ids": ["child"],
                }
            ],
        },
        "child": {
            "parent_conversation_id": "root",
            "agent_depth": 1,
            "turns": [{"timestamp_ms": 4 if mode == "spawn" else 12, "api_time_ms": 5}],
        },
    }


def fixture(folder):
    tree = templates()
    (folder / "tree.json").write_text(json.dumps(tree))
    event = {
        "phase": "warmup",
        "event": "setup",
        "producer_revision": PRODUCER,
        "templates": {"root": "tree.json"},
        "session_markers": {"actual-root": "marker"},
        "trajectories": [
            {
                "conversation_id": "root",
                "snapshot": {
                    "t_star_ms": 5,
                    "states": [{"conversation_id": "root", "x_correlation_id": "actual-root"}],
                },
            }
        ],
    }
    (folder / "events-1.jsonl").write_text(json.dumps(event) + "\n")
    provenance = dict(
        producer_revision=PRODUCER,
        corpus_sha256="original-fixture-corpus",
        runtime_image_digest="original-fixture-runtime",
        formatter_sha256="original-fixture-formatter",
        encoder_sha256="original-fixture-encoder",
        tokenizer_revision="original-fixture-tokenizer",
    )
    materializations = [
        {
            **{key: value for key, value in provenance.items() if key not in ("producer_revision", "corpus_sha256")},
            "lane_id": 0,
            "conversation_id": cid,
            "turn_index": index,
            "session_id": "actual-" + cid,
            "root_session_id": "actual-root",
            "input_token_ids": [1, 2, 3, 4, 20 + index],
            "input_identity": "native_formatter",
            "payload_sha256": "original-payload",
            "cache_bust_marker": "marker",
            "cache_bust_target": "first_turn_prefix",
            "output_length": 2,
            "output_length_basis": "observed_usage",
            "source": {},
        }
        for cid, meta in tree.items()
        for index in range(len(meta["turns"]))
    ]
    return materializations, provenance


class ImportTests(unittest.TestCase):
    def test_fork_join_delays_remain_independent(self):
        gates = lower_dependencies(templates())
        self.assertEqual(gates["child", 0][0]["trigger"], "completion")
        self.assertEqual(gates["child", 0][0]["delay_ms"], 0)
        join = gates["root", 1]
        self.assertEqual([(edge["relation"], edge["delay_ms"]) for edge in join], [("sequence", 10), ("join", 0)])

    def test_overlapping_spawn_uses_dispatch_clock(self):
        edge = lower_dependencies(templates("spawn"))["child", 0][0]
        self.assertEqual((edge["trigger"], edge["delay_ms"]), ("dispatch", 4))

    def test_normal_payload_preserves_unknown_outputs(self):
        record = raw()
        record["cache_bust_marker"] = "marker"
        record["cache_bust_target"] = "first_turn_prefix"
        record["payload"]["messages"][0]["content"] = "marker original fixture"
        record["responses"] = [
            {
                "packets": [
                    {"name": "data", "value": json.dumps({"usage": {"prompt_tokens": 2, "completion_tokens": 2}})}
                ]
            }
        ]
        result = materialize_raw_record(record, [1, 2], "formatter", "image")
        self.assertEqual(result["input_token_ids"], [1, 2])
        self.assertEqual(result["output_length"], 2)
        self.assertIsNone(result["output_token_ids"])
        with self.assertRaisesRegex(ValueError, "server usage"):
            materialize_raw_record(record, [1], "formatter", "image")
        record["payload"]["messages"][0]["content"] = "missing"
        with self.assertRaisesRegex(ValueError, "cache marker"):
            materialize_raw_record(record, [1, 2], "formatter", "image")

    def test_complete_selected_tree_keeps_source_identity_without_arrivals(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            materializations, provenance = fixture(folder)
            header, rows, manifest = lower(folder, materializations, provenance, "fixture-model", block_size=4)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all("output_token_ids" not in row for row in rows))
            self.assertEqual([row["not_before_ms"] for row in rows], [0, 20, 12])
            self.assertFalse(manifest["scoring_eligible"])
            self.assertEqual(manifest["input_tokens"], 15)
            self.assertNotEqual(header["source"]["digest"], provenance["corpus_sha256"])
            identity = {
                key: manifest[key] for key in ("source", "block_size", "node_count", "play_count", "source_models")
            }
            identity["graph_digest"] = "native-only-fixture-digest"
            options = bind_sampled(manifest, identity)
            self.assertEqual(options["sampled"]["lanes"][0]["t_star_ms"], 5)
            self.assertEqual(
                set(options["sampled"]["lanes"][0]["session_ids"].values()), {"actual-root", "actual-child"}
            )
            identity["node_count"] = 4
            with self.assertRaisesRegex(ValueError, "node_count"):
                bind_sampled(manifest, identity)

    def test_lowered_tree_runs_through_canonical_native_graph_and_cut_api(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            materializations, provenance = fixture(folder)
            header, rows, manifest = lower(folder, materializations, provenance, "fixture-model", block_size=4)
            path = folder / "lowered.jsonl"
            path.write_text("\n".join(map(json.dumps, [header, *rows])) + "\n")
            deployment = BackendDeploymentSpec(
                deployment_mode="agg",
                backend="vllm",
                backend_version="fixture",
                num_workers=1,
                agg_engine_args=dict(
                    worker_type="aggregated",
                    engine_type="vllm",
                    aic_backend="vllm",
                    aic_model_path="fixture-model",
                    aic_system="fixture",
                    aic_tp_size=1,
                    block_size=4,
                    num_gpu_blocks=64,
                    timing_model=dict(type="fixed", prefill_ms=2.0, decode_ms=1.0),
                ),
            )
            workload = dict(
                source_type="trace",
                load_type="trace_timestamps",
                trace_path=str(path),
                trace_format="agentic_mooncake",
                agentic_lanes=1,
            )
            runner = EngineReplayRunnerFactory().create(0)
            try:
                full = runner.run(
                    ReplaySpec(backend_deployment=deployment, workload=workload, goal={"target": "throughput"})
                )
                self.assertEqual(full.metrics["completed_requests"], 3)
                sampled = {**workload, "agentic_snapshot": {"seed": 42}}
                baseline = runner.run(
                    ReplaySpec(backend_deployment=deployment, workload=sampled, goal={"target": "throughput"})
                )
                options = bind_sampled(manifest, baseline.metadata["agentic_snapshots"][0]["graph"])
                replay = runner.run(
                    ReplaySpec(
                        backend_deployment=deployment,
                        workload={**sampled, "agentic_snapshot": options},
                        goal={"target": "throughput"},
                    )
                )
                snapshot = replay.metadata["agentic_snapshots"][0]
                self.assertEqual(snapshot["t_star_ms"], 5)
                self.assertEqual(snapshot["sampled"], options["sampled"]["lanes"][0])
                self.assertTrue(
                    all(record["identity"]["conversation_id"].startswith("actual-") for record in snapshot["requests"])
                )
            finally:
                runner.close()

    def test_missing_turns_duplicate_identity_and_memory_bound_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows, provenance = fixture(folder)
            for altered, message in [(rows[:-1], "missing exact"), ([*rows, rows[0]], "duplicate")]:
                with self.assertRaisesRegex(ValueError, message):
                    lower(folder, altered, provenance, "fixture-model")
            with self.assertRaisesRegex(ValueError, "token budget"):
                lower(folder, rows, provenance, "fixture-model", max_input_tokens=14)
            for key in ("formatter_sha256", "encoder_sha256", "tokenizer_revision", "runtime_image_digest"):
                altered = copy.deepcopy(rows)
                altered[0][key] = "different"
                with self.assertRaisesRegex(ValueError, key):
                    lower(folder, altered, provenance, "fixture-model")
            altered = copy.deepcopy(rows)
            altered[-1]["root_session_id"] = "recycled-root"
            with self.assertRaisesRegex(ValueError, "recycled"):
                lower(folder, altered, provenance, "fixture-model")

    def test_unknown_branch_and_nonfinite_clocks_rejected(self):
        tree = templates()
        tree["root"]["turns"][1]["delay_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            lower_dependencies(tree)
        tree = templates()
        tree["root"]["branches"][0]["child_conversation_ids"] = ["absent"]
        with self.assertRaisesRegex(ValueError, "children"):
            lower_dependencies(tree)


if __name__ == "__main__":
    unittest.main()
