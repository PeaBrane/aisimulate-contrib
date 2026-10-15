# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for collected token bursts and acceptance-window accounting."""

import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acceptance import measured_acceptance
from campaign import cells
from collect_fixed import one_request
from freeze_acceptance import freeze
from pd_driver import load_role_receipts, selected_windows
from runtime_receipt import DsparkReceiptWorker, primitive
from serve_role import main as role_main
from serve_smoke import main as serve_main
from serve_smoke import server_command


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
    def test_explicit_windows_resume_without_repeating_completed_point(self):
        windows = [(1, 2), (1, 3), (4, 1), (4, 2), (4, 3)]
        self.assertEqual(selected_windows("initial", windows=windows), windows)
        self.assertEqual(selected_windows("initial"), [(1, 1), (1, 2), (1, 3), (4, 1), (4, 2), (4, 3)])
        self.assertEqual(selected_windows("smoke"), [(1, 1)])
        for kwargs in (
            {"windows": [(1, 1), (1, 1)]},
            {"windows": [(16, 1)]},
            {"windows": [(1, 4)]},
            {"windows": [(1, 2)], "concurrencies": [1]},
            {"windows": [(1, 2)], "repetitions": [2]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                selected_windows("initial", **kwargs)

    def test_role_preflight_resolves_command_without_gpu_or_server_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                "serve_role.py",
                "--model-key",
                "v4-pro",
                "--role",
                "prefill",
                "--k",
                "6",
                "--port",
                "8010",
                "--output",
                temp,
                "--preflight-only",
            ]
            with (
                patch("sys.argv", argv),
                patch("serve_role.server_command", return_value=(["server"], {}, {"path": "/model"})) as build,
                patch("serve_role.subprocess.run") as gpu,
                patch("serve_role.os.execvpe") as launch,
            ):
                role_main()
            build.assert_called_once()
            gpu.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(json.loads((Path(temp) / "prefill-preflight.json").read_text())["command"], ["server"])
            self.assertFalse((Path(temp) / "prefill-identity.json").exists())

    def test_source_pool_observer_is_explicit_and_rejects_prefix_caching(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                "serve_role.py",
                "--model-key",
                "v4-pro",
                "--role",
                "prefill",
                "--k",
                "6",
                "--port",
                "8010",
                "--output",
                temp,
                "--preflight-only",
                "--observe-source-pool",
            ]
            with (
                patch("sys.argv", argv),
                patch("serve_role.server_command", return_value=(["server"], {}, {})),
            ):
                role_main()
            receipt = json.loads((Path(temp) / "prefill-preflight.json").read_text())
            self.assertEqual(
                receipt["command"][-2:],
                [
                    "--scheduler-cls",
                    "source_pool_observer.SourcePoolObserverScheduler",
                ],
            )
            self.assertTrue(receipt["source_pool_observer"]["enabled"])
            self.assertEqual(receipt["runtime_environment"]["AISIM_POOL_OBSERVER_ROLE"], "prefill")
            self.assertEqual(len(receipt["source_pool_observer"]["source_pool_observer.py"]), 64)
            with (
                patch("sys.argv", argv + ["--prefix-cache"]),
                patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                role_main()

    def test_pd_prefill_memory_budget_and_reference_control_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                "serve_role.py",
                "--model-key",
                "v4-pro",
                "--role",
                "prefill",
                "--k",
                "6",
                "--port",
                "8010",
                "--output",
                temp,
                "--preflight-only",
                "--synthetic-al",
                "3.77",
                "--gpu-memory-utilization",
                "0.92",
            ]
            with (
                patch("sys.argv", argv),
                patch("serve_smoke.inspect_checkpoint", return_value={"path": "/pinned/model"}),
                patch.dict("os.environ", {"VLLM_NIXL_SIDE_CHANNEL_HOST": "private-p-node"}),
            ):
                role_main()
            identity = json.loads((Path(temp) / "prefill-preflight.json").read_text())
            command = identity["command"]
            self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.92")
            self.assertEqual(identity["synthetic_control"]["accepted_tokens"], 2.77)
            self.assertTrue(identity["synthetic_control"]["frozen_before_measurement"])
            config = json.loads(command[command.index("--speculative-config") + 1])
            self.assertEqual(config["synthetic_acceptance_length"], 3.77)

    def test_kimi_fp8_prefill_uses_compatible_explicit_backend(self):
        with (
            patch("serve_smoke.inspect_checkpoint", return_value={"path": "/pinned/model"}),
            patch("serve_smoke.checkpoint_path", return_value=Path("/pinned/draft")),
        ):
            command, _, _ = server_command("kimi-k3", 2, 8010)
        config = json.loads(command[command.index("--attention-config") + 1])
        self.assertTrue(config["use_prefill_query_quantization"])
        self.assertEqual(config["mla_prefill_backend"], "TRTLLM_RAGGED")
        self.assertEqual(command[command.index("--attention-backend") + 1], "FLASHINFER_MLA")
        draft = json.loads(command[command.index("--speculative-config") + 1])
        self.assertEqual(draft["attention_backend"], "FLASHINFER_MLA")

    def test_entrypoint_forwards_loader_and_internal_startup_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                "serve_smoke.py",
                "--model-key",
                "kimi-k3",
                "--k",
                "2",
                "--output",
                str(Path(temp) / "run"),
                "--load-strategy",
                "eager",
                "--startup-timeout",
                "6000",
                "--graph-policy",
                "runtime-default",
            ]
            with (
                patch("sys.argv", argv),
                patch("serve_smoke.server_command", side_effect=RuntimeError("stop before launch")) as build,
                self.assertRaisesRegex(RuntimeError, "stop before launch"),
            ):
                serve_main()
            self.assertEqual(build.call_args.kwargs["load_strategy"], "eager")
            self.assertEqual(build.call_args.kwargs["startup_timeout_s"], 6000)
            self.assertEqual(build.call_args.kwargs["graph_policy"], "runtime-default")

    def test_pd_launcher_preserves_fixed_k_and_private_rpc_topology(self):
        with (
            patch("serve_smoke.inspect_checkpoint", return_value={"path": "/pinned/model"}),
            patch.dict("os.environ", {"VLLM_NIXL_SIDE_CHANNEL_HOST": "private-p-node"}),
        ):
            for role, tp in (("prefill", "1"), ("decode", "8")):
                command, env, _ = server_command("v4-pro", 6, 8010, role=role, graph_policy="runtime-default")
                self.assertEqual(command[command.index("--tensor-parallel-size") + 1], tp)
                self.assertEqual(command[command.index("--host") + 1], "private-p-node")
                self.assertEqual(env["VLLM_SERVER_DEV_MODE"], "1")
                self.assertEqual("--enable-expert-parallel" in command, role == "prefill")
                if role == "prefill":
                    self.assertEqual(command[command.index("--data-parallel-size") + 1], "8")
                spec = json.loads(command[command.index("--speculative-config") + 1])
                self.assertEqual(spec["num_speculative_tokens"], 6)
                self.assertFalse(spec["enable_adaptive_verification"])
                self.assertNotIn("--enforce-eager", command)
            command, _, _ = server_command("v4-pro", 6, 8010)
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        with (
            patch("serve_smoke.inspect_checkpoint", return_value={"path": "/pinned/model"}),
            patch.dict("os.environ", {"VLLM_NIXL_SIDE_CHANNEL_HOST": "0.0.0.0"}),
            self.assertRaisesRegex(ValueError, "private role host"),
        ):
            server_command("v4-pro", 6, 8010, role="prefill")

    def test_pd_cache_receipts_keep_all_dp_workers_and_reject_missing_or_duplicate_ranks(self):
        @dataclasses.dataclass
        class Cache:
            num_blocks: int = 100
            transfer_group_ids: tuple = (0, 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dp in range(8):
                worker = SimpleNamespace(
                    rank=0,
                    local_rank=dp,
                    parallel_config=SimpleNamespace(
                        data_parallel_index=dp,
                        tensor_parallel_size=1,
                        data_parallel_size=8,
                        pipeline_parallel_size=1,
                        enable_expert_parallel=True,
                    ),
                    model_runner=SimpleNamespace(kv_cache_config=Cache(), get_kv_cache_spec=lambda: {"layer": Cache()}),
                )
                DsparkReceiptWorker.dspark_campaign_receipt(worker, str(root))
            self.assertEqual(len(load_role_receipts(root, "v4-pro", "prefill")), 8)
            self.assertEqual(list(root.glob("*.tmp")), [])
            last = root / "dp-7-tp-0.json"
            last.unlink()
            with self.assertRaisesRegex(ValueError, "missing or duplicates"):
                load_role_receipts(root, "v4-pro", "prefill")
            last.write_text((root / "dp-0-tp-0.json").read_text())
            with self.assertRaisesRegex(ValueError, "missing or duplicates"):
                load_role_receipts(root, "v4-pro", "prefill")

    def test_cache_receipt_preserves_computed_page_padding(self):
        @dataclasses.dataclass
        class PaddedState:
            block_size: int = 1536
            page_size_padded: int = 4096

            @property
            def page_size_bytes(self):
                return self.page_size_padded

        result = primitive(PaddedState())
        self.assertEqual(result["type"], "PaddedState")
        self.assertEqual(result["page_size_bytes"], 4096)
        self.assertEqual(result["block_size"], 1536)
        with self.assertRaisesRegex(TypeError, "unsupported cache receipt"):
            primitive(object())

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
