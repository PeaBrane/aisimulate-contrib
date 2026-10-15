# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reconcile_pro_cache import read_pool_evidence, reconcile, reconcile_pd

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig


def receipt():
    groups, tensors = [], []
    for index, (block, window, count) in enumerate(
        [(64, 128, 32), (64, 128, 32), (256, None, 91), (8, 128, 31), (4, 8, 60)]
    ):
        members = {
            f"g{index}.layer{layer}": {
                "type": "MLAAttentionSpec" if window is None else "SlidingWindowMLASpec",
                "block_size": block,
                "sliding_window": window,
                "page_size_bytes": 1,
            }
            for layer in range(count)
        }
        groups.append(
            {"host_resident": False, "layer_names": list(members), "kv_cache_spec": {"kv_cache_specs": members}}
        )
        tensors.append(
            {
                "block_stride": 1025,
                "size": 10250000,
                "host_resident": False,
                "offset": 0,
                "layer_stride": 1,
                "layers": list(members),
            }
        )
    config = {
        "kv_cache_groups": groups,
        "kv_cache_tensors": tensors,
        "kv_cache_layout": "BLHNC",
        "kv_tp_replicas": 8,
        "hisparse_host_num_blocks": None,
        "num_blocks": 10000,
    }
    return {
        "results": [
            {"rank": rank, "kv_cache_config": copy.deepcopy(config), "available_kv_cache_memory_bytes": 10250000}
            for rank in range(8)
        ]
    }


def run(value, concurrency=1, input_length=1024):
    return reconcile(value, k=6, concurrency=concurrency, input_length=input_length, output_length=1024)


def pd_receipt():
    result = {}
    for role, dp, tp in (("prefill", 8, 1), ("decode", 1, 8)):
        ranks = receipt()["results"]
        for index, rank in enumerate(ranks):
            config = rank["kv_cache_config"]
            config["kv_tp_replicas"] = tp
            for group_index in (0, 1):
                group = config["kv_cache_groups"][group_index]
                names = [f"model.layers.{2 * i + group_index}.attn.swa_cache" for i in range(32)]
                group["kv_cache_spec"]["kv_cache_specs"] = dict(
                    zip(names, group["kv_cache_spec"]["kv_cache_specs"].values(), strict=True)
                )
                group["layer_names"] = names
                config["kv_cache_tensors"][group_index]["layers"] = names
            rank.update(
                schema="dspark-native-cache-v1",
                data_parallel_index=index if dp == 8 else 0,
                tensor_parallel_rank=index if tp == 8 else 0,
                parallelism={
                    "tensor_parallel_size": tp,
                    "data_parallel_size": dp,
                    "pipeline_parallel_size": 1,
                    "enable_expert_parallel": role == "prefill",
                },
                transfer_group_ids=list(range(5)),
                layer_specs={
                    name: spec
                    for group in config["kv_cache_groups"]
                    for name, spec in group["kv_cache_spec"]["kv_cache_specs"].items()
                },
            )
        result[role] = ranks
    return result


def run_pd(value, concurrency=4):
    return reconcile_pd(value, k=6, concurrency=concurrency, input_length=1024, output_length=1024)


def pool_evidence():
    root = Path(__file__).parent
    rows = {}
    for role, dp, tp in (("prefill", 8, 1), ("decode", 1, 8)):
        rows[role] = [
            {
                "schema": "dspark-native-pool-observation-v1",
                "run_id": "original-test-serving-run",
                "role": role,
                "data_parallel_index": rank,
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
                "async_scheduler": True,
                "prefix_caching": False,
                "initial_null_block_included": True,
                "runtime_revision": "ac9126e58aa7bbab1856ba6593ba4d5003fea516",
                "source_hashes": json.loads((root / "source_pool_sources.json").read_text()),
                "observer_sha256": hashlib.sha256((root / "source_pool_observer.py").read_bytes()).hexdigest(),
                "num_blocks": 10000,
                "complete": True,
                "reason": "shutdown",
                "started_ns": 1,
                "updated_ns": 100,
                "peak_used_blocks": 500,
                "current_used_blocks": 1,
                "pool_mutation_calls": 300,
                "allocation_attempts": 150,
                "allocation_denials": 0,
                "observation_or_allocation_errors": 0,
                "preemptions": 0,
            }
            for rank in range(dp)
        ]
    return {
        "schema": "dspark-native-pool-evidence-v1",
        "serving_run_id": "original-test-serving-run",
        "measurement_identity_sha256": "d" * 64,
        "window": {"start_unix_ns": 10, "end_unix_ns": 90, "source": "window.json", "sha256": "a" * 64},
        "roles": rows,
    }


class CacheReceiptTests(unittest.TestCase):
    def test_pool_observation_bounds_native_holds_without_changing_replay_bound(self):
        evidence = pool_evidence()
        result = reconcile_pd(
            pd_receipt(),
            k=6,
            concurrency=4,
            input_length=1024,
            output_length=1024,
            pool_observation_evidence=evidence,
        )
        self.assertTrue(result["native_capacity_proven_nonbinding"])
        self.assertTrue(result["replay_capacity_proven_nonbinding"])
        self.assertFalse(result["source_residency"]["native_bound_established"])
        pool = result["roles"]["prefill"]["rank_pools"][0]
        self.assertEqual(pool["native_observed_peak_blocks"], 500)
        self.assertEqual(pool["static_rolling_reserve_blocks"], 4 * 431)
        self.assertIsNone(pool["native_resident_requests_bound"])
        large = reconcile_pd(
            pd_receipt(),
            k=6,
            concurrency=64,
            input_length=8192,
            output_length=1024,
            pool_observation_evidence=evidence,
        )
        self.assertTrue(large["native_capacity_proven_nonbinding"])
        self.assertFalse(large["replay_capacity_proven_nonbinding"])

    def test_pool_admission_denials_preemption_or_full_pool_prevent_qualification(self):
        for field, value in (
            ("allocation_denials", 1),
            ("preemptions", 1),
            ("observation_or_allocation_errors", 1),
            ("peak_used_blocks", 10000),
        ):
            with self.subTest(field=field):
                evidence = pool_evidence()
                evidence["roles"]["prefill"][3][field] = value
                result = reconcile_pd(
                    pd_receipt(),
                    k=6,
                    concurrency=4,
                    input_length=1024,
                    output_length=1024,
                    pool_observation_evidence=evidence,
                )
                self.assertTrue(result["replay_capacity_proven_nonbinding"])
                self.assertFalse(result["native_capacity_proven_nonbinding"])

    def test_pool_observation_rejects_incomplete_stale_or_mismatched_evidence(self):
        mutations = (
            ("complete", False),
            ("started_ns", 11),
            ("updated_ns", 89),
            ("num_blocks", 10001),
            ("run_id", "other-run"),
            ("observer_sha256", "b" * 64),
            ("source_hashes", {}),
            ("prefix_caching", True),
            ("data_parallel_index", 1),
            ("peak_used_blocks", -1),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                evidence = pool_evidence()
                evidence["roles"]["prefill"][0][field] = value
                with self.assertRaises(ValueError):
                    reconcile_pd(
                        pd_receipt(),
                        k=6,
                        concurrency=4,
                        input_length=1024,
                        output_length=1024,
                        pool_observation_evidence=evidence,
                    )

    def test_pool_window_file_and_runtime_receipt_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = pool_evidence()
            evidence["runtime_cache_receipt_sha256"] = "c" * 64
            window = {"serving_run_id": evidence["serving_run_id"], "start_unix_ns": 10, "end_unix_ns": 90}
            window["identity_sha256"] = evidence["measurement_identity_sha256"]
            raw = json.dumps(window).encode()
            (root / "window.json").write_bytes(raw)
            evidence["window"]["sha256"] = hashlib.sha256(raw).hexdigest()
            path = root / "pool.json"
            path.write_text(json.dumps(evidence))
            self.assertEqual(read_pool_evidence(path, "c" * 64), evidence)
            with self.assertRaisesRegex(ValueError, "cache receipt hash"):
                read_pool_evidence(path, "d" * 64)
            (root / "window.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "window hash"):
                read_pool_evidence(path, "c" * 64)

    def test_aliases_are_not_summed_and_pool_blocks_are_preserved(self):
        result = run(receipt())
        self.assertEqual(result["physical_pool_bytes_per_rank"], 10250000)
        self.assertEqual(result["replay_kv_cache"]["capacity"]["blocks"], 10000)
        self.assertEqual(result["approximate_steady_state_blocks_per_request"], 31)
        self.assertEqual(result["representation_rounding_bytes_per_block"], 255)
        self.assertTrue(result["capacity_proven_nonbinding"])

    def test_transient_bound_includes_async_final_rounds_and_draft_lookahead(self):
        result = run(receipt())
        # Two PP1 in-flight rounds each retain all K+1 query positions, even
        # when the requested output would end partway through the round.
        self.assertEqual(result["conservative_retained_tokens_per_group"], 1024 + 1024 + 2 * 7 + 6)
        self.assertEqual(result["max_concurrent_batches_bound"], 2)
        self.assertEqual(result["conservative_peak_blocks"], 857)

    def test_incomplete_or_inconsistent_ranks_are_rejected(self):
        value = receipt()
        value["results"].pop()
        with self.assertRaisesRegex(ValueError, "eight unique"):
            run(value)
        value = receipt()
        value["results"][1]["kv_cache_config"]["num_blocks"] -= 1
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            run(value)

    def test_missing_physical_layer_and_overflow_are_rejected(self):
        for overflow in (False, True):
            value = receipt()
            for rank in value["results"]:
                tensor = rank["kv_cache_config"]["kv_cache_tensors"][0]
                if overflow:
                    tensor["offset"] = 1025
                else:
                    tensor["layers"].pop()
            with self.assertRaises(ValueError):
                run(value)

    def test_larger_workload_cannot_inherit_small_run_capacity_qualification(self):
        result = run(receipt(), concurrency=64, input_length=8192)
        self.assertFalse(result["capacity_proven_nonbinding"])

    def test_pd_independent_source_pools_use_minimum_without_summing(self):
        value = pd_receipt()
        limiting = value["prefill"][3]
        limiting["kv_cache_config"]["num_blocks"] = 9000
        for tensor in limiting["kv_cache_config"]["kv_cache_tensors"]:
            tensor["size"] = 9000 * 1025
        result = run_pd(value)
        prefill, decode = result["roles"]["prefill"], result["roles"]["decode"]
        self.assertEqual(prefill["independent_request_pools"], 8)
        self.assertEqual(decode["independent_request_pools"], 1)
        self.assertEqual(prefill["replay_kv_cache"]["capacity"]["blocks"], 9000 - 431 * 4)
        self.assertEqual(decode["replay_kv_cache"]["capacity"]["blocks"], 10000 - 846 * 4)
        self.assertNotIn("state_cache", prefill["replay_kv_cache"])
        self.assertNotIn("state_cache", decode["replay_kv_cache"])
        self.assertEqual(prefill["physical_pool_bytes_across_role"], (7 * 10000 + 9000) * 1025)
        self.assertEqual(decode["physical_pool_bytes_across_role"], 8 * 10000 * 1025)
        for pool in prefill["rank_pools"]:
            self.assertEqual(pool["workload"]["concurrency"], 4)
            self.assertEqual(pool["workload"]["output_length"], 1)
        self.assertEqual(decode["rank_pools"][0]["workload"]["output_length"], 1024)
        self.assertIsNone(result["transport"]["request_wire_bytes"])
        self.assertFalse(result["transport"]["bandwidth_qualified"])

    def test_pd_missing_duplicate_or_wrong_topology_is_rejected(self):
        for mutation in ("missing", "duplicate", "topology"):
            with self.subTest(mutation=mutation):
                value = pd_receipt()
                if mutation == "missing":
                    value["prefill"].pop()
                elif mutation == "duplicate":
                    value["prefill"][7]["data_parallel_index"] = 0
                else:
                    value["prefill"][0]["parallelism"]["tensor_parallel_size"] = 8
                with self.assertRaises(ValueError):
                    run_pd(value)

    def test_pd_target_only_transfer_and_missing_draft_are_rejected(self):
        for mutation in ("transfer", "draft", "shape"):
            with self.subTest(mutation=mutation):
                value = pd_receipt()
                rank = value["prefill"][0]
                if mutation == "transfer":
                    rank["transfer_group_ids"].pop()
                elif mutation == "draft":
                    del rank["layer_specs"]["model.layers.63.attn.swa_cache"]
                else:
                    rank["kv_cache_config"]["kv_tp_replicas"] = 8
                with self.assertRaises(ValueError):
                    run_pd(value)

    def test_pd_capacity_bound_does_not_divide_concurrency_by_dp(self):
        result = run_pd(pd_receipt(), concurrency=64)
        self.assertFalse(result["roles"]["prefill"]["capacity_proven_nonbinding"])
        self.assertFalse(result["capacity_proven_nonbinding"])
        self.assertIsNone(result["roles"]["prefill"]["replay_kv_cache"])
        self.assertIsNone(result["roles"]["decode"]["replay_kv_cache"])

    def test_pd_transfer_bound_includes_all_groups_once_without_fanout_multiplier(self):
        result = run_pd(pd_receipt())
        groups = result["transport"]["groups"]
        self.assertEqual([g["block_count_bound"] for g in groups], [3, 3, 5, 17, 3])
        expected = 32 * 3 * 2 + 91 * 5 + 31 * 17 + 60 * 3
        self.assertEqual(result["transport"]["per_reader_page_payload_bound_bytes"], expected)
        registered = (3 + 3 + 5 + 17 + 3) * 1025
        self.assertEqual(result["transport"]["per_reader_registered_region_bound_bytes"], registered)
        self.assertGreater(registered, expected)
        self.assertEqual(result["replay_kv_transfer"]["bytes_per_token"], (registered + 1023) // 1024)
        self.assertIsNone(result["replay_kv_transfer"]["bandwidth_gb_per_second"])
        self.assertEqual(result["replay_kv_transfer"]["timing_mode"], "full_prompt")
        names = [name for group in groups for name in group["layers"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("model.layers.63.attn.swa_cache", names)

    def test_pd_bandwidth_requires_explicit_evidence_without_claiming_accuracy(self):
        evidence = {"kind": "assumed", "gb_per_second": 50, "source": "test scenario; not measured"}
        result = reconcile_pd(
            pd_receipt(), k=6, concurrency=1, input_length=1024, output_length=1024, bandwidth_evidence=evidence
        )
        self.assertEqual(result["replay_kv_transfer"]["bandwidth_gb_per_second"], 50)
        self.assertFalse(result["transport"]["bandwidth_qualified"])
        with self.assertRaisesRegex(ValueError, "bandwidth evidence"):
            reconcile_pd(
                pd_receipt(),
                k=6,
                concurrency=1,
                input_length=1024,
                output_length=1024,
                bandwidth_evidence={"kind": "measured", "gb_per_second": 50},
            )

    def test_pd_native_source_bound_is_separate_from_client_concurrency(self):
        default = run_pd(pd_receipt())
        self.assertTrue(default["replay_capacity_proven_nonbinding"])
        self.assertFalse(default["native_capacity_proven_nonbinding"])
        self.assertFalse(default["source_residency"]["native_bound_established"])
        evidence = {
            "kind": "conservative",
            "max_resident_requests_per_pool": 10,
            "includes_active_and_finished_holds": True,
            "source": "test bound for complete run including warmup",
        }
        result = reconcile_pd(
            pd_receipt(),
            k=6,
            concurrency=4,
            input_length=1024,
            output_length=1024,
            source_residency_evidence=evidence,
        )
        self.assertTrue(result["native_capacity_proven_nonbinding"])
        self.assertTrue(result["replay_capacity_proven_nonbinding"])
        pool = result["roles"]["prefill"]["rank_pools"][0]
        self.assertEqual(pool["static_rolling_reserve_blocks"], 431 * 10)
        self.assertGreater(pool["native_conservative_peak_blocks"], pool["replay_peak_with_static_reserve_blocks"])
        evidence["max_resident_requests_per_pool"] = 50
        failed = reconcile_pd(
            pd_receipt(),
            k=6,
            concurrency=4,
            input_length=1024,
            output_length=1024,
            source_residency_evidence=evidence,
        )
        self.assertTrue(failed["capacity_proven_nonbinding"])  # Original C-request bound is retained.
        self.assertFalse(failed["native_capacity_proven_nonbinding"])
        self.assertFalse(failed["replay_capacity_proven_nonbinding"])
        self.assertIsNone(failed["roles"]["prefill"]["replay_kv_cache"])
        del evidence["includes_active_and_finished_holds"]
        with self.assertRaisesRegex(ValueError, "active and finished"):
            reconcile_pd(
                pd_receipt(),
                k=6,
                concurrency=4,
                input_length=1024,
                output_length=1024,
                source_residency_evidence=evidence,
            )

    def test_pd_storage_shape_mismatch_between_roles_is_rejected(self):
        value = pd_receipt()
        for rank in value["decode"]:
            specs = rank["kv_cache_config"]["kv_cache_groups"][0]["kv_cache_spec"]["kv_cache_specs"]
            next(iter(specs.values()))["page_size_bytes"] = 2
        with self.assertRaisesRegex(ValueError, "storage geometries differ"):
            run_pd(value)

    def test_pd_cumulative_issued_bounds_remain_per_dp_pool(self):
        evidence = {
            "kind": "conservative",
            "max_resident_requests_by_dp_rank": {str(i): i for i in range(8)},
            "includes_active_and_finished_holds": True,
            "source": "all issued requests since startup, including warmup, counted per DP engine",
        }
        result = reconcile_pd(
            pd_receipt(),
            k=6,
            concurrency=1,
            input_length=1024,
            output_length=1024,
            source_residency_evidence=evidence,
        )
        pools = result["roles"]["prefill"]["rank_pools"]
        self.assertEqual([p["native_resident_requests_bound"] for p in pools], list(range(8)))
        self.assertEqual([p["static_rolling_reserve_blocks"] for p in pools], [431 * max(1, i) for i in range(8)])
        self.assertTrue(result["native_capacity_proven_nonbinding"])
        del evidence["max_resident_requests_by_dp_rank"]["7"]
        with self.assertRaisesRegex(ValueError, "eight DP-rank"):
            reconcile_pd(
                pd_receipt(),
                k=6,
                concurrency=1,
                input_length=1024,
                output_length=1024,
                source_residency_evidence=evidence,
            )

    def test_pd_adapter_constructs_canonical_b0_r1_config_without_duplicate_transfer_state(self):
        result = run_pd(pd_receipt())
        base = CorePredictionConfig.from_yaml(Path(__file__).parent / "pro-initial-c1-r1/replay-config.yaml")
        payload = base.model_dump(mode="json", exclude_none=True)
        engine = payload["engine"]
        template = engine["workers"]["aggregated"]
        engine["mode"] = "disaggregated"
        engine["kv_transfer"] = result["replay_kv_transfer"]
        engine["workers"] = {}
        for role, tp, dp, ep in (("prefill", 1, 8, 8), ("decode", 8, 1, 1)):
            worker = copy.deepcopy(template)
            worker["parallelism"].update(tensor=tp, attention_data=dp, moe_expert=ep, moe_tensor=tp)
            worker["kv_cache"] = result["roles"][role]["replay_kv_cache"]
            engine["workers"][role] = worker
        config = CorePredictionConfig.model_validate(payload)
        spec = prediction_to_replay_spec(config)
        for role in ("prefill", "decode"):
            args = getattr(spec.backend_deployment, f"{role}_engine_args")
            self.assertEqual(args["kv_transfer_decode_recompute_tokens"], 1)
            self.assertNotIn("state_cache", args)
            self.assertNotIn("kv_transfer_draft_bytes_per_token", args)
            self.assertEqual(args["kv_transfer_bytes_per_token"], result["replay_kv_transfer"]["bytes_per_token"])
            self.assertEqual(args["num_gpu_blocks"], result["roles"][role]["replay_kv_cache"]["capacity"]["blocks"])
        # The recurrent-state safety guard remains intact.
        for role in ("prefill", "decode"):
            engine["workers"][role]["kv_cache"]["state_cache"] = {"bytes_per_request": 100}
        with self.assertRaisesRegex(ValueError, "state_cache requires decode_recompute_tokens"):
            CorePredictionConfig.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
