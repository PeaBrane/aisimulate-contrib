# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from reconcile_pro_cache import reconcile, reconcile_pd


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


class CacheReceiptTests(unittest.TestCase):
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
        self.assertEqual(prefill["replay_kv_cache"]["capacity"]["blocks"], 9000)
        self.assertEqual(decode["replay_kv_cache"]["capacity"]["blocks"], 10000)
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


if __name__ == "__main__":
    unittest.main()
