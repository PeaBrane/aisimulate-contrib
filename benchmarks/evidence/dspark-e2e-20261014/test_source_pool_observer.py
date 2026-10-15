# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU behavior tests; the installed-image preflight checks real vLLM imports."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class FakePool:
    def __init__(self, size=100):
        self.num_gpu_blocks, self.free = size, size - 1

    def get_num_free_blocks(self):
        return self.free

    def get_new_blocks(self, count):
        if count > self.free:
            raise ValueError("no free blocks")
        self.free -= count
        return [object() for _ in range(count)]

    def touch(self, blocks):
        self.free -= len(blocks)


class FakeAsyncScheduler:
    def __init__(self, config, manager, cache):
        self.vllm_config, self.kv_cache_manager, self.kv_cache_config = config, manager, cache
        self.async_state = "original async initialization"

    def _preempt_request(self, request, *, drop_stale_output=False):
        return request, drop_stale_output

    def shutdown(self):
        return "original shutdown"


def load_observer():
    modules = {}
    for name, attribute, value in (
        ("vllm.v1.core.block_pool", "BlockPool", FakePool),
        ("vllm.v1.core.sched.async_scheduler", "AsyncScheduler", FakeAsyncScheduler),
    ):
        modules[name] = ModuleType(name)
        setattr(modules[name], attribute, value)
    spec = importlib.util.spec_from_file_location(
        "source_pool_observer_tested", Path(__file__).with_name("source_pool_observer.py")
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SourcePoolObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_observer()

    def test_peak_includes_null_detached_blocks_and_touch(self):
        with tempfile.TemporaryDirectory() as root:
            pool = FakePool()
            path = Path(root) / "p.json"
            observer = self.module.PoolObservation(pool, path, {})
            observer.wrap(pool, "get_new_blocks")
            observer.wrap(pool, "touch")
            self.assertEqual(json.loads(path.read_text())["peak_used_blocks"], 1)
            previous = path.read_bytes()
            with patch.object(observer, "write", side_effect=AssertionError("hot-path filesystem write")):
                detached = pool.get_new_blocks(5)
                # Removing a request does not release its deferred physical blocks.
                self.assertEqual(len(detached), 5)
                pool.get_new_blocks(2)
                pool.touch([object(), object()])
                self.assertEqual(observer.peak, 10)
                pool.free += 6
                pool.get_new_blocks(1)
            self.assertEqual(path.read_bytes(), previous)
            observer.closed = True
            observer.write("shutdown")
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["peak_used_blocks"], 10)
            self.assertEqual(receipt["current_used_blocks"], 5)
            self.assertEqual(receipt["pool_mutation_calls"], 4)
            self.assertTrue(receipt["complete"])

    def test_admission_denial_with_free_capacity_and_exception_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            pool = FakePool()
            path = Path(root) / "p.json"
            observer = self.module.PoolObservation(pool, path, {})
            marker = object()
            error = ValueError("original admission failure")

            def allocate_slots(request, *, deny=False, fail=False):
                self.assertIs(request, marker)
                if fail:
                    raise error
                return None if deny else marker

            manager = SimpleNamespace(allocate_slots=allocate_slots)
            observer.wrap(manager, "allocate_slots", admission=True)
            self.assertIs(manager.allocate_slots(marker), marker)
            self.assertIsNone(manager.allocate_slots(marker, deny=True))
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["allocation_denials"], 1)
            self.assertEqual(receipt["current_used_blocks"], 1)
            with self.assertRaises(ValueError) as raised:
                manager.allocate_slots(marker, fail=True)
            self.assertIs(raised.exception, error)
            self.assertEqual(json.loads(path.read_text())["observation_or_allocation_errors"], 1)

    def test_async_inheritance_and_original_calls_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            pool = FakePool()
            manager = SimpleNamespace(
                block_pool=pool,
                coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_pool=pool)]),
                allocate_slots=lambda *args, **kwargs: "original allocation",
            )
            config = SimpleNamespace(
                scheduler_config=SimpleNamespace(async_scheduling=True),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
                parallel_config=SimpleNamespace(
                    pipeline_parallel_size=1, data_parallel_index=3, tensor_parallel_size=1, data_parallel_size=8
                ),
                kv_transfer_config=SimpleNamespace(kv_connector="NixlConnector", kv_connector_extra_config={}),
            )
            cache = SimpleNamespace(
                kv_cache_groups=[SimpleNamespace(host_resident=False)], hisparse_shared_host_pool=False
            )
            env = {
                "AISIM_POOL_OBSERVER_DIR": root,
                "AISIM_POOL_OBSERVER_ROLE": "prefill",
                "AISIM_POOL_OBSERVER_RUN_ID": "unit",
            }
            with patch.dict(os.environ, env), patch.object(self.module, "verify_sources", return_value={}):
                scheduler = self.module.SourcePoolObserverScheduler(config, manager, cache)
                self.assertIsInstance(scheduler, FakeAsyncScheduler)
                self.assertEqual(scheduler.async_state, "original async initialization")
                self.assertEqual(manager.allocate_slots("request"), "original allocation")
                self.assertEqual(scheduler._preempt_request("request", drop_stale_output=True), ("request", True))
                self.assertEqual(scheduler.shutdown(), "original shutdown")
                receipt = json.loads((Path(root) / "prefill-dp-3.json").read_text())
                self.assertEqual(receipt["preemptions"], 1)
                self.assertEqual(receipt["allocation_attempts"], 1)
                self.assertTrue(receipt["complete"])
                config.cache_config.enable_prefix_caching = True
                with self.assertRaisesRegex(ValueError, "prefix OFF"):
                    self.module.SourcePoolObserverScheduler(config, manager, cache)
                config.cache_config.enable_prefix_caching = False
                config.scheduler_config.async_scheduling = False
                with self.assertRaisesRegex(ValueError, "async PP1"):
                    self.module.SourcePoolObserverScheduler(config, manager, cache)

    def test_changed_installed_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            changed = Path(root) / "changed.py"
            changed.write_text("# different runtime\n")
            module = ModuleType("changed_runtime")
            module.__file__ = str(changed)
            with (
                patch.object(self.module.importlib, "import_module", return_value=module),
                self.assertRaisesRegex(ValueError, "runtime source mismatch"),
            ):
                self.module.verify_sources()

    def test_existing_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "p.json"
            path.write_text("existing run")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.module.PoolObservation(FakePool(), path, {})
            self.assertEqual(path.read_text(), "existing run")


if __name__ == "__main__":
    unittest.main()
