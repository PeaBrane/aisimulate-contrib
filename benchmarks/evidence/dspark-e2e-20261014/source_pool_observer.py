# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe pinned native physical cache occupancy without changing scheduling.

Original adapter for vLLM ac9126e58aa7bbab1856ba6593ba4d5003fea516,
v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator}.py and
v1/core/sched/{scheduler,async_scheduler}.py, Apache-2.0.
See COLLECTION.md and the canonical THIRD_PARTY_NOTICES.md.
"""

import functools
import hashlib
import importlib
import inspect
import json
import os
import time
from pathlib import Path

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def verify_sources():
    expected = json.loads(Path(__file__).with_name("source_pool_sources.json").read_text())
    for name, digest in expected.items():
        source = Path(inspect.getsourcefile(importlib.import_module(name)))
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"cache observer runtime source mismatch: {name}")
    return expected


class PoolObservation:
    """Constant-time observations after the original mutation returns."""

    def __init__(self, pool, output, identity):
        self.pool, self.output, self.identity = pool, output, identity
        self.peak = pool.num_gpu_blocks - pool.get_num_free_blocks()
        self.mutations = self.admissions = self.denials = self.errors = self.preemptions = 0
        self.started_ns = time.time_ns()
        self.closed = False
        if output.exists():
            raise ValueError(f"refusing to overwrite an existing cache observation: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.write("start")

    def write(self, reason):
        used = self.pool.num_gpu_blocks - self.pool.get_num_free_blocks()
        self.peak = max(self.peak, used)
        receipt = self.identity | {
            "schema": "dspark-native-pool-observation-v1",
            "started_ns": self.started_ns,
            "updated_ns": time.time_ns(),
            "pid": os.getpid(),
            "reason": reason,
            "complete": self.closed,
            "num_blocks": self.pool.num_gpu_blocks,
            "initial_null_block_included": True,
            "peak_used_blocks": self.peak,
            "current_used_blocks": used,
            "pool_mutation_calls": self.mutations,
            "allocation_attempts": self.admissions,
            "allocation_denials": self.denials,
            "observation_or_allocation_errors": self.errors,
            "preemptions": self.preemptions,
            "qualification": (
                "Observation only; receipt completeness and Replay capacity require separate reconciliation."
            ),
            "overhead": (
                "Python wrappers and O(1) in-memory counter/free-count reads; atomic JSON writes only at "
                "start, shutdown and invalidating failures/preemptions. No latency correction applied."
            ),
        }
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        temporary.replace(self.output)

    def wrap(self, owner, name, *, admission=False):
        original = getattr(owner, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if admission:
                self.admissions += 1
            else:
                self.mutations += 1
            try:
                result = original(*args, **kwargs)
            except Exception:
                self.errors += 1
                self.write("allocation_error")
                raise
            if admission and result is None:
                self.denials += 1
                self.write("allocation_denied")
            else:
                used = self.pool.num_gpu_blocks - self.pool.get_num_free_blocks()
                if used > self.peak:
                    self.peak = used
            return result

        setattr(owner, name, observed)


class SourcePoolObserverScheduler(AsyncScheduler):
    """Preserve the pinned async scheduler; observe its existing physical pool."""

    def __init__(self, *args, **kwargs):
        sources = verify_sources()
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        parallel = config.parallel_config
        transfer = config.kv_transfer_config
        manager = self.kv_cache_manager
        pool = manager.block_pool
        if (
            not config.scheduler_config.async_scheduling
            or config.cache_config.enable_prefix_caching
            or parallel.pipeline_parallel_size != 1
            or transfer is None
            or transfer.kv_connector != "NixlConnector"
            or transfer.kv_connector_extra_config
            or type(pool) is not BlockPool
            or any(m.block_pool is not pool for m in manager.coordinator.single_type_managers)
            or any(g.host_resident for g in self.kv_cache_config.kv_cache_groups)
            or self.kv_cache_config.hisparse_shared_host_pool
        ):
            raise ValueError(
                "cache observer requires async PP1, prefix OFF, one native pool and plain NIXL without offload"
            )
        root = Path(os.environ["AISIM_POOL_OBSERVER_DIR"])
        role, run_id = os.environ["AISIM_POOL_OBSERVER_ROLE"], os.environ["AISIM_POOL_OBSERVER_RUN_ID"]
        if role not in ("prefill", "decode") or not run_id:
            raise ValueError("cache observer needs an explicit role and run identity")
        identity = {
            "run_id": run_id,
            "role": role,
            "data_parallel_index": parallel.data_parallel_index,
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "async_scheduler": True,
            "prefix_caching": False,
            "scope": "Fixed text-token Pro P/D; physical device KV pool only, excluding weights and activations.",
            "runtime_revision": "ac9126e58aa7bbab1856ba6593ba4d5003fea516",
            "source_hashes": sources,
            "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self._pool_observation = PoolObservation(
            pool, root / f"{role}-dp-{parallel.data_parallel_index}.json", identity
        )
        # These are the only free-queue depletion paths after initialization.
        # All original calls, arguments, return values and exceptions remain.
        self._pool_observation.wrap(pool, "get_new_blocks")
        self._pool_observation.wrap(pool, "touch")
        self._pool_observation.wrap(manager, "allocate_slots", admission=True)

    def _preempt_request(self, *args, **kwargs):
        self._pool_observation.preemptions += 1
        try:
            return super()._preempt_request(*args, **kwargs)
        finally:
            self._pool_observation.write("preemption")

    def shutdown(self):
        try:
            result = super().shutdown()
        except Exception:
            self._pool_observation.errors += 1
            raise
        else:
            self._pool_observation.closed = True
            return result
        finally:
            self._pool_observation.write("shutdown")
