# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only worker extension for exact native cache receipts before measurement.

Original adapter for vLLM ac9126e58aa7bbab1856ba6593ba4d5003fea516 public
worker-extension and collective-RPC APIs. No forward methods are replaced.
"""

import dataclasses
import enum


def primitive(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        result = {field.name: primitive(getattr(value, field.name)) for field in dataclasses.fields(value)}
        result["type"] = type(value).__name__
        for name in ("page_size_bytes", "block_table_token_alignment", "prefix_cacheable", "prefix_replay_tokens"):
            if hasattr(value, name):
                result[name] = primitive(getattr(value, name))
        return result
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [primitive(item) for item in value]
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    raise TypeError(f"unsupported cache receipt value: {type(value).__name__}")


class DsparkReceiptWorker:
    def dspark_campaign_receipt(self):
        runner = self.model_runner
        config = runner.kv_cache_config
        if config is None:
            raise RuntimeError("cache has not been initialized")
        return {
            "schema": "dspark-native-cache-v1",
            "rank": self.rank,
            "local_rank": self.local_rank,
            "kv_cache_config": primitive(config),
            "layer_specs": primitive(runner.get_kv_cache_spec()),
            "available_kv_cache_memory_bytes": getattr(self, "available_kv_cache_memory_bytes", None),
            "scope": "Read after server readiness and before warmup; no forward instrumentation.",
        }
