# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper


def configuration(backoff=1, backend="vllm", recompute=0):
    return {
        "engine": {
            "model": "Qwen/Qwen3-32B",
            "hardware": "b200_sxm",
            "backend": backend,
            "mode": "disaggregated",
            "context_length": 4096,
            "kv_transfer": {
                "bytes_per_token": 16,
                "prefill_backoff_tokens": backoff,
                "decode_recompute_tokens": recompute,
            },
            "workers": {
                role: {
                    "parallelism": {"tensor": 1},
                    "kv_cache": {"capacity": {"type": "fixed", "blocks": 256}},
                    "timing": {"type": "fixed", "prefill_ms": 7, "decode_ms": 3},
                }
                for role in ("prefill", "decode")
            },
        },
    }


def test_backoff_roundtrip_and_both_role_lowering():
    config = CorePredictionConfig.model_validate(configuration())
    reloaded = CorePredictionConfig.model_validate(config.model_dump(mode="json"))
    deployment = prediction_to_replay_spec(reloaded).backend_deployment
    assert deployment.prefill_engine_args["kv_transfer_prefill_backoff_tokens"] == 1
    assert deployment.decode_engine_args["kv_transfer_prefill_backoff_tokens"] == 1


def test_default_backoff_preserves_existing_wire_payload():
    config = CorePredictionConfig.model_validate(configuration(backoff=0))
    deployment = prediction_to_replay_spec(config).backend_deployment
    assert "kv_transfer_prefill_backoff_tokens" not in deployment.prefill_engine_args


@pytest.mark.parametrize("backoff", [-1, True, 0.5])
def test_invalid_backoff_is_rejected(backoff):
    with pytest.raises(ValueError):
        CorePredictionConfig.model_validate(configuration(backoff))


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_backoff_requires_vllm(backend):
    with pytest.raises(ValueError, match="prefill_backoff_tokens requires"):
        CorePredictionConfig.model_validate(configuration(backend=backend))


def test_recommendation_preserves_backoff():
    payload = configuration(recompute=128)
    for worker in payload["engine"]["workers"].values():
        worker["parallelism"]["preset"] = False
    payload["optimization"] = {"target": "throughput", "constraints": {"max_candidate_gpus": 16}}
    config = CoreRecommendationConfig.model_validate(payload)
    search = recommendation_to_sweeper(config).search_space
    assert search.kv_transfer_prefill_backoff_tokens == 1
    assert search.kv_transfer_decode_recompute_tokens == 128


def test_destination_recompute_is_independent_of_source_backoff():
    config = CorePredictionConfig.model_validate(configuration(backoff=0, recompute=128))
    deployment = prediction_to_replay_spec(config).backend_deployment
    for args in (deployment.prefill_engine_args, deployment.decode_engine_args):
        assert "kv_transfer_prefill_backoff_tokens" not in args
        assert args["kv_transfer_decode_recompute_tokens"] == 128
    assert CorePredictionConfig.model_validate(config.model_dump()).engine.kv_transfer.decode_recompute_tokens == 128


@pytest.mark.parametrize("recompute", [-1, True, 0.5])
def test_invalid_destination_recompute_is_rejected(recompute):
    with pytest.raises(ValueError):
        CorePredictionConfig.model_validate(configuration(recompute=recompute))


@pytest.mark.parametrize("backend", ["sglang", "trtllm"])
def test_destination_recompute_requires_vllm(backend):
    with pytest.raises(ValueError, match="decode_recompute_tokens requires"):
        CorePredictionConfig.model_validate(configuration(backoff=0, backend=backend, recompute=1))


def test_recompute_cannot_cross_received_recurrent_state_boundary():
    payload = configuration(backoff=1, recompute=2)
    for worker in payload["engine"]["workers"].values():
        worker["kv_cache"].update(
            block_size=16, bytes_per_token=8, state_cache={"bytes_per_request": 128}, prefix_caching=True
        )
    with pytest.raises(ValueError, match="decode_recompute_tokens <= prefill_backoff_tokens"):
        CorePredictionConfig.model_validate(payload)
