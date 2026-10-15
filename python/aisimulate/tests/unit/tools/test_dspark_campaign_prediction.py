# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

from aisimulate.config import CorePredictionConfig

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "dspark_predict", ROOT / "benchmarks/evidence/dspark-e2e-20261014/predict.py"
)
prediction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prediction)


def inputs():
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
                "hardware": "b200_sxm",
                "estimation_mode": "op_level",
                "fallback_policy": "deny",
                "speculation": {"kind": "dspark", "num_speculative_tokens": 3, "accepted_tokens": 1.25},
                "workers": {
                    "aggregated": {
                        "parallelism": {"tensor": 8},
                        "kv_cache": {"prefix_caching": False, "capacity": {"type": "fixed", "blocks": 128}},
                    }
                },
            },
        }
    )
    identity = {
        "model": {"model": config.engine.model},
        "cell": {
            "expected_requests": 1,
            "num_speculative_tokens": 3,
            "deployment": "agg",
            "hardware": "b200",
            "decode": {"tp": 8, "attention_dp": 1, "ep": 1},
            "input_length": 4,
            "output_length": 3,
        },
    }
    records = [
        {
            "request_id": "measured-1",
            "status": "success",
            "prompt_token_ids": [5, 8, 2, 9],
            "output_token_ids": [10, 20, 30],
        }
    ]
    return config, identity, records


def test_measured_token_ids_are_preserved_for_closed_loop_replay():
    config, identity, records = inputs()
    assert prediction.validate_inputs(config, identity, records) == [
        {
            "id": "measured-1",
            "arrival_time_ms": 0.0,
            "input_token_ids": [5, 8, 2, 9],
            "output_tokens": 3,
            "output_token_ids": [10, 20, 30],
        }
    ]


def test_pinned_artifacts_reject_changed_loader_or_draft(tmp_path, monkeypatch):
    config, identity, _ = inputs()
    raw = {"hidden_size": 7168, "dspark_block_size": 3}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    identity["model"]["revision"] = "immutable-revision"
    pins = {
        config.engine.model: {
            "revision": "immutable-revision",
            "config_sha256": prediction.digest(path),
        }
    }
    receipt = {
        f"{role}_config_{field}": value
        for role in ("target", "draft")
        for field, value in (("path", path.name), ("sha256", prediction.digest(path)))
    }
    config.engine.speculation.draft_config = dict(raw)
    monkeypatch.setattr(prediction, "get_model_config_from_model_path", lambda _: {"raw_config": dict(raw)})
    prediction.validate_artifacts(config, identity, receipt, tmp_path, pins)
    config.engine.speculation.draft_config["hidden_size"] = 4096
    with pytest.raises(ValueError, match="exact pinned draft_config"):
        prediction.validate_artifacts(config, identity, receipt, tmp_path, pins)
    config.engine.speculation.draft_config = dict(raw)
    monkeypatch.setattr(prediction, "get_model_config_from_model_path", lambda _: {"raw_config": {}})
    with pytest.raises(ValueError, match="canonical target config"):
        prediction.validate_artifacts(config, identity, receipt, tmp_path, pins)
    path.write_text(json.dumps({"hidden_size": 4096}))
    with pytest.raises(ValueError, match="immutable checkpoint pin"):
        prediction.validate_artifacts(config, identity, receipt, tmp_path, pins)


@pytest.mark.parametrize("mismatch", ["model", "hardware", "draft_width", "topology", "prefix", "tokens", "failure"])
def test_identity_or_workload_mismatches_cannot_be_scored(mismatch):
    config, identity, records = inputs()
    if mismatch == "model":
        identity["model"]["model"] = "other-model"
    elif mismatch == "hardware":
        identity["cell"]["hardware"] = "b300"
    elif mismatch == "draft_width":
        identity["cell"]["num_speculative_tokens"] = 6
    elif mismatch == "topology":
        identity["cell"]["decode"]["ep"] = 8
    elif mismatch == "prefix":
        config.engine.workers.aggregated.kv_cache.prefix_caching = True
    elif mismatch == "tokens":
        records[0]["output_token_ids"].pop()
    else:
        records[0]["status"] = "failed"
    with pytest.raises(ValueError):
        prediction.validate_inputs(config, identity, records)


def test_acceptance_cannot_be_changed_to_fit_latency():
    config, _, _ = inputs()
    speculation = config.engine.speculation
    prediction.validate_acceptance(speculation, {"accepted_tokens": 1.25}, "real")
    with pytest.raises(ValueError, match="mean acceptance"):
        prediction.validate_acceptance(speculation, {"accepted_tokens": 1.5}, "real")
    with pytest.raises(ValueError, match="frozen before"):
        prediction.validate_acceptance(speculation, {"accepted_tokens": 1.25}, "synthetic")
    prediction.validate_acceptance(
        speculation, {"accepted_tokens": 1.25, "frozen_before_measurement": True}, "synthetic"
    )


def test_conditional_rates_take_precedence_over_mean_only_approximation():
    config, _, _ = inputs()
    evidence = {"acceptance_rates": [0.75, 0.5, 0.1], "accepted_tokens": 1.25}
    with pytest.raises(ValueError, match="conditional acceptance"):
        prediction.validate_acceptance(config.engine.speculation, evidence, "real")
    speculation = config.engine.speculation.model_copy(
        update={"accepted_tokens": None, "acceptance_rates": [0.75, 0.5, 0.1]}
    )
    prediction.validate_acceptance(speculation, evidence, "real")


@pytest.mark.parametrize("scope", ["engine", "worker"])
def test_empirical_or_sol_timing_cannot_enter_measured_table_scoring(scope):
    config, identity, records = inputs()
    if scope == "engine":
        config.engine.database_mode = "SOL"
    else:
        config.engine.workers.aggregated.timing.database_mode = "HYBRID"
    with pytest.raises(ValueError, match="measured-table"):
        prediction.validate_inputs(config, identity, records)
