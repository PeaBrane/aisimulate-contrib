# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

from aisimulate.config import CorePredictionConfig
from aisimulate.config.engine import KvTransferConfig

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
                "context_length": 16384,
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
        "server_command": ["--max-model-len", "16384", "--max-num-seqs", "256", "--max-num-batched-tokens", "8192"],
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


def test_smoke_can_be_predicted_without_changing_production_matrix():
    config, identity, records = inputs()
    identity["cell"]["expected_requests"] = 128
    identity["phase"] = "smoke"
    assert len(prediction.validate_inputs(config, identity, records)) == 1
    identity["phase"] = "initial"
    with pytest.raises(ValueError, match="request count"):
        prediction.validate_inputs(config, identity, records)


@pytest.mark.parametrize("flag", ["--max-model-len", "--max-num-seqs", "--max-num-batched-tokens"])
def test_context_and_scheduler_must_match_the_actual_launch(flag):
    config, identity, records = inputs()
    command = identity["server_command"]
    command[command.index(flag) + 1] = "1048576"
    with pytest.raises(ValueError, match="frozen launch"):
        prediction.validate_inputs(config, identity, records)


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


def test_qualified_cache_receipt_binds_workload_capacity_and_runtime_bytes(tmp_path):
    config, identity, _ = inputs()
    identity["cell"].update(model_key="v4-pro", concurrency=1)
    raw = tmp_path / "runtime.json"
    raw.write_text('{"runtime": "cache"}')
    mapping = {
        "schema": "dspark-pro-cache-reconciliation-v1",
        "workload": {"k": 3, "concurrency": 1, "input_length": 4, "output_length": 3},
        "runtime_receipt_sha256": prediction.digest(raw),
        "capacity_proven_nonbinding": True,
        "replay_kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(mapping))
    receipt = {
        "qualification": {"eligible": True},
        "cache_reconciliation_path": path.name,
        "cache_reconciliation_sha256": prediction.digest(path),
        "runtime_cache_receipt_path": raw.name,
    }
    prediction.validate_cache(config, identity, receipt, tmp_path)
    config.engine.workers.aggregated.kv_cache.capacity.blocks = 256
    with pytest.raises(ValueError, match="reconciled runtime capacity"):
        prediction.validate_cache(config, identity, receipt, tmp_path)
    config.engine.workers.aggregated.kv_cache.capacity.blocks = 128
    identity["cell"]["concurrency"] = 4
    with pytest.raises(ValueError, match="workload"):
        prediction.validate_cache(config, identity, receipt, tmp_path)
    identity["cell"]["concurrency"] = 1
    raw.write_text("changed")
    with pytest.raises(ValueError, match="runtime cache receipt digest"):
        prediction.validate_cache(config, identity, receipt, tmp_path)


def test_qualified_prediction_requires_cache_evidence(tmp_path):
    config, identity, _ = inputs()
    with pytest.raises(ValueError, match="verified cache reconciliation"):
        prediction.validate_cache(config, identity, {"qualification": {"eligible": True}}, tmp_path)


def test_pd_cache_receipt_checks_both_role_capacities(tmp_path):
    config, identity, _ = inputs()
    identity["cell"].update(model_key="v4-pro", concurrency=1, deployment="disagg")
    workers = config.engine.workers
    workers.prefill = workers.aggregated.model_copy(deep=True)
    workers.decode = workers.aggregated.model_copy(deep=True)
    mapping = {
        "schema": "dspark-pro-pd-cache-reconciliation-v1",
        "workload": {"k": 3, "concurrency": 1, "input_length": 4, "output_length": 3},
        "roles": {
            role: {"replay_kv_cache": {"capacity": {"type": "fixed", "blocks": 128}}} for role in ("prefill", "decode")
        },
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(mapping))
    receipt = {"cache_reconciliation_path": path.name, "cache_reconciliation_sha256": prediction.digest(path)}
    prediction.validate_cache(config, identity, receipt, tmp_path)
    workers.prefill.kv_cache.capacity.blocks = 1024
    with pytest.raises(ValueError, match="prefill Replay cache capacity"):
        prediction.validate_cache(config, identity, receipt, tmp_path)
    workers.prefill.kv_cache.capacity.blocks = 128
    mapping["replay_kv_transfer"] = {"bytes_per_token": 8192, "decode_recompute_tokens": 1}
    path.write_text(json.dumps(mapping))
    receipt["cache_reconciliation_sha256"] = prediction.digest(path)
    config.engine.kv_transfer = KvTransferConfig(bytes_per_token=8192, decode_recompute_tokens=1)
    prediction.validate_cache(config, identity, receipt, tmp_path)
    config.engine.kv_transfer.bytes_per_token = "auto"
    with pytest.raises(ValueError, match="reconciled target/draft state contract"):
        prediction.validate_cache(config, identity, receipt, tmp_path)
    config.engine.kv_transfer = None
    with pytest.raises(ValueError, match="Replay transfer is missing"):
        prediction.validate_cache(config, identity, receipt, tmp_path)
    config.engine.kv_transfer = KvTransferConfig(bytes_per_token=8192, decode_recompute_tokens=1)
    raw = tmp_path / "runtime.json"
    raw.write_text('{"runtime": "cache"}')
    receipt.update(qualification={"eligible": True}, runtime_cache_receipt_path=raw.name)
    mapping.update(
        runtime_receipt_sha256=prediction.digest(raw),
        capacity_proven_nonbinding=True,
        replay_capacity_proven_nonbinding=True,
        native_capacity_proven_nonbinding=True,
    )
    for proof in ("replay_capacity_proven_nonbinding", "native_capacity_proven_nonbinding"):
        mapping[proof] = False
        path.write_text(json.dumps(mapping))
        receipt["cache_reconciliation_sha256"] = prediction.digest(path)
        with pytest.raises(ValueError, match="capacity-bound workload"):
            prediction.validate_cache(config, identity, receipt, tmp_path)
        mapping[proof] = True
    path.write_text(json.dumps(mapping))
    receipt["cache_reconciliation_sha256"] = prediction.digest(path)
    prediction.validate_cache(config, identity, receipt, tmp_path)


@pytest.mark.parametrize("scope", ["engine", "worker"])
def test_empirical_or_sol_timing_cannot_enter_measured_table_scoring(scope):
    config, identity, records = inputs()
    if scope == "engine":
        config.engine.database_mode = "SOL"
    else:
        config.engine.workers.aggregated.timing.database_mode = "HYBRID"
    with pytest.raises(ValueError, match="measured-table"):
        prediction.validate_inputs(config, identity, records)
