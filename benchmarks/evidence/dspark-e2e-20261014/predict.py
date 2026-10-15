# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run canonical Engine Replay for one measured fixed-length DSpark window."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import aisimulate._runtime as native
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements
from aisimulate_core.sdk.utils import get_model_config_from_model_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot():
    root = Path(__file__).resolve().parents[3]
    paths = [
        "crates/core",
        "python/aisimulate/src",
        str(Path(__file__).resolve().relative_to(root)),
    ]
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        cwd=root,
        text=True,
    )
    if untracked.strip():
        raise ValueError("commit new predictor source files before recording reproducible predictions")
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *paths], cwd=root)
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
        "native_sha256": digest(Path(native.__file__)),
        "source_paths": paths,
    }, patch


def validate_artifacts(config, identity, receipt, receipt_dir, pins):
    """Verify source configs before the canonical loader constructs the model."""
    model = identity["model"]
    target = model["model"]
    draft = model.get("draft_model", target)
    resolved = {}
    for role, name, revision in (
        ("target", target, model["revision"]),
        ("draft", draft, model.get("draft_revision", model["revision"])),
    ):
        pin = pins[name]
        path = receipt_dir / receipt[f"{role}_config_path"]
        if (
            revision != pin["revision"]
            or digest(path) != pin["config_sha256"]
            or digest(path) != receipt[f"{role}_config_sha256"]
        ):
            raise ValueError(f"{role} config differs from its immutable checkpoint pin")
        resolved[role] = json.loads(path.read_text())
    actual = get_model_config_from_model_path(config.engine.model)["raw_config"]
    if any(key not in actual or actual[key] != value for key, value in resolved["target"].items()):
        raise ValueError("canonical target config differs from the pinned serving config")
    if config.engine.speculation.draft_config != resolved["draft"]:
        raise ValueError("campaign requires the exact pinned draft_config in Replay YAML")


def validate_inputs(config, identity, records):
    cell = identity["cell"]
    model = identity["model"]
    if config.engine.model != model["model"] or config.engine.backend != "vllm":
        raise ValueError("Replay target model/backend differs from the measured identity")
    if config.engine.hardware != cell["hardware"] + "_sxm":
        raise ValueError("Replay hardware differs from the measured cell")
    expected = 1 if identity.get("phase") == "smoke" else cell["expected_requests"]
    if len(records) != expected:
        raise ValueError("request count does not match the fixed-length matrix cell")
    if config.engine.estimation_mode != "op_level" or config.engine.fallback_policy != "deny":
        raise ValueError("campaign requires explicit op_level timing and fallback_policy=deny")
    if config.engine.database_mode != "SILICON":
        raise ValueError("campaign scoring requires measured-table SILICON mode")
    speculation = config.engine.speculation
    if speculation is None or speculation.kind != "dspark" or speculation.adaptive_verification:
        raise ValueError("campaign requires fixed-length DSpark with adaptive verification disabled")
    if speculation.num_speculative_tokens != cell["num_speculative_tokens"]:
        raise ValueError("draft length differs from the measured cell")
    if model.get("draft_model") is not None and (
        speculation.draft_model_path != model["draft_model"] or speculation.draft_revision != model["draft_revision"]
    ):
        raise ValueError("Replay draft identity differs from the measured checkpoint")
    expected_mode = "aggregated" if cell["deployment"] == "agg" else "disaggregated"
    if config.engine.mode != expected_mode:
        raise ValueError("deployment differs from the measured cell")
    if expected_mode == "disaggregated":
        transfer = config.engine.kv_transfer
        backoff = 1 if cell["model_key"] == "kimi-k3" else 0
        recompute = 128 if cell["model_key"] == "v41-flash" else 1
        if transfer is None or (
            transfer.prefill_backoff_tokens,
            transfer.decode_recompute_tokens,
        ) != (
            backoff,
            recompute,
        ):
            raise ValueError("P/D prompt boundaries differ from the pinned NIXL contract")
    roles = ["aggregated"] if expected_mode == "aggregated" else ["prefill", "decode"]
    for role in roles:
        worker = getattr(config.engine.workers, role)
        command = (
            identity.get("server_command", [])
            if role == "aggregated"
            else identity.get("roles", {}).get(role, {}).get("command", [])
        )
        for flag, value in (
            ("--max-model-len", config.engine.context_length),
            ("--max-num-seqs", worker.scheduler.max_sequences),
            ("--max-num-batched-tokens", worker.scheduler.max_batched_tokens),
        ):
            supplied = [
                command[index + 1] if part == flag and index + 1 < len(command) else part[len(flag) + 1 :]
                for index, part in enumerate(command)
                if part == flag or part.startswith(flag + "=")
            ]
            if len(supplied) != 1 or supplied[0] != str(value):
                raise ValueError(f"{role} Replay {flag} differs from the frozen launch configuration")
        if (worker.hardware or config.engine.hardware) != config.engine.hardware:
            raise ValueError(f"{role} hardware differs from the measured cell")
        if (
            worker.timing.estimation_mode not in (None, "op_level")
            or worker.timing.fallback_policy not in (None, "deny")
            or worker.timing.database_mode not in (None, "SILICON")
        ):
            raise ValueError(f"{role} overrides the measured-table op_level/deny contract")
        topology = cell["decode" if role == "aggregated" else role]
        parallel = worker.parallelism
        if (
            parallel.tensor,
            parallel.attention_data,
            parallel.moe_expert,
            parallel.pipeline,
            parallel.replicas,
        ) != (
            topology["tp"],
            topology["attention_dp"],
            topology["ep"],
            1,
            1,
        ):
            raise ValueError(f"{role} topology differs from the measured cell")
        if worker.kv_cache.prefix_caching or worker.kv_cache.host_offload or worker.kv_cache.g3_offload:
            raise ValueError("fixed-length campaign requires prefix caching and external offload disabled")
        if worker.timing.type != "default":
            raise ValueError("measured scoring requires the canonical op-based model, not fixed timing")
    requests = []
    for row in records:
        if row["status"] != "success":
            raise ValueError("failed measurements cannot become completed Replay requests")
        if (
            len(row["prompt_token_ids"]) != cell["input_length"]
            or len(row["output_token_ids"]) != cell["output_length"]
        ):
            raise ValueError("measurement token lengths differ from the matrix")
        requests.append(
            {
                "id": row["request_id"],
                "arrival_time_ms": 0.0,
                "input_token_ids": row["prompt_token_ids"],
                "output_tokens": cell["output_length"],
                "output_token_ids": row["output_token_ids"],
            }
        )
    return requests


def validate_acceptance(speculation, evidence, acceptance_kind):
    rates = evidence.get("acceptance_rates")
    if acceptance_kind == "synthetic":
        if evidence.get("frozen_before_measurement") is not True:
            raise ValueError("synthetic control must be frozen before measurement")
        if rates is not None:
            raise ValueError("synthetic control requires the producer's adjacent-integer mean")
    if rates is not None:
        supplied = speculation.acceptance_rates
        if (
            supplied is None
            or len(supplied) != len(rates)
            or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(supplied, rates, strict=True))
        ):
            raise ValueError("Replay conditional acceptance differs from the evidence")
    elif speculation.accepted_tokens is None or not math.isclose(
        speculation.accepted_tokens,
        evidence["accepted_tokens"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Replay mean acceptance differs from the evidence")


def validate_cache(config, identity, receipt, receipt_dir):
    """Bind qualified Pro cache mapping to both runtime bytes and Replay fields."""
    eligible = receipt.get("qualification", {}).get("eligible") is True
    path = receipt.get("cache_reconciliation_path")
    if path is None:
        if eligible:
            raise ValueError("qualified scoring requires a verified cache reconciliation")
        return
    path = receipt_dir / path
    if digest(path) != receipt["cache_reconciliation_sha256"]:
        raise ValueError("cache reconciliation digest mismatch")
    mapping = json.loads(path.read_text())
    deployment = identity["cell"]["deployment"]
    schema = {
        "agg": "dspark-pro-cache-reconciliation-v1",
        "disagg": "dspark-pro-pd-cache-reconciliation-v1",
    }.get(deployment)
    if mapping["schema"] != schema:
        raise ValueError("cache reconciliation requires an adapter for this deployment")
    cell = identity["cell"]
    if cell["model_key"] != "v4-pro" or mapping["workload"] != {
        "k": cell["num_speculative_tokens"],
        "concurrency": cell["concurrency"],
        "input_length": cell["input_length"],
        "output_length": cell["output_length"],
    }:
        raise ValueError("cache reconciliation workload differs from measurement")
    if eligible:
        raw = receipt_dir / receipt["runtime_cache_receipt_path"]
        if digest(raw) != mapping["runtime_receipt_sha256"]:
            raise ValueError("runtime cache receipt digest mismatch")
        proofs = (
            ("replay_capacity_proven_nonbinding", "native_capacity_proven_nonbinding")
            if deployment == "disagg"
            else ("capacity_proven_nonbinding",)
        )
        if any(mapping.get(proof) is not True for proof in proofs):
            raise ValueError("approximate cache mapping cannot qualify a capacity-bound workload")
    roles = {"aggregated": mapping} if deployment == "agg" else mapping["roles"]
    if deployment == "disagg" and set(roles) != {"prefill", "decode"}:
        raise ValueError("P/D cache reconciliation must contain both roles")
    for role, role_mapping in roles.items():
        if role_mapping["replay_kv_cache"] is None:
            raise ValueError(f"{role} cache mapping has no nonbinding-capacity proof")
        actual = getattr(config.engine.workers, role).kv_cache.model_dump(mode="json")
        for key, expected in role_mapping["replay_kv_cache"].items():
            value = actual[key]
            matches = (
                isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items())
                if isinstance(expected, dict)
                else value == expected
            )
            if not matches:
                raise ValueError(f"{role} Replay cache {key} differs from the reconciled runtime capacity")
    if deployment == "disagg" and "replay_kv_transfer" in mapping:
        if config.engine.kv_transfer is None:
            raise ValueError("Replay transfer is missing for the reconciled target/draft state contract")
        actual = config.engine.kv_transfer.model_dump(mode="json")
        if any(actual.get(key) != value for key, value in mapping["replay_kv_transfer"].items()):
            raise ValueError("Replay transfer differs from the reconciled target/draft state contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", required=True, type=Path, help="Measured summary.json")
    parser.add_argument("--config", required=True, type=Path, help="Frozen canonical Replay YAML")
    parser.add_argument("--match-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source, patch = source_snapshot()
    (args.output_dir / "predictor-source.patch").write_bytes(patch)
    measured = json.loads(args.measurement.read_text())
    identity = json.loads((args.measurement.parent / "identity.json").read_text())
    receipt = json.loads(args.match_receipt.read_text())
    records_path = args.measurement.parent / measured["request_records"]
    summary = {
        key: measured[key]
        for key in (
            "cell_id",
            "repetition",
            "identity_sha256",
            "metrics_contract",
            "total_gpus",
            "expected_requests",
        )
    }
    summary.update(
        run_id=args.output_dir.name,
        predictor="aisimulate.engine_replay",
        status="failed",
        metrics={},
    )
    reasons = list(measured.get("qualification", {}).get("reasons", []))
    if identity.get("phase") == "smoke":
        reasons.append("smoke diagnostic; excluded from production scoring")
    if measured.get("qualification", {}).get("eligible") is not True:
        reasons.append("measurement is not qualified")
    if receipt.get("qualification", {}).get("eligible") is not True:
        reasons.extend(receipt.get("qualification", {}).get("reasons", ["replay match not qualified"]))
        reasons.append("replay match not qualified")
    summary["qualification"] = {"eligible": False, "reasons": reasons}
    try:
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if identity_hash != measured["identity_sha256"]:
            raise ValueError("measurement identity digest mismatch")
        if receipt["measurement_identity_sha256"] != identity_hash or receipt["replay_config_sha256"] != digest(
            args.config
        ):
            raise ValueError("replay-match receipt does not match measured identity/configuration")
        acceptance_path = args.match_receipt.parent / receipt["acceptance_path"]
        if digest(acceptance_path) != receipt["acceptance_sha256"]:
            raise ValueError("acceptance evidence digest mismatch")
        config = CorePredictionConfig.from_yaml(args.config)
        pins = {
            item["model"]: item
            for item in json.loads((Path(__file__).parent / "checkpoint-references.json").read_text())
        }
        validate_artifacts(config, identity, receipt, args.match_receipt.parent, pins)
        validate_cache(config, identity, receipt, args.match_receipt.parent)
        acceptance = json.loads(acceptance_path.read_text())
        validate_acceptance(config.engine.speculation, acceptance, identity["cell"]["acceptance"])
        summary["acceptance"] = {
            "kind": identity["cell"]["acceptance"],
            "sampling_model": "conditional_prefix_rates"
            if acceptance.get("acceptance_rates") is not None
            else "adjacent_integer_mean_only",
            "evidence_sha256": digest(acceptance_path),
        }
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        requests = validate_inputs(config, identity, records)
        spec = prediction_to_replay_spec(config)
        concurrency = identity["cell"]["concurrency"]
        spec = replace(
            spec,
            workload={"requests": requests, "replay_concurrency": concurrency},
            concurrency=concurrency,
        )
        (args.output_dir / "replay-spec.json").write_text(json.dumps(asdict(spec), indent=2, allow_nan=False) + "\n")
        runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
        start = time.perf_counter()
        try:
            result = runner.run(
                spec,
                output_requirements=ReplayOutputRequirements(
                    include_raw_report=True,
                    capture_per_request=True,
                    capture_memory_diagnostics=True,
                    capture_performance_diagnostics=True,
                ),
            )
        finally:
            runner.close()
        summary["host_wall_seconds"] = time.perf_counter() - start
        (args.output_dir / "replay-report.json").write_text(
            json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
        )
        metrics = result.metrics
        completed = int(metrics["completed_requests"])
        expected = measured["expected_requests"]
        if completed != expected or int(metrics["total_output_tokens"]) != expected * identity["cell"]["output_length"]:
            raise ValueError("Replay did not complete the exact measured workload")
        summary["metrics"] = {key: metrics[key] for key in ("mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s")}
        summary["metrics"].update(completed_requests=completed, failed_requests=0)
        if source_snapshot()[0] != source:
            raise ValueError("predictor source or native library changed during prediction; repeat on a stable build")
        summary["status"] = "success"
        summary["qualification"]["eligible"] = not reasons
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["qualification"]["reasons"].append(summary["error"])
    summary["provenance"] = {
        **source,
        "source_patch": "predictor-source.patch",
        "replay_config_sha256": digest(args.config),
        "match_receipt_sha256": digest(args.match_receipt),
        "measurement_summary_sha256": digest(args.measurement),
        "request_records_sha256": digest(records_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
