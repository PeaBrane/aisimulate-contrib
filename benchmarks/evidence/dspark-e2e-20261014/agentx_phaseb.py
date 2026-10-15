# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare or execute one evidence-gated AgentX client run; never allocate GPUs.

Original adapter for SemiAnalysisAI/aiperf@754356e9a39acc6cc6afb242d123bb57c3fb6f75
CLI/plugin contracts (Apache-2.0). Source anchors and preparation: COLLECTION.md.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from campaign import IMAGE, MODELS, RUNTIME_REVISION, cells
from compare import CONTRACT, METRICS, qualification_error

PRODUCER = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"
CORPUS = {
    "repository": "semianalysisai/cc-traces-weka-062126",
    "revision": "23f152f6f0f9399a85901b89a6458def0ef16729",
    "file": "traces.jsonl",
    "sha256": "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276",
    "bytes": 1847151435,
    "rows": 393,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def context_limit(identity):
    commands = (
        [identity["server_command"]]
        if "server_command" in identity
        else [row["command"] for row in identity["roles"].values()]
    )
    limits = {int(command[command.index("--max-model-len") + 1]) for command in commands}
    if len(limits) != 1:
        raise ValueError("serving roles have different context limits")
    return limits.pop()


def scheduling_limits(identity):
    roles = {"agg": {"command": identity["server_command"]}} if "server_command" in identity else identity["roles"]
    result = {}
    for role, row in roles.items():
        command = row["command"]
        result[role] = {
            name: int(command[command.index(flag) + 1])
            for name, flag in (
                ("max_sequences", "--max-num-seqs"),
                ("max_batched_tokens", "--max-num-batched-tokens"),
            )
        }
    return result


def acceptance_contract(model_key, k, acceptance):
    if acceptance == "real":
        return {"kind": "real", "rejection_sample_method": "block"}
    model = MODELS[model_key]
    if acceptance != "synthetic" or k != model["reference_k"]:
        raise ValueError("Phase B synthetic acceptance requires the frozen reference K")
    return {
        "kind": "synthetic",
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": model["reference_al"],
        "accepted_tokens": model["reference_al"] - 1,
        "frozen_before_measurement": True,
        "source": f"AgentX result {model['reference_result']}",
        "producer": f"vllm/config/speculative.py::_acceptance_length_to_rates@{RUNTIME_REVISION}",
    }


def validate_acceptance(command, control, model_key, k, acceptance):
    expected = acceptance_contract(model_key, k, acceptance)
    spec = json.loads(command[command.index("--speculative-config") + 1])
    if spec.get("rejection_sample_method") != expected["rejection_sample_method"]:
        raise ValueError("server acceptance method differs from requested Phase B control")
    if acceptance == "synthetic":
        if spec.get("synthetic_acceptance_length") != expected["synthetic_acceptance_length"] or any(
            (control or {}).get(key) != value
            for key, value in expected.items()
            if key not in ("kind", "rejection_sample_method")
        ):
            raise ValueError("server synthetic control is not the frozen reference identity")
    elif control is not None or spec.get("synthetic_acceptance_length") is not None:
        raise ValueError("real acceptance cannot retain a synthetic control")
    return expected


def context_gate(path, server, fixed, model_key):
    if path is None:
        raise ValueError(
            "context audit or native runtime preflight is required; current 16384-token server is insufficient"
        )
    audit = read_json(path)
    model = MODELS[model_key]
    shared_identity = (
        audit.get("corpus") == CORPUS
        and audit.get("tokenizer") == model["model"]
        and audit.get("tokenizer_revision") == model["revision"]
        and audit.get("truncation_or_filtering") is False
        and audit.get("output_policy") == "recorded_out_minimum_one_ignore_eos"
        and type(audit.get("max_prompt_plus_requested_output")) is int
        and audit["max_prompt_plus_requested_output"] > 0
        and bool(audit.get("evidence"))
    )
    exhaustive = audit.get("all_generated_payloads_verified") is True
    runtime_validation = audit.get("validation_policy") == "native_runtime_per_request"
    if not shared_identity or not (exhaustive or runtime_validation):
        raise ValueError("corpus context audit does not cover the pinned preflight policy")
    if runtime_validation:
        witness_path = Path(path).parent / audit["native_witness"]["path"]
        witness = read_json(witness_path)
        source_path = Path(path).parent / audit["source_audit"]["path"]
        source = read_json(source_path)
        if (
            digest(witness_path) != audit["native_witness"]["sha256"]
            or digest(source_path) != audit["source_audit"]["sha256"]
            or source.get("corpus") != CORPUS
            or source.get("local_content_verified") is not True
            or witness.get("corpus") != CORPUS
            or witness.get("producer_revision") != PRODUCER
            or witness.get("model") != model["model"]
            or witness.get("revision") != model["revision"]
            or witness.get("native_encoder_sha256") != audit.get("native_formatter_sha256")
            or not audit.get("native_formatter_sha256")
            or audit.get("image") != IMAGE
            or audit.get("runtime_revision") != RUNTIME_REVISION
            or audit.get("preserve_all_context_failures") is not True
            or audit.get("unsampled_corpus_unverified") is not True
            or audit["max_prompt_plus_requested_output"] != witness["native_prompt_plus_output_without_session_marker"]
        ):
            raise ValueError("native runtime context preflight identity or witness differs")
    if server.get("prefix_cache") is not True:
        raise ValueError("AgentX server must enable prefix cache")
    limit = server["max_model_len"]
    if runtime_validation and (
        audit.get("context_length") != limit
        or source["statistics"]["max_input_plus_requested_output"] > limit
        or source["models"][model_key]["max_position_embeddings"] < limit
    ):
        raise ValueError("native runtime context limit differs from the source/model preflight")
    if audit["max_prompt_plus_requested_output"] > limit or any(
        row["max_model_len"] != limit or row.get("prefix_cache") is not False for row in fixed
    ):
        raise ValueError("corpus, server context limit and qualified fixed anchors do not match")
    if server.get("scheduling_limits") is not None and any(
        row.get("scheduling_limits") != server["scheduling_limits"] for row in fixed
    ):
        raise ValueError("AgentX scheduling limits differ from native fixed anchors")
    for row in audit["evidence"]:
        if digest(Path(path).parent / row["path"]) != row["sha256"]:
            raise ValueError("corpus context audit evidence digest mismatch")
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest(path),
        "validation_policy": "native_runtime_per_request" if runtime_validation else "exhaustive_payload_audit",
    }


def fixed_gate(
    pairs, model_key, deployment, k, required_concurrencies=(1, 4), *, collection_only=False, acceptance=None
):
    """Keep native collection readiness separate from paired scoring readiness."""
    matrix = {cell["cell_id"]: cell for cell in cells()}
    seen, receipts = set(), []
    for measured_path, predicted_path in pairs:
        measured = read_json(measured_path)
        predicted = None if predicted_path is None else read_json(predicted_path)
        cell = matrix.get(measured.get("cell_id"))
        if cell is None or (
            cell["model_key"],
            cell["deployment"],
            cell["num_speculative_tokens"],
            cell["input_length"],
        ) != (model_key, deployment, k, 1024):
            raise ValueError("fixed-length pair does not match model/deployment/K/initial workload")
        if acceptance is not None and cell["acceptance"] != acceptance:
            raise ValueError("fixed anchor acceptance differs from Phase B control")
        if collection_only:
            if predicted is not None:
                raise ValueError("collection-only anchors accept measurements, not prediction pairs")
            if (
                measured.get("status") != "success"
                or measured.get("metrics_contract") != CONTRACT
                or measured.get("expected_requests") != cell["expected_requests"]
                or measured.get("total_gpus") != cell["total_gpus"]
                or measured.get("metrics", {}).get("completed_requests") != cell["expected_requests"]
                or measured.get("metrics", {}).get("failed_requests") != 0
            ):
                raise ValueError("native fixed anchor is incomplete or has a different measurement contract")
        else:
            error = qualification_error(measured, predicted, cell["expected_requests"], cell["total_gpus"])
            if error is not None:
                raise ValueError(f"fixed-length gate failed: {error}")
            if (predicted["cell_id"], predicted["repetition"]) != (
                measured["cell_id"],
                measured["repetition"],
            ):
                raise ValueError("fixed-length pair keys differ")
        identity_path = Path(measured_path).parent / "identity.json"
        identity = read_json(identity_path)
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if identity_hash != measured["identity_sha256"] or identity.get("phase") == "smoke":
            raise ValueError("fixed-length identity is altered or smoke-only")
        if identity["model"] != MODELS[model_key] or identity.get("image") != IMAGE:
            raise ValueError("fixed-length model/image identity differs from campaign")
        if identity.get("runtime_revision") != RUNTIME_REVISION:
            raise ValueError("fixed-length serving runtime differs")
        if acceptance is not None:
            roles = (
                [{"command": identity["server_command"], "synthetic_control": identity.get("synthetic_control")}]
                if "server_command" in identity
                else identity["roles"].values()
            )
            for role in roles:
                validate_acceptance(role["command"], role.get("synthetic_control"), model_key, k, acceptance)
        records_path = Path(measured_path).parent / measured["request_records"]
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        if len(records) != cell["expected_requests"] or any(
            row.get("status") != "success"
            or len(row.get("prompt_token_ids", [])) != 1024
            or len(row.get("output_token_ids", [])) != 1024
            for row in records
        ):
            raise ValueError("fixed-length functional records are missing or incomplete")
        for side in (measured,) if collection_only else (measured, predicted):
            if any(
                type(side["metrics"].get(metric)) not in (int, float)
                or not math.isfinite(side["metrics"][metric])
                or side["metrics"][metric] <= 0
                for metric in METRICS
            ):
                raise ValueError("qualified pair needs finite positive metrics, without an accuracy threshold")
        seen.add(cell["concurrency"])
        prefix_cache = identity.get("prefix_cache")
        if collection_only:
            commands = (
                [identity["server_command"]]
                if "server_command" in identity
                else [row["command"] for row in identity["roles"].values()]
            )
            if any(
                "--no-enable-prefix-caching" not in command or "--enable-prefix-caching" in command
                for command in commands
            ):
                raise ValueError("native fixed anchor must explicitly disable prefix caching in every role")
            for command in commands:
                spec = json.loads(command[command.index("--speculative-config") + 1])
                if (
                    spec.get("method") != "dspark"
                    or spec.get("num_speculative_tokens") != k
                    or spec.get("enable_adaptive_verification") is not False
                ):
                    raise ValueError("native fixed anchor must execute matching fixed DSpark")
            prefix_cache = False
        paths = [measured_path, identity_path, records_path]
        if predicted_path is not None:
            paths.append(predicted_path)
        receipts.append(
            {
                "cell_id": cell["cell_id"],
                "max_model_len": context_limit(identity),
                "prefix_cache": prefix_cache,
                "gate": "native_functional_collection" if collection_only else "qualified_prediction_pair",
                "scheduling_limits": scheduling_limits(identity),
                "repetition": measured["repetition"],
                "artifacts": {str(Path(path).resolve()): digest(path) for path in paths},
            }
        )
    if not set(required_concurrencies) <= seen:
        raise ValueError("Phase B requires completed fixed anchors for the selected concurrency gate")
    return receipts


def profile_gate(paths, model_key, deployment, k, *, max_model_len=None, acceptance=None):
    seen = set()
    receipts = []
    for path in paths:
        check = read_json(path)
        run_path = Path(path).parent / check["run_receipt"]
        run = read_json(run_path)
        if digest(run_path) != check["run_receipt_sha256"] or run.get("status") != "completed_client_run":
            raise ValueError("pilot receipt is altered or incomplete")
        identity = run["identity"]
        if (
            identity["model_key"],
            identity["deployment"],
            identity["k"],
            identity["phase"],
        ) != (model_key, deployment, k, "pilot"):
            raise ValueError("pilot belongs to another model/deployment/K")
        if (
            identity.get("producer_revision") != PRODUCER
            or identity.get("corpus") != CORPUS
            or (max_model_len is not None and identity.get("max_model_len") != max_model_len)
        ):
            raise ValueError("pilot producer, corpus or context differs from this profile")
        if acceptance is not None and identity.get("acceptance_control") != acceptance_contract(
            model_key, k, acceptance
        ):
            raise ValueError("pilot acceptance control differs from this profile")
        required = (
            "dependencies",
            "fork_join",
            "warmup_frontier",
            "prefix_reuse",
            "stop_and_cancel",
        )
        if any(check.get("checks", {}).get(name) is not True for name in required) or not check.get("evidence"):
            raise ValueError("pilot lifecycle/cache review is incomplete")
        for evidence in check["evidence"]:
            if digest(Path(path).parent / evidence["path"]) != evidence["sha256"]:
                raise ValueError("pilot lifecycle evidence digest mismatch")
        seen.add(identity["concurrency"])
        receipts.append({"path": str(Path(path).resolve()), "sha256": digest(path)})
    if not {1, 4} <= seen:
        raise ValueError("one-hour profiles require reviewed completed pilots at C1 and C4")
    return receipts


def producer_command(executable, model_key, endpoint, concurrency, phase, output, metrics_urls):
    model = MODELS[model_key]
    command = [
        str(executable),
        "profile",
        "--scenario",
        "inferencex-agentx-mvp",
        "--url",
        endpoint,
        "--endpoint",
        "/v1/chat/completions",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--model",
        model["model"],
        "--tokenizer",
        model["model"],
        "--tokenizer-revision",
        model["revision"],
        "--tokenizer-trust-remote-code",
        "--public-dataset",
        "semianalysis_cc_traces_weka_062126",
        "--num-dataset-entries",
        "393",
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        "600" if phase == "pilot" else "3600",
        "--benchmark-grace-period",
        "30",
        "--failed-request-threshold",
        "0.1",
        "--random-seed",
        "42",
        "--trajectory-start-min-ratio",
        "0.25",
        "--trajectory-start-max-ratio",
        "0.75",
        "--warmup-requests-per-lane",
        "10",
        "--agentic-warmup-grace-period",
        "1800",
        "--trace-idle-gap-cap-seconds",
        "300",
        "--system-idle-gap-cap-seconds",
        "10",
        "--cache-bust",
        "first_turn_prefix",
        "--use-server-token-count",
        "--no-gpu-telemetry",
        "--export-level",
        "raw",
        "--stats-interval",
        "30",
        "--slice-duration",
        "1.0",
        "--output-artifact-dir",
        str(output / "aiperf"),
    ]
    if phase == "pilot":
        # Pinned scenario minimum is 900s. The requested 600s pilot is a
        # diagnostic and cannot be reported as a scenario-valid submission.
        command.append("--unsafe-override")
    if metrics_urls:
        command += ["--server-metrics", *metrics_urls]
    return command


def verify_producer(executable, checkout, command):
    """Check the executable's environment and parse the exact command without serving."""
    python = executable.parent / "python"
    code = """
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
import aiperf
from aiperf.cli import app
from aiperf.plugin import plugins
loader = plugins.get_class("public_dataset_loader", "semianalysis_cc_traces_weka_062126")
strategy = plugins.get_class("timing_strategy", "agentic_replay")
app.parse_args(sys.argv[1:], exit_on_error=False)
print(json.dumps({"producer_file": str(Path(aiperf.__file__).resolve()),
                  "loader_file": str(Path(inspect.getfile(loader)).resolve()),
                  "loader_revision": loader.hf_revision,
                  "capture_file": str(Path(inspect.getfile(strategy)).resolve()),
                  "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}))
"""
    proof = json.loads(subprocess.check_output([str(python), "-c", code, *command[1:]], text=True))
    expected = checkout.resolve() / "src" / "aiperf" / "__init__.py"
    loader = Path(__file__).parent / "agentx_corpus_pin" / "aisim_agentx_pin" / "__init__.py"
    if (
        Path(proof["producer_file"]) != expected
        or proof["loader_revision"] != CORPUS["revision"]
        or digest(proof["loader_file"]) != digest(loader)
    ):
        raise ValueError("aiperf executable does not load the pinned producer and campaign corpus adapter")
    capture = loader.parent / "capture.py"
    if digest(proof["capture_file"]) != digest(capture):
        raise ValueError("aiperf executable does not load the campaign capture hooks")
    proof["capture_sha256"] = digest(capture)
    proof["loader_sha256"] = digest(loader)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--deployment", choices=("agg", "disagg"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--acceptance", choices=("real", "synthetic"), default="real")
    parser.add_argument("--phase", choices=("pilot", "profile"), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 16), required=True)
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--fixed-pair",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("MEASURED", "PREDICTED"),
    )
    parser.add_argument("--fixed-measurement", type=Path, action="append", default=[])
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Collect silicon using native functional anchors without requiring simulator qualification",
    )
    parser.add_argument("--pilot-check", type=Path, action="append", default=[])
    parser.add_argument("--server-identity", type=Path, required=True)
    parser.add_argument("--context-audit", type=Path)
    parser.add_argument("--producer-checkout", type=Path, required=True)
    parser.add_argument("--aiperf", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--metrics-url", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-token-echo",
        action="store_true",
        help="Unscored diagnostic: return exact prompt/output IDs, increasing response payload size",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the client against the supplied existing server",
    )
    args = parser.parse_args()
    if args.k not in MODELS[args.model_key]["k"] or (args.phase == "pilot" and args.concurrency not in (1, 4)):
        parser.error("K or pilot concurrency is outside the approved matrix")
    if args.collection_only:
        if args.fixed_pair or not args.fixed_measurement:
            parser.error("collection-only requires --fixed-measurement and excludes --fixed-pair")
        anchors = [(path, None) for path in args.fixed_measurement]
    else:
        if args.fixed_measurement or not args.fixed_pair:
            parser.error("paired validation requires --fixed-pair; native anchors need --collection-only")
        anchors = args.fixed_pair
    fixed = fixed_gate(
        anchors,
        args.model_key,
        args.deployment,
        args.k,
        (args.concurrency,) if args.phase == "pilot" else (1, 4),
        collection_only=args.collection_only,
        acceptance=args.acceptance,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(args.producer_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.producer_checkout), "status", "--porcelain"], text=True)
    if revision != PRODUCER or dirty.strip():
        raise ValueError("AgentX producer checkout must be clean at the immutable pin")
    server = read_json(args.server_identity)
    if args.collection_only and not server.get("scheduling_limits"):
        raise ValueError("collection-only requires a fresh server receipt with per-role scheduling limits")
    if (
        server.get("model") != MODELS[args.model_key]
        or server.get("deployment") != args.deployment
        or server.get("k") != args.k
        or server.get("acceptance_control") != acceptance_contract(args.model_key, args.k, args.acceptance)
        or server.get("prefix_cache") is not True
        or server.get("adaptive_verification") is not False
        or server.get("external_kv_offload") is not False
        or server.get("image") != IMAGE
        or server.get("runtime_revision") != RUNTIME_REVISION
        or not server.get("cache_receipts")
    ):
        raise ValueError(
            "AgentX requires matching fixed DSpark server identity, prefix cache and native cache receipts"
        )
    for receipt in server["cache_receipts"]:
        if digest(args.server_identity.parent / receipt["path"]) != receipt["sha256"]:
            raise ValueError("AgentX runtime cache receipt digest mismatch")
    pilots = (
        profile_gate(
            args.pilot_check,
            args.model_key,
            args.deployment,
            args.k,
            max_model_len=server["max_model_len"],
            acceptance=args.acceptance,
        )
        if args.phase == "profile"
        else []
    )
    context = (
        context_gate(args.context_audit, server, fixed, args.model_key) if args.execute or args.context_audit else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    command = producer_command(
        args.aiperf.resolve(),
        args.model_key,
        args.endpoint,
        args.concurrency,
        args.phase,
        args.output.resolve(),
        args.metrics_url,
    )
    if args.diagnostic_token_echo:
        command += ["--extra-inputs", "return_token_ids:true"]
    proof = verify_producer(args.aiperf.resolve(), args.producer_checkout, command)
    identity = {
        "producer_environment": proof,
        "capture": {
            "boundary_hooks": True,
            "diagnostic_token_echo": args.diagnostic_token_echo,
            "metric_equivalence_claimed": False if args.diagnostic_token_echo else None,
            "normal_mode_input_identity": "raw payload plus pinned native formatter; offline reconstruction pending",
        },
        "context_gate": context,
        "max_model_len": server["max_model_len"],
        "model_key": args.model_key,
        "deployment": args.deployment,
        "k": args.k,
        "acceptance_control": server["acceptance_control"],
        "phase": args.phase,
        "concurrency": args.concurrency,
        "repetition": args.repetition,
        "producer_revision": PRODUCER,
        "corpus": CORPUS,
        "command": command,
        "fixed_length_gate": fixed,
        "collection_only": args.collection_only,
        "pilot_gate": pilots,
        "server_identity_sha256": digest(args.server_identity),
        "strict_scenario_requested": args.phase == "profile",
        "replay_policy": {
            "warmup": "live_trajectories",
            "output_policy": "recorded_out_minimum_one_ignore_eos",
            "truncation_or_filtering": False,
            "failed_request_threshold": 0.1,
            "seed": 42,
            "snapshot_ratio": [0.25, 0.75],
            "burst_phase_starts": False,
            "warmup_requests_per_lane": 10,
            "response_grace_seconds": 30,
            "cancel_drain_seconds": 10,
            "tree_idle_cap_seconds": 300,
            "global_idle_cap_seconds": 10,
        },
        "scoring_eligible": False,
        "remaining_scoring_gates": [
            "Check every sampled request native usage/context validation and all failures; "
            "unsampled corpus remains unverified.",
            "Match actual producer snapshot/frontier and synthesized request/prefix identities to Replay.",
            "Match server cleanup versus drained-wire warmup boundary and lifecycle/cache behavior.",
            "Match output-stop, cancellation, acceptance, routing and client metric cohorts.",
        ],
    }
    receipt = {
        "schema": "dspark-agentx-phaseb-v1",
        "status": "prepared",
        "identity": identity,
    }
    path = args.output / "run.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.execute:
        env = {
            **os.environ,
            "DSPARK_AGENTX_CAPTURE_DIR": str((args.output / "capture").resolve()),
            "DSPARK_AGENTX_CORPUS_RECEIPT": str((args.output / "corpus.json").resolve()),
            "AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT": "10",
            "AIPERF_DATASET_MMAP_CACHE_ENABLED": "false",
        }
        try:
            with (args.output / "client.log").open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt["returncode"] = completed.returncode
            corpus = read_json(args.output / "corpus.json")
            if completed.returncode or any(corpus.get(key) != value for key, value in CORPUS.items()):
                raise ValueError("AgentX client failed or did not use the content-verified corpus")
            receipt["status"] = "completed_client_run"
            receipt["artifacts"] = {
                str(p.relative_to(args.output)): digest(p)
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and p != path
            }
        except Exception as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
