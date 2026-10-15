# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned AgentX P/D artifacts without producing unmatched accuracy scores."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

API = "https://inferencex.semianalysis.com/api/v1/"
ARTIFACTS = {
    "metrics": (
        11092304949,
        "113cf89286ff95e60f2464be69de299fd6f9fb5a583837134476c42600b7fab2",
    ),
    "agentic": (
        11092647256,
        "207500a11f1eeb59ee129393299a317139e3f32e2d8c4572ba4e3b25f2a910d5",
    ),
    "server-logs": (
        11092552440,
        "6eb1766a96e9017dc9bda11c9a5905845121e65b982581eb6182a617a6fd5a7a",
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def api(raw, filename, route, params, download):
    path = raw / filename
    url = API + route + "?" + urllib.parse.urlencode(params)
    if download:
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text()), {"url": url, "sha256": digest(path)}


def launches(text):
    result = []
    for line in text.splitlines():
        if "Command: python3 -m dynamo.vllm" not in line:
            continue
        words = shlex.split(line.split("Command: ", 1)[1])

        def value(flag):
            return words[words.index(flag) + 1] if flag in words else None

        result.append(
            {
                "role": value("--disaggregation-mode"),
                "tp": int(value("--tensor-parallel-size")),
                "attention_dp": int(value("--data-parallel-size")),
                "local_dp": int(value("--data-parallel-size-local")),
                "speculation": json.loads(value("--speculative-config")),
                "connector": json.loads(value("--kv-transfer-config")),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw/agentx-pd")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "agentx-pd-audit.json")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch pinned archives and current API receipts",
    )
    args = parser.parse_args()
    raw = args.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    receipts = []
    for name, (artifact_id, expected) in ARTIFACTS.items():
        path = raw / f"{name}.zip"
        if args.download and not path.exists():
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                subprocess.run(
                    [
                        "gh",
                        "api",
                        f"repos/SemiAnalysisAI/InferenceX/actions/artifacts/{artifact_id}/zip",
                    ],
                    stdout=stream,
                    check=True,
                )
            temporary.replace(path)
        observed = digest(path)
        if observed != expected:
            raise ValueError(f"Artifact {artifact_id} checksum mismatch: {observed}")
        receipts.append({"artifact_id": artifact_id, "sha256": observed})

    selected = {}
    with (
        zipfile.ZipFile(raw / "server-logs.zip") as archive,
        archive.open("multinode_server_logs.tar.gz") as compressed,
        tarfile.open(fileobj=compressed, mode="r|gz") as files,
    ):
        for member in files:
            name = Path(member.name).name
            wanted = name in {"26842.json", "sweep_26842.log"} or (
                name.startswith(("watchtower-navy-", "fingerprint_")) and name.endswith(".json")
            )
            if not member.isfile() or not wanted or name in selected:
                continue
            content = files.extractfile(member).read()
            selected[name] = content
            (raw / f"archive-{name}").write_bytes(content)
    configs = {}
    for name, content in selected.items():
        if not name.endswith("_config.json"):
            continue
        config = json.loads(content)
        engine = config["config"]["engine_args"]
        keys = (
            "model",
            "revision",
            "tensor_parallel_size",
            "data_parallel_size",
            "data_parallel_start_rank",
            "enable_expert_parallel",
            "speculative_config",
            "kv_transfer_config",
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "kv_cache_dtype",
            "kv_cache_memory_bytes",
            "enable_prefix_caching",
            "prefill_schedule_interval",
            "compilation_config",
            "attention_config",
            "dtype",
            "quantization",
            "moe_backend",
            "async_scheduling",
            "disable_hybrid_kv_cache_manager",
        )
        configs[name] = {
            "role": config["config"]["disaggregation_mode"],
            "runtime": config["vllm_version"],
            "engine": {key: engine.get(key) for key in keys},
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    with zipfile.ZipFile(raw / "metrics.zip") as archive:
        metrics = json.loads(archive.read(archive.namelist()[0]))
    with zipfile.ZipFile(raw / "agentic.zip") as archive:
        prefix = "conc_32/aiperf_artifacts/"
        export = json.loads(archive.read(prefix + "profile_export_aiperf.json"))
        rows = [json.loads(line) for line in archive.read(prefix + "profile_export.jsonl").splitlines()]
        benchmark = archive.read("conc_32/benchmark.log").decode()
    counts = Counter((row["metadata"]["benchmark_phase"], bool(row.get("error"))) for row in rows)
    if counts["profiling", False] != metrics["num_requests_successful"]:
        raise ValueError("Pinned metrics and request records disagree")

    siblings, receipt = api(raw, "siblings.json", "benchmark-siblings", {"id": 444916}, args.download)
    receipts.append(receipt)
    api_names, receipt = api(raw, "api-files-444916.json", "server-log-files", {"id": 444916}, args.download)
    receipts.append(receipt)
    api_sweep = next(name for name in api_names if name.startswith("sweep_"))
    api_log, receipt = api(
        raw,
        "api-sweep-444916.json",
        "server-log",
        {"id": 444916, "file": api_sweep, "offset": 0, "limit": 262144},
        args.download,
    )
    receipts.append(receipt)
    api_job = int(re.search(r"sweep_(\d+)", api_sweep).group(1))
    api_attempt = re.search(r"srt-slurm-36425419107-(\d+)-", api_log["serverLog"])
    candidates = []
    for point in siblings["siblings"]:
        if not point["disagg"] or point["offload_mode"] != "off":
            continue
        bid = point["id"]
        if bid == 444916:
            commands = launches(selected["sweep_26842.log"].decode())
            source = {
                "artifact_id": ARTIFACTS["server-logs"][0],
                "job_id": 26842,
                "attempt": 3,
            }
        else:
            names, receipt = api(raw, f"files-{bid}.json", "server-log-files", {"id": bid}, args.download)
            receipts.append(receipt)
            name = next(name for name in names if name.startswith("sweep_"))
            log, receipt = api(
                raw,
                f"sweep-{bid}.json",
                "server-log",
                {"id": bid, "file": name, "offset": 0, "limit": 262144},
                args.download,
            )
            receipts.append(receipt)
            commands = launches(log["serverLog"])
            source = {
                "file": name,
                "partial": log.get("nextOffset") is not None,
                "attempt_alignment": "not_qualified",
            }
        candidates.append({"metadata": point, "launches": commands, "launch_source": source})
    result = {
        "benchmark_id": 444916,
        "workflow_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36425419107/attempts/3",
        "workflow_revision": "f4ad60fba73439cf4833a674e6a84feb2014bfaf",
        "job_id": 26842,
        "source_receipts": receipts,
        "archive_entries": {name: hashlib.sha256(value).hexdigest() for name, value in selected.items()},
        "configs": configs,
        "fingerprints": {
            name: {k: json.loads(value)[k] for k in ("frameworks", "model")}
            for name, value in selected.items()
            if name.startswith("fingerprint_")
        },
        "candidate_points": candidates,
        "profiling_success": counts["profiling", False],
        "profiling_error": counts["profiling", True],
        "warmup_records": counts["warmup", False] + counts["warmup", True],
        "aiperf_measured": {
            k: export[k]
            for k in (
                "time_to_first_token",
                "inter_token_latency",
                "output_token_throughput",
                "benchmark_duration",
                "completed_request_count",
                "branch_stats",
                "run_info",
            )
        },
        "published_measured": metrics["request_metrics"],
        "workload_observations": {
            "dataset_revision": sorted(set(re.findall(r"revision/([a-f0-9]{40})", benchmark))),
            "accelerated_warmup": "replaying live trajectories" in benchmark,
            "handoff_live_streams": re.findall(r"handoff: persisted (\d+) live streams", benchmark),
        },
        "eligible_paired_points": 0,
        "predictions": [],
        "mape": None,
        "signed_bias": None,
        "status": "pending_runtime_and_workload_contract",
        "api_log_alignment": {
            "benchmark_id": 444916,
            "observed_api_job": api_job,
            "observed_api_attempt": int(api_attempt.group(1)) if api_attempt else None,
            "selected_artifact_job": 26842,
            "selected_artifact_attempt": 3,
            "job_matches": api_job == 26842,
            "action": "Use pinned attempt-3 artifacts for the selected point; "
            "do not combine API logs with its metrics.",
        },
        "blockers": [
            "Adaptive verification is absent from launch/config dumps; "
            "exact vLLM revision 20ebb2850 is not publicly resolvable.",
            "Synthetic AL=3.61 is observed; its bonus-token convention still needs the exact runtime source.",
            "Producer drains issued wire requests then preserves 63 unfinished streams across 32 lanes; "
            "the live_trajectories policy now models this DAG handoff, but snapshot/input identity "
            "and extra server-cleanup delay remain unqualified.",
            "Random routing with session affinity and the NIXL target/draft payload "
            "need matched-runtime qualification.",
            "Recorded token lengths do not establish token/prefix identity, exact session replay or completion policy.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
