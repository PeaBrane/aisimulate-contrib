# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired complete-call Replay benchmark on synthetic FPM profiles."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from types import SimpleNamespace

from pytest import MonkeyPatch

from aisim.runner import _materialize_engine_execution_spec

root = Path(os.environ["AISIM_BENCHMARK_DIR"])
repo = Path.cwd()
test_path = repo / "python/aisim/tests/unit/sdk/test_fpm_forward.py"
test_spec = importlib.util.spec_from_file_location("fpm_benchmark_fixtures", test_path)
fixtures = importlib.util.module_from_spec(test_spec)
sys.modules[test_spec.name] = fixtures
sys.path.insert(0, str(repo / "tests"))
test_spec.loader.exec_module(fixtures)


def runtime(name, path):
    spec = importlib.util.spec_from_file_location(f"{name}._runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def input_grid(row):
    rows = [
        row("prefill", b, t, k, 0.9 + t * 0.014 + k * 0.0004, 440 + 0.1 * t + k * 0.03)
        for b in range(1, 33)
        for t in sorted({b, 32, 64, 128, 256, 384, 512})
        for k in (
            0,
            b,
            4 * b,
            8 * b,
            16 * b,
            32 * b,
            64 * b,
            128 * b,
            256 * b,
            512 * b,
            1024 * b,
        )
    ]
    rows.extend(
        row("decode", b, 0, k, 0.9 + 0.20 * b + 0.004 * k / b, 350 + 8 * b + 0.004 * k)
        for b in range(1, 33)
        for k in (b, 128 * b, 512 * b, 1024 * b, 2048 * b)
    )
    return rows


def make_case(dp, method, powered, requests):
    data = root / f"dp{dp}-{method}-v{8 if powered else 6}"
    data.mkdir(exist_ok=True)
    if method == "direct":
        return make_direct_case(data, dp, requests)
    systems = data / "systems"
    systems.mkdir(exist_ok=True)
    shutil.copy(Path(fixtures._CORE_SYSTEMS) / f"{fixtures.SYSTEM}.yaml", systems)
    os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = str(systems)
    os.environ["AIC_ALLOW_UNLISTED_VERSIONS"] = "1"
    model = fixtures.models.get_model(
        "Qwen/Qwen3-0.6B",
        fixtures._model_config(forward_model="fpm", attention_dp_size=dp, moe_ep_size=dp),
        fixtures.BACKEND,
    )
    identity = dict(
        zip(
            fixtures._CELL_MATCH_COLUMNS,
            model.context_ops[0]._match_identity,
            strict=True,
        )
    )

    def make_row(phase, batch, tokens, kv, time_ms, power):
        row = fixtures._row(
            phase,
            batch,
            tokens,
            kv,
            time_ms,
            model_path=model.model_path,
            identity=identity,
        )
        if powered:
            policy = "dynamo_native_single_sample_power_window_v1"
            row.update(
                model_config_sha256="",
                execution_profile="full",
                engram_residency="none",
                input_modality="text",
                measurement_policy=policy,
                measurement_repeats=1,
                power_w=power,
            )
        return row

    rows = input_grid(make_row)
    overrides = (
        {
            "schema_version": 8,
            "measurement_policy": "dynamo_native_single_sample_power_window_v1",
        }
        if powered
        else None
    )
    path = fixtures._write_pair(str(data), rows, sidecar_overrides=overrides)
    engine = {
        "model": model.model_path,
        "hardware": fixtures.SYSTEM,
        "backend": fixtures.BACKEND,
        "backend_version": fixtures.VERSION,
        "context_length": 1024,
        "workers": {
            "aggregated": {
                "parallelism": {
                    "replicas": 1,
                    "tensor": 1,
                    "pipeline": 1,
                    "attention_data": dp,
                    "moe_tensor": 1,
                    "moe_expert": dp,
                },
                "scheduler": {"max_batched_tokens": 512, "max_sequences": 32},
                "kv_cache": {
                    "prefix_caching": False,
                    "capacity": {"type": "fixed", "blocks": 4096},
                },
                "timing": {
                    "type": "default",
                    "forward_model": "fpm",
                    "estimator_config": {
                        "fpm_interpolation": {
                            "fpm_parquet_path": path,
                            "method": method,
                        }
                    },
                },
            }
        },
    }
    return make_replay_args(engine, requests), path


def make_direct_case(data, dp, requests):
    import pyarrow.parquet as pq
    import test_fpm_profile_workflow as workflow

    factory = SimpleNamespace(mktemp=lambda name: Path(mkdtemp(prefix=name, dir=data)))
    profile = workflow.profile.__wrapped__(factory)
    profile["deployments"][1]["resources"]["max_batch_size"] = 32
    temp = Path(mkdtemp(prefix="timing-", dir=data))
    with MonkeyPatch.context() as patch:
        systems = workflow.timing_systems.__wrapped__(profile, temp, patch)
    os.environ["AIC_ALLOW_UNLISTED_VERSIONS"] = "1"
    table = Path(systems) / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    template = next(row for row in pq.read_table(table).to_pylist() if row["dp"] == dp)
    policy = "dynamo_native_single_sample_power_window_v1"

    def row(phase, b, t, k, time_ms, power):
        return {
            **template,
            "cell_id": f"{phase}-{b}-{t}-{k}",
            "workload_kind": phase,
            "batch_size": b,
            "total_prefill_tokens": t,
            "total_kv_read_tokens": k,
            "latency_ms": time_ms,
            "power_w": power,
            "model_config_sha256": "",
            "execution_profile": "full",
            "engram_residency": "none",
            "input_modality": "text",
            "measurement_policy": policy,
            "measurement_repeats": 1,
        }

    rows = input_grid(row)
    path = fixtures._write_pair(
        str(table.parent),
        rows,
        sidecar_overrides={
            "schema_version": 8,
            "measurement_policy": policy,
            "backend_version": "0.25.1",
        },
    )
    engine = workflow._engine(profile)
    engine.update(systems_paths=[systems], context_length=1024)
    worker = workflow._worker(dep=dp == 2)
    worker["scheduler"] = {"max_batched_tokens": 512, "max_sequences": 32}
    engine["workers"]["aggregated"] = worker
    return make_replay_args(engine, requests), path


def make_replay_args(engine, requests):
    raw = {
        "engine": engine,
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 512, "output_tokens": 32},
            "load": {"type": "poisson", "requests_per_second": 200, "seed": 7},
            "stop": {"requests": requests},
        },
    }
    config = fixtures.CorePredictionConfig.model_validate(raw)
    spec = fixtures.prediction_to_replay_spec(config)
    args = _materialize_engine_execution_spec(spec, trace_block_size=512, record_per_request=False)
    if "spec" not in args:
        args = {"spec": args}
    args["determinism"] = "canonical_v1"
    return args


def summarize(result, requests):
    assert result["completed_requests"] == requests, result["completed_requests"]
    return {
        key: result[key]
        for key in (
            "completed_requests",
            "duration_ms",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "power_w",
            "power_coverage",
        )
    }


def complete_call(module, spec):
    start = perf_counter()
    output = module.run_replay_json(spec)
    return (perf_counter() - start) * 1000, json.loads(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=20, type=int)
    options = parser.parse_args()
    baseline = runtime("baseline", root / "base-runtime.so")
    candidate = runtime("candidate", root / "change-runtime.so")
    for dp, method, powered in [
        (2, "sol", True),
        (2, "direct", True),
        (1, "sol", False),
    ]:
        requests = 1024 if powered else 128
        args, path = make_case(dp, method, powered, requests)
        serial = json.dumps(args, separators=(",", ":"))
        base_warm = [complete_call(baseline, serial)[1] for _ in range(2)][-1]
        change_warm = [complete_call(candidate, serial)[1] for _ in range(2)][-1]
        first, second = summarize(base_warm, requests), summarize(change_warm, requests)
        for key in first.keys() & second.keys() - {"power_w", "power_coverage"}:
            assert first[key] == second[key], (key, first[key], second[key])
        assert (second["power_w"] is not None) == powered, second
        assert second["power_coverage"] == (1.0 if powered else None), second
        detail = json.loads(candidate.run_replay_json(json.dumps({**args, "capture_performance_diagnostics": True})))
        estimates = [
            rec
            for group in detail["fpm_query_evidence"]["providers"]
            for phase in group["phases"]
            for op in phase["operations"]
            for rec in op["fpm_estimates"]
        ]
        mixed = sum(
            rec["count"]
            for rec in estimates
            if any(
                rank["prefill_ms"] is not None and rank["decode_baseline_ms"] is not None
                for rank in rec["estimate"]["ranks"]
            )
        )
        assert mixed > 0, "workload produced no mixed rank"
        print(
            json.dumps(
                {
                    "case": [dp, method, powered],
                    "requests": requests,
                    "mixed_rank_calls": mixed,
                    "input_sha256": hashlib.sha256(serial.encode()).hexdigest(),
                    "parquet_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "baseline": first,
                    "candidate": second,
                }
            ),
            flush=True,
        )
        gc.collect()
        gc.disable()
        try:
            samples = []
            for i in range(options.pairs):
                modules = [(baseline, "baseline"), (candidate, "candidate")]
                if i % 2:
                    modules.reverse()
                sample = {}
                for module, name in modules:
                    elapsed, output = complete_call(module, serial)
                    assert summarize(output, requests) == (first if name == "baseline" else second)
                    sample[name] = elapsed
                samples.append(sample)
        finally:
            gc.enable()
        print(
            json.dumps(
                {
                    "case": [dp, method, powered],
                    "samples_ms": samples,
                    "baseline_median_ms": statistics.median(s["baseline"] for s in samples),
                    "candidate_median_ms": statistics.median(s["candidate"] for s in samples),
                    "paired_delta_pct": statistics.median((s["candidate"] / s["baseline"] - 1) * 100 for s in samples),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
