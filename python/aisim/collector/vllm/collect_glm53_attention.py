# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the vLLM GLM-5.3-Flash attention collector where only ``collect.py`` can be started.

The table is measured by ``glm53flash_attention_runner`` on the real checkpoint under a frozen
manifest (``glm53flash_attention_launch.freeze``), one attempt per deployment (checkpoint, TP)
and context class. The launcher renders a Slurm script for that; this module does the same
steps inside a collection job:

1. fetch the pinned checkpoint revision (``fetch_checkpoint``),
2. freeze an attempt manifest and run the runner in a subprocess that owns the visible GPUs
   (``run_attempt``), with the serving environment the launcher exports,
3. repeat with the request sets that are still open (``run_deployment``): a set that fails
   is skipped for the rest of the pass and retried at the next ``gpu_memory_utilization`` of
   ``UTILIZATION_LADDER``. That knob only moves memory between vLLM's KV pool and the
   headroom the per-target module graphs need; the 0.31.0 tables needed two to four values
   per deployment.

The attempt directories stay in the output directory. ``glm53flash_attention_contract
finalize`` admits them into the table (``--partial`` for an attempt without a completion
receipt); no row goes through ``log_perf``, because the table has its own schema and writer.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from collector.glm53flash_attention_contract import (
    CHECKPOINT_CONFIG_SHA256,
    CHECKPOINTS,
    MACHINE_ARCH,
    OP_NAME,
    build_plan,
    context_class_sets,
    sha256_json,
)

# GPU name (torch.cuda.get_device_name) -> system YAML of the systems this table ships for.
SYSTEM_OF_DEVICE = {
    "NVIDIA B300 SXM6 AC": "b300_sxm",
    "NVIDIA GB200": "gb200",
    "NVIDIA GB300": "gb300",
    "NVIDIA H100 80GB HBM3": "h100_sxm",
    "NVIDIA H200": "h200_sxm",
}
# vLLM --gpu-memory-utilization values tried in order (capacity only, recorded per attempt).
UTILIZATION_LADDER = (0.85, 0.7, 0.92, 0.55)
SWEEP = Path(__file__).resolve().parents[1] / "cases" / "base_ops" / "glm53flash_attention.yaml"
# A pre-staged checkpoint root (<root>/<fp8|nvfp4>) replaces the download. The manifest pins
# the revision and the runner checks config.json and tokenizer.json either way.
CHECKPOINT_ROOT_ENV = "GLM53_ATTN_CHECKPOINT_ROOT"
RUNNER = "collector.vllm.glm53flash_attention_runner"
# The output directory is archived every few minutes to a volume other jobs share. A vLLM
# worker that dies on a CUDA error dumps core (17 GB a worker on GB200, into the working
# directory), and the runner's log has no size bound. So the runner dumps no core, and a file
# past this size leaves the output directory; the measurements are a few MB a rank.
OVERSIZE_BYTES = 64 << 20
PARK_TAIL_BYTES = 64 << 10


def _job_volume() -> Path:
    # /dev/shm is host memory under the pipeline's --ipc=host; the job volume is the only
    # disk a collection job owns besides the result directory, which is archived.
    volume = Path("/mnt/ci_job")
    return volume if volume.is_dir() else Path.home()


def checkpoint_root() -> Path:
    forced = os.environ.get(CHECKPOINT_ROOT_ENV)
    return Path(forced) if forced else _job_volume() / "glm53flash-checkpoints"


def park_oversize(directory: Path, parking: Path | None = None) -> list[dict]:
    """Move every file under ``directory`` larger than ``OVERSIZE_BYTES`` to ``parking`` and
    leave ``<name>.parked`` (its size, where it went and its tail) in its place. A rename on
    the job volume, so a process that has the file open keeps writing to it there."""
    attempt = directory
    parking = parking or _job_volume() / "glm53flash-oversize" / attempt.name
    parked = []
    for path in sorted(p for p in attempt.rglob("*") if p.is_file() and not p.is_symlink()):
        size = path.stat().st_size
        if size <= OVERSIZE_BYTES:
            continue
        target = parking / path.relative_to(attempt)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as stream:
            stream.seek(max(0, size - PARK_TAIL_BYTES))
            tail = stream.read().decode(errors="replace")
        os.replace(path, target)
        note = {"file": str(path.relative_to(attempt)), "bytes": size, "moved_to": str(target)}
        path.with_name(path.name + ".parked").write_text(json.dumps({**note, "tail": tail}, indent=1))
        print("GLM53_OVERSIZE " + json.dumps(note), flush=True)
        parked.append(note)
    return parked


def _config_ok(path: Path, checkpoint: str) -> bool:
    config = path / "config.json"
    return config.is_file() and sha256_json(json.loads(config.read_text())) == CHECKPOINT_CONFIG_SHA256[checkpoint]


def fetch_checkpoint(checkpoint: str, root: Path | None = None) -> tuple[Path, dict]:
    """The pinned revision of ``checkpoint`` under ``root``, downloaded from the Hub when absent."""
    model_id, revision = CHECKPOINTS[checkpoint]
    path = (root or checkpoint_root()) / checkpoint
    receipt = path / ".glm53-fetched.json"
    if (
        receipt.is_file()
        and json.loads(receipt.read_text()).get("revision") == revision
        and _config_ok(path, checkpoint)
    ):
        return path, {"reused": True, **json.loads(receipt.read_text())}
    if os.environ.get(CHECKPOINT_ROOT_ENV):
        if not _config_ok(path, checkpoint):
            raise RuntimeError(f"{path} does not hold config.json of {model_id}@{revision[:8]}")
        return path, {"reused": True, "staged": True, "revision": revision}
    from huggingface_hub import snapshot_download

    started = time.time()
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(path), max_workers=16)
    if not _config_ok(path, checkpoint):
        raise RuntimeError(f"downloaded config.json is not that of {model_id}@{revision[:8]}")
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    info = {
        "model_id": model_id,
        "revision": revision,
        "gib": round(size / (1 << 30), 1),
        "seconds": round(time.time() - started, 1),
    }
    receipt.write_text(json.dumps(info))
    return path, {"reused": False, **info}


def device_system() -> tuple[str, int]:
    """(system YAML name, visible GPU count) of the node the job runs on."""
    import torch

    name = torch.cuda.get_device_name(0)
    if name not in SYSTEM_OF_DEVICE:
        raise RuntimeError(f"no GLM-5.3-Flash attention system for GPU {name!r}")
    return SYSTEM_OF_DEVICE[name], torch.cuda.device_count()


def serving_kv_cache_dtype(sm_version: int) -> str:
    """The table's KV identity on this SM: BF16 on SM90 (the only KV the stock runtime
    serves there, see ``glm53flash_attention_contract.KV_CACHE_DTYPES``), FP8 elsewhere."""
    return "bfloat16" if 90 <= sm_version < 100 else "fp8"


def class_sets(context_class: str) -> list[str]:
    import yaml

    sweep = yaml.safe_load(SWEEP.read_text())["common_case_values"][OP_NAME]
    return context_class_sets(build_plan(sweep), context_class)


def attempt_state(attempt: Path) -> dict:
    """What one attempt directory measured: complete receipt, the sets whose every target
    passed, and the set that failed (with its error)."""
    raw = attempt / "raw"
    manifest = json.loads((attempt / "manifest.json").read_text())
    selected = manifest.get("only_sets") or [s["set_id"] for s in manifest["plan"]["sets"]]
    targets = {s["set_id"]: len(s["targets"]) for s in manifest["plan"]["sets"]}
    passed: dict[str, int] = {}
    failed: dict[str, str] = {}
    progress = raw / "progress.jsonl"
    for line in progress.read_text().splitlines() if progress.is_file() else []:
        entry = json.loads(line) if line.strip() else {}
        if entry.get("status") == "passed":
            passed[entry["set_id"]] = passed.get(entry["set_id"], 0) + 1
        elif entry.get("status") == "failed":
            failed[entry["set_id"]] = f"{entry.get('error_type')}: {entry.get('error')}"[:600]
    done = sorted(s for s in selected if passed.get(s, 0) == targets[s] and s not in failed)
    return {
        "attempt": attempt.name,
        "complete": (raw / "COMPLETE").is_file(),
        "utilization": manifest.get("vllm_gpu_memory_utilization"),
        "selected": selected,
        "done": done,
        "failed": failed,
        "targets_passed": sum(passed.values()),
    }


INTERRUPTED = "Interrupted"


def close_attempt(attempt: Path, error_type: str, error: str, every_set: bool = False) -> str | None:
    """Name the set an attempt was measuring when it ended without a completion receipt and
    without a failed set, so ``finalize --partial`` can admit what it finished. Returns the set.

    ``every_set`` names every selected set of an attempt that passed no target: its runner
    ended before the first measurement (the engine did not start), which no set caused, so
    the pass moves to the next utilization instead of spending one engine start per set."""
    state = attempt_state(attempt)
    if state["complete"] or state["failed"]:
        return None
    manifest = json.loads((attempt / "manifest.json").read_text())
    order = [s["set_id"] for s in manifest["plan"]["sets"] if s["set_id"] in set(state["selected"])]
    in_flight = next((s for s in order if s not in state["done"]), None)
    if in_flight is None:
        return None
    named = order if every_set and not state["targets_passed"] else [in_flight]
    (attempt / "raw").mkdir(exist_ok=True)
    with (attempt / "raw" / "progress.jsonl").open("a") as stream:
        for set_id in named:
            entry = {"set_id": set_id, "status": "failed", "error_type": error_type, "error": error}
            stream.write(json.dumps(entry) + "\n")
    return in_flight


def run_attempt(
    *,
    output: Path,
    name: str,
    system: str,
    checkpoint: str,
    tp_size: int,
    kv_cache_dtype: str,
    context_class: str,
    only_sets: list[str] | None,
    utilization: float,
    model_path: Path,
    timeout: float | None = None,
) -> dict:
    """Freeze one attempt under ``output/name`` and run the runner on it. Returns ``attempt_state``
    plus the runner's exit code; a failing set leaves a partial attempt, it does not raise."""
    from collector.glm53flash_attention_launch import freeze

    output = Path(output).resolve()
    attempt = output / name
    args = SimpleNamespace(
        backend="vllm",
        checkpoint=checkpoint,
        tp=tp_size,
        system=system,
        systems_dir=None,
        arch=MACHINE_ARCH[platform.machine()],
        gpus_per_node=None,
        kv_cache_dtype=kv_cache_dtype,
        config=str(model_path / "config.json"),
        sweep=str(SWEEP),
        context_class=context_class,
        attempt=str(attempt),
        source_commit=(os.environ.get("AISIM_COLLECTOR_REF") or os.environ.get("AIC_COLLECTOR_REF") or "unknown"),
        allocator_max_split_mb=None,
        sglang_mem_fraction=None,
        only_sets=only_sets,
        skip_sets=None,
        vllm_gpu_memory_utilization=utilization,
    )
    _, _, serving_env = freeze(args)
    root = str(Path(__file__).resolve().parents[2])
    env = {
        **os.environ,
        "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        **serving_env,
    }
    command = [
        sys.executable,
        "-m",
        RUNNER,
        "--manifest",
        str(attempt / "manifest.json"),
        "--output",
        str(attempt / "raw"),
        "--model-path",
        str(model_path),
        "--gpu-memory-utilization",
        str(utilization),
    ]
    started = time.time()
    finished = threading.Event()

    def keep_small() -> None:
        while not finished.wait(30):
            park_oversize(output)

    # inherited by the runner and its workers; set here, not in a preexec hook, because the
    # watcher thread is running when the runner is forked
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    watcher = threading.Thread(target=keep_small, daemon=True)
    watcher.start()
    try:
        with (attempt / "runner.log").open("w") as log:
            try:
                code = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                ).returncode
            except subprocess.TimeoutExpired:
                code = None
    finally:
        finished.set()
        watcher.join()
        park_oversize(output)
    if code != 0:
        # the runner names the failing set itself; when it dies earlier (engine start) or is
        # stopped, the set in flight is recorded here
        close_attempt(
            attempt,
            "RunnerExit",
            f"the runner ended with exit code {code} before naming a set; see runner.log",
            every_set=True,
        )
    state = attempt_state(attempt)
    kept = sum(p.stat().st_size for p in attempt.rglob("*") if p.is_file())
    state.update(returncode=code, seconds=round(time.time() - started, 1), output_bytes=kept)
    (attempt / "attempt_state.json").write_text(json.dumps(state, indent=1))
    return state


def open_sets(output: Path, checkpoint: str, tp_size: int, context_class: str) -> tuple[list[str], dict]:
    """The sets of the class no attempt under ``output`` measured, and per set the utilizations
    it failed at. Reads the attempt directories, so a resumed job continues where it stopped."""
    done: set[str] = set()
    failures: dict[str, dict[str, str]] = {}
    prefix = f"{checkpoint}-tp{tp_size}-{context_class}-"
    for attempt in sorted(p for p in output.glob(f"{prefix}*") if (p / "manifest.json").is_file()):
        if not (attempt / "attempt_state.json").is_file():
            # the job was stopped while this attempt ran (the caller holds the GPU lock, so it
            # is not running now): its set in flight is open again at the same utilization
            close_attempt(attempt, INTERRUPTED, "the job ended while this attempt was running")
        state = attempt_state(attempt)
        done.update(state["done"])
        for set_id, error in state["failed"].items():
            if not error.startswith(INTERRUPTED):
                failures.setdefault(set_id, {})[str(state["utilization"])] = error
    return [s for s in class_sets(context_class) if s not in done], failures


def run_deployment(
    *,
    output: Path,
    system: str,
    checkpoint: str,
    tp_size: int,
    kv_cache_dtype: str,
    context_class: str,
    model_path: Path,
) -> dict:
    """Passes over the open sets of one deployment and class until every set is measured or
    has failed at every utilization of the ladder. Returns the summary also written to
    ``<checkpoint>-tp<N>-<class>.summary.json``."""
    passes = []
    while True:
        remaining, failures = open_sets(output, checkpoint, tp_size, context_class)
        if not remaining:
            break
        choice = None
        for utilization in UTILIZATION_LADDER:
            todo = [s for s in remaining if str(utilization) not in failures.get(s, {})]
            if todo:
                choice = (utilization, todo)
                break
        if choice is None:
            break
        utilization, todo = choice
        index = len(list(output.glob(f"{checkpoint}-tp{tp_size}-{context_class}-*")))
        name = f"{checkpoint}-tp{tp_size}-{context_class}-{index:02d}-u{int(utilization * 100)}"
        state = run_attempt(
            output=output,
            name=name,
            system=system,
            checkpoint=checkpoint,
            tp_size=tp_size,
            kv_cache_dtype=kv_cache_dtype,
            context_class=context_class,
            only_sets=todo,
            utilization=utilization,
            model_path=model_path,
        )
        passes.append({k: state[k] for k in ("attempt", "complete", "utilization", "returncode", "seconds")})
        passes[-1].update(done=len(state["done"]), failed=state["failed"])
        if not state["done"] and not state["failed"]:
            # nothing measured and no set named: stop instead of repeating the same pass
            passes[-1]["aborted"] = True
            break
    remaining, failures = open_sets(output, checkpoint, tp_size, context_class)
    summary = {
        "deployment": f"{checkpoint}-tp{tp_size}",
        "context_class": context_class,
        "system": system,
        "kv_cache_dtype": kv_cache_dtype,
        "passes": passes,
        "open_sets": remaining,
        "failures": {s: failures[s] for s in remaining if s in failures},
    }
    (output / f"{checkpoint}-tp{tp_size}-{context_class}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def exclusive(output: Path):
    """One native deployment at a time: the runner owns every visible GPU."""
    lock = (output / ".glm53-attention-gpus.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock
