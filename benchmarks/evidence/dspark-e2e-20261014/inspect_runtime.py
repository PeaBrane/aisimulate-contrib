# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record package/source and read-only checkpoint identity on a CPU allocation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from campaign import IMAGE, MODELS, RUNTIME_REVISION

SHARED_MODELS = Path("/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model")


def checkpoint_path(model, revision):
    root = SHARED_MODELS / model.lower().replace("/", "_") / "hf"
    candidates = [root / f"hf-{revision}_orig", root / f"hf-{revision[:7]}_orig"]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"exact revision not available in {root}")


def inspect_checkpoint(model, revision):
    path = checkpoint_path(model, revision)
    index = path / "model.safetensors.index.json"
    data = json.loads(index.read_text()) if index.exists() else {}
    weights = sorted(set(data.get("weight_map", {}).values())) or [p.name for p in path.glob("*.safetensors")]
    if not weights:
        raise ValueError(f"no checkpoint weights in {path}")
    missing = [name for name in weights if not (path / name).is_file()]
    if missing:
        raise ValueError(f"missing weight shards in {path}: {missing}")
    config = path / "config.json"
    references = json.loads(Path(__file__).with_name("checkpoint-references.json").read_text())
    expected = next(row["config_sha256"] for row in references if row["model"] == model and row["revision"] == revision)
    if hashlib.sha256(config.read_bytes()).hexdigest() != expected:
        raise ValueError(f"shared config content differs from pinned Hugging Face config: {path}")
    return {
        "model": model,
        "revision": revision,
        "path": str(path),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "architectures": json.loads(config.read_text()).get("architectures"),
        "quantization": json.loads(config.read_text()).get("quantization_config"),
        "weight_files": len(weights),
        "weight_bytes": sum((path / name).stat().st_size for name in weights),
        "revision_evidence": (
            "shared revision directory and pinned Hugging Face config hash verified; full weight content not rehashed"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", choices=list(MODELS), action="append")
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("vllm")
    root = Path(distribution.locate_file("vllm"))
    paths = [
        "config/speculative.py",
        "models/deepseek_v4/nvidia/dspark.py",
        "models/deepseek_v41/nvidia/dspark.py",
        "models/kimi_k3/nvidia/dspark_mla.py",
        "v1/worker/gpu/spec_decode/dspark/speculator.py",
        "distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py",
    ]
    sources = {}
    for name in paths:
        path = root / name
        sources[name] = {
            "present": path.is_file(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    configs = []
    for key in args.model_key or MODELS:
        model = MODELS[key]
        configs.append(inspect_checkpoint(model["model"], model["revision"]))
        if "draft_model" in model:
            configs.append(inspect_checkpoint(model["draft_model"], model["draft_revision"]))
    result = {
        "candidate_image": IMAGE,
        "candidate_runtime_revision": RUNTIME_REVISION,
        "actual_vllm_version": distribution.version,
        "sources": sources,
        "checkpoints": configs,
        "gpu_execution": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "actual_vllm_version": distribution.version,
                "checkpoints": [{"model": c["model"], "weight_bytes": c["weight_bytes"]} for c in configs],
            }
        )
    )


if __name__ == "__main__":
    main()
