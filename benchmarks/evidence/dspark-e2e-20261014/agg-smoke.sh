#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${storage_root:?Set authorized personal storage root}"
: "${DSPARK_MODEL_KEY:?Set approved model key}"
: "${DSPARK_K:?Set approved K}"
cd "$storage_root"
mkdir -p "$storage_root/.cache/dspark-e2e-20261014"
export HF_HOME=/workspace/.cache/dspark-e2e-20261014/huggingface
export XDG_CACHE_HOME=/workspace/.cache/dspark-e2e-20261014
run_args=(--mode "${DSPARK_RUN_MODE:-smoke}" --graph-policy "${DSPARK_GRAPH_POLICY:-eager}"
  --startup-timeout "${DSPARK_STARTUP_TIMEOUT:-3600}")
if [[ -n "${DSPARK_LOAD_STRATEGY:-}" ]]; then
  run_args+=(--load-strategy "$DSPARK_LOAD_STRATEGY")
fi
if [[ -n "${DSPARK_CONCURRENCY:-}" ]]; then
  run_args+=(--concurrency "$DSPARK_CONCURRENCY")
fi
if [[ -n "${DSPARK_REPETITION:-}" ]]; then
  run_args+=(--repetition "$DSPARK_REPETITION")
fi
if [[ -n "${DSPARK_SYNTHETIC_CONTROL:-}" ]]; then
  run_args+=(--synthetic-control "$DSPARK_SYNTHETIC_CONTROL")
fi
if [[ -n "${DSPARK_SYNTHETIC_AL:-}" ]]; then
  run_args+=(--synthetic-al "$DSPARK_SYNTHETIC_AL")
fi
if [[ -n "${DSPARK_INPUT_LENGTH:-}" ]]; then
  run_args+=(--input-length "$DSPARK_INPUT_LENGTH")
fi
if [[ -n "${DSPARK_CELL_IDS:-}" ]]; then
  read -r -a selected_cells <<< "$DSPARK_CELL_IDS"
  for cell in "${selected_cells[@]}"; do run_args+=(--cell-id "$cell"); done
fi
if [[ -n "${DSPARK_COMPLETED_RUNS:-}" ]]; then
  run_args+=(--completed-runs "$DSPARK_COMPLETED_RUNS")
fi
bundle="${DSPARK_BUNDLE_DIR:-$storage_root/jobs/dspark-e2e-20261014}"
if [[ "$bundle" != "$storage_root/"* ]]; then
  echo "DSPARK_BUNDLE_DIR must be under storage_root" >&2; exit 2
fi
container_bundle="/workspace/${bundle#"$storage_root/"}"
run_args+=(--context-length "${DSPARK_CONTEXT_LENGTH:-16384}")
srun --ntasks=1 \
  --container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh" \
  --container-mounts="$storage_root:/workspace,/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:ro" \
  --container-workdir="$container_bundle" \
  --container-env=HF_HOME,XDG_CACHE_HOME \
  /usr/bin/python3 serve_smoke.py --model-key "$DSPARK_MODEL_KEY" --k "$DSPARK_K" \
  --output "/workspace/results/dspark-e2e-20261014/${DSPARK_RUN_MODE:-smoke}-$DSPARK_MODEL_KEY-k$DSPARK_K-$SLURM_JOB_ID" "${run_args[@]}"
