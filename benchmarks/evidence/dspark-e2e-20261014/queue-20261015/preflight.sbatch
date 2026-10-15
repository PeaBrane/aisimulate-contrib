#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${storage_root:?}"
: "${DSPARK_PLATFORM:?}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 2;; esac
export HF_HOME=/workspace/.cache/dspark-e2e-20261014/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --nodes=1 --ntasks=1 \
  --container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh" \
  --container-mounts="$storage_root:/workspace,/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:ro" \
  --container-workdir=/workspace/jobs/dspark-full-20261015/bundle \
  --container-env=HF_HOME,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,OMP_NUM_THREADS,MKL_NUM_THREADS \
  /usr/bin/python3 /workspace/jobs/dspark-full-20261015/queue/preflight.py \
  --bundle /workspace/jobs/dspark-full-20261015/bundle \
  --queue /workspace/jobs/dspark-full-20261015/queue \
  --platform "$DSPARK_PLATFORM" \
  --output "/workspace/results/dspark-full-20261015/preflight-$DSPARK_PLATFORM-$SLURM_JOB_ID"
