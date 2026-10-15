#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${storage_root:?Set authorized personal storage root}"
cd "$storage_root"
srun --ntasks=1 \
  --container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh" \
  --container-mounts="$storage_root:/workspace,/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:ro" \
  --container-workdir=/workspace/jobs/dspark-e2e-20261014 \
  /usr/bin/python3 inspect_runtime.py --output "/workspace/results/dspark-e2e-20261014/runtime-$SLURM_JOB_ID.json"
