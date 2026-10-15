#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run only inside an authorized CPU data-mover allocation.
set -euo pipefail
: "${storage_root:?Set the authorized personal storage root}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 1 ;; esac
export ENROOT_CONFIG_PATH="$storage_root/.config/enroot"
export ENROOT_CACHE_PATH="$storage_root/.cache/enroot"
export ENROOT_DATA_PATH="$storage_root/.local/share/enroot"
export TMPDIR="$storage_root/tmp/dspark-e2e-20261014"
mkdir -p "$ENROOT_CONFIG_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$TMPDIR" "$storage_root/images"
image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh"
if [[ ! -s "$image" ]]; then
  enroot import -o "$image.partial-$SLURM_JOB_ID" docker://vllm/vllm-openai@sha256:3d17635c5aa0340e2f2502c067ce4b587d8e7dc678195e90b71142490dc6d71c
  mv "$image.partial-$SLURM_JOB_ID" "$image"
fi
sha256sum "$image" > "$image.sha256"
cat "$image.sha256"
# Preserve the unmodified upstream proxy's license/copyright header. Its bytes
# are a runtime dependency outside this repository, verified before execution.
proxy_dir="$storage_root/jobs/dspark-e2e-20261014/upstream"
mkdir -p "$proxy_dir"
curl --fail --location --silent --show-error \
  https://raw.githubusercontent.com/vllm-project/vllm/ac9126e58aa7bbab1856ba6593ba4d5003fea516/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --output "$proxy_dir/toy_proxy_server.py.partial"
printf '%s  %s\n' 1210e5525cc0c57f891734b682d3f8939aa7456a672bea16b44f03b6707136ff "$proxy_dir/toy_proxy_server.py.partial" | sha256sum --check
mv "$proxy_dir/toy_proxy_server.py.partial" "$proxy_dir/toy_proxy_server.py"
