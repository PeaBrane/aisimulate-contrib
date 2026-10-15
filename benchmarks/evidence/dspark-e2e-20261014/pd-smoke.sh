#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Allocate both P/D roles together: Flash=1 node/8 GPUs; Pro/Kimi=2 nodes/8 GPUs each.
set -euo pipefail
: "${storage_root:?Set authorized personal storage root}"
: "${DSPARK_MODEL_KEY:?Set approved model key}"
: "${DSPARK_K:?Set approved K}"
cd "$storage_root"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
prefill_host="${nodes[0]}"
decode_host="${nodes[1]:-${nodes[0]}}"
width=8
role_memory=1200G
if [[ "$DSPARK_MODEL_KEY" == v41-flash ]]; then width=4; role_memory=768G; fi
if [[ "$DSPARK_MODEL_KEY" == kimi-k3 ]]; then role_memory=2048G; fi
if [[ "$width" == 8 && "${#nodes[@]}" != 2 ]]; then
  echo "Pro/Kimi P/D requires two nodes in the same allocation" >&2; exit 1
fi
export HF_HOME=/workspace/.cache/dspark-e2e-20261014/huggingface
export XDG_CACHE_HOME=/workspace/.cache/dspark-e2e-20261014
bundle="${DSPARK_BUNDLE_DIR:-$storage_root/jobs/dspark-e2e-20261014}"
result="/workspace/results/dspark-e2e-20261014/pd-${DSPARK_RUN_MODE:-smoke}-$DSPARK_MODEL_KEY-k$DSPARK_K-$SLURM_JOB_ID"
mkdir -p "$storage_root/${result#/workspace/}/source"
cp "$bundle/"*.py \
  "$bundle/"*.json \
  "$storage_root/${result#/workspace/}/source/"
container=(--container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh"
  --container-mounts="$storage_root:/workspace,/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:ro"
  --container-workdir="$result/source"
  --container-env=HF_HOME,XDG_CACHE_HOME,VLLM_NIXL_SIDE_CHANNEL_HOST,VLLM_NIXL_SIDE_CHANNEL_PORT)
role_args=()
if [[ "${DSPARK_OBSERVE_SOURCE_POOL:-0}" == 1 ]]; then
  role_args+=(--observe-source-pool)
fi
if [[ -n "${DSPARK_SYNTHETIC_AL:-}" ]]; then
  role_args+=(--synthetic-al "$DSPARK_SYNTHETIC_AL")
fi
if [[ -n "${DSPARK_SYNTHETIC_CONTROL:-}" ]]; then
  role_args+=(--synthetic-control "$DSPARK_SYNTHETIC_CONTROL")
fi
driver_args=(--mode "${DSPARK_RUN_MODE:-smoke}")
if [[ -n "${DSPARK_CONCURRENCY:-}" ]]; then
  driver_args+=(--concurrency "$DSPARK_CONCURRENCY")
fi
if [[ -n "${DSPARK_REPETITION:-}" ]]; then
  driver_args+=(--repetition "$DSPARK_REPETITION")
fi
if [[ -n "${DSPARK_WINDOWS:-}" ]]; then
  read -r -a selected_windows <<< "$DSPARK_WINDOWS"
  for window in "${selected_windows[@]}"; do
    if [[ ! "$window" =~ ^[14]:[123]$ ]]; then
      echo "Invalid DSPARK_WINDOWS entry: $window" >&2
      exit 2
    fi
    driver_args+=(--window "${window%:*}" "${window#*:}")
  done
fi
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
for role in prefill decode; do
  host="$prefill_host"; port=8010; side_port=5557
  gpu_memory_utilization="${DSPARK_PREFILL_GPU_MEMORY_UTILIZATION:-0.90}"
  if [[ "$role" == decode ]]; then
    host="$decode_host"; port=8020; side_port=5657
    gpu_memory_utilization="${DSPARK_DECODE_GPU_MEMORY_UTILIZATION:-0.90}"
  fi
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$host" VLLM_NIXL_SIDE_CHANNEL_PORT="$side_port"
  srun --nodes=1 --ntasks=1 --nodelist="$host" --exclusive --gres="gpu:$width" --cpus-per-task=24 --mem="$role_memory" \
    "${container[@]}" /usr/bin/python3 serve_role.py --model-key "$DSPARK_MODEL_KEY" \
    --role "$role" --k "$DSPARK_K" --port "$port" --output "$result" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --graph-policy "${DSPARK_GRAPH_POLICY:-runtime-default}" "${role_args[@]}" \
    > "$storage_root/${result#/workspace/}/$role.log" 2>&1 &
  pids+=("$!")
done
# The small proxy/client shares P's allocation; GPU workers remain in their own steps.
srun --nodes=1 --ntasks=1 --nodelist="$prefill_host" --overlap --gres=gpu:0 --cpus-per-task=2 --mem=4G \
  "${container[@]}" /usr/bin/python3 pd_driver.py --model-key "$DSPARK_MODEL_KEY" --k "$DSPARK_K" \
  --prefill-host "$prefill_host" --decode-host "$decode_host" --output "$result" "${driver_args[@]}" \
  --proxy-script /workspace/jobs/dspark-e2e-20261014/upstream/toy_proxy_server.py &
driver_pid=$!
while kill -0 "$driver_pid" 2>/dev/null; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "A P/D role exited before the client completed" >&2
      kill "$driver_pid" 2>/dev/null || true
      wait "$driver_pid" || true
      exit 1
    fi
  done
  sleep 5
done
wait "$driver_pid"
