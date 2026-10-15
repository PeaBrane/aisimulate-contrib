#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run one gated AgentX window in an existing correctly sized allocation; never submit.
set -euo pipefail
: "${storage_root:?}" "${DSPARK_BUNDLE_DIR:?}" "${DSPARK_MODEL_KEY:?}"
: "${DSPARK_AGENTX_DEPLOYMENT:?}" "${DSPARK_AGENTX_PHASE:?}"
: "${DSPARK_AGENTX_CONCURRENCY:?}" "${DSPARK_AGENTX_REPETITION:?}"
: "${DSPARK_AGENTX_CONTEXT_AUDIT:?}" "${DSPARK_AGENTX_AIPERF:?}" "${DSPARK_AGENTX_PRODUCER:?}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 2;; esac
[[ "$DSPARK_BUNDLE_DIR" == "$storage_root/"* ]] || exit 2
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
prefill_host="${nodes[0]}"; decode_host="${nodes[1]:-${nodes[0]}}"
case "$DSPARK_MODEL_KEY" in
  v4-pro) width=8; k=6; al=3.77; role_memory=1200G;;
  v41-flash) width=4; k=5; al=3.51; role_memory=768G;;
  kimi-k3) width=8; k=2; al=2.51; role_memory=2048G;;
  *) exit 2;;
esac
case "$DSPARK_AGENTX_DEPLOYMENT" in
  agg) roles=(agg); [[ "${#nodes[@]}" == 1 ]] || exit 2;;
  disagg) roles=(prefill decode); [[ "$width" == 4 || "${#nodes[@]}" == 2 ]] || exit 2;;
  *) exit 2;;
esac
result="/workspace/results/dspark-agentx-20261015/$DSPARK_MODEL_KEY-$DSPARK_AGENTX_DEPLOYMENT-$DSPARK_AGENTX_PHASE-c$DSPARK_AGENTX_CONCURRENCY-r$DSPARK_AGENTX_REPETITION-$SLURM_JOB_ID"
host_result="$storage_root/${result#/workspace/}"
mkdir -p "$host_result/source" "$host_result/roles"
cp "$DSPARK_BUNDLE_DIR/"*.py "$DSPARK_BUNDLE_DIR/"*.json "$host_result/source/"
export HF_HOME=/workspace/.cache/dspark-e2e-20261014/huggingface
export XDG_CACHE_HOME=/workspace/.cache/dspark-e2e-20261014
container=(--container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh"
  --container-mounts="$storage_root:/workspace,/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model:ro"
  --container-workdir="$result/source"
  --container-env=HF_HOME,XDG_CACHE_HOME,VLLM_NIXL_SIDE_CHANNEL_HOST,VLLM_NIXL_SIDE_CHANNEL_PORT)
client_python="${DSPARK_AGENTX_AIPERF%/aiperf}/python"
client_args=(--model-key "$DSPARK_MODEL_KEY" --deployment "$DSPARK_AGENTX_DEPLOYMENT" --phase "$DSPARK_AGENTX_PHASE"
  --window "$DSPARK_AGENTX_CONCURRENCY" "$DSPARK_AGENTX_REPETITION"
  --context-audit "$DSPARK_AGENTX_CONTEXT_AUDIT" --aiperf "$DSPARK_AGENTX_AIPERF"
  --producer-checkout "$DSPARK_AGENTX_PRODUCER" --roles-output "$result/roles" --output "$result/client"
  --prefill-host "$prefill_host" --decode-host "$decode_host" --startup-timeout 6000
  --proxy-script /workspace/jobs/dspark-e2e-20261014/upstream/toy_proxy_server.py)
if [[ "$DSPARK_AGENTX_PHASE" == profile ]]; then
  : "${DSPARK_AGENTX_ANCHOR_C1:?}" "${DSPARK_AGENTX_ANCHOR_C4:?}"
  client_args+=(--fixed-measurement "$DSPARK_AGENTX_ANCHOR_C1" --fixed-measurement "$DSPARK_AGENTX_ANCHOR_C4")
  : "${DSPARK_AGENTX_PILOT_C1:?}" "${DSPARK_AGENTX_PILOT_C4:?}"
  client_args+=(--pilot-check "$DSPARK_AGENTX_PILOT_C1" --pilot-check "$DSPARK_AGENTX_PILOT_C4")
else
  anchor_name="DSPARK_AGENTX_ANCHOR_C$DSPARK_AGENTX_CONCURRENCY"
  [[ -n "${!anchor_name:-}" ]] || { echo "Missing $anchor_name" >&2; exit 2; }
  client_args+=(--fixed-measurement "${!anchor_name}")
fi
# Native anchors and reviewed pilots are checked before reserving role steps.
srun --nodes=1 --ntasks=1 --nodelist="$prefill_host" --overlap --gres=gpu:0 --cpus-per-task=4 --mem=32G \
  "${container[@]}" "$client_python" agentx_run.py "${client_args[@]}" --preflight-only
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
for role in "${roles[@]}"; do
  host="$prefill_host"; port=8010; side=5557; budget=8192; utilization=0.90
  if [[ "$role" == decode ]]; then host="$decode_host"; port=8020; side=5657; fi
  if [[ "$role" == prefill && "$DSPARK_MODEL_KEY" == v4-pro ]]; then budget=2048; utilization=0.92; fi
  export VLLM_NIXL_SIDE_CHANNEL_HOST="$host" VLLM_NIXL_SIDE_CHANNEL_PORT="$side"
  srun --nodes=1 --ntasks=1 --nodelist="$host" --exclusive --gres="gpu:$width" --cpus-per-task=24 --mem="$role_memory" \
    "${container[@]}" /usr/bin/python3 serve_role.py --model-key "$DSPARK_MODEL_KEY" --role "$role" --k "$k" \
    --port "$port" --synthetic-al "$al" --prefix-cache --context-length 1048576 --max-batched-tokens "$budget" \
    --gpu-memory-utilization "$utilization" --graph-policy runtime-default --load-strategy eager \
    --startup-timeout 6000 --output "$result/roles" > "$host_result/$role.log" 2>&1 &
  pids+=("$!")
done
srun --nodes=1 --ntasks=1 --nodelist="$prefill_host" --overlap --gres=gpu:0 --cpus-per-task=4 --mem=32G \
  "${container[@]}" "$client_python" agentx_run.py "${client_args[@]}" > "$host_result/client.log" 2>&1 &
client_pid=$!
pids+=("$client_pid")
while kill -0 "$client_pid" 2>/dev/null; do
  for pid in "${pids[@]:0:${#roles[@]}}"; do
    if ! kill -0 "$pid" 2>/dev/null; then echo "AgentX role exited before client" >&2; exit 1; fi
  done
  sleep 5
done
wait "$client_pid"
