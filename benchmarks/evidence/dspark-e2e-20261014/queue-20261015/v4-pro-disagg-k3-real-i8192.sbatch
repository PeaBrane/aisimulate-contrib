#!/usr/bin/env bash
set -euo pipefail
: "${storage_root:?}"
: "${DSPARK_BUNDLE_DIR:?}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 2;; esac
for setting in "${!DSPARK_@}"; do
  [[ "$setting" == DSPARK_BUNDLE_DIR ]] || unset "$setting"
done
export DSPARK_COMPLETED_RUNS="/workspace/jobs/dspark-full-20261015/bundle/completed-runs.json"
export DSPARK_MODEL_KEY=v4-pro
export DSPARK_K=3
export DSPARK_RUN_MODE=sweep
export DSPARK_GRAPH_POLICY=runtime-default
export DSPARK_STARTUP_TIMEOUT=6000
export DSPARK_LOAD_STRATEGY=eager
export DSPARK_INPUT_LENGTH=8192
export DSPARK_CELL_IDS='v4-pro-disagg-k3-i8192-o1024-c1-real v4-pro-disagg-k3-i8192-o1024-c4-real v4-pro-disagg-k3-i8192-o1024-c8-real v4-pro-disagg-k3-i8192-o1024-c16-real v4-pro-disagg-k3-i8192-o1024-c64-real'
export DSPARK_PREFILL_GPU_MEMORY_UTILIZATION=0.92
export DSPARK_OBSERVE_SOURCE_POOL=1
bash "$DSPARK_BUNDLE_DIR/pd-smoke.sh"
