#!/usr/bin/env bash
set -euo pipefail
: "${storage_root:?}"
: "${DSPARK_BUNDLE_DIR:?}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 2;; esac
for setting in "${!DSPARK_@}"; do
  [[ "$setting" == DSPARK_BUNDLE_DIR ]] || unset "$setting"
done
(cd "$DSPARK_BUNDLE_DIR" && sha256sum --check source.sha256)
export DSPARK_MODEL_KEY=kimi-k3
export DSPARK_K=2
export DSPARK_RUN_MODE=initial
export DSPARK_GRAPH_POLICY=runtime-default
export DSPARK_STARTUP_TIMEOUT=6000
export DSPARK_LOAD_STRATEGY=eager
export DSPARK_CONTEXT_LENGTH=1048576
export DSPARK_SYNTHETIC_AL=2.51
export DSPARK_INPUT_LENGTH=1024
export DSPARK_REPETITION=1
export DSPARK_CELL_IDS='kimi-k3-disagg-k2-i1024-o1024-c1-synthetic kimi-k3-disagg-k2-i1024-o1024-c4-synthetic'
export DSPARK_PREFILL_MAX_BATCHED_TOKENS=8192
export DSPARK_DECODE_MAX_BATCHED_TOKENS=8192
bash "$DSPARK_BUNDLE_DIR/pd-smoke.sh"
