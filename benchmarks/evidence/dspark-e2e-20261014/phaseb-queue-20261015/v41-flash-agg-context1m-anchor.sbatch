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
export DSPARK_MODEL_KEY=v41-flash
export DSPARK_K=5
export DSPARK_RUN_MODE=initial
export DSPARK_GRAPH_POLICY=runtime-default
export DSPARK_STARTUP_TIMEOUT=6000
export DSPARK_LOAD_STRATEGY=eager
export DSPARK_CONTEXT_LENGTH=1048576
export DSPARK_SYNTHETIC_AL=3.51
export DSPARK_INPUT_LENGTH=1024
export DSPARK_REPETITION=1
export DSPARK_CELL_IDS='v41-flash-agg-k5-i1024-o1024-c1-synthetic v41-flash-agg-k5-i1024-o1024-c4-synthetic'
bash "$DSPARK_BUNDLE_DIR/agg-smoke.sh"
