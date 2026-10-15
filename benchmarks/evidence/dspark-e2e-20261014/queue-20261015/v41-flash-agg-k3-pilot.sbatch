#!/usr/bin/env bash
set -euo pipefail
: "${storage_root:?}"
: "${DSPARK_BUNDLE_DIR:?}"
cd "$storage_root"
case "$PWD/" in "$HOME"/*) exit 2;; esac
for setting in "${!DSPARK_@}"; do
  [[ "$setting" == DSPARK_BUNDLE_DIR ]] || unset "$setting"
done
export DSPARK_MODEL_KEY=v41-flash
export DSPARK_K=3
export DSPARK_RUN_MODE=pilot
export DSPARK_GRAPH_POLICY=runtime-default
export DSPARK_STARTUP_TIMEOUT=6000
export DSPARK_LOAD_STRATEGY=eager
export DSPARK_INPUT_LENGTH=1024
bash "$DSPARK_BUNDLE_DIR/agg-smoke.sh"
srun --nodes=1 --ntasks=1 --gres=gpu:0 --cpus-per-task=2 --mem=4G --container-image="$storage_root/images/vllm-ac9126e58aa7-3d17635c.sqsh" --container-mounts="$storage_root:/workspace" /usr/bin/python3 /workspace/jobs/dspark-full-20261015/bundle/freeze_acceptance.py --pilot "/workspace/results/dspark-e2e-20261014/pilot-v41-flash-k3-$SLURM_JOB_ID/pilot-c1-r1" --output "/workspace/jobs/dspark-full-20261015/controls/synthetic-control-v41-flash-k3.json"
