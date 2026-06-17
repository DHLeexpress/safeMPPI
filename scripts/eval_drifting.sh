#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p results/benchmark logs

[[ -f output_dir/drifting_generator/checkpoint_best.pth ]] || { echo "Missing output_dir/drifting_generator/checkpoint_best.pth" >&2; exit 1; }

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset "${DATASET:-sfm}" \
  --dynamics "${DYNAMICS:-doubleintegrator}" \
  --methods drifting \
  --num-episodes "${NUM_EPISODES:-100}" \
  --seed "${SEED:-0}" \
  --output-root "${OUTPUT_ROOT:-results/benchmark}" \
  "$@" 2>&1 | tee logs/eval_drifting.log
