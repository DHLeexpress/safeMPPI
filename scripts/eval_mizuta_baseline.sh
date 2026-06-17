#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p results/benchmark logs

[[ -f output_dir/cfm_transformer/checkpoint.pth ]] || { echo "Missing output_dir/cfm_transformer/checkpoint.pth" >&2; exit 1; }

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset "${DATASET:-sfm}" \
  --dynamics "${DYNAMICS:-doubleintegrator}" \
  --methods mizuta_cfm_mppi \
  --num-episodes "${NUM_EPISODES:-100}" \
  --seed "${SEED:-0}" \
  --output-root "${OUTPUT_ROOT:-results/benchmark}" \
  "$@" 2>&1 | tee logs/eval_mizuta_baseline.log
