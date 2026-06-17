#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p results/benchmark logs

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset "${DATASET:-sfm}" \
  --dynamics "${DYNAMICS:-doubleintegrator}" \
  --methods safemppi_gamma \
  --num-episodes "${NUM_EPISODES:-100}" \
  --seed "${SEED:-0}" \
  --output-root "${OUTPUT_ROOT:-results/benchmark}" \
  --gamma-schedule "${GAMMA_SCHEDULE:-safeGPC_v4_2}" \
  "$@" 2>&1 | tee logs/eval_safemppi_gamma.log
