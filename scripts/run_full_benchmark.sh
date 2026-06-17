#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p results/benchmark logs

[[ -f output_dir/cfm_transformer/checkpoint.pth ]] || { echo "Missing Mizuta checkpoint" >&2; exit 1; }
[[ -f output_dir/safe_contextual_cfm/checkpoint_best.pth ]] || { echo "Missing safe CFM checkpoint" >&2; exit 1; }
[[ -f output_dir/drifting_generator/checkpoint_best.pth ]] || { echo "Missing Drifting checkpoint" >&2; exit 1; }

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset "${DATASET:-sfm}" \
  --dynamics "${DYNAMICS:-doubleintegrator}" \
  --methods mizuta_cfm_mppi safemppi_gamma safe_cfm drifting \
  --num-episodes "${NUM_EPISODES:-100}" \
  --seed "${SEED:-0}" \
  --output-root "${OUTPUT_ROOT:-results/benchmark}" \
  --gamma-schedule "${GAMMA_SCHEDULE:-safeGPC_v4_2}" \
  "$@" 2>&1 | tee logs/run_full_benchmark.log
