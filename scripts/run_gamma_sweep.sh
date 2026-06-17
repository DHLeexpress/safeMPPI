#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p logs results/visualization/gamma_sweep

DATASET="${DATASET:-sfm}"
DYNAMICS="${DYNAMICS:-doubleintegrator}"
NUM_EPISODES="${NUM_EPISODES:-20}"
GAMMA_COUNT="${GAMMA_COUNT:-21}"
SAFEMPPI_NUM_SAMPLES="${SAFEMPPI_NUM_SAMPLES:-2048}"
SAFEMPPI_HORIZON="${SAFEMPPI_HORIZON:-20}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

python -m cfm_mppi.visualization.live_gamma_compare \
  --dataset "$DATASET" \
  --dynamics "$DYNAMICS" \
  --num-episodes "$NUM_EPISODES" \
  --gamma-count "$GAMMA_COUNT" \
  --safemppi-num-samples "$SAFEMPPI_NUM_SAMPLES" \
  --safemppi-horizon "$SAFEMPPI_HORIZON" \
  --device "$DEVICE" \
  $EXTRA_ARGS 2>&1 | tee logs/gamma_sweep_${DATASET}_${DYNAMICS}.log
