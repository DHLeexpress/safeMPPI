#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p logs

pytest -q 2>&1 | tee logs/pytest.log

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset sfm \
  --dynamics doubleintegrator \
  --methods mizuta_cfm_mppi \
  --num-episodes 2 \
  --seed 0 \
  --smoke \
  --device cpu 2>&1 | tee logs/smoke_mizuta.log

python -m cfm_mppi.evaluation.eval_benchmark \
  --dataset sfm \
  --dynamics doubleintegrator \
  --methods safemppi_gamma \
  --num-episodes 2 \
  --seed 0 \
  --gamma-grid 0.1 0.5 \
  --smoke \
  --device cpu 2>&1 | tee logs/smoke_safemppi.log
