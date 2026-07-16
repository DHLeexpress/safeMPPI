#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$HERE/stage_results/04_frozen_ood"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH="/home/dohyun/miniforge3/lib:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

mkdir -p "$OUT/logs"
cd "$ROOT"

python -u giant_obstacle_ood/stage4_frozen_ood.py \
  --device cuda:0 \
  "$@" 2>&1 | tee "$OUT/logs/automation.log"

python -u giant_obstacle_ood/validate_stage4.py 2>&1 | tee -a "$OUT/logs/automation.log"
