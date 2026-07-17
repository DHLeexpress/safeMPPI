#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 RBF_RUN ENSEMBLE_RUN NEW_OUTPUT_ROOT" >&2
  exit 2
fi

RBF_RUN=$(realpath "$1")
ENSEMBLE_RUN=$(realpath "$2")
OUT=$3
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}

if [ -e "$OUT" ]; then
  echo "evaluation output root must be absent/new: $OUT" >&2
  exit 3
fi
if [ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]; then
  echo "set CUDA_VISIBLE_DEVICES=1 so process cuda:0 maps to physical GPU 1" >&2
  exit 4
fi
if [ -n "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "physical GPU 1 has an active compute process; refusing to share it" >&2
  exit 5
fi

"$PYTHON_BIN" paper_results/afe100_m20_eval.py \
  --rbf-run "$RBF_RUN" \
  --ensemble-run "$ENSEMBLE_RUN" \
  --outdir "$OUT" \
  --base-source-commit 1ca51e2bfbce01d09b5d8a45e8c4e44e156dbc6e

"$PYTHON_BIN" paper_results/afe100_m20_eval.py \
  --outdir "$OUT" \
  --validate-only
