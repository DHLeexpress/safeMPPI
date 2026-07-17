#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 SCENE_PROFILE AUTHENTICATED_AFE_RUN NEW_OUTPUT_ROOT" >&2
  exit 2
fi

PROFILE=$1
RUN=$(realpath "$2")
OUT=$3
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}

if [ -e "$OUT" ]; then
  echo "evaluation output root must be absent/new: $OUT" >&2
  exit 3
fi
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  echo "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index" >&2
  exit 4
fi
if [ -n "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "physical GPU $CUDA_VISIBLE_DEVICES has an active compute process; refusing to share it" >&2
  exit 5
fi

"$PYTHON_BIN" paper_results/afe_m20_eval.py \
  --scene-profile "$PROFILE" \
  --run-root "$RUN" \
  --outdir "$OUT"

"$PYTHON_BIN" paper_results/afe_m20_eval.py \
  --outdir "$OUT" \
  --validate-only
