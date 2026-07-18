#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 SCENE_PROFILE COMPLETED_RBF_RUN NEW_OUTPUT_ROOT PHYSICAL_INDEX EXPECTED_UUID" >&2
  exit 2
fi

PROFILE=$1
RUN=$(realpath "$2")
OUT=$3
PHYSICAL_INDEX=$4
EXPECTED_UUID=$5
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}

if [ -e "$OUT" ]; then
  echo "evaluation output root must be absent/new: $OUT" >&2
  exit 3
fi
if [[ ! "$PHYSICAL_INDEX" =~ ^[0-9]+$ || ! "$EXPECTED_UUID" == GPU-* ]]; then
  echo "physical GPU index or expected UUID is invalid" >&2
  exit 4
fi
GPU_LINE=$(nvidia-smi -i "$PHYSICAL_INDEX" --query-gpu=index,uuid --format=csv,noheader,nounits)
IFS=',' read -r ACTUAL_INDEX ACTUAL_UUID <<<"$GPU_LINE"
ACTUAL_INDEX=${ACTUAL_INDEX//[[:space:]]/}
ACTUAL_UUID=${ACTUAL_UUID//[[:space:]]/}
if [[ "$ACTUAL_INDEX" != "$PHYSICAL_INDEX" || "$ACTUAL_UUID" != "$EXPECTED_UUID" ]]; then
  echo "GPU identity mismatch: got index=$ACTUAL_INDEX uuid=$ACTUAL_UUID" >&2
  exit 5
fi
if [ -n "$(nvidia-smi -i "$PHYSICAL_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)" ]; then
  echo "physical GPU $PHYSICAL_INDEX has an active compute process; refusing to share it" >&2
  exit 5
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$HERE"
"$PYTHON_BIN" paper_results/low7_raw_m50_eval.py \
  --scene-profile "$PROFILE" \
  --run-root "$RUN" \
  --outdir "$OUT"

"$PYTHON_BIN" paper_results/low7_raw_m50_eval.py \
  --outdir "$OUT" \
  --validate-only
