#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CKPT SHA OUT PHYSICAL_INDEX EXPECTED_UUID" >&2
  exit 2
fi

CKPT=$1
EXPECTED_SHA=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
OUT=$3
PHYSICAL_INDEX=$4
EXPECTED_UUID=$5
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
VERIFIER_WORKERS=${VERIFIER_WORKERS:-64}
HERE=$(cd "$(dirname "$0")" && pwd)

if [[ ! -f "$CKPT" || ! "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint or SHA-256 is invalid" >&2
  exit 2
fi
if [[ ! "$PHYSICAL_INDEX" =~ ^[0-9]+$ || ! "$EXPECTED_UUID" == GPU-* ]]; then
  echo "physical GPU index or expected UUID is invalid" >&2
  exit 2
fi
if [[ ! "$VERIFIER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "VERIFIER_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "output root must be absent/new: $OUT" >&2
  exit 2
fi

GPU_LINE=$(nvidia-smi -i "$PHYSICAL_INDEX" \
  --query-gpu=index,uuid --format=csv,noheader,nounits)
IFS=',' read -r ACTUAL_INDEX ACTUAL_UUID <<<"$GPU_LINE"
ACTUAL_INDEX=${ACTUAL_INDEX//[[:space:]]/}
ACTUAL_UUID=${ACTUAL_UUID//[[:space:]]/}
if [[ "$ACTUAL_INDEX" != "$PHYSICAL_INDEX" || "$ACTUAL_UUID" != "$EXPECTED_UUID" ]]; then
  echo "GPU identity mismatch: got index=$ACTUAL_INDEX uuid=$ACTUAL_UUID" >&2
  exit 2
fi
ACTIVE_PIDS=$(nvidia-smi -i "$PHYSICAL_INDEX" \
  --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "${ACTIVE_PIDS//[[:space:]]/}" ]]; then
  echo "physical GPU $PHYSICAL_INDEX is not exclusive: $ACTIVE_PIDS" >&2
  exit 2
fi

CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
OUT=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUT")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$HERE"

exec "$PYTHON_BIN" analysis/low7_rbf_v2_smoke_driver.py \
  --ckpt "$CKPT" \
  --expected-ckpt-sha256 "$EXPECTED_SHA" \
  --out "$OUT" \
  --verifier-workers "$VERIFIER_WORKERS" \
  --python "$PYTHON_BIN"
