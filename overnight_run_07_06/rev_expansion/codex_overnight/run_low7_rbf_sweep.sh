#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 SCENE_PROFILE CKPT SHA OUT PHYSICAL_INDEX EXPECTED_UUID" >&2
  exit 2
fi

SCENE_PROFILE=$1
CKPT=$2
EXPECTED_SHA=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
OUT=$4
PHYSICAL_INDEX=$5
EXPECTED_UUID=$6
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
MAX_JOBS=${MAX_JOBS:-2}
VERIFIER_WORKERS=${VERIFIER_WORKERS:-8}
OUT=$(
  "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUT"
)

for REQUIRED_COMMAND in nvidia-smi ffmpeg ffprobe; do
  command -v "$REQUIRED_COMMAND" >/dev/null || {
    echo "required command is unavailable: $REQUIRED_COMMAND" >&2
    exit 2
  }
done

case "$SCENE_PROFILE" in
  low7_radius1_canonical_v1|low7_radius03_canonical_v1) ;;
  *) echo "scene must be low7_radius1_canonical_v1 or low7_radius03_canonical_v1" >&2; exit 2 ;;
esac
if [[ ! -f "$CKPT" || ! "$EXPECTED_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint or SHA-256 is invalid" >&2
  exit 2
fi
if [[ ! "$PHYSICAL_INDEX" =~ ^[0-9]+$ || ! "$EXPECTED_UUID" == GPU-* ]]; then
  echo "physical GPU index or expected UUID is invalid" >&2
  exit 2
fi
if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ || ! "$VERIFIER_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS and VERIFIER_WORKERS must be positive integers" >&2
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
ACTIVE_COMPUTE_PIDS=$(nvidia-smi -i "$PHYSICAL_INDEX" \
  --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "${ACTIVE_COMPUTE_PIDS//[[:space:]]/}" ]]; then
  echo "physical GPU $PHYSICAL_INDEX already has active compute PIDs: $ACTIVE_COMPUTE_PIDS" >&2
  exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
CKPT=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$PHYSICAL_INDEX
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$HERE"

exec "$PYTHON_BIN" analysis/low7_rbf_sweep_driver.py \
  --scene-profile "$SCENE_PROFILE" \
  --ckpt "$CKPT" \
  --expected-ckpt-sha256 "$EXPECTED_SHA" \
  --out "$OUT" \
  --physical-index "$PHYSICAL_INDEX" \
  --expected-gpu-uuid "$EXPECTED_UUID" \
  --max-jobs "$MAX_JOBS" \
  --verifier-workers "$VERIFIER_WORKERS"
