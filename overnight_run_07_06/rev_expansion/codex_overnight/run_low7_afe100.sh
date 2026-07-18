#!/usr/bin/env bash
# Canonical single-arm low7 AFE100 giant-obstacle study.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 AUTHENTICATED_LOW7_CHECKPOINT EXPECTED_SHA256 NEW_OUTPUT_ROOT" >&2
  exit 2
fi

export OMP_NUM_THREADS=${AFE_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${AFE_MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${AFE_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${AFE_NUMEXPR_NUM_THREADS:-1}

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-/home/dohyun/miniforge3/envs/cfm_mppi/bin/python}
PROFILE=low7_radius1_canonical_v1
EXPECTED_UUID=GPU-50fb5dae-52a8-5843-bc81-b869586dccde
CKPT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
EXPECTED=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
OUT=$3

if [[ "${CUDA_DEVICE_ORDER:-}" != PCI_BUS_ID || "${CUDA_VISIBLE_DEVICES:-}" != 1 ]]; then
  echo "canonical low7 run requires CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=1" >&2
  exit 2
fi
if [[ ! -f "$CKPT" || ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint or SHA-256 is invalid" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "output root must be absent/new: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$HERE"

"$PYTHON_BIN" analysis/write_gpu_provenance.py \
  --physical-index 1 --expected-uuid "$EXPECTED_UUID" \
  --out "$OUT/gpu_provenance.json"
"$PYTHON_BIN" analysis/preflight_low7_afe.py \
  --checkpoint "$CKPT" --expected-sha256 "$EXPECTED" \
  --scene-profile "$PROFILE" --out "$OUT/preflight.json"

"$PYTHON_BIN" grid_expand_afe_ensemble.py \
  --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
  --scene-profile "$PROFILE" --outdir "$OUT/run" \
  --rounds 100 --rollout-replicas 2 --K 64 --B 8 --T 300 --M-eval 2 \
  --batch 128 --afe-steps 250 --afe-lr 1e-4 --verifier-workers 16 \
  --seed 910 --replay-window 5 --adaptive-ess-target 0.5 \
  --conditioning-schema low7_closest_boundary --freeze-visual-encoder

"$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$OUT/run" --out "$OUT/report.png"
"$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$OUT/run" --out "$OUT/report.pdf"
"$PYTHON_BIN" video_afe2.py --run "$OUT/run" --out "$OUT/video.mp4" \
  --dense-until 10 --every-after 10
"$PYTHON_BIN" analysis/validate_afe_ensemble_run.py \
  --run "$OUT/run" --report "$OUT/report.png" --report-pdf "$OUT/report.pdf" \
  --gpu-provenance "$OUT/gpu_provenance.json" --video "$OUT/video.mp4" \
  --expected-video-frames 20 --out "$OUT/TRAINER_DELIVERY_COMPLETE.json"

"$HERE/run_afe_m20_eval.sh" "$PROFILE" "$OUT/run" "$OUT/evaluation"
"$PYTHON_BIN" analysis/finalize_afe_endpoint_delivery.py --root "$OUT"

echo "Canonical low7 AFE100 delivery complete: $OUT"
