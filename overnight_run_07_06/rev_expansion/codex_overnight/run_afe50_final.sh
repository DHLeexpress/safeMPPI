#!/usr/bin/env bash
# Canonical 50-round single-arm confirmation: adaptive ensemble, ESS=.5, W=5.
set -euo pipefail

export OMP_NUM_THREADS=${AFE_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${AFE_MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${AFE_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${AFE_NUMEXPR_NUM_THREADS:-1}

if [[ $# -ne 8 ]]; then
  echo "usage: $0 SCENE CHECKPOINT SHA256 OUTPUT_ROOT REPLICAS M_EVAL WORKERS SEED" >&2
  exit 2
fi

PROFILE="$1"
CKPT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
EXPECTED=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
OUT="$4"
REPLICAS="$5"
M_EVAL="$6"
WORKERS="$7"
SEED="$8"
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}
ROUNDS=50
VIDEO_FRAMES=14

case "$PROFILE" in
  codex_radius1_v1|codex_radius03_v1) ;;
  *) echo "unsupported final scene: $PROFILE" >&2; exit 2 ;;
esac
if [[ ! -f "$CKPT" || ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint or SHA-256 is invalid" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
  echo "set CUDA_VISIBLE_DEVICES to exactly one physical GPU index" >&2
  exit 2
fi
if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
RUN="$OUT/run"
cd "$HERE"

nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,uuid,name,driver_version,memory.total \
  --format=csv,noheader >"$OUT/gpu_provenance.csv"

"$PYTHON_BIN" grid_expand_afe_ensemble.py \
  --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
  --scene-profile "$PROFILE" --outdir "$RUN" --rounds "$ROUNDS" \
  --rollout-replicas "$REPLICAS" --K 64 --B 8 --T 300 --M-eval "$M_EVAL" \
  --batch 128 --afe-steps 250 --afe-lr 1e-4 --verifier-workers "$WORKERS" \
  --seed "$SEED" --replay-window 5 --adaptive-ess-target 0.5

"$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$RUN" --out "$OUT/report.png"
"$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$RUN" --out "$OUT/report.pdf"
"$PYTHON_BIN" video_afe2.py --run "$RUN" --out "$OUT/video.mp4" \
  --dense-until 10 --every-after 10
"$PYTHON_BIN" analysis/validate_afe_ensemble_run.py \
  --run "$RUN" --report "$OUT/report.png" --report-pdf "$OUT/report.pdf" \
  --gpu-provenance "$OUT/gpu_provenance.csv" \
  --video "$OUT/video.mp4" --expected-video-frames "$VIDEO_FRAMES" \
  --out "$OUT/DELIVERY_COMPLETE.json"

echo "Canonical 50-round AFE confirmation complete: $OUT"
