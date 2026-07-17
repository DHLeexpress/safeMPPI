#!/usr/bin/env bash
# One five-round adaptive-estimator run at replay W=1, W=5, or cumulative.
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 rbf|ensemble 1|5|all SCENE CHECKPOINT SHA256 OUTPUT_ROOT REPLICAS M_EVAL WORKERS" >&2
  exit 2
fi

ESTIMATOR="$1"
WINDOW="$2"
PROFILE="$3"
CKPT=$(cd "$(dirname "$4")" && pwd)/$(basename "$4")
EXPECTED=$(printf '%s' "$5" | tr '[:upper:]' '[:lower:]')
OUT="$6"
REPLICAS="$7"
M_EVAL="$8"
WORKERS="$9"
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}
ROUNDS=5

case "$ESTIMATOR" in rbf|ensemble) ;; *) echo "estimator must be rbf or ensemble" >&2; exit 2 ;; esac
case "$WINDOW" in 1|5) REPLAY=(--replay-window "$WINDOW") ;; all) REPLAY=() ;; *) echo "W must be 1, 5, or all" >&2; exit 2 ;; esac
if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
RUN="$OUT/run"
cd "$HERE"

if [[ "$ESTIMATOR" == rbf ]]; then
  "$PYTHON_BIN" grid_expand_afe_rbf.py \
    --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
    --scene-profile "$PROFILE" --outdir "$RUN" --rounds "$ROUNDS" \
    --rollout-replicas "$REPLICAS" --K 64 --B 8 --T 300 --M-eval "$M_EVAL" \
    --batch 128 --afe-steps 250 --afe-lr 1e-4 --gp-cap 512 --gp-lam 1e-2 \
    --verifier-workers "$WORKERS" --seed 910 --adaptive-ess-target 0.5 "${REPLAY[@]}"
  "$PYTHON_BIN" analysis/afe_rbf_report.py --run "$RUN" --out "$OUT/report.png"
  VALIDATOR=analysis/validate_afe_rbf_run.py
else
  "$PYTHON_BIN" grid_expand_afe_ensemble.py \
    --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
    --scene-profile "$PROFILE" --outdir "$RUN" --rounds "$ROUNDS" \
    --rollout-replicas "$REPLICAS" --K 64 --B 8 --T 300 --M-eval "$M_EVAL" \
    --batch 128 --afe-steps 250 --afe-lr 1e-4 --verifier-workers "$WORKERS" \
    --seed 910 --adaptive-ess-target 0.5 "${REPLAY[@]}"
  "$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$RUN" --out "$OUT/report.png"
  VALIDATOR=analysis/validate_afe_ensemble_run.py
fi

"$PYTHON_BIN" video_afe2.py --run "$RUN" --out "$OUT/video.mp4"
"$PYTHON_BIN" "$VALIDATOR" \
  --run "$RUN" --report "$OUT/report.png" --video "$OUT/video.mp4" \
  --expected-video-frames "$ROUNDS" --out "$OUT/DELIVERY_COMPLETE.json"
echo "Five-round replay screen complete: estimator=$ESTIMATOR W=$WINDOW output=$OUT"
