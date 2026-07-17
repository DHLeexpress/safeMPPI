#!/usr/bin/env bash
# Five-round causal screen at W=5: uniform, adaptive RBF, adaptive ensemble.
set -euo pipefail

# Prevent each independent arm and each spawned verifier worker from creating a
# full-machine BLAS/OpenMP pool. Callers may override the conservative caps.
export OMP_NUM_THREADS=${AFE_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${AFE_MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${AFE_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${AFE_NUMEXPR_NUM_THREADS:-1}

if [[ $# -ne 7 ]]; then
  echo "usage: $0 SCENE_PROFILE CHECKPOINT SHA256 OUTPUT_ROOT REPLICAS M_EVAL VERIFIER_WORKERS" >&2
  exit 2
fi

PROFILE="$1"
CKPT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
EXPECTED=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
OUT="$4"
REPLICAS="$5"
M_EVAL="$6"
WORKERS="$7"
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}
ROUNDS=5
PARALLEL_ARMS=${AFE_SCREEN_PARALLEL:-0}

case "$PROFILE" in
  codex_radius1_v1|codex_radius03_v1) ;;
  *) echo "unsupported screen scene: $PROFILE" >&2; exit 2 ;;
esac
if [[ ! -f "$CKPT" || ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint or SHA-256 is invalid" >&2
  exit 2
fi
if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$HERE"

run_rbf() {
  local name="$1"
  shift
  local root="$OUT/$name"
  local run="$root/run"
  "$PYTHON_BIN" grid_expand_afe_rbf.py \
    --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
    --scene-profile "$PROFILE" --outdir "$run" --rounds "$ROUNDS" \
    --rollout-replicas "$REPLICAS" --K 64 --B 8 --T 300 --M-eval "$M_EVAL" \
    --batch 128 --afe-steps 250 --afe-lr 1e-4 --gp-cap 512 --gp-lam 1e-2 \
    --verifier-workers "$WORKERS" --seed 910 --replay-window 5 "$@"
  "$PYTHON_BIN" analysis/afe_rbf_report.py --run "$run" --out "$root/report.png"
  "$PYTHON_BIN" video_afe2.py --run "$run" --out "$root/video.mp4"
  "$PYTHON_BIN" analysis/validate_afe_rbf_run.py \
    --run "$run" --report "$root/report.png" --video "$root/video.mp4" \
    --expected-video-frames "$ROUNDS" --out "$root/DELIVERY_COMPLETE.json"
}

run_ensemble() {
  local root="$OUT/ensemble_adaptive"
  local run="$root/run"
  "$PYTHON_BIN" grid_expand_afe_ensemble.py \
    --ckpt "$CKPT" --expected-ckpt-sha256 "$EXPECTED" \
    --scene-profile "$PROFILE" --outdir "$run" --rounds "$ROUNDS" \
    --rollout-replicas "$REPLICAS" --K 64 --B 8 --T 300 --M-eval "$M_EVAL" \
    --batch 128 --afe-steps 250 --afe-lr 1e-4 --verifier-workers "$WORKERS" \
    --seed 910 --replay-window 5 --adaptive-ess-target 0.5
  "$PYTHON_BIN" analysis/afe_ensemble_report.py --run "$run" --out "$root/report.png"
  "$PYTHON_BIN" video_afe2.py --run "$run" --out "$root/video.mp4"
  "$PYTHON_BIN" analysis/validate_afe_ensemble_run.py \
    --run "$run" --report "$root/report.png" --video "$root/video.mp4" \
    --expected-video-frames "$ROUNDS" --out "$root/DELIVERY_COMPLETE.json"
}

if [[ "$PARALLEL_ARMS" == 1 ]]; then
  mkdir -p "$OUT/uniform" "$OUT/rbf_adaptive" "$OUT/ensemble_adaptive"
  run_rbf uniform --acquisition-mode uniform >"$OUT/uniform/launcher.log" 2>&1 &
  P_UNIFORM=$!
  run_rbf rbf_adaptive --adaptive-ess-target 0.5 --rbf-offline-sweep >"$OUT/rbf_adaptive/launcher.log" 2>&1 &
  P_RBF=$!
  run_ensemble >"$OUT/ensemble_adaptive/launcher.log" 2>&1 &
  P_ENSEMBLE=$!
  STATUS=0
  wait "$P_UNIFORM" || STATUS=1
  wait "$P_RBF" || STATUS=1
  wait "$P_ENSEMBLE" || STATUS=1
  if [[ "$STATUS" -ne 0 ]]; then
    echo "at least one parallel arm failed; inspect each launcher.log" >&2
    exit "$STATUS"
  fi
elif [[ "$PARALLEL_ARMS" == 0 ]]; then
  run_rbf uniform --acquisition-mode uniform
  run_rbf rbf_adaptive --adaptive-ess-target 0.5 --rbf-offline-sweep
  run_ensemble
else
  echo "AFE_SCREEN_PARALLEL must be 0 or 1" >&2
  exit 2
fi

echo "Five-round acquisition screen complete: $OUT"
