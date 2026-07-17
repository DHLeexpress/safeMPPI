#!/usr/bin/env bash
# One-arm AFE-RBF launcher.  All task and compute choices are explicit arguments.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 SCENE_PROFILE CHECKPOINT SHA256 OUTPUT_ROOT ROUNDS REPLICAS M_EVAL VERIFIER_WORKERS" >&2
  exit 2
fi

PROFILE="$1"
case "$PROFILE" in
  claude_grid_v1|codex_radius1_v1|codex_radius03_v1|codex_radius04_v1) ;;
  *) echo "unknown scene profile: $PROFILE" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
CKPT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
EXPECTED=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
OUT="$4"
ROUNDS="$5"
REPLICAS="$6"
M_EVAL="$7"
WORKERS="$8"
PYTHON_BIN=${PYTHON:-python}

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi
if [[ ! "$EXPECTED" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint SHA-256 must be 64 lowercase hex characters" >&2
  exit 2
fi
if [[ -e "$OUT" && ! -d "$OUT" ]]; then
  echo "output root is not a directory: $OUT" >&2
  exit 2
fi
if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output root must be new or empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
cd "$HERE"

RUN="$OUT/afe_rbf_s910"
"$PYTHON_BIN" grid_expand_afe_rbf.py \
  --ckpt "$CKPT" \
  --expected-ckpt-sha256 "$EXPECTED" \
  --scene-profile "$PROFILE" \
  --outdir "$RUN" \
  --rounds "$ROUNDS" \
  --rollout-replicas "$REPLICAS" \
  --K 64 \
  --B 8 \
  --T 300 \
  --M-eval "$M_EVAL" \
  --batch 128 \
  --afe-steps 250 \
  --afe-lr 1e-4 \
  --gp-cap 512 \
  --gp-lam 1e-2 \
  --verifier-workers "$WORKERS" \
  --seed 910

"$PYTHON_BIN" analysis/afe_rbf_report.py \
  --run "$RUN" \
  --out "$OUT/afe_rbf_${PROFILE}_report.png"

"$PYTHON_BIN" video_afe2.py \
  --run "$RUN" \
  --out "$OUT/afe_rbf_${PROFILE}.mp4" \
  --dense-until 10 \
  --every-after 10

"$PYTHON_BIN" analysis/validate_afe_rbf_run.py \
  --run "$RUN" \
  --report "$OUT/afe_rbf_${PROFILE}_report.png" \
  --video "$OUT/afe_rbf_${PROFILE}.mp4" \
  --expected-video-frames "$(( ROUNDS <= 10 ? ROUNDS : 10 + ROUNDS / 10 - 1 ))" \
  --out "$OUT/DELIVERY_COMPLETE.json"

echo "AFE-RBF single-arm delivery complete: $OUT"
