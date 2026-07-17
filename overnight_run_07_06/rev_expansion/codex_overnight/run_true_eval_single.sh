#!/usr/bin/env bash
# True-evaluation launcher for one completed single-arm AFE-RBF run.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 SCENE_PROFILE RUN_ROOT OUTPUT_ROOT ROUNDS M pilot|canonical" >&2
  exit 2
fi

PROFILE="$1"
RUN_ROOT=$(cd "$2" && pwd)
OUT="$3"
ROUNDS="$4"
M="$5"
MODE="$6"
case "$PROFILE" in
  claude_grid_v1|codex_radius1_v1|codex_radius04_v1) ;;
  *) echo "unknown scene profile: $PROFILE" >&2; exit 2 ;;
esac
case "$MODE" in
  pilot) PILOT=(--pilot) ;;
  canonical) PILOT=() ;;
  *) echo "mode must be pilot or canonical" >&2; exit 2 ;;
esac
if [[ -e "$OUT" && ! -d "$OUT" ]]; then
  echo "output root is not a directory: $OUT" >&2
  exit 2
fi
if [[ -d "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "true evaluation output must be new or empty: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}
cd "$HERE"

"$PYTHON_BIN" paper_results/true_eval_run.py \
  --scene-profile "$PROFILE" \
  --run-root "$RUN_ROOT" \
  --outdir "$OUT/cells" \
  --rounds "$ROUNDS" --M "$M" --T 300 --reach 0.15 \
  "${PILOT[@]}"

"$PYTHON_BIN" paper_results/true_eval_fig.py \
  --scene-profile "$PROFILE" \
  --eval-dir "$OUT/cells" \
  --rounds "$ROUNDS" --reach 0.15 \
  --out-prefix "$OUT/true_eval_${PROFILE}" \
  "${PILOT[@]}"

echo "single-arm true evaluation complete [$PROFILE, $MODE]: $OUT"
