#!/usr/bin/env bash
# Shared TRUE-evaluation launcher (integration/afe2-terminal-dualscene-v1).
# Runs the portable bare-policy/oracle/baseline evaluation for ONE scene profile against the
# exact round checkpoints of ONE AFE2 arm directory, then renders the gallery + curves.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SCENE_PROFILE /absolute/path/to/arm_run_dir /absolute/output/root" >&2
  echo "  arm_run_dir must contain EXACT ckpt_0.pt .. ckpt_10.pt (no substitutions)" >&2
  exit 2
fi

PROFILE="$1"
case "$PROFILE" in
  claude_grid_v1|codex_radius1_v1) ;;
  *) echo "unknown scene profile: $PROFILE" >&2; exit 2 ;;
esac
CKPT_DIR=$(cd "$2" && pwd)
if [[ -e "$3" && ! -d "$3" ]]; then
  echo "output root exists and is not a directory: $3" >&2
  exit 2
fi
if [[ -d "$3" && -n "$(find "$3" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "true evaluation requires a new or empty output root (stale outputs are rejected): $3" >&2
  exit 2
fi
mkdir -p "$3"
OUT=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}
cd "$HERE"

for n in $(seq 0 10); do
  if [[ ! -f "$CKPT_DIR/ckpt_${n}.pt" ]]; then
    echo "missing required round checkpoint: $CKPT_DIR/ckpt_${n}.pt" >&2
    exit 2
  fi
done

"$PYTHON_BIN" paper_results/true_eval_run.py \
  --scene-profile "$PROFILE" \
  --ckpt-dir "$CKPT_DIR" \
  --outdir "$OUT/cells" \
  --rounds 10 --M 100 --T 300 --reach 0.15

"$PYTHON_BIN" paper_results/true_eval_fig.py \
  --scene-profile "$PROFILE" \
  --eval-dir "$OUT/cells" \
  --rounds 10 --reach 0.15 \
  --out-prefix "$OUT/true_eval_${PROFILE}"

echo "TRUE evaluation complete [$PROFILE]: $OUT"
