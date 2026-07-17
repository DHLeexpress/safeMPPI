#!/usr/bin/env bash
# Sequential 100-round AFE study: exact RBF first, deep ensemble second.
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 SCENE_PROFILE CHECKPOINT SHA256 OUTPUT_ROOT REPLICAS M_EVAL VERIFIER_WORKERS" >&2
  exit 2
fi

PROFILE="$1"
case "$PROFILE" in
  codex_radius1_v1|codex_radius03_v1) ;;
  *) echo "the 100-round study supports codex_radius1_v1 or codex_radius03_v1: $PROFILE" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
CKPT=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
EXPECTED=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
OUT="$4"
REPLICAS="$5"
M_EVAL="$6"
WORKERS="$7"
ROUNDS=100

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

"$HERE/run_afe_rbf_single.sh" \
  "$PROFILE" "$CKPT" "$EXPECTED" "$OUT/rbf" \
  "$ROUNDS" "$REPLICAS" "$M_EVAL" "$WORKERS"

"$HERE/run_afe_ensemble_single.sh" \
  "$PROFILE" "$CKPT" "$EXPECTED" "$OUT/ensemble" \
  "$ROUNDS" "$REPLICAS" "$M_EVAL" "$WORKERS"

echo "Both 100-round AFE uncertainty estimators completed sequentially: $OUT"
