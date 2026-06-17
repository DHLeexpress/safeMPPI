#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p dataset/canonical logs

SOURCE="${SOURCE:-safeGPC}"
INPUT="${INPUT:-/home/dohyun/projects/safeGPC/artifacts/mppi_seq_newcsv_keep_nobs_20250828_174641.pkl}"

if [[ ! -e "$INPUT" ]]; then
  echo "Missing dataset input: $INPUT" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${MAX_EPISODES:-}" ]]; then
  EXTRA_ARGS+=(--max-episodes "$MAX_EPISODES")
fi

python -m cfm_mppi.data.build_canonical_dataset \
  --source "$SOURCE" \
  --input "$INPUT" \
  --output-dir dataset/canonical \
  --seed "${SEED:-0}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee logs/build_canonical_dataset.log
