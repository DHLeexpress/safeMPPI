#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p logs

GPC_EPOCHS="${GPC_EPOCHS:-300}"
GPC_BATCH_SIZE="${GPC_BATCH_SIZE:-256}"
GPC_DEVICE="${GPC_DEVICE:-cuda}"
GPC_EXTRA_ARGS="${GPC_EXTRA_ARGS:-}"

if [[ "${SKIP_GPC_TRAIN:-0}" != "1" ]]; then
  [[ -f dataset/canonical/train.pt ]] || { echo "Missing dataset/canonical/train.pt. Run scripts/build_canonical_dataset.sh first." >&2; exit 1; }
  [[ -f dataset/canonical/val.pt ]] || { echo "Missing dataset/canonical/val.pt. Run scripts/build_canonical_dataset.sh first." >&2; exit 1; }
  python -m cfm_mppi.training.train_safe_cfm \
    --epochs "$GPC_EPOCHS" \
    --batch-size "$GPC_BATCH_SIZE" \
    --device "$GPC_DEVICE" \
    $GPC_EXTRA_ARGS 2>&1 | tee logs/train_gpc_overnight.log
fi

DEVICE="${DEVICE:-$GPC_DEVICE}" scripts/run_gamma_sweep.sh
