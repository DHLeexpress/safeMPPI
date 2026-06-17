#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p output_dir/drifting_generator logs

[[ -f dataset/canonical/train.pt ]] || { echo "Missing dataset/canonical/train.pt" >&2; exit 1; }
[[ -f dataset/canonical/val.pt ]] || { echo "Missing dataset/canonical/val.pt" >&2; exit 1; }

python -m cfm_mppi.training.train_drifting "$@" 2>&1 | tee logs/train_drifting.log
