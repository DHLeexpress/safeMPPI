#!/usr/bin/env bash
# Faithful window-level recipe → iter100, fresh from the pretrained (2026-07-15).
# Recipe = faithful_g47 EXACTLY (reach 0.15, RBF sigma, no retune) + every-iter viz snapshots +
# the new batch-used / path-status logging. Reused for OURS and the 3 ablation brothers.
#   usage: faithful_it100_driver.sh <gpu> <outdir> <viz_every> [extra flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
GPU="$1"; OUTDIR="$2"; VIZ="$3"; shift 3
PRE=../../results/hp_repr/pretrained_a32uni.pt
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 python grid_expand_hardtail.py \
  --ckpt "$PRE" --outdir "$OUTDIR" --iters 100 --start-iter 0 --drop-train-state --seed 910 --lr 2e-5 \
  --window-level --wall-plugs 8 --start-eps 0.3 --goal-xy 4.7 4.7 --reach 0.15 \
  --rollouts-per-iter 14 --gather-attempt-cap 400 --batch 64 --gp-buf 200 --qbuf-cap 200 \
  --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.30 --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta 0.2 \
  --early-until 100 --cooldown-from 400 --demo-frac 0.0 --lwf-eta 0.0 \
  --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
  --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 \
  --log-comp-every 1 \
  --viz-db-every "$VIZ" --ckpt-every 10 --measure-every 10 --tag "$(basename "$OUTDIR")" "$@"
echo "DONE $(basename "$OUTDIR")"
touch "$OUTDIR/IT100_DONE"
