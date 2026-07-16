#!/usr/bin/env bash
# Diversity-preserving window-level recipe → iter100 (user 2026-07-15). Three targeted changes vs
# faithful_g47_it100 (which mode-collapsed R→U and over-conservatived γ0.1):
#   1. PRESERVE MULTI-MODE DIVERSITY: re-enable the δ/η anchor (demo-frac .125 + LwF .05 to a frozen copy
#      of the pretrained) — the proven anti-collapse anchor. Keeps the policy near the diverse base so it
#      can't drift into a single (U-first) mode.
#   2. MORE TIME for strict low-γ: --T 300 gather / 350 eval (rollout horizon + eval success cap) so the slow-but-safe γ0.1
#      policy can reach before timeout instead of SR 0.05.
#   3. MORE INNER STEPS but STABLE: --early-inner 4 (from 2; the whole run is early-phase since
#      early_until=100) → 2× the gradient steps/iter, using more of the sparse valid pool. Trust gates
#      RE-ENABLED (max-functional-step .03 / max-anchor-drift .02, were 999=off) so destabilizing steps
#      roll back.
#   usage: faithful_div_driver.sh <gpu> <outdir> <viz_every> [extra flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
GPU="$1"; OUTDIR="$2"; VIZ="$3"; shift 3
PRE=../../results/hp_repr/pretrained_a32uni.pt
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 python grid_expand_hardtail.py \
  --ckpt "$PRE" --outdir "$OUTDIR" --iters 100 --start-iter 0 --drop-train-state --seed 910 --lr 2e-5 \
  --window-level --wall-plugs 8 --start-eps 0.3 --goal-xy 4.7 4.7 --reach 0.15 --T 300 \
  --rollouts-per-iter 14 --gather-attempt-cap 400 --batch 64 --gp-buf 200 --qbuf-cap 200 \
  --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.30 --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta 0.2 \
  --early-until 100 --cooldown-from 400 \
  --demo-frac 0.125 --lwf-eta 0.05 --demo-prefix dr05_ \
  --early-inner 4 --inner-steps 4 --cooldown-inner 4 \
  --max-functional-step 0.03 --max-anchor-drift 0.02 --field-grad-clip 1.0 --nfe-explore 8 \
  --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 \
  --log-comp-every 1 \
  --viz-db-every "$VIZ" --ckpt-every 10 --measure-every 10 --probe-cov 10 --tag "$(basename "$OUTDIR")" "$@"
echo "DONE $(basename "$OUTDIR")"
touch "$OUTDIR/IT100_DONE"
