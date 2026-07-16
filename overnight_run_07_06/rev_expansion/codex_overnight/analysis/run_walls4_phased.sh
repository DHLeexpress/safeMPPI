#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# New from-scratch scientific arm. Never point this at a pure-arm directory.
gpu="${1:-3}"
out="${2:-results/p2/walls4_scratch_phased_s824}"
log="${3:-logs/walls4_scratch_phased_s824.log}"
if [[ -e "$out/final.pt" || -e "$out/recipe.json" ]]; then
  echo "refusing to overwrite existing phased arm: $out" >&2
  exit 2
fi
mkdir -p "$out" "$(dirname "$log")"

CUDA_VISIBLE_DEVICES="$gpu" LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=8 \
python grid_expand_hardtail.py \
  --ckpt ../../results/hp_repr/pretrained_a32uni.pt --outdir "$out" --iters 220 \
  --no-freeze --enc-lr-mult 0.3 --m-measure 5 --measure-every 2 \
  --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 \
  --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.50 200:0.60 400:0.70 \
  --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 \
  --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 \
  --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt ../../results/hp_repr/pretrained_a32uni.pt \
  --lr 1e-4 --nfe-explore 8 --field-grad-clip 1.0 \
  --max-functional-step 999 --max-anchor-drift 999 \
  --targeted-frac 0.5 --n-target 40 --align-temp 0.45 --min-modes-per-gamma 0 \
  --recovery-frac 0.3 --recovery-origin-band 0 1 -0.05 0.18 0 0.45 -0.28 0.05 \
  --recovery-goal-band 4.3 5 4.6 5.06 -0.3 0.3 -0.05 0.35 \
  --hard-quota 12 --hard-x0 oob --hard-x0-cand 64 --strip-probe-every 2 \
  --phased-curriculum --phase-sr-threshold 0.85 --phase-sr-patience 2 \
  --wall-plugs 4 --probe-cov 2 --viz-db-every 1 --ckpt-every 4 --log-comp-every 1 \
  --seed 824 --tag walls4_scratch_phased_s824 > "$log" 2>&1
