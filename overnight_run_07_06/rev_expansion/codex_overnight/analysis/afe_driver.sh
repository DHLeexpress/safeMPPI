#!/usr/bin/env bash
# PURE AFE-minimal Safe Flow Expansion (user 2026-07-16: no demo_frac / LwF / anchoring / encoder
# freezing — the prox term is the only regularizer).  Arms: lam=10 (live sigma, from the measured
# lam study) at seeds 910/911/912 + lam=0.01 reference (sigma saturates -> tilt ~uniform) at s910.
# C baseline for the comparison = existing results/p2/faithful_div_it100 (curriculum recipe).
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
export CUDA_VISIBLE_DEVICES=3
mkdir -p results/afe logs

COMMON="--ckpt ../../results/hp_repr/pretrained_a32uni.pt --rounds 100 --episodes 8 --T 300 --T-eval 350
 --reach 0.15 --K 64 --B 8 --beta 0.2 --n-theta 180 --exec-rule progress
 --lr 2e-5 --eta 0.01 --batch 128 --max-inner 40 --fstep-stop 0.03
 --audit-every 5 --audit-pos 12 --audit-plans 4 --measure-every 10 --M-measure 8
 --ckpt-every 10 --viz-every 1 --wall-plugs 8 --start-eps 0.3 --goal-xy 4.7 4.7 --max-hours 20"

echo "[afe_driver] launching PURE arms on GPU 3 ..."
for s in 910 911 912; do
  setsid nohup python grid_expand_afe.py $COMMON --lam 10 --seed $s \
    --outdir results/afe/pure_s$s > logs/afe_pure_s$s.log 2>&1 &
  echo "  pure_s$s (lam10) pid $!"
done
setsid nohup python grid_expand_afe.py $COMMON --lam 1e-2 --seed 910 \
  --outdir results/afe/pure_lam001_s910 > logs/afe_pure_lam001_s910.log 2>&1 &
echo "  pure_lam001_s910 pid $!"
echo "[afe_driver] tail -f logs/afe_pure_s910.log"
