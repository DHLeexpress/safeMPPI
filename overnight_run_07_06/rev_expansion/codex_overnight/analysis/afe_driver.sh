#!/usr/bin/env bash
# AFE-minimal Safe Flow Expansion — arms A (spec-faithful) and B (+demo-replay ablation) on GPU 3.
# C baseline = existing results/p2/faithful_div_it100 (curriculum recipe), no re-run needed.
# Knobs fixed by the measured component probe (2026-07-16): SOCP 2.3 ms/plan -> B=8; eta=0.01
# self-limits the round update to ~0.02 relative field displacement (prox is the trust region).
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
export CUDA_VISIBLE_DEVICES=3
mkdir -p results/afe logs

COMMON="--ckpt ../../results/hp_repr/pretrained_a32uni.pt --rounds 100 --episodes 8 --T 300 --T-eval 350
 --reach 0.15 --K 64 --B 8 --beta 0.2 --lam 1e-2 --n-theta 180 --exec-rule progress
 --lr 2e-5 --eta 0.01 --batch 128 --max-inner 40 --fstep-stop 0.03
 --audit-every 5 --audit-pos 12 --audit-plans 4 --measure-every 10 --M-measure 8
 --ckpt-every 10 --viz-every 1 --wall-plugs 8 --start-eps 0.3 --goal-xy 4.7 4.7 --max-hours 20"

echo "[afe_driver] launching arm A (AFE-minimal) and arm B (+demo 0.125) on GPU 3 ..."
setsid nohup python grid_expand_afe.py $COMMON --seed 910 \
  --outdir results/afe/A_s910 > logs/afe_A_s910.log 2>&1 &
echo "  A pid $!"
setsid nohup python grid_expand_afe.py $COMMON --seed 910 --demo-frac 0.125 --demo-prefix dr05_ \
  --outdir results/afe/B_s910 > logs/afe_B_s910.log 2>&1 &
echo "  B pid $!"
echo "[afe_driver] tail -f logs/afe_A_s910.log logs/afe_B_s910.log"
