#!/bin/bash
# FAITHFUL it10 sweep (2026-07-14): pretrained -> 8-plug WALLED scene, safe flow expansion, NO rec / NO
# rev / NO strips (frontier-knobs ONLY). Sweep beta {0.2,0.3,0.4} (lower=explore toward high-sigma frontier).
# EMERGENT gamma-curriculum (--emergent-gamma): keep ALL 7 gammas, uniform, but the update is NOT blocked by
# gammas with no certified windows yet. From the pretrained, gamma 0.1/0.2 are 0% SOCP-valid (their SOCP
# needs interior clearance the policy lacks — pushing the perimeter can't fix that AND breaks the encoder,
# see grand_final_reports_rev/push_deploy.png); as the frontier lifts interior clearance the low gammas join
# on their own -> curriculum emerges from the certificate, no per-gamma special-casing. In-distribution
# 8-plug scene (walls +/-0.2 are load-bearing context). Start +eps 0.05 (origin on corner plugs). gp==qbuf
# 200 (deterministic sigma). quantile 0.30 + mix 0.4/0.6 (frontier-heavy). Frozen encoder, lr 2e-5, demo
# 0.125 + LwF 0.05 anchor (holds gamma 0.1/0.2 meanwhile). pretrained -> it10, ckpt/5, viz-db/iter. Goal:
# beat the demo expert (results/expert_gt, kept as-is) per gamma on a-d.  ~3GB/arm, 3 concurrent << 50% GPU3.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=${GPU:-3}
ITERS=${ITERS:-10}
mkdir -p logs results/p2

launch() {   # tag beta
  local tag=$1 beta=$2
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
    --ckpt "$PRE" --outdir "results/p2/${tag}" \
    --iters "$ITERS" --seed 910 --lr 2e-5 --wall-plugs 8 \
    --start-eps 0.05 --reach 0.2 \
    --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 --gp-buf 500 --qbuf-cap 500 \
    --emergent-gamma \
    --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
    --quantile-schedule 0:0.30 \
    --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta "$beta" \
    --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
    --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" \
    --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
    --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 \
    --m-measure 5 --measure-every 5 --probe-cov 2 --log-comp-every 1 \
    --viz-db-every 1 --ckpt-every 5 --tag "$tag" > "logs/${tag}.log" 2>&1 &
  echo "$(date '+%H:%M') launched $tag (beta=$beta wall8 eps0.05 faithful) PID $!" >> logs/faithful_sweep.log
}

echo "$(date '+%H:%M') FAITHFUL sweep start (beta 0.2/0.3/0.4 -> it$ITERS, 3-concurrent GPU$GPU)" >> logs/faithful_sweep.log
launch fsw_b02 0.2
launch fsw_b03 0.3
launch fsw_b04 0.4

# wait for all 3 finals (or all procs gone)
while true; do
  done=0
  for t in fsw_b02 fsw_b03 fsw_b04; do [ -f "results/p2/${t}/final.pt" ] && done=$((done+1)); done
  [ "$done" -ge 3 ] && break
  empty=0; for k in 1 2 3; do pgrep -f "grid_expand_hardtail.*fsw_" >/dev/null || empty=$((empty+1)); sleep 3; done
  [ "$empty" -ge 3 ] && { echo "$(date '+%H:%M') fsw procs gone (done=$done)" >> logs/faithful_sweep.log; break; }
  sleep 20
done
echo "$(date '+%H:%M') FAITHFUL sweep it$ITERS done (finals=$done)" >> logs/faithful_sweep.log
touch results/p2/FAITHFUL_SWEEP_IT${ITERS}_DONE
