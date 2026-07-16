#!/bin/bash
# WALL8 ABLATION (corrected, 2026-07-14): full vs -SOCP vs -progress vs -curriculum, ALL sharing one
# recipe so the rollouts figure is apples-to-apples. The earlier arms STARVED (n_frontier=0) only because
# they started at the EXACT origin (start_eps 0), which OVERLAPS the corner plugs (0,-0.2)&(-0.2,0) whose
# edges reach x=0/y=0 -> the real SOCP correctly rejects window 0 -> no valid windows. FIX = --start-eps
# 0.05 (clears the plug; socp_sanity confirms pretrained 75% valid2 WITH SOCP once cleared) + --reach 0.2.
# The geometric-clearance ablation (-SOCP) tolerated the origin, which is why ONLY it trained before.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=${GPU:-3}
ITERS=${ITERS:-10}
mkdir -p logs results/p2

launch() {   # tag  extra_ablation_flags...
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
    --ckpt "$PRE" --outdir "results/p2/${tag}" \
    --iters "$ITERS" --seed 910 --lr 2e-5 --wall-plugs 8 \
    --start-eps 0.05 --reach 0.2 \
    --rollouts-per-iter 10 --gather-attempt-cap 400 --batch 64 --gp-buf 200 --qbuf-cap 200 \
    --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
    --quantile-schedule 0:0.30 \
    --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta 0.2 \
    --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
    --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" \
    --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
    --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 \
    --m-measure 5 --measure-every 5 --probe-cov 2 --log-comp-every 1 \
    --viz-db-every 1 --ckpt-every 5 --tag "$tag" "$@" > "logs/${tag}.log" 2>&1 &
  echo "$(date '+%H:%M') launched $tag ($*) PID $!" >> logs/wall8_ablation.log
}

wait_all() {
  local tags=("$@")
  while true; do
    local done=0
    for t in "${tags[@]}"; do [ -f "results/p2/${t}/final.pt" ] && done=$((done+1)); done
    [ "$done" -ge "${#tags[@]}" ] && break
    local empty=0
    for k in 1 2 3; do pgrep -f "grid_expand_hardtail.*wa8_" >/dev/null || empty=$((empty+1)); sleep 2; done
    [ "$empty" -ge 3 ] && { echo "$(date '+%H:%M') procs gone (done=$done)" >> logs/wall8_ablation.log; break; }
    sleep 25
  done
}

echo "$(date '+%H:%M') WALL8 ABLATION start (eps0.05 reach0.2, 2-concurrent, iters=$ITERS)" >> logs/wall8_ablation.log
launch wa8_full                                   # real SOCP + progress + curriculum
launch wa8_nosocp   --ablate-socp                 # geometric clearance
wait_all wa8_full wa8_nosocp
launch wa8_noprog   --ablate-progress             # drop the goal-progress condition
launch wa8_nocur    --ablate-curriculum           # single class (no easy/frontier split)
wait_all wa8_noprog wa8_nocur
echo "$(date '+%H:%M') WALL8 ABLATION done" >> logs/wall8_ablation.log
touch results/p2/WALL8_ABLATION_DONE
