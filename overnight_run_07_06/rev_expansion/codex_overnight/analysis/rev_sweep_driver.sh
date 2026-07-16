#!/bin/bash
# REV sweep (user 2026-07-13): fundamentally fix the OOB ill-conditioning (pretrained flies out of the
# [0,5] box 64-80% of the time = goal overshoot; CR~0 hides it). Hypotheses: (a) too many EASY samples
# -> use FRONTIER early + keep easy LOW; (b) early beta + mixing ratio matter; (c) goal-band recovery
# targets the overshoot. Each config: pretrained -> iter 10 (frozen, lr 2e-5, NO phased), then faithful
# taxonomy. 3 concurrent, 2 waves. Ranker (rev_rank.py) evals + keeps best.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=3

launch() {   # tag beta frontier recovery
  local tag=$1 beta=$2 f=$3 rec=$4
  local e=$(python3 -c "print(round(1-$f,4))")
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
    --ckpt "$PRE" --outdir "results/p2/${tag}" \
    --iters 10 --seed 900 --lr 2e-5 \
    --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 \
    --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
    --quantile-schedule 0:0.50 200:0.60 400:0.70 \
    --mix-start "$e" "$f" --mix-end "$e" "$f" --beta "$beta" --gp-buf 500 \
    --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
    --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" \
    --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
    --targeted-frac 0.5 --n-target 40 --align-temp 0.45 --min-modes-per-gamma 2 --target-perp-brake \
    --recovery-frac "$rec" --recovery-origin-band 0.0 1.0 -0.05 0.18 0.0 0.45 -0.28 0.05 \
    --recovery-goal-band 4.3 5.0 4.6 5.06 -0.30 0.30 -0.05 0.35 \
    --hard-quota 12 --hard-x0 oob --hard-x0-cand 64 --strip-probe-every 2 \
    --m-measure 5 --measure-every 5 --probe-cov 2 --log-comp-every 1 \
    --viz-db-every 1 --ckpt-every 5 --tag "$tag" > "logs/${tag}.log" 2>&1 &
  echo "$(date '+%H:%M') launched $tag (b$beta f$f r$rec) PID $!" >> logs/rev_sweep.log
}

wait_wave() {   # wait until all given tags have final.pt or died
  local tags=("$@")
  while true; do
    local done=0
    for t in "${tags[@]}"; do
      [ -f "results/p2/${t}/final.pt" ] && done=$((done+1))
    done
    [ "$done" -ge "${#tags[@]}" ] && break
    pgrep -f "grid_expand_hardtail.*rev_" > /dev/null || { echo "$(date '+%H:%M') wave procs gone (done=$done/${#tags[@]})" >> logs/rev_sweep.log; break; }
    sleep 30
  done
}

echo "$(date '+%H:%M') REV sweep start" >> logs/rev_sweep.log
# wave 1
launch rev_b3_f50_r3 0.3 0.5  0.3
launch rev_b3_f75_r3 0.3 0.75 0.3
launch rev_b2_f75_r3 0.2 0.75 0.3
wait_wave rev_b3_f50_r3 rev_b3_f75_r3 rev_b2_f75_r3
# wave 2
launch rev_b2_f50_r3 0.2 0.5  0.3
launch rev_b3_f75_r5 0.3 0.75 0.5
launch rev_b3_f25_r3 0.3 0.25 0.3
wait_wave rev_b2_f50_r3 rev_b3_f75_r5 rev_b3_f25_r3
echo "$(date '+%H:%M') REV sweep TRAINING done -> ranking" >> logs/rev_sweep.log
touch results/p2/REV_SWEEP_TRAINED
