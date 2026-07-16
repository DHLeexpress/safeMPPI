#!/bin/bash
# WALLED (8-plug) sanity sweep (user 2026-07-13): close the OOB escape routes (origin/goal corner
# openings) so OOB becomes a detectable COLLISION; forget rec (recovery_frac 0); reduce rollouts
# DRASTICALLY (28 -> 6, the 64 batch only needs ~56 fresh); INCREASE lr (trust the frontier). gp_buf500
# (deterministic sigma). Each config: pretrained -> iter 10, then faithful taxonomy on the WALLED scene.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
GPU=3

launch() {   # tag lr frontier
  local tag=$1 lr=$2 f=$3
  local e=$(python3 -c "print(round(1-$f,4))")
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
    --ckpt "$PRE" --outdir "results/p2/${tag}" \
    --iters 10 --seed 910 --lr "$lr" --wall-plugs 8 \
    --rollouts-per-iter 6 --gather-attempt-cap 120 --batch 64 --gp-buf 500 \
    --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
    --quantile-schedule 0:0.50 200:0.60 400:0.70 \
    --mix-start "$e" "$f" --mix-end "$e" "$f" --beta 0.2 \
    --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
    --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" \
    --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
    --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 --strip-probe-every 2 \
    --m-measure 5 --measure-every 5 --probe-cov 2 --log-comp-every 1 \
    --viz-db-every 1 --ckpt-every 5 --tag "$tag" > "logs/${tag}.log" 2>&1 &
  echo "$(date '+%H:%M') launched $tag (lr$lr f$f wall8 rec0 roll6) PID $!" >> logs/wall8_sweep.log
}

wait_all() {
  local tags=("$@")
  while true; do
    local done=0
    for t in "${tags[@]}"; do [ -f "results/p2/${t}/final.pt" ] && done=$((done+1)); done
    [ "$done" -ge "${#tags[@]}" ] && break
    pgrep -f "grid_expand_hardtail.*w8_" > /dev/null || { echo "$(date '+%H:%M') procs gone (done=$done)" >> logs/wall8_sweep.log; break; }
    sleep 25
  done
}

echo "$(date '+%H:%M') WALL8 sweep start (cap120, 2-concurrent)" >> logs/wall8_sweep.log
launch w8_lr5e5_f50 5e-5 0.5
launch w8_lr1e4_f50 1e-4 0.5
wait_all w8_lr5e5_f50 w8_lr1e4_f50
launch w8_lr5e5_f75 5e-5 0.75
launch w8_lr1e4_f75 1e-4 0.75
wait_all w8_lr5e5_f75 w8_lr1e4_f75
echo "$(date '+%H:%M') WALL8 sweep TRAINING done" >> logs/wall8_sweep.log
touch results/p2/WALL8_SWEEP_TRAINED
