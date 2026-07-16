#!/bin/bash
# Extend the winning beta -> it50 (emergent-gamma, 8-plug, faithful no-rec) and REPORT every it10:
# per-gamma a-d scorecard vs expert_gt (7 gammas @ reach 0.15, so we track whether gamma 0.1/0.2 join),
# faithful raw taxonomy, and the curriculum video at the end. Fresh run from pretrained (clean it10..it50
# checkpoints) rather than a resume, since the sweep it10 was cheap.
#   bash analysis/extend_report_driver.sh <beta> [tag]
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
BETA=${1:?need beta}; TAG=${2:-win50_b${BETA/./}}
GPU=${GPU:-3}; PRE=../../results/hp_repr/pretrained_a32uni.pt
mkdir -p logs results/p2 grand_final_reports_rev/${TAG}
rm -rf results/p2/${TAG}

CUDA_VISIBLE_DEVICES=$GPU setsid nohup python grid_expand_hardtail.py \
  --ckpt "$PRE" --outdir "results/p2/${TAG}" \
  --iters 50 --seed 910 --lr 2e-5 --wall-plugs 8 --start-eps 0.05 --reach 0.2 \
  --rollouts-per-iter 28 --gather-attempt-cap 600 --batch 64 --gp-buf 500 --qbuf-cap 500 \
  --emergent-gamma \
  --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
  --quantile-schedule 0:0.30 --mix-start 0.4 0.6 --mix-end 0.4 0.6 --beta "$BETA" \
  --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 1 --cooldown-inner 1 \
  --demo-frac 0.125 --lwf-eta 0.05 --teacher-ckpt "$PRE" \
  --nfe-explore 8 --field-grad-clip 1.0 --max-functional-step 999 --max-anchor-drift 999 \
  --targeted-frac 0.0 --min-modes-per-gamma 0 --recovery-frac 0.0 --hard-quota 0 \
  --m-measure 5 --measure-every 5 --probe-cov 2 --log-comp-every 1 \
  --viz-db-every 1 --ckpt-every 5 --tag "$TAG" > "logs/${TAG}.log" 2>&1 &
echo "$(date '+%H:%M') launched extend $TAG (beta=$BETA -> it50) PID $!" >> logs/extend.log

# report at each it10 boundary as the checkpoints land
for N in 10 20 30 40 50; do
  CK="results/p2/${TAG}/ckpt_${N}.pt"
  waited=0
  while [ ! -f "$CK" ]; do
    pgrep -f "grid_expand_hardtail.*${TAG}" >/dev/null || { echo "$(date '+%H:%M') $TAG proc gone before it$N" >> logs/extend.log; break; }
    sleep 20; waited=$((waited+20)); [ $waited -gt 3600 ] && break
  done
  [ -f "$CK" ] || continue
  echo "$(date '+%H:%M') $TAG it$N ckpt landed -> reporting" >> logs/extend.log
  python analysis/report_at.py --ckpt "$CK" --tag "${TAG}_it${N}" --M 60 \
    --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --reach 0.15 --concurrency 5 --gpu $GPU \
    > "grand_final_reports_rev/${TAG}/scorecard_it${N}.txt" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python analysis/faithful_taxonomy.py --ckpt "$CK" --tag "${TAG}_it${N}" \
    --M 40 --gammas 0.1 0.5 1.0 --reach 0.15 --wall-plugs 8 --start-eps 0.05 \
    --out-dir "grand_final_reports_rev/${TAG}" > "grand_final_reports_rev/${TAG}/taxonomy_it${N}.txt" 2>&1 || true
done

# curriculum video over the whole run
python video_curriculum_fixed.py --run "results/p2/${TAG}" \
  --out "grand_final_reports_rev/${TAG}/curriculum_${TAG}.mp4" \
  --title "Safe Flow Expansion (emergent-gamma, beta=$BETA)" > "logs/${TAG}_video.log" 2>&1 || true
echo "$(date '+%H:%M') $TAG DONE (reports + video)" >> logs/extend.log
touch "results/p2/${TAG}/EXTEND_DONE"
