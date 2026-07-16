#!/bin/bash
# Rank the emergent-gamma it10 beta-sweep arms vs expert_gt, on the 8-plug walled scene.
# 7 gammas @ reach 0.15, M40 -> so we SEE per-gamma a-d AND whether the untrained-fresh low gammas
# (0.1/0.2) improved via the SHARED weights (user's key question: emergent-gamma trains each gamma's own
# samples; gamma 0.1 can only ride the shared representation until it becomes certifiable). Sequential.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
PRE=../../results/hp_repr/pretrained_a32uni.pt
M=${M:-40}

run() { # tag ckpt
  echo "===== $1 ====="
  python analysis/report_at.py --ckpt "$2" --tag "$1" --M "$M" \
    --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --reach 0.15 --concurrency 6 --gpu 3 2>&1 | tail -13
  echo
}

run pretrained_it0     "$PRE"
run fsw_b02_it10 results/p2/fsw_b02/final.pt
run fsw_b03_it10 results/p2/fsw_b03/final.pt
run fsw_b04_it10 results/p2/fsw_b04/final.pt
echo "=== per-gamma valid2 movement (pretrained -> best arm): the emergent-curriculum signal ==="
echo "RANK_IT10_DONE"; touch results/p2/RANK_IT10_DONE
