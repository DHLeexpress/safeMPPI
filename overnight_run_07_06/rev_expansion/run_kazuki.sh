#!/bin/bash
# Phase 2.5 protocol (user): apples-to-apples on the SAME gamma-conditioned a32uni policy.
#   HIS  = guidance+select+refine (kazuki_baseline.py), 9 runs: gamma_ctx {0.1,0.5,1.0} x w_safe {0.1,0.5,0.9}
#   OURS = faithful FM deploy (no guidance, no MPPI), 3 runs: gamma {0.1,0.5,1.0}   [ours_faithful.py]
# GPU2 (shared with uni_base), 3 concurrent. M=25, T=250, reach 0.1. No git/wandb.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/kazuki; mkdir -p $OUT/logs
K(){ CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 LD_LIBRARY_PATH=$PYLIB setsid nohup python -u kazuki_baseline.py \
      --gamma-ctx $1 --w-safe $2 --M 25 --T 250 --tag kaz_g$1_w$2 --outdir $OUT \
      > $OUT/logs/kaz_g$1_w$2.log 2>&1 </dev/null & }
for g in 0.1 0.5 1.0; do
  for w in 0.1 0.5 0.9; do K $g $w; done
  wait   # 3 concurrent per gamma batch
done
echo KAZUKI_9RUNS_DONE
