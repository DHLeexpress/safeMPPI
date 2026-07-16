#!/bin/bash
# Stage-1 schedule sweep (2026-07-07): β-schedule × mix-ramp factors isolated around the a32_unf baseline.
# 5k iters each, locked a32 model + winning recipe; collapse-termination on; viz_db saved every 1000.
# helios GPU0: B1,M1 · GPU1: B2,M2 · GPU3: B3,M3 · nyx GPU0: BASE re-run (seed0) · nyx GPU1: BASE seed1 (noise).
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06
N=dohyunlee@dhcp-101-145.caltech.edu; R=projects/cfm_mppi/overnight_run_07_06
PYLIB=/home/dohyun/miniforge3/lib
BASE="--ckpt results/hp_repr/pretrained_a32.pt --iters 5000 --measure-every 500 --m-measure 100 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict"
mkdir -p results/stage1/logs
rsync -az grid_expand_cur.py $N:$R/ >/dev/null 2>&1
ssh $N "mkdir -p $R/results/stage1/logs" >/dev/null 2>&1
L(){ CUDA_VISIBLE_DEVICES=$1 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur.py $BASE \
      --tag $2 --outdir results/stage1/$2 "${@:3}" >results/stage1/logs/$2.log 2>&1 </dev/null & }
# --- β arms (mix at baseline) ---
L 0 S1_B1 --beta-fracs 0 0.15 0.3 0.5                 # earlier steps: β=.2 by it1500, .1 by 2500
L 1 S1_B2 --beta-fracs 0 0.35 0.6 0.85                # later/protective: long β=1 easy phase
L 3 S1_B3 --beta-steps 1.0 0.6 0.35 0.2               # finer staircase, never fully greedy
# --- mix arms (β at baseline) ---
L 0 S1_M1 --early-frac 0.05 --cooldown-frac 0.5       # earlier ramp: frontier 33% by it2500
L 1 S1_M2 --early-frac 0.2 --cooldown-frac 0.9        # later ramp: protect easy phase longer
L 3 S1_M3 --mix-end 0.25 0.30 0.45                    # frontier-heavier END only
# --- baseline anchors on nyx (each ssh has its own cd — no shared-cd bug) ---
ssh $N "cd $R && CUDA_VISIBLE_DEVICES=0 setsid nohup /usr/bin/python grid_expand_cur.py $BASE --tag S1_BASE --outdir results/stage1/S1_BASE >results/stage1/logs/S1_BASE.log 2>&1 </dev/null &"
ssh $N "cd $R && CUDA_VISIBLE_DEVICES=1 setsid nohup /usr/bin/python grid_expand_cur.py $BASE --seed 1 --tag S1_BASEs1 --outdir results/stage1/S1_BASEs1 >results/stage1/logs/S1_BASEs1.log 2>&1 </dev/null &"
echo STAGE1_LAUNCHED
