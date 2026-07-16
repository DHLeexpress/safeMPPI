#!/bin/bash
# A + C expansion-mechanism sweep (2026-07-07), locked a32 model. Default = a32_unf recipe; each arm changes one knob.
# helios GPU3: A1 (inner8+lr5e-5) + C3 (mix 55/30/15). nyx GPU0/1: C1 (β smooth-exp) + C2 (β aggressive).
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06
N=dohyunlee@dhcp-101-145.caltech.edu; R=projects/cfm_mppi/overnight_run_07_06
PYLIB=/home/dohyun/miniforge3/lib
BASE="--ckpt results/hp_repr/pretrained_a32.pt --iters 5000 --measure-every 500 --m-measure 100 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict"
mkdir -p results/sweep_ac/logs
rsync -az *.py $N:$R/ >/dev/null 2>&1        # push the updated code to nyx
# --- helios GPU3: A1 + C3 ---
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur.py $BASE \
  --inner-steps 8 --lr 5e-5 --tag A1 --outdir results/sweep_ac/A1 >results/sweep_ac/logs/A1.log 2>&1 </dev/null &
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur.py $BASE \
  --mix-start 0.55 0.30 0.15 --tag C3 --outdir results/sweep_ac/C3 >results/sweep_ac/logs/C3.log 2>&1 </dev/null &
# --- nyx GPU0/1: C1 + C2 ---
ssh $N "cd $R && mkdir -p results/sweep_ac/logs && \
  CUDA_VISIBLE_DEVICES=0 setsid nohup /usr/bin/python grid_expand_cur.py $BASE --beta-smooth exp --tag C1 --outdir results/sweep_ac/C1 >results/sweep_ac/logs/C1.log 2>&1 </dev/null & \
  CUDA_VISIBLE_DEVICES=1 setsid nohup /usr/bin/python grid_expand_cur.py $BASE --beta-smooth aggressive --tag C2 --outdir results/sweep_ac/C2 >results/sweep_ac/logs/C2.log 2>&1 </dev/null & \
  sleep 2; echo nyx-launched"
echo "SWEEP_AC_LAUNCHED"
