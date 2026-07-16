#!/bin/bash
# micro100 (user 2026-07-09): 6 arms x 100 iters, INSTANTANEOUS per-iter measurement — M=50 faithful @γ0.5
# SR/CR + staircase-id coverage (NOT cumulative) + escape probe + composition/rid + loss/gradRMS -> probe.jsonl.
# Goal: smooth gradients -> STABLE SR *and* coverage. GPU3, no git/wandb.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/micro100; mkdir -p $OUT/logs

REC="--ckpt ../results/hp_repr/pretrained_a32uni.pt --batch 64 --early-inner 1 --inner-steps 2 \
--cooldown-inner 1 --lr 1e-4 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict \
--valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --m-measure 25 --measure-every 25"

M(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $REC \
      --iters 100 --seed 0 --viz-db-every 20 --probe-escape 1 --probe-cov 1 --log-comp-every 1 \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

M m_base
M m_bup    --beta-steps 0.3 0.5 0.7 1.0
M m_strat  --strat-rid
M m_sigabs --easy-sig-abs 0.25 --easy-demo-backfill
M m_skipf  --easy-skip-first 2
M m_combo  --beta-steps 0.3 0.5 0.7 1.0 --strat-rid --easy-sig-abs 0.25 --easy-demo-backfill --easy-skip-first 2
echo "LAUNCHED 6 micro100 arms on GPU3"
