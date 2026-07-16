#!/bin/bash
# P-series micro (user 2026-07-09): pile revival axis on the LOCKED uni_A recipe. fresh_frac {.3,.5,.7} +
# warm-up(10, no-GD) + bounded pile (FIFO 3000, LRU replace=False, relabel/10) — expected: STABLE increase of
# SR50 AND coverage. P5_nw isolates warm-up; P5_rT isolates replace=False. Baseline = micro100/m_base.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/micro_pile; mkdir -p $OUT/logs

REC="--ckpt ../results/hp_repr/pretrained_a32uni.pt --batch 64 --early-inner 1 --inner-steps 2 \
--cooldown-inner 1 --lr 1e-4 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict \
--valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --m-measure 25 --measure-every 25"

P(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $REC \
      --iters 100 --seed 0 --viz-db-every 20 --probe-escape 1 --probe-cov 1 --log-comp-every 1 \
      --pile-cap 3000 --pile-relabel-every 10 --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

P P3    --fresh-frac 0.3 --warmup-gather 10
P P5    --fresh-frac 0.5 --warmup-gather 10
P P7    --fresh-frac 0.7 --warmup-gather 10
P P5_nw --fresh-frac 0.5 --warmup-gather 0
P P5_rT --fresh-frac 0.5 --warmup-gather 10 --pile-replace
echo "LAUNCHED 5 P-series arms on GPU3 (baseline = micro100/m_base)"
