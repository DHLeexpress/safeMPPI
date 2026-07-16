#!/bin/bash
# Early-β / initial-IS diagnostic (2026-07-08): does stronger early exploration (β>1) break the below-diagonal
# mode-lock in the FIRST iters? SHORT runs (150 iters) with DENSE viz_db (every 10) so we can read the
# below-% + first-action angle at it10/20/50. Stable-ish base (batch 64, inner 1/2/1, lr 5e-5). GPU3. No git.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/earlybeta; mkdir -p $OUT/logs
BASE="--ckpt ../results/hp_repr/pretrained_a32.pt --iters 150 --measure-every 25 --m-measure 25 \
--no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict --viz-db-every 10 --seed 0 \
--batch 64 --early-inner 1 --inner-steps 2 --cooldown-inner 1 --lr 5e-5"
L(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $BASE \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

L eb1 --beta-steps 1.0 0.8 0.6 0.5      # baseline β (starts 1.0)
L eb2 --beta-steps 2.0 1.5 1.0 0.5      # β>1 early
L eb3 --beta-steps 3.0 2.0 1.0 0.5      # β>>1 early
echo "EARLYBETA_LAUNCHED 3 arms -> $OUT"
