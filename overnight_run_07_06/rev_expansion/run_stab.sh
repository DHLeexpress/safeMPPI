#!/bin/bash
# Stability diagnostic (2026-07-08): 2x2 batch{32,64} x inner{1/2/1, 2/4/2} on the baseline recipe.
# base (results/rev_sweep/base) already IS the (batch32, inner2/4/2) corner -> run the other 3. Judge on
# last-3-mean + SR-std (want high mean, LOW std = holds its peak, no oscillation). GPU3, concurrent. No git/wandb.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/stab; mkdir -p $OUT/logs
BASE="--ckpt ../results/hp_repr/pretrained_a32.pt --iters 1000 --measure-every 100 --m-measure 25 \
--no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict --viz-db-every 100 --seed 0"
L(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $BASE \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

L b32_i121 --batch 32 --early-inner 1 --inner-steps 2 --cooldown-inner 1   # gentler inner, small batch
L b64_i242 --batch 64                                                       # bigger batch, default inner 2/4/2
L b64_i121 --batch 64 --early-inner 1 --inner-steps 2 --cooldown-inner 1    # bigger batch + gentler inner
echo "STAB_LAUNCHED 3 arms -> $OUT (base = the batch32/inner2/4/2 corner, already in results/rev_sweep/base)"
