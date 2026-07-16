#!/bin/bash
# Jiggle diagnostic (2026-07-08): β schedule + learning-rate arms to reduce the SR/loss oscillation. All on the
# BASE recipe (batch 32, inner 2/4/2) varying ONLY β or lr, so vs `base` each isolates that lever. Judge on
# SR-std (want LOW) + last-3 mean. Runs in parallel with the batch×inner stab diagnostic. GPU3. No git/wandb.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/jiggle; mkdir -p $OUT/logs
BASE="--ckpt ../results/hp_repr/pretrained_a32.pt --iters 1000 --measure-every 100 --m-measure 25 \
--no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict --viz-db-every 100 --seed 0"
L(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $BASE \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

L lr5e5    --lr 5e-5                                        # half lr -> smaller steps
L lr3e5    --lr 3e-5                                        # third lr
L bfloor5  --beta-steps 1.0 0.8 0.6 0.5                     # β never goes greedy (floor 0.5)
L bexp     --beta-smooth exp                                # smooth anneal, no step jumps
L gentle   --lr 5e-5 --beta-steps 1.0 0.8 0.6 0.5          # combo: gentle lr + non-greedy β
echo "JIGGLE_LAUNCHED 5 arms -> $OUT"
