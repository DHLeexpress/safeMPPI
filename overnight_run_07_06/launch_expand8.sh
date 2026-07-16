#!/bin/bash
# 8-arm curriculum safe-flow expansion across GPU 0,1,3 (2026-07-07).
# repr{10,15,20} x {frozen, unfrozen} (6) + repr20 frozen schedule-ablations {gentle-β, flat-β} (2).
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06
export PYLIB=/home/dohyun/miniforge3/lib
CK=results/hp_repr
OD=results/expand_cur
mkdir -p $OD
C="--iters 2000 --measure-every 200"
P() { echo "$CK/pretrained_repr${1}_dr05.pt"; }
L() { CUDA_VISIBLE_DEVICES=$1 LD_LIBRARY_PATH=$PYLIB nohup python grid_expand_cur.py \
        --ckpt "$(P $2)" $C --tag "$4" --outdir "$OD/$4" "${@:5}" > "$OD/$4.log" 2>&1 & }

# GPU 0: repr20 frozen / unfrozen / gentle-β
L 0 20 x r20_frz    --freeze
L 0 20 x r20_unf    --no-freeze --enc-lr-mult 0.3
L 0 20 x r20_gentle --freeze --beta-steps 1.0 0.7 0.5 0.3
# GPU 1: repr15 frozen / unfrozen ; repr20 flat-β (no ramp)
L 1 15 x r15_frz    --freeze
L 1 15 x r15_unf    --no-freeze --enc-lr-mult 0.3
L 1 20 x r20_flat   --freeze --beta-steps 1.0 1.0 1.0 1.0
# GPU 3: repr10 frozen / unfrozen
L 3 10 x r10_frz    --freeze
L 3 10 x r10_unf    --no-freeze --enc-lr-mult 0.3

wait
echo ALL_EXPAND_DONE
