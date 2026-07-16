#!/bin/bash
# Anchor (δ demo-replay + η LwF) safe-flow expansion across GPU 0,1,3 (2026-07-07).
# Tests whether the pretraining anchor SUSTAINS the enhancement that the un-anchored curriculum forgets.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06
export PYLIB=/home/dohyun/miniforge3/lib
CK=results/hp_repr; OD=results/expand_cur; mkdir -p $OD
C="--iters 2000 --measure-every 200 --freeze"
P() { echo "$CK/pretrained_repr${1}_dr05.pt"; }
L() { CUDA_VISIBLE_DEVICES=$1 LD_LIBRARY_PATH=$PYLIB nohup python grid_expand_cur.py \
        --ckpt "$(P $2)" $C --tag "$3" --outdir "$OD/$3" "${@:4}" > "$OD/$3.log" 2>&1 & }

# GPU 0: repr20 δ-only, repr20 strong-both
L 0 20 a_r20_delta --demo-frac 0.25
L 0 20 a_r20_hi    --demo-frac 0.40 --lwf-eta 0.10
# GPU 1: repr20 η-only, repr15 both
L 1 20 a_r20_lwf   --lwf-eta 0.05
L 1 15 a_r15_both  --demo-frac 0.25 --lwf-eta 0.05
# GPU 3: repr20 both (the headline), repr15 δ-only
L 3 20 a_r20_both  --demo-frac 0.25 --lwf-eta 0.05
L 3 15 a_r15_delta --demo-frac 0.25

wait
echo ALL_ANCHOR_DONE
