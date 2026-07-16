#!/bin/bash
# rev_expansion 9-arm OFAT sweep (fresh-only 2-class curriculum). All arms on GPU3 (256 cores, GPU idle) —
# run concurrently. Baseline A0 aligns with a32_unf (rollout 10, mix 7:3->5:5, β step, early/cool 0.1/0.75).
# No git / no wandb. Run from rev_expansion/.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/rev_sweep
mkdir -p $OUT/logs
BASE="--ckpt ../results/hp_repr/pretrained_a32.pt --iters 1000 --measure-every 100 --m-measure 25 \
--no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict --viz-db-every 100 --seed 0"

L(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $BASE \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

L base                                                    # A0 baseline
L sig_off    --sigma-off                                  # A1 σ-ablation: frontier drops the σ criterion
L sig_strong --frontier-qsig 0.5                          # A2 σ stronger: top-50% σ -> frontier
L pf20       --prog-floor 0.2                             # A3
L pf40       --prog-floor 0.4                             # A4
L beta_lo    --beta-steps 0.7 0.5 0.3 0.1                 # A5
L beta_hi    --beta-steps 0.8 0.6 0.4 0.2                 # A6
L ph_early   --early-frac 0.05 --cooldown-frac 0.60       # A7
L ph_late    --early-frac 0.20 --cooldown-frac 0.85       # A8
echo "REV_SWEEP_LAUNCHED 9 arms -> $OUT"
