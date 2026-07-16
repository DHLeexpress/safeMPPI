#!/bin/bash
# Fix-and-extend (2026-07-07): re-run the 5 repr20 arms on the KNOWN-GOOD a20 model, and move the
# trunk/encoder-depth sweep onto the robust repr32 base (repr20-depth was OOD-origin-fragile).
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06
export PYLIB=/home/dohyun/miniforge3/lib
CK=results/hp_repr; OD=results/sweep_overnight; mkdir -p $OD/logs
# --- pretrain repr32 depth variants (robust base) ---
( CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$PYLIB python pretrain_repr.py --repr 32 --trunk-hidden 160 160 96 --enc-depth 2 --epochs 120 --tag a32T > $OD/logs/pre_a32T.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$PYLIB python pretrain_repr.py --repr 32 --trunk-hidden 160 96 --enc-depth 3 --epochs 120 --tag a32E > $OD/logs/pre_a32E.log 2>&1 ) &
wait
C="--iters 5000 --measure-every 500 --m-measure 100"
run() { CUDA_VISIBLE_DEVICES=$1 LD_LIBRARY_PATH=$PYLIB nohup python grid_expand_cur.py \
          --ckpt $CK/pretrained_$2.pt $C --tag $3 --outdir $OD/$3 "${@:4}" > $OD/logs/$3.log 2>&1 & }
# repr20 anchor/freeze/strict sweep (fixed a20)
run 0 a20  a20_hero  --freeze --demo-frac 0.25 --lwf-eta 0.05 --easy-strict
run 0 a20  a20_unf   --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict
run 1 a20  a20_comp  --freeze --demo-frac 0.25 --lwf-eta 0.05
run 1 a20  a20_noanc --freeze --easy-strict
run 3 a20  a20_hi    --freeze --demo-frac 0.40 --lwf-eta 0.10 --easy-strict
# repr32 depth sweep (robust base)
run 3 a32T a32T_hero --freeze --demo-frac 0.25 --lwf-eta 0.05 --easy-strict
run 0 a32E a32E_hero --freeze --demo-frac 0.25 --lwf-eta 0.05 --easy-strict
wait
echo FIX_DONE
