#!/bin/bash
# Micro-experiments (user 2026-07-09): the warm-up pathology = noisy near-origin initial windows hammered as
# easy -> unstable origin escape. 6x 20-iter arms with PER-ITER composition + escape-probe logging, focused on
# the early pattern; + 2 FULL 1000-iter beta-ASCENDING arms (0.3->1.0: explore early when all is unknown,
# faithful late to HOLD the peak — the locked recipe's descending beta ends at max-explore right when the
# mid-run collapse happens). GPU3, no git/wandb.
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion
PYLIB=/home/dohyun/miniforge3/lib
OUT=results/micro20; mkdir -p $OUT/logs results/uni_expand/logs

# LOCKED uni_A recipe base (a32uni, batch64, inner 1/2/1, lr 1e-4, vpf .15, knobs OFF)
REC="--ckpt ../results/hp_repr/pretrained_a32uni.pt --batch 64 --early-inner 1 --inner-steps 2 \
--cooldown-inner 1 --lr 1e-4 --no-freeze --enc-lr-mult 0.3 --demo-frac 0.25 --lwf-eta 0.05 --easy-strict \
--valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 --m-measure 25 --measure-every 100"

M(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $REC \
      --iters 20 --seed 0 --viz-db-every 5 --probe-escape 1 --log-comp-every 1 \
      --tag $1 --outdir $OUT/$1 "${@:2}" >$OUT/logs/$1.log 2>&1 </dev/null & }

M m_base                                                        # control (locked recipe)
M m_bup    --beta-steps 0.3 0.5 0.7 1.0                         # beta ASCENDING (low beta = explore FIRST)
M m_strat  --strat-rid                                          # batch round-robins across source rollouts
M m_sigabs --easy-sig-abs 0.25 --easy-demo-backfill             # ABSOLUTE sigma easy gate + demo backfill
M m_skipf  --easy-skip-first 2                                  # initial escape windows never easy
M m_combo  --beta-steps 0.3 0.5 0.7 1.0 --strat-rid --easy-sig-abs 0.25 --easy-demo-backfill --easy-skip-first 2

# FULL beta-ascending comparison (user: "i think it is prominent"), 2 seeds incl. the collapsed seed 1
F(){ CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$PYLIB setsid nohup python grid_expand_cur_rev.py $REC \
      --iters 1000 --viz-db-every 100 --probe-escape 25 --log-comp-every 25 \
      --beta-steps 0.3 0.5 0.7 1.0 --tag $1 --outdir results/uni_expand/$1 "${@:2}" \
      >results/uni_expand/logs/$1.log 2>&1 </dev/null & }
F uni_bup_s0 --seed 0
F uni_bup_s1 --seed 1
echo "LAUNCHED 6 micro (20it, per-iter probe) + 2 full beta-up (1000it) on GPU3"
