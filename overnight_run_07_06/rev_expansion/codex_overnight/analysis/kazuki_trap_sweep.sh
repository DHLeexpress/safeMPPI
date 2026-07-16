#!/bin/bash
# Kazuki TRAP sweep (user 2026-07-15): cleared goal (4.7,4.7)/start-eps 0.3, 8-plug scene.
# Goal = the LOCAL-MINIMUM regime BETWEEN the two known anchors — w09 (w_safe .9 -> SR0, too catastrophic)
# and g47-tuned (w_safe .3/cw5/gw5/gc1 -> SR1, too good). Push collision-cost UP + goal-pull DOWN so the
# MPPI circles to the T=250 timeout on the hard episodes (partial SR) while the reached episodes still post
# clean clearance/time. Sweep at gamma_ctx 0.5; the winner is re-run at 3 gammas + eval'd for the vizs.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
GPU=${GPU:-3}
OUT=results/kazuki_g47_trap; mkdir -p "$OUT" logs
# (w_safe coll_w goal_w goal_coef reach) — higher safety/collision, lower goal-pull than the SR1 tuned arm
CONFIGS=(
  "0.45 10 4 1.0 0.15"
  "0.50 15 3 0.8 0.15"
  "0.55 15 3 0.8 0.15"
  "0.60 20 2 0.7 0.15"
  "0.70 30 2 0.6 0.15"
)
: > logs/kaz_trap_summary.log
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; ws=$1; cw=$2; gw=$3; gc=$4; rc=$5
  tag="ws${ws}_cw${cw}_gw${gw}_gc${gc}"
  CUDA_VISIBLE_DEVICES=$GPU python kazuki_baseline.py \
    --w-safe "$ws" --gamma-ctx 0.5 --M 20 --T 250 --tag "$tag" \
    --coll-w "$cw" --goal-w "$gw" --goal-coef "$gc" --beta-mppi 20 \
    --reach "$rc" --wall-plugs 8 --start-eps 0.3 --goal-xy 4.7 4.7 --outdir "$OUT" >> logs/kaz_trap.log 2>&1
  echo "$(date '+%H:%M') $tag: $(python3 -c "import json;d=json.load(open('$OUT/$tag.json'));print('SR%.2f CR%.2f steps%.0f'%(d['SR'],d['CR'],d['mean_steps']))" 2>/dev/null)" | tee -a logs/kaz_trap_summary.log
done
touch "$OUT/TRAP_SWEEP_DONE"
echo "$(date '+%H:%M') Kazuki trap sweep DONE" | tee -a logs/kaz_trap_summary.log
