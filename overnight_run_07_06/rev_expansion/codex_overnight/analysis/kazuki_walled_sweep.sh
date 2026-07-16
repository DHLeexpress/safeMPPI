#!/bin/bash
# Kazuki on the 8-plug WALLED scene: wide aggressive sweep to extract SOME success (user 2026-07-14).
# The dense 72-obstacle scene + goal-corner plugs defeat its MPPI (SR 0/timeout at default); push goal-pull
# up + collision-cost down + larger reach. Eval each with saved-worker (reach 0.15) -> row for the panel.
set -u
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=4
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
GPU=${GPU:-2}
OUT=results/kazuki_walls8_sweep; mkdir -p "$OUT" logs
# (w_safe coll_w goal_w goal_coef reach)  — aggressive goal-pull, low collision-aversion
CONFIGS=(
  "0.3 5 5 1.0 0.15"
  "0.3 2 8 1.5 0.20"
  "0.5 5 8 1.5 0.15"
  "0.1 10 5 1.0 0.15"
  "0.3 3 10 2.0 0.20"
)
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; ws=$1; cw=$2; gw=$3; gc=$4; rc=$5
  tag="ws${ws}_cw${cw}_gw${gw}_gc${gc}_r${rc}"
  CUDA_VISIBLE_DEVICES=$GPU python kazuki_baseline.py \
    --w-safe "$ws" --gamma-ctx 0.5 --M 20 --T 250 --tag "$tag" \
    --coll-w "$cw" --goal-w "$gw" --goal-coef "$gc" --beta-mppi 20 \
    --reach "$rc" --wall-plugs 8 --start-eps 0.05 --outdir "$OUT" >> logs/kaz_sweep.log 2>&1
  echo "$(date '+%H:%M') $tag done: $(python3 -c "import json;d=json.load(open('$OUT/$tag.json'));print('SR%.2f CR%.2f steps%.0f'%(d['SR'],d['CR'],d['mean_steps']))" 2>/dev/null)" >> logs/kaz_sweep_summary.log
done
touch "$OUT/SWEEP_DONE"
echo "$(date '+%H:%M') Kazuki walled sweep DONE" >> logs/kaz_sweep_summary.log
