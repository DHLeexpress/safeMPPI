#!/bin/bash
# Paper-grade milestone eval of a greedy checkpoint: FULL 7-gamma at M100 (the signal the paper reports
# need, NOT the M8/M20 selection). Then OOD rollouts + the three _v3 reports + greedy internals, all
# copied into grand_final_reports/.  Usage: bash analysis/greedy_milestone.sh <ckpt> <iterN> <gpu>
set -u
CKPT="$1"; N="$2"; GPU="${3:-3}"
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=8
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
OUT="results/p2/eval_greedy_it${N}_m100"
echo "[milestone] full 7-gamma M100 eval of $CKPT -> $OUT"
for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do
  CUDA_VISIBLE_DEVICES=$GPU python eval_ae.py policy-worker --gamma "$g" --M 100 --reach 0.1 --seed0 0 \
    --method "greedy-it${N}" --outdir "$OUT" --ckpt "$CKPT" --device cuda --T 250 \
    > "logs/eval_greedy_it${N}_g${g}.log" 2>&1
done
echo "[milestone] OOD-start rollouts"
CUDA_VISIBLE_DEVICES=$GPU python analysis/ood_start_rollouts.py --ckpt "$CKPT" --tag grandfinal \
  > "logs/ood_greedy_it${N}.log" 2>&1
# point the _v3 paper modules (OURS_DIR=eval_grandfinal_m100) at this milestone
rm -f results/p2/eval_grandfinal_m100
ln -sfn "eval_greedy_it${N}_m100" results/p2/eval_grandfinal_m100
python - "$N" << 'PY'
import sys, json, glob, os
N = sys.argv[1]; d = f"results/p2/eval_greedy_it{N}_m100"
print(f"\n=== it{N} PAPER-GRADE (full 7-gamma, M100) ===")
for f in sorted(glob.glob(f"{d}/row_g*.json")):
    r = json.load(open(f))
    print(f"  g{r['gamma']}: SR {r['SR']*100:3.0f}% CR {r['CR']*100:2.0f}% "
          f"clr {r['clearance_mean']:.3f} time {r['time_mean_s']:5.2f} cov {r['coverage']}")
PY
