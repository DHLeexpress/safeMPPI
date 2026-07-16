#!/bin/bash
# One-shot candidate gate: M25 all-gamma a-e eval (7 parallel workers) + fixed-seed diff vs t104.
# Usage: bash run_gate.sh <ckpt> <tag> <gpu>
set -u
CKPT="$1"; TAG="$2"; GPU="${3:-3}"
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16
OUT="results/p2/eval_${TAG}_m25"
for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do
  CUDA_VISIBLE_DEVICES=$GPU python eval_ae.py policy-worker --gamma "$g" --M 25 --reach 0.1 --seed0 0 \
    --method "Flow-hardtail-${TAG}" --outdir "$OUT" --ckpt "$CKPT" --device cuda --T 250 \
    > "logs/eval_${TAG}_g${g}.log" 2>&1 &
done
wait
python analysis/fixed_seed_gate.py --eval-dir "$OUT" --out "analysis/fixed_seed_gate_${TAG}.json" | tail -20
python - << EOF
import json, glob
print("\n${TAG} — SR / CR / clearance / time / coverage   (P1: 100/0/.281-.333/10.5-15.1/6-11)")
for f in sorted(glob.glob("$OUT/row_g*.json")):
    r = json.load(open(f))
    print(f"  γ{r['gamma']}: {r['SR']*100:.0f}% / {r['CR']*100:.0f}% / {r['clearance_mean']:.3f} / {r['time_mean_s']:.2f} s / {r['coverage']}")
EOF
