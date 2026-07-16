#!/usr/bin/env bash
# Final closed-loop evals for the AFE arms (report_at vs expert, same protocol as the div assembly:
# M=40/gamma, 7 gammas, T=350, reach .15, cleared goal, 8 wall plugs).  Then the validity report and
# per-arm expansion videos.  Usage: bash analysis/afe_assemble.sh [arm ...]  (default: the trio)
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
G="0.1 0.2 0.3 0.4 0.5 0.7 1.0"
COMMON="--M 40 --gammas $G --reach 0.15 --start-eps 0.3 --goal-xy 4.7 4.7 --wall-plugs 8 --T 350 --expert-dir results/expert_g47"
ARMS=${@:-"pure_s910 pure_s911 pure_s912 pure_lam001_s910"}

for a in $ARMS; do
  if [ ! -f "results/afe/$a/final.pt" ]; then echo "skip $a (no final.pt)"; continue; fi
  echo "[assemble] eval $a ..."
  python analysis/report_at.py --ckpt results/afe/$a/final.pt --tag afe_$a --gpu 3 --concurrency 4 \
    $COMMON > logs/eval_afe_$a.log 2>&1
  grep POOLED logs/eval_afe_$a.log | tail -n 1
done

echo "[assemble] validity report + videos ..."
CUDA_VISIBLE_DEVICES="" python analysis/afe_report.py \
  --arms results/afe/pure_s910 results/afe/pure_s911 results/afe/pure_s912 results/afe/pure_lam001_s910 \
  --labels pure-s910 pure-s911 pure-s912 lam001-ref --upfrac --out paper_results/afe_validity_v1.png | tail -n 1
for a in $ARMS; do
  CUDA_VISIBLE_DEVICES="" python video_afe.py --run results/afe/$a --out paper_results/afe_${a}_expansion.mp4 --fps 3 | tail -n 1
done
touch results/afe/ASSEMBLE_DONE
echo "[assemble] DONE"
