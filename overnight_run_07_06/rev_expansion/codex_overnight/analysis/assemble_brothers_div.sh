#!/usr/bin/env bash
# Eval the 3 diversity-recipe brothers at T350 → eval_faithbro_div_*, then re-render rollouts_v5 + scatter_v5
# (already repointed to the div-brother dirs). Run once all 3 have IT100_DONE.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
G="0.1 0.2 0.3 0.4 0.5 0.7 1.0"
COMMON="--M 40 --gammas $G --reach 0.15 --start-eps 0.3 --goal-xy 4.7 4.7 --wall-plugs 8 --T 350 --expert-dir results/expert_g47"

echo "[1/2] eval 3 brothers at T350 (GPU 1/3) ..."
python analysis/report_at.py --ckpt results/p2/faithbro_div_nosocp/final.pt --tag faithbro_div_nosocp --gpu 1 --concurrency 4 $COMMON > logs/eval_div_nosocp.log 2>&1 &
python analysis/report_at.py --ckpt results/p2/faithbro_div_noprog/final.pt --tag faithbro_div_noprog --gpu 3 --concurrency 4 $COMMON > logs/eval_div_noprog.log 2>&1 &
python analysis/report_at.py --ckpt results/p2/faithbro_div_nocur/final.pt  --tag faithbro_div_nocur  --gpu 1 --concurrency 4 $COMMON > logs/eval_div_nocur.log 2>&1 &
wait
for b in nosocp noprog nocur; do echo -n "   $b: "; grep POOLED logs/eval_div_${b}.log | tail -1; done

echo "[2/2] re-render rollouts_v5 + scatter_v5 ..."
CUDA_VISIBLE_DEVICES="" python paper_results/rollouts_v5.py 2>/dev/null | tail -1
CUDA_VISIBLE_DEVICES="" python paper_results/scatter_v5.py 2>/dev/null | tail -1
touch results/p2/BROTHERS_DIV_DONE
echo "DONE"
