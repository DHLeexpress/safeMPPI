#!/usr/bin/env bash
# Final assembly for the it0→100 window-level run (2026-07-15): evals + U-bias + centerpiece analyses +
# the four vizs + curriculum video. Run once all 4 training runs have IT100_DONE.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
PRE=../../results/hp_repr/pretrained_a32uni.pt
G="0.1 0.2 0.3 0.4 0.5 0.7 1.0"
COMMON="--M 40 --gammas $G --reach 0.15 --start-eps 0.3 --goal-xy 4.7 4.7 --wall-plugs 8 --expert-dir results/expert_g47"

echo "[1/6] parallel evals (ours + 3 brothers), one GPU each ..."
python analysis/report_at.py --ckpt results/p2/faithful_g47_it100/final.pt --tag faithful_it100 --gpu 0 --concurrency 4 $COMMON > logs/eval_ours.log 2>&1 &
python analysis/report_at.py --ckpt results/p2/faithbro_nosocp/final.pt --tag faithbro_nosocp --gpu 1 --concurrency 4 $COMMON > logs/eval_nosocp.log 2>&1 &
python analysis/report_at.py --ckpt results/p2/faithbro_noprog/final.pt --tag faithbro_noprog --gpu 2 --concurrency 4 $COMMON > logs/eval_noprog.log 2>&1 &
python analysis/report_at.py --ckpt results/p2/faithbro_nocur/final.pt --tag faithbro_nocur --gpu 3 --concurrency 4 $COMMON > logs/eval_nocur.log 2>&1 &
wait
echo "  eval scorecards:"
for t in faithful_it100 faithbro_nosocp faithbro_noprog faithbro_nocur; do
  echo -n "   $t: "; grep -E "POOLED" logs/eval_${t#faithful_}.log 2>/dev/null | tail -1 || grep -E "POOLED" logs/eval_ours.log | tail -1
done

echo "[2/6] U-bias probe (it0/it50/it100) ..."
CUDA_VISIBLE_DEVICES=0 python analysis/ubias_probe.py --ckpts \
  it0=$PRE it50=results/p2/faithful_g47_it100/ckpt_50.pt it100=results/p2/faithful_g47_it100/final.pt \
  > logs/ubias.log 2>&1; tail -5 logs/ubias.log

echo "[3/6] draw-sparsity + sigma-field ..."
CUDA_VISIBLE_DEVICES="" python analysis/draw_sparsity.py --run results/p2/faithful_g47_it100 --snap-iter 50 2>>logs/analyses.log | tail -2
CUDA_VISIBLE_DEVICES="" python analysis/sigma_field.py --run results/p2/faithful_g47_it100 2>>logs/analyses.log | tail -3

echo "[4/6] scatter + rollouts + internals ..."
CUDA_VISIBLE_DEVICES="" python paper_results/scatter_v4.py 2>>logs/vizs.log | tail -1
CUDA_VISIBLE_DEVICES="" python paper_results/rollouts_v4.py 2>>logs/vizs.log | tail -1
CUDA_VISIBLE_DEVICES="" python paper_results/internals_v4.py 2>>logs/vizs.log | tail -1

echo "[5/6] curriculum video -> paper_results ..."
python video_curriculum_fixed.py --run results/p2/faithful_g47_it100 \
  --out paper_results/faithful_g47_it100_curriculum.mp4 --ckpt $PRE \
  --goal-xy 4.7 4.7 --start-eps 0.3 --gamma 0.5 \
  --iters 0,1,2,3,4,5,7,10,15,20,30,40,50,60,70,80,90,100 \
  --title "Safe Flow Expansion — window-level curriculum (fresh it0→100, start 0.3 → goal 4.7)" \
  > logs/video.log 2>&1 && echo "  video: $(ls -la paper_results/faithful_g47_it100_curriculum.mp4 | awk '{print $5}') bytes"

echo "[6/6] DONE — assembly complete"
touch results/p2/ASSEMBLE_DONE
