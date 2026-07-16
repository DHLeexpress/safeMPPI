#!/usr/bin/env bash
# Assembly for the diversity-preserving run (faithful_div_it100). Eval at T350 (more time for γ0.1),
# U-bias trajectory (did diversity hold?), draw-sparsity (did more inner steps raise usage?), video.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
PRE=../../results/hp_repr/pretrained_a32uni.pt
RUN=results/p2/faithful_div_it100
G="0.1 0.2 0.3 0.4 0.5 0.7 1.0"

echo "[1/5] eval div at T350 (goal 4.7,4.7, reach 0.15) vs expert ..."
python analysis/report_at.py --ckpt $RUN/final.pt --tag faithful_div_it100 --gpu 0 --concurrency 5 \
  --M 40 --gammas $G --reach 0.15 --start-eps 0.3 --goal-xy 4.7 4.7 --wall-plugs 8 --T 350 \
  --expert-dir results/expert_g47 > logs/eval_div.log 2>&1
grep -E "^ 0\.1 | 0\.5 | 1\.0 |POOLED" logs/eval_div.log | tail -4

echo "[2/5] U-bias it0/50/100 (diversity held?) ..."
CUDA_VISIBLE_DEVICES=0 python analysis/ubias_probe.py --ckpts \
  it0=$PRE it50=$RUN/ckpt_50.pt it100=$RUN/final.pt > logs/ubias_div.log 2>&1
grep -E "it0:|it50:|it100:|moved|CONFIRMS" logs/ubias_div.log | tail -4

echo "[3/5] draw-sparsity + sigma-field (div) ..."
CUDA_VISIBLE_DEVICES="" python analysis/draw_sparsity.py --run $RUN --snap-iter 50 2>/dev/null | tail -2
CUDA_VISIBLE_DEVICES="" python analysis/sigma_field.py --run $RUN 2>/dev/null | tail -1 || true

echo "[4/5] curriculum video -> paper_results ..."
python video_curriculum_fixed.py --run $RUN \
  --out paper_results/faithful_div_it100_curriculum.mp4 --ckpt $PRE \
  --goal-xy 4.7 4.7 --start-eps 0.3 --gamma 0.5 \
  --iters 0,1,2,3,4,5,7,10,15,20,30,40,50,60,70,80,90,100 \
  --title "Safe Flow Expansion — diversity-preserving (anchor+T350+inner4, it0→100)" \
  > logs/video_div.log 2>&1 && echo "  video: $(ls -la paper_results/faithful_div_it100_curriculum.mp4 | awk '{print $5}')b"

echo "[5/5] DIV vs COLLAPSED comparison ..."
python3 -c "
import json
def pooled(d):
    import glob,os
    rs=[json.load(open(f)) for f in glob.glob(d+'/row_g*.json')]
    srs=[r['SR'] for r in rs]; g01=[json.load(open(d+'/row_g0.1.json'))] if os.path.exists(d+'/row_g0.1.json') else []
    return (sum(srs)/len(srs) if srs else 0), (g01[0]['SR'] if g01 else -1), (g01[0]['clearance_mean'] if g01 and g01[0]['clearance_mean'] else -1)
for name,d in [('COLLAPSED (no-anchor,T250,inner2)','results/p2/eval_faithful_it100'),('DIVERSITY (anchor,T350,inner4)','results/p2/eval_faithful_div_it100')]:
    sr,g01sr,g01clr=pooled(d); print('  %-38s pooled-SR %.2f | γ0.1 SR %.2f clr %.3f'%(name,sr,g01sr,g01clr))
"
touch $RUN/ASSEMBLE_DONE
echo "DONE"
