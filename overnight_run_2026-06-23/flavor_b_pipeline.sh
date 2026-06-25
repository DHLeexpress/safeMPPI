#!/bin/bash
# Flavor B end-to-end: distill Mizuta -> train our CFM/drifting -> eval distilled
# proposal + certificate on held-out 100-199 -> aggregate. Uses GPUs 0,2,3 (GPU1 busy).
set -e
cd /home/dohyun/projects/cfm_mppi
export WANDB_MODE=disabled
OUT=overnight_run_2026-06-23
RUN="conda run -n cfm_mppi python -m"

echo "[1/5] distilling Mizuta over eps 0-99 (3 GPUs)..."
CUDA_VISIBLE_DEVICES=0 $RUN cfm_mppi.data.distill_mizuta_dataset --episode-start 0  --episode-end 33  --steps 80 --repeats 6 --success-only --device cuda --output-dir dataset/mzd_a > $OUT/fb_gen_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN cfm_mppi.data.distill_mizuta_dataset --episode-start 33 --episode-end 66  --steps 80 --repeats 6 --success-only --device cuda --output-dir dataset/mzd_b > $OUT/fb_gen_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $RUN cfm_mppi.data.distill_mizuta_dataset --episode-start 66 --episode-end 100 --steps 80 --repeats 6 --success-only --device cuda --output-dir dataset/mzd_c > $OUT/fb_gen_c.log 2>&1 &
wait

echo "[2/5] merging..."
$RUN cfm_mppi.data.merge_canonical --inputs dataset/mzd_a dataset/mzd_b dataset/mzd_c --output dataset/mizuta_distill_big > $OUT/fb_merge.log 2>&1

echo "[3/5] training CFM + drifting on distilled data..."
CUDA_VISIBLE_DEVICES=0 $RUN cfm_mppi.training.train_safe_cfm --train-data dataset/mizuta_distill_big/train.pt --val-data dataset/mizuta_distill_big/val.pt --output-dir output_dir/safe_cfm_mizd --epochs 250 --batch-size 128 --lr 2e-4 --device cuda > $OUT/fb_train_cfm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN cfm_mppi.training.train_drifting --train-data dataset/mizuta_distill_big/train.pt --val-data dataset/mizuta_distill_big/val.pt --output-dir output_dir/drifting_mizd --epochs 250 --batch-size 128 --lr 2e-4 --device cuda > $OUT/fb_train_drift.log 2>&1 &
wait

echo "[4/5] evaluating on held-out 100-199 (3 GPUs)..."
A="--methods mizuta_cfm_mppi cfm_proposal_mppi guided_drifting --gamma-grid 0.2 0.4 --guided-eta 0.6 --guided-extra-margin 0.25 --guided-progress-weight 9 --guided-terminal-goal-weight 200 --guided-running-goal-weight 0.4 --guided-guidance-horizon 10 --safemppi-samples 512 --safemppi-horizon 30 --device cuda --safe-cfm-checkpoint output_dir/safe_cfm_mizd/checkpoint_best.pth --drifting-checkpoint output_dir/drifting_mizd/checkpoint_best.pth --dataset ucy --dynamics doubleintegrator"
CUDA_VISIBLE_DEVICES=0 $RUN cfm_mppi.evaluation.eval_pedestrian_benchmark $A --episode-list $(seq -s ' ' 100 132) --output $OUT/fb_h0 > $OUT/fb_eval0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN cfm_mppi.evaluation.eval_pedestrian_benchmark $A --episode-list $(seq -s ' ' 133 166) --output $OUT/fb_h1 > $OUT/fb_eval1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $RUN cfm_mppi.evaluation.eval_pedestrian_benchmark $A --episode-list $(seq -s ' ' 167 199) --output $OUT/fb_h2 > $OUT/fb_eval2.log 2>&1 &
wait

echo "[5/5] aggregating..."
$RUN cfm_mppi.evaluation.aggregate_chunks --dirs $OUT/fb_h0 $OUT/fb_h1 $OUT/fb_h2 --out $OUT/FLAVORB_RESULT.txt > $OUT/fb_aggregate.log 2>&1
echo "DONE. Result in $OUT/FLAVORB_RESULT.txt"
cat $OUT/FLAVORB_RESULT.txt
