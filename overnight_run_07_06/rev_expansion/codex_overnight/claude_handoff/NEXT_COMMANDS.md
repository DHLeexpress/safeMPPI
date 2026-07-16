# Exact bootstrap commands

Run from the parent experiment folder:

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
export OMP_NUM_THREADS=16
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 260 PROGRESS.md
```

Reproduce the semantic regression and origin/failure report:

```bash
CUDA_VISIBLE_DEVICES=2 python analysis/test_corrected_trainer.py --json analysis/test_corrected_trainer.claude.json
python analysis/origin_window_failure_probe.py \
  --viz mode2_it102=results/p2/corrected_mode2_target50_s81/viz_db/it102.pt \
  --viz mode2_it103=results/p2/corrected_mode2_target50_s81_to106/viz_db/it103.pt \
  --viz mode2_it104=results/p2/corrected_mode2_target50_s81_from103_to105/viz_db/it104.pt \
  --viz mode2_it105=results/p2/corrected_mode2_target50_s81_from103_to105/viz_db/it105.pt \
  --eval-dir rollback_it100=results/p2/eval_finalunit_s15_it100 \
  --eval-dir mode2_it103=results/p2/eval_corrected_mode2_it103_m25 \
  --eval-dir mode2_it104=results/p2/eval_corrected_mode2_it104_m25 \
  --out analysis/origin_window_failure_probe.claude.json \
  --markdown analysis/origin_window_failure_probe.claude.md
```

Inspect the selected result and recipe without rerunning evaluation:

```bash
sed -n '1,80p' tables/_T2_corrected_mode2_it104_m25.md
python -m json.tool results/p2/corrected_mode2_target50_s81_from103_to105/recipe.json
python -m json.tool analysis/origin_window_failure_probe.json | less
```

Rebuild the latest 2×4 visualization if needed:

```bash
python video_curriculum_fixed.py \
  --run results/p2/corrected_mode2_target50_s81_from103_to105 \
  --out video/p2_corrected_mode2_it104_105_curriculum.mp4 \
  --ckpt ../../results/hp_repr/pretrained_a32uni.pt \
  --title 'Safe Flow Expansion — exact-valid 2 modes/gamma, targeted 50%, guarded updates' \
  --iters 0,104,105
```

Do **not** blindly resume unchanged from t104: that exact next update already exists as t105 and regressed.
Make the sampler/batch diagnosis first. If a controlled arm is implemented, give it a new output directory,
include the change in the resume signature/recipe, use `--drop-train-state` only when the recipe signature truly
requires a deliberate model-only branch, and document why exact optimizer continuation was impossible.

