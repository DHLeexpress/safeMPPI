# Curriculum-video + internals regeneration (for codex)

## The curriculum video (panel A scene/rollouts+σ, B σ-hist, C 3D margin×progress×σ with planes, D batch bins; real-time β/count/mix/lr traces)

Source runs must be launched with `--viz-db-every 1or2 --log-comp-every 1` (all unit generations already are).
To span MULTIPLE ratchet generations in one video, merge snapshots + probe lines first (symlinks are fine):

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
mkdir -p results/p2/unit_merged_viz/viz_db && cd results/p2/unit_merged_viz/viz_db
for f in ../../unit_s792_esc64_s801/viz_db/*.pt ../../unit_ratchet_gen*/viz_db/*.pt; do ln -sf "$f" .; done
cd ../../../..
cat results/p2/unit_s792_esc64_s801/probe.jsonl results/p2/unit_ratchet_gen*/probe.jsonl \
  > results/p2/unit_merged_viz/probe.jsonl
CUDA_VISIBLE_DEVICES=<free> python video_curriculum_fixed.py \
  --run results/p2/unit_merged_viz \
  --out video/unit_ratchet_curriculum.mp4 \
  --ckpt ../../results/hp_repr/pretrained_a32uni.pt \
  --title 'Safe Flow Expansion — guarded ratcheted unit' \
  --iters 0,136,140,144,150,160,170,180,190,200
```

`--iters 0` renders the pretrained intro frame; then pick the cadence the user asked for
(1,5,10,20,30,…,100 RELATIVE to the unit start = absolute 136,140,145,155,165,…,235 once enough
generations exist — regenerate the merge + rerun, same command). Fonts were enlarged in
`video_curriculum_fixed.py` rcParams (font.size 15 / titles 19); keep them.

## The internals report figure (Image-1 style)

`python analysis/report_internals_v3.py` → `figures/internals_v3_unit.png`. It auto-discovers every
`results/p2/unit_ratchet_gen*` directory, so after more generations just rerun it — panels: loss, field/enc
grad-RMS, trust telemetry vs bounds (ratchet points visible as anchor saw-teeth), batch portions, M5-SR/SR50,
CR, coverage+strip probes. Also: `analysis/unit_internals_plot.py` (unit + bounded-arm loss curves + gate
bars) and `analysis/repair_lineage_internals.py` (all gated experiments chronological).
