# Paste this prompt into Claude

You are taking over Safe Flow Expansion in:

`/home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight`

Read, in order: `GOAL.md`, `claude_handoff/START_HERE_LATEST.md`, the last 220 lines of `PROGRESS.md`,
`analysis/origin_window_failure_probe.md`, `analysis/seed12_tail_trace.md`, and
`analysis/coverage_iteration_diagnosis.md`. Treat
`claude_handoff/START_HERE_LATEST.md` as the current source of truth; older `NEXT_CODEX.md` is historical.

Hard constraints: write only inside this folder; GPUs 2/3 only after checking availability;
`OMP_NUM_THREADS=16`; no wandb/push; faithful temp=1/NFE8/reach=.1 evaluation; do not loosen Valid2 or the
SOCP certificate; M25 seeds 0-24 are evaluation-only; append every command/result/decision to `PROGRESS.md`.
Mizuta/Kazuki is benchmark-only: do not flow-expand or modify it.

Current production selection is t104. Best diagnostic is s766: all 11 original failures flip and gamma
.1-.7 are SR100/CR0, but gamma1 is SR92 because formerly-successful seeds 5 and 14 regress near the goal.
s790 is rejected: gamma1 remains 92 and gamma-.1 seed22 reopens. Do not promote either checkpoint.

First reproduce the evidence with these exact commands:

```bash
cd /home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight
export LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib
export OMP_NUM_THREADS=16
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
python -m py_compile analysis/one_step_from_viz.py
python analysis/origin_window_failure_probe.py \
  --viz mode2_it102=results/p2/corrected_mode2_target50_s81/viz_db/it102.pt \
  --viz mode2_it103=results/p2/corrected_mode2_target50_s81_to106/viz_db/it103.pt \
  --viz mode2_it104=results/p2/corrected_mode2_target50_s81_from103_to105/viz_db/it104.pt \
  --viz mode2_it105=results/p2/corrected_mode2_target50_s81_from103_to105/viz_db/it105.pt \
  --eval-dir rollback_it100=results/p2/eval_finalunit_s15_it100 \
  --eval-dir mode2_it103=results/p2/eval_corrected_mode2_it103_m25 \
  --eval-dir mode2_it104=results/p2/eval_corrected_mode2_it104_m25 \
  --out analysis/origin_window_failure_probe.claude2.json \
  --markdown analysis/origin_window_failure_probe.claude2.md
CUDA_VISIBLE_DEVICES=3 python analysis/seed12_tail_trace.py \
  --device cuda --n-latents 512 \
  --out analysis/seed12_tail_trace.claude2 \
  --fig figures/seed12_trace_claude2.png
```

Before training, implement the visualization-level diagnosis described in
`claude_handoff/START_HERE_LATEST.md` as `analysis/latent_support_map.py` and
`figures/current_goal_latent_support.png`. Use >=4096 deterministic latents, true NFE8 integration, fixed
contexts from the faithful traces, and compare t104/s671/s766. Report accepted-window numerical condition,
sigma/curriculum classification, latent OOB tail probability, and empty-strip absorption separately.

Then build independent-seed, all-gamma on-policy teacher replay. Run one gamma at a time on a free GPU:

```bash
mkdir -p analysis/goal_replay/s766_t104
for g in 0.1 0.2 0.3 0.4 0.5 0.7 1.0; do
  CUDA_VISIBLE_DEVICES=3 python analysis/build_onpolicy_teacher_replay.py \
    --candidate results/p2/goal_brake_gammaaug_s766.pt \
    --teacher results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt \
    --gamma "$g" --seed0 100 --M 100 --device cuda \
    --region 4.0 5.0 4.0 5.12 \
    --out "analysis/goal_replay/s766_t104/g${g}.pt"
done
python analysis/build_escape_replay.py \
  --inputs analysis/goal_replay/s766_t104/g0.1.pt \
           analysis/goal_replay/s766_t104/g0.2.pt \
           analysis/goal_replay/s766_t104/g0.3.pt \
           analysis/goal_replay/s766_t104/g0.4.pt \
           analysis/goal_replay/s766_t104/g0.5.pt \
           analysis/goal_replay/s766_t104/g0.7.pt \
           analysis/goal_replay/s766_t104/g1.0.pt \
  --out analysis/goal_replay/onpolicy_s766_teacher_t104_allg_seed100_199.pt
```

Audit that replay rows retain metadata, use only seeds 100-199, and that every training target is certified
at its actual gamma. If the existing builder does not enforce destination-gamma certification, add that
assertion before training.

Run exactly one bounded diagnostic update from s766 with immutable t104 preservation:

```bash
CUDA_VISIBLE_DEVICES=3 python analysis/one_step_from_viz.py \
  --ckpt results/p2/goal_brake_gammaaug_s766.pt \
  --guard-teacher-ckpt results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt \
  --viz-db results/p2/hardtail_guard_b256_t104_s86/viz_db/it105.pt \
  --escape-replay analysis/goal_replay/onpolicy_s766_teacher_t104_allg_seed100_199.pt \
  --escape-quota 256 --escape-eta 1.0 \
  --out results/p2/goal_gamma1_preserve_t104_s791.pt --seed 791 \
  --batch 256 --hard-quota 48 --guard-quota 0 \
  --boundary-adapter --adapter-hidden 64 --adapter-side goal \
  --hard-side goal-brake --hard-gamma-augment --hard-focus-gamma 1.0 \
  --hard-x0-cand 128 --hard-x0-select random-oob --hard-x0-allow-majority \
  --endpoint-eta 1.0 --cfm-eta 0 --steps 10 --lr 5e-5
bash run_gate.sh results/p2/goal_gamma1_preserve_t104_s791.pt goal_gamma1_preserve_t104_s791 3
```

Promotion ladder: first require every gamma M25 SR100 and CR0, all 11 original failures flipped, and zero
regressions versus t104. Then require an independent M100 SR100/CR0 audit: t104's new M100 audit exposed
CR1% at gamma .2/.3 and 18 near-goal failures, so M25 is not sufficient. If this arm fails, diagnose its
latent/state support using the new figure; do not stack blind updates. Keep t104 selected until both gates
pass.

After the short gate passes, integrate the mechanism into a resumable corrected trainer and complete the
actual GOAL: one fixed-recipe 100-update unit, then final M>=100 per gamma with SR100/CR0, clearance>P1,
time<P1, coverage>=14 and >P1, `T2_expanded`, `T_ALL`, final video, and audit. Every status report must show
SR, CR, clearance, time, and coverage.
