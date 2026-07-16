# Copy/paste prompt for Claude

You are taking over the Safe Flow Expansion experiment directly in:

`/home/dohyun/projects/cfm_mppi/overnight_run_07_06/rev_expansion/codex_overnight`

Complete `GOAL.md`, but first diagnose and correct the faithful generative tails that cause CR=0 with SR<1.
Read, in order: `GOAL.md`; the last 260 lines of `PROGRESS.md`; `claude_handoff/README.md`;
`analysis/origin_window_failure_probe.md`; `analysis/corrected_trainer_regression.md`; and
`results/p2/corrected_mode2_target50_s81_from103_to105/recipe.json`.

The selected corrected resumable checkpoint is
`results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt`. Its M25 result is SR
`{.92,.96,.96,.92,.92,.92,.96}` for γ `{.1,.2,.3,.4,.5,.7,1}`, CR0 all, aggregate 93.7%, coverage
`{6,4,3,3,3,2,3}`. Do not select it105: its M5 gate regressed. P1 and P3 are complete. Mizuta/Kazuki is an
untouched-pretrained benchmark and must never be flow-expanded.

Start read-only. Reproduce `analysis/origin_window_failure_probe.py`, then create a focused sampler trace for
faithful seed 12 at the origin versus successful seeds, NFE=8, across all γ and checkpoints it100/it104. The
current evidence is specific: accepted origin windows are high-σ but not more low-rank by a 10×2 control SVD;
seed 12 exits below the origin boundary for all γ and already failed before corrected tuning; four other M25
failures overshoot near the goal. Keep these as separate failure strata.

Instrument the exact selected training batch indices because the current viz database stores only the pool.
Determine whether the defect is origin-data weighting, rare base-noise-tail coverage, or both. Then implement
one minimal controlled local repair, preferably a training-only hard-tail CFM sampling arm using exact-valid
origin and late-goal targets. Never add inference clipping/safety filtering, loosen Valid2, alter reach=.1,
or train on unexecuted/uncertified proposal tails. Preserve full-state resume and the 2.5% per-step / 1.6%
cumulative fixed-origin rollback gates.

Use GPUs 2 and 3 only and `OMP_NUM_THREADS=16`; work only in this folder; no wandb or git push. Append every
command and decision to `PROGRESS.md`. Gate changes on fixed failure seeds, then independent M25. Do not begin
the 100-update unit until every γ has M25 SR≥95%, CR0, with coverage and time improving. Final certification is
M≥100 per γ with SR100%, CR0, clearance>P1, time<P1, coverage≥14 and >P1. Then generate `T2_expanded`, `T_ALL`,
the final 2×4 curriculum video, figures, and run `audit_p2_goals.py`.

Run the exact bootstrap commands in `claude_handoff/NEXT_COMMANDS.md` first.

