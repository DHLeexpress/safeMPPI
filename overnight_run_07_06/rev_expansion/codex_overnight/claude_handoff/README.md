# Claude handoff — Safe Flow Expansion

> **Latest entry point (2026-07-10 22:21 PDT):** read `START_HERE_LATEST.md`, then paste
> `CLAUDE_EXACT_PROMPT_CURRENT.md` into Claude. The state below is retained as historical context.

# Historical handoff — 2026-07-10 13:35 PDT

Claude can work directly in the parent folder. This directory is only the handoff packet; all authoritative
code, checkpoints, tables, and diagnostics remain one level above it.

## Current result in one table

| Item | Authoritative artifact | Result / status |
|---|---|---|
| P1 SafeMPPI | `../tables/T1_expert.{md,csv}` | M100, every γ SR100%, CR0; complete |
| P3 Mizuta/Kazuki | `../tables/T3_kazuki.{md,csv}` | Untouched pretrained, M200, every γ SR100%, CR0; complete and **never flow-expand** |
| P2 rollback initialization | `../results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt` | Pre-correction it100; M100 SR 91–96%, not final evidence |
| P2 best corrected resumable state | `../results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt` | M25 SR `{92,96,96,92,92,92,96}%`, CR0; aggregate 93.7%; coverage `{6,4,3,3,3,2,3}` |
| P2 rejected next state | `../results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_105.pt` | M5 fell from 97.1% to 94.3%; retain only for diagnosis |
| Origin/failure diagnosis | `../analysis/origin_window_failure_probe.{md,json}` | Seed 12 exits below the origin for every γ; separate near-goal overshoot failures also exist |
| 2×4 curriculum video | `../video/p2_corrected_mode2_it104_105_curriculum.mp4` | Latest accepted-block internals and traces |

The corrected pipeline itself is not the open question: `../analysis/corrected_trainer_regression.md` and
`../analysis/test_corrected_trainer.gpu2.modequota.json` pass all 14 semantic/resume gates.

## What the new origin diagnosis says

- Accepted near-origin windows are stable at 20.2–21.3% of the pool, have higher σ (`.745–.772` vs
  `.442–.456`), and 89–94% are labelled easy because frontier is a three-way AND cell.
- Their median centered 10×2 control condition is `1.54–1.67`, essentially the same as away-from-origin
  `1.57–1.61`. Thus “numerically low-rank target windows” is not supported by this proxy.
- The behavioral boundary-tail hypothesis **is** supported: faithful latent seed 12 goes below y=−.12 at the
  origin for all seven γ at rollback it100, corrected it103, and corrected it104.
- This bad seed predates corrected fine-tuning. Continuing ordinary guarded updates has not removed it.
- Four additional it104/M25 failures overshoot the upper boundary near the goal. Treat this as a second
  mechanism; do not average it together with the origin failure.

## Non-negotiable constraints

- Work only in the parent `codex_overnight/` folder; copy locally before changing shared behavior.
- GPU 2/3 only; `OMP_NUM_THREADS=16`; no wandb, push, or destructive git commands.
- Preserve strict reach `.1`, unchanged `grid_metrics2.traj_valid2`, exact target certificates, faithful
  evaluation (`temp=1`, NFE=8, no inference safety/clip filter), full-state resume, and rollback gates.
- Do not loosen the verifier or relabel unsafe/nonprogress windows to manufacture SR.
- Mizuta/Kazuki is a frozen benchmark. Do not flow-expand or retune it.
- Append every command/result/decision to `PROGRESS.md`; never rewrite its old entries.

## Immediate work sequence

1. Reproduce and localize the faithful seed-12 origin tail before training anything. Trace the NFE=8 ODE
   sample at the origin and compare seeds `{12}` vs successful seeds, per γ, for it100/it104.
2. Add diagnostic telemetry for the actual 64 samples chosen by a training update: selected source index,
   origin radius, γ, mode, label, target first action, σ/progress/margin. Existing `viz_db` contains the pool,
   not the exact chosen indices.
3. Test whether the failing latent is underrepresented by ordinary CFM `x0` draws. Separate data-location
   imbalance (origin windows) from base-noise-tail coverage (rare latent mapping).
4. Make one controlled local remediation arm. A legitimate candidate is hard-tail CFM importance sampling:
   retain only exact-valid positive origin target windows, oversample base-noise states whose current faithful
   map points out of bounds, and keep the standard CFM target. Log it as a controlled optimizer/data-sampling
   change. Do not clip generated actions at inference.
5. Simultaneously guard the near-goal overshoot tail using exact-valid late-goal targets; report the two strata
   separately. Do not allow origin repair to regress late-goal first-action direction or cumulative field drift.
6. Gate every candidate on the fixed seeds first, then independent M25. Required short gate: every γ SR≥95%,
   CR0, increasing coverage, decreasing time. Only after that run the corrected stateful 100-update unit.
7. Final claim requires M≥100 for every γ: SR100%, CR0, clearance>P1, time<P1, coverage≥14 and >P1, followed
   by `T2_expanded`, `T_ALL`, final 2×4 video/figures, and `audit_p2_goals.py`.

## Useful diagnosis tips

- `sr_cr_eval.eval_policy` resets `torch.manual_seed(seed0+i)` for every rollout, so seed 12 is exactly
  reproducible. At it104 its endpoint is roughly `(0.15, −0.14)` after 10–11 controls for all γ.
- In `grid_rollout.fm_deploy`, faithful deployment uses `policy.sample_window(..., n=1, nfe=8)` and executes
  only `U[0]`. Inspect the inherited `FlowPolicy.sample` integration states; do not diagnose from the unused
  proposal tail alone.
- Compare the same latent/context through it100 → it102 → it103 → it104. The persisted failure shows ordinary
  mean CFM improvement is not sufficient to cover this base-noise tail.
- A passing M5 can hide the defect because seed 12 is absent. Use explicit fixed-seed probes plus M25.
- The current t104 trust telemetry is safe: per-step drift 1.06%, cumulative anchor drift .97%. T105 remains
  under the 1.6% bound (1.24%) yet performs worse, showing the drift bound is necessary but not sufficient.
- Coverage counts in the training history are cumulative accepted modes, not faithful deployed coverage.
