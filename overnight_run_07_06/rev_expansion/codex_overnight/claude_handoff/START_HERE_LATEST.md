# Safe Flow Expansion — latest Claude handoff (2026-07-10 22:21 PDT)

This file supersedes `NEXT_CODEX.md` for the current state. Work directly in the parent
`codex_overnight/` folder. Mizuta/Kazuki is a frozen comparison benchmark: **do not flow-expand it**.

## Current state

| Checkpoint | Role | SR by gamma (`.1,.2,.3,.4,.5,.7,1`) | CR | Clearance mean | Time mean (s) | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| `results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt` | Selected resumable production state | `92,96,96,92,92,92,96` | 0 all | `.294-.307` | `11.73-18.20` | `2-6` | Keep selected; not goal-complete |
| `results/p2/origin_tightgate_production_s671.pt` | Independent-seed origin repair | `96,100,100,100,100,92,88` | 0 all | `.292-.305` | `11.53-17.70` | `2-5` | Diagnostic branch; origin fixed, goal regressions |
| `results/p2/goal_brake_gammaaug_s766.pt` | Best exact-valid capacity result | `100,100,100,100,100,100,92` | 0 all | `.292-.306` | `11.54-17.80` | `2-5` | All 11 original failures flip, but gamma-1 seeds 5/14 regress; do not promote |
| `results/p2/goal_gamma1_brake_focus_s790.pt` | Gamma-1 focused follow-up | `96,100,100,100,100,100,92` | 0 all | `.292-.305` | `11.54-17.72` | `2-5` | Reopens gamma-.1 seed22 and does not fix gamma-1; reject |

Authoritative fixed-seed results are
`analysis/fixed_seed_gate_goal_brake_gammaaug_s766.json` and
`analysis/fixed_seed_gate_goal_gamma1_brake_focus_s790.json`. Adapter checkpoints are explicitly
non-resumable diagnostics. Production remains t104.

### New M100 coverage/reliability audit

Faithful t104 seeds 0--99 give coverage `{10,7,4,4,4,5,6}`, SR `{94,96,97,96,97,96,97}%`, and
CR `{0,1,1,0,0,0,0}%`. Thus M25 undercounted rare modes, but the current field remains far below coverage
14 and M25 also missed rare collisions. Read `analysis/coverage_iteration_diagnosis.md` before deciding to
add iterations. The current target mechanism had zero exact target hits at corrected iterations 101--105;
unchanged iterations are not a credible coverage solution.

## Origin / “ill-conditioned window” diagnosis

| Question | Measured evidence | Conclusion |
|---|---|---|
| Are accepted windows concentrated near the origin? | Radius `<1 m` is `20.2-21.3%` at t102-t105. | Moderately enriched, not dominant. |
| Are they curriculum-hard? | Near-origin sigma is `.745-.772` vs `.442-.456` away; `89-94%` become easy under the three-axis AND rule. | Yes: high uncertainty is present, but high sigma alone does not place a row in the frontier. |
| Are target controls numerically ill-conditioned? | Centered `10x2` control SVD condition median is `1.54-1.67` near vs `1.57-1.61` away. | Not supported by this proxy. Do not claim a low-rank target problem without a stronger test. |
| Why can CR be zero while SR is below 100%? | Seed 12 is a `1-2%` origin latent tail with raw `u0_y=-1.593` (deployment clamps to `-1`). After it enters `y<0`, window-OOB probability is `.83-1.00`. Near-goal contexts have a separate empty upper-boundary strip. | Rare trigger plus data-empty absorbing strip; valid local windows do not guarantee closed-loop completion for every latent fiber. |
| Did ordinary corrected updates cover the bad fiber? | The same seed-12 action is saturated at pretrained, it100, t103, and t104. Exact replay shows the true t104 batch had 13/56 near-origin rows but zero goal-strip rows in 2,540 pool rows. | Origin quantity alone was not the bottleneck; latent support and boundary-state support were. |

Primary evidence: `analysis/origin_window_failure_probe.md`, `analysis/seed12_tail_trace.md`,
`figures/seed12_trace.png`, and `analysis/runs/replay_t104_trace/batch_trace_it104.npz`.

## Pinpointed code caveats

1. `grid_expand_fixed.py:163-214`: frontier is the per-gamma intersection
   `sigma>=plane AND margin<=plane AND progress>=plane`; every other exact-valid row is easy. Thus a
   high-sigma origin row is still easy unless it also crosses the other two planes. This is a curriculum
   classification caveat, not evidence that Valid2 accepted an invalid target.
2. `grid_expand_fixed.py:783-805`: acceptance checks the gathered trajectory and each extracted window, but
   normal gathering starts at the origin and terminates when it leaves the task space. It cannot supply
   recovery contexts in `y<0` or the missing goal strip by reweighting existing rows.
3. Formerly, `analysis/one_step_from_viz.py:130-153` gamma-relabelled five braking rows without explicitly
   recertifying the destination gamma. The certificate is gamma-dependent at `grid_metrics2.py:121-123`.
   A retrospective audit found all `5x7=35` relabels certified with positive margin, so s766 remains valid;
   the script now enforces this check and records `gamma_aug_certified` for all future runs.

## Explicit assumptions and limits

- The SVD condition number is only a numerical proxy for target-action rank; it does not measure closed-loop
  reachability, latent density, or model Jacobian conditioning.
- Fixed M25 seeds `0-24` are evaluation-only. Training/replay uses independent seeds `100+`; never train on
  seed 12, 22, 5, 8, 3, or 14 merely to pass the gate.
- Gamma relabelling is allowed only after the exact destination-gamma certificate passes. Never rely on
  “same physical path” alone.
- Compact boundary adapters prove capacity but are not a resumable expansion recipe. A result is promotable
  only after zero fixed-seed regressions and independent all-gamma M25 SR100/CR0.
- M25 is a short gate, not final evidence. Final claims require the stateful 100-update unit and M>=100.
- M100 now shows rare collisions at gamma .2/.3 and 18 near-goal failures across all gamma values. Any
  promoted repair must pass an independent M100 gate before coverage training.

## Visualization-level goal before another optimizer change

Create `figures/current_goal_latent_support.png` (and optionally a short MP4) with fixed axes and four panels:

1. accepted-window start radius vs sigma, colored easy/frontier, with origin share and condition medians;
2. t104 origin `u0_y` distribution for >=4096 deterministic NFE8 base latents, seed-12 marked, and OOB tail rate;
3. the same latent-support view at representative last-in near-goal states, comparing t104/s671/s766;
4. faithful trajectories for the 11 original failures plus the two s766 regressions, t104 vs s766, annotated
   with boundary exit state and failure taxonomy.

The plot must distinguish: accepted-target conditioning, latent-tail conditioning, and empty-state support.
Do not collapse them into one “ill-conditioning” label.

## Nearest scientific task

The next controlled arm should preserve successful t104 behavior on states induced by s766 while applying
goal braking. Build immutable teacher replay using candidate=s766, teacher=t104, all seven gamma values, and
independent seeds `100-199`; keep only exact destination-gamma-certified rows. Then run one bounded s766
goal-adapter update and gate it. Do not launch long training until it yields all-gamma M25 SR100/CR0 with
zero regressions.

If it passes, integrate the mechanism into a resumable corrected trainer, run the required 100-update unit,
then pursue coverage >=14 and >P1, time <P1, clearance >P1, and final M>=100. Coverage is still the largest
unaddressed goal gap.
