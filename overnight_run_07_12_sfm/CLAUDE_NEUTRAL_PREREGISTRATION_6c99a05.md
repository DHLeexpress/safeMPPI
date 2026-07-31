# Pre-registration: Claude neutral-expansion arms (frozen base 6c99a05)

Author: Claude (agent/claude-neutral-expansion-20260731). Output root:
`/data3/research1/claude_sfm_neutral_6c99a05/`. Nothing outside that root is
written; all shared datasets, checkpoints, locked baselines, and existing
`/data3/research1` results are read-only.

## Motivation (from cached, read-only evidence)

The canonical `lr1em5_s01` control (reference run
`sfm_neutral_multiround_round50_a36dfe7/lr1em5_s01`, M20 disjoint sweep
ep0=270000) monotonically trades Validity (+.081 over 50 rounds) against
liveness: time-to-goal +1.90 s, clearance −.019 m, SR .707→.664. Its
fresh-M50 confirmation (`sfm_neutral_gamma_temp_0e441d6`) shows the selected
round-2 checkpoint statistically indistinguishable from pretrained
(all four paired CIs straddle zero, `objective_achieved=false`).
Mechanistic reading: the update imitates max-Hp-margin certified windows
(no progress preference) plus D0 neutral continuations (certified-unsafe
teacher actions, ~20–40% of D+ mass, `D0_actual_repairs=0` every round).

## Arms (each one isolated change vs the exact canonical control)

Common: pretrained checkpoint sha 1b5179c9…b215, `--rounds 50`,
`--scenario-ep0 260000`, `--eval-ep0 270000 --eval-M 20
--eval-rounds 0,1,2,5,10,20,30,40,50 --noise-seed 20260733` (identical M20
bank to the reference control), `--lr 1e-5 --inner-steps 1
--probe-per-gamma 4 --device cuda --workers 40`, seeds
sample/audit/train/probe = 700000/20260730/20260731/20260732. Single-shot
50 rounds (matches reference milestones).

1. `control` — exact canonical recipe, unchanged semantics. GPU 0.
2. `pgm` — `--selector progress_gated_margin` (existing frozen selector,
   StudyConfig-allowed; H10 displacement/progress gate, max-margin fallback,
   symmetric in D+ selection and D0 ranking). Hypothesis: imitating
   progress-making certified windows lowers time-to-goal and holds SR.
   GPU 0.
3. `noD0` — `--no-neutral-replay` (existing flag; D0 still collected,
   audited, stored — only its replay pass is skipped). Hypothesis: removing
   imitation of verifier-negative windows lowers CR and clearance loss.
   GPU 2.
4. `anchor` — `--id-anchor-windows 448` (new args-only flag, default 0=off;
   γ-balanced 64/γ ID pretraining windows per round, trajectory→window mass
   within γ, one extra Adam step after the D0 pass, dataset manifest
   sha-verified, encoder-frozen assertion, distinct seed streams). D+/D0
   collection, stores, GP, probes untouched; D0 never relabeled or leaked.
   Hypothesis: anchoring to the ID demonstration distribution arrests the
   liveness/clearance drift while D+ still adds Validity. GPU 2.

## Bank ledger (all disjoint; both ep0 and noise_seed always advanced)

- Training scenarios: 260000–260099 (canonical, per arm identical).
- Screening M20: ep0 270000, noise seed 20260733 (canonical in-run bank,
  same as reference — enables direct comparison to the reference M20 rows).
- Confirmation M50 (temp-1, scientific): ep0 810000, noise seed 20260802.
- Calibration M50 (only if a temperature stage is later run): ep0 800000,
  noise seed 20260801.
- Tuned-schedule confirmation (only if a schedule is frozen on the
  calibration bank): ep0 820000, noise seed 20260803.
- ep0 ≥ 800000 is unclaimed anywhere in the repo (max declared 620000;
  700000 appears only as MPPI sample_seed). Cached reference bank
  485000/20260741 is never reused for any selection or confirmation.

## Selection rule (frozen before any training result is seen)

From each arm's in-run disjoint M20 sweep (rounds 0,1,2,5,10,20,30,40,50;
r0 = pretrained on the same bank/process):

- Eligibility of (arm, round>0): SR ≥ SR_r0 − .05 AND timeout ≤
  timeout_r0 + .05.
- Score: number of strict wins vs the r0 row on the four primary metrics
  (CR lower, Validity higher, successful clearance higher, successful
  time-to-goal lower). Rank: wins desc, then ΔCR asc, then Δtime asc.
- Select the top (arm, round) overall → "winner"; also record best round of
  `control` by the same rule as the like-for-like comparator.

## Confirmation protocol (canonical raw, temperature 1)

On ep0=810000/seed=20260802, M50, `double_density_velocity_ood`, temp 1.0,
NFE 8, single GPU, sequential jobs, per-cell cache under the confirmation
root: pretrained r0, control best round, winner, each via the frozen
`sfm_b1_offline_eval.py` (never edited); locked Kazuki on the same bank via
`run_sfm_neutral_temperature_m50.py kazuki`. Report pooled + per-γ rows,
paired scenario-cluster deltas vs pretrained, the four-family γ-trend test
(tolerances rate .1 / clearance .02 m / time 1 s, ≥75% adjacent pairs per
family), and the liveness contract (SR ≥ min reference SR − .05, timeout ≤
max reference + .05). The cached handoff table (ep0 485000) is reported
alongside with the explicit caveat that its pretrained/expanded rows are
per-γ temperature-tuned, not temp-1.

Optional temperature stage (only if the winner is close but not CI-clean at
temp-1): select per-γ schedule on the calibration bank only, freeze
(schedule sha), confirm on ep0 820000; temp-1 result reported alongside.
Never select a temperature after reading a confirmation bank.

## Honest-reporting commitments

- Same-lineage M2 and the paired trigger probes are fit diagnostics, never
  evaluations. M20 is a screening metric, never final confirmation.
- Successful-only clearance/time are conditioned on different success sets
  when SR differs; SR is always reported next to them.
- D0 keeps audit truth y=0 everywhere; any arm found leaking D0 into D+ or
  the GP invalidates that arm, not the label.
- Negative results (no arm beats pretrained CI-clean) are reported as such.
