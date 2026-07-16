# P2 run-data forensics: collapse, recovery, and missing exploration

Date: 2026-07-10 PDT

This is a read-only analysis of the P2 artifacts. It covers all 36 runs with a
`probe.jsonl`, all available histories, all 87 non-viz P2 checkpoint headers,
and every one of the 164 per-iteration viz databases in the complete seed-15
and seed-16 final units. No production code was changed. Reproduce the CSVs
with:

```bash
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 \
  python analysis/run_data_forensics.py
```

Generated evidence:

- `analysis/forensics_run_summary.csv`
- `analysis/forensics_finalunit_viz.csv`
- `analysis/forensics_checkpoint_headers.csv`
- `analysis/forensics_field_checks.csv`

## Decision summary

1. **Keep `finalunit_q50_k14_s15_from_it18/safe_best.pt` at iteration 100 as
   the incumbent, but not as a final model.** It is the best all-gamma-safe
   M=25 checkpoint and recovered to aggregate SR=.903/CR=0. Its independent
   M=100 evaluation is only SR 91--96%, CR 0--2%, coverage 4--9, and it is
   slower and generally less clear than the expert. It therefore does not
   satisfy GOAL.md.

2. **Reject seed 16 and both 100-to-200 resumes.** Seed 16 ends at aggregate
   M=25 SR=.531/CR=.086. The seed-17 live resume fell from the incumbent
   gamma=.5 SR50=.94 to .60, .22, and .06 in its first three updates; seed 18
   reached only .68, .62, .60. Both jobs were no longer running at the final
   inspection, and neither produced a periodic checkpoint beyond iteration
   100.

3. **The main instability is receding-horizon first-action inconsistency, not
   origin dither or exploding gradients.** The verifier and progress rule score
   the full 10-control plan, while deployment executes only `U[0]` and replans.
   A plan can compensate later for a biased first action, but those compensating
   controls are never executed. The resulting repeated first-action bias sends
   failed policies through the top boundary near the goal.

4. **The current AND frontier structurally misses the samples that could add
   staircase coverage.** Near-origin/branch-decision windows are genuinely
   high uncertainty, but their progress is below the global median. Only about
   1.7% of near-origin windows become frontier; frontier windows occur around
   in-trajectory index 72. More frontier weight therefore emphasizes late
   transit/goal behavior, not new initial branches.

5. **Attempt rotation does not provide condition balance.** Gamma=.1 receives
   roughly balanced attempts but only 2--3% of training windows, versus 14.3%
   under a balanced target. The batch is also dominated by one staircase mode.
   `rid_n` looks healthy because it measures rollout IDs, not gamma or homotopy
   diversity.

## 1. What exactly collapses

The paired final units start from the exact same iteration-18 checkpoint and
use the same recipe. Their M=25 all-gamma histories are:

| abs. iter | seed 15 SR / CR | seed 16 SR / CR |
|---:|---:|---:|
| 18 | .863 / .011 | .863 / .011 |
| 20 | .834 / .029 | .823 / .011 |
| 30 | .343 / .006 | .263 / .011 |
| 40 | .594 / .006 | .000 / .074 |
| 50 | .634 / .023 | .314 / .086 |
| 60 | .543 / .017 | .829 / .034 |
| 70 | .503 / .011 | .663 / .029 |
| 80 | .766 / .006 | .206 / .177 |
| 90 | .863 / .000 | .514 / .080 |
| 100 | **.903 / .000** | .531 / .086 |

This is oscillatory collapse and recovery, not monotonic overfitting. Gathered
rollouts remain selected-safe throughout: online gather SR is approximately
.98--1.00 and CR is zero even when faithful SR is zero. That statistic is
selection-biased because gather draws 64 candidates with uncertainty tilt and
a cheap safety filter; it does not measure the faithful policy distribution.

Failure replay at gamma=.5 shows the physical failure:

| checkpoint | faithful SR / CR / OOB (M=50) | dominant endpoint behavior |
|---|---:|---|
| seed15 it30 | .52 / .04 / .44 | overshoots top boundary near goal |
| seed15 it100 | .96 / .02 / .02 | reaches the goal |
| seed16 it40 | .00 / .18 / .82 | 38/50 paths end above y=5; typical endpoint about (4.85, 5.14) |
| seed16 it80 | .32 / .26 / .42 | top overshoot plus earlier collisions |

The near-origin diagnostic does **not** exhibit the warned-about dither
signature in these final units: `near0_e` averages .23--.24, and only about 1%
of easy windows have `widx<2`. This differs from early failed arms, where
`near0_e` commonly reached .5--.8.

## 2. True flow-field diagnosis

The policy is Cond-OT flow matching over a 20-dimensional, 10-control sequence:

`x_tau = (1-tau)x0 + tau*x1`, with target velocity `v* = x1 - x0`.

The CFM loss weights every temporal control dimension equally. It does not
reflect that receding-horizon deployment causally executes only the first two
dimensions before conditioning changes.

I froze 64 verified late-goal contexts from the shared iteration-19 database,
used 128 identical noise draws per context, and integrated each checkpoint's
actual velocity field with the deployment's 8 Euler steps. The decisive metric
is the generated first-action directional bias `a0_x-a0_y`:

| checkpoint | mean first `a_x-a_y` | y-dominant samples | full-plan `sum(a_x)-sum(a_y)` | all-gamma SR / CR |
|---|---:|---:|---:|---:|
| seed15 it20 | +.157 | 40.6% | +4.070 | .834 / .029 |
| seed15 it30 | -.098 | 55.5% | +2.013 | .343 / .006 |
| seed15 it80 | +.094 | 44.0% | +2.551 | .766 / .006 |
| seed15 it100 | **+.154** | 40.2% | +3.539 | **.903 / .000** |
| seed16 it30 | -.110 | 55.8% | +3.323 | .263 / .011 |
| seed16 it40 | **-.331** | **68.0%** | **+3.015** | **.000 / .074** |
| seed16 it60 | +.105 | 43.0% | +3.373 | .829 / .034 |
| seed16 it80 | -.249 | 64.4% | +3.785 | .206 / .177 |
| seed16 it100 | -.083 | 54.2% | +4.062 | .531 / .086 |

Across all 18 periodic checkpoints, late-context first-action bias correlates
with all-gamma SR at Pearson **r=.948** (Spearman .934), and with gamma=.5 SR50
at Pearson **r=.906**. The full plan remains strongly x-compensating even in
the failed checkpoints. That sign mismatch is direct evidence of a plan that
looks valid over the horizon but repeatedly executes the wrong early action.

The common aggregate diagnostics miss it:

- Fixed verified-window CFM MSE improves slightly even for collapsed seed 16
  (start .739; seed16 it90 .716). A lower offline CFM error is not a closed-loop
  guarantee.
- Mean field gradient RMS is about .012 and encoder gradient RMS about .017--.019;
  neither explodes nor separates the seeds.
- By iteration 100, both seeds have moved only about 0.97--0.98% in relative
  field-parameter norm and about 0.65% in encoder norm from iteration 18.
  Seed15 and seed16 field weights differ by only .66%. Small field changes cross
  a closed-loop behavioral bifurcation.

Therefore checkpoint gating needs a causal `U[0]` diagnostic and faithful
rollouts; loss, gradient RMS, and selected online SR are insufficient.

## 3. Why the frontier does not yield coverage

Across all 164 final-unit viz databases after the cold first iteration:

| statistic | seed 15 | seed 16 |
|---|---:|---:|
| mean actual frontier fraction | 11.1% | 12.3% |
| mean sigma near origin / elsewhere | .678 / .506 | .667 / .507 |
| near-origin windows passing high-sigma plane | 77.3% | 74.8% |
| near-origin windows passing high-progress plane | 11.8% | 9.3% |
| near-origin windows passing all three planes | **1.55%** | **1.80%** |
| near-origin fraction among frontier | 3.0% | 3.2% |
| mean frontier `widx` | 72.2 | 71.8 |
| sigma--progress correlation | -.437 | -.403 |

Thus the uncertainty signal does find the OOD origin, but the global progress
quantile vetoes it. With the 42-easy/14-frontier fresh batch, only roughly ten
origin windows enter through the easy pool and fewer than one through the
frontier pool. Moving the mix toward 50:50 reduces rather than increases causal
branch-decision training.

The tilted gather does visit many nominal IDs (union 16 in seed 15 and 22 in
seed 16), but its frequency is highly concentrated:

| statistic | seed 15 | seed 16 |
|---|---:|---:|
| valid gathered paths with an ID | 1,137 | 1,124 |
| canonical `RURURURURU` count | 826 (72.6%) | 674 (60.0%) |
| median per-iteration dominant-mode share | 71.4% | 64.3% |
| mean distinct IDs per iteration | 3.62 | 4.07 |

Sampling 13--14 rollout IDs per update does not correct this homotopy imbalance.
Union coverage also overstates the effective distribution: rare IDs seen once
do not survive as faithful modes.

## 4. Gamma balance failure

The absolute-iteration rotation balances **attempts**, but validity acceptance
is strongly condition-dependent:

| final unit | gamma=.1 attempts | valid gamma=.1 rollouts | gamma=.1 windows | window share | `gamma_ready` iterations |
|---|---:|---:|---:|---:|---:|
| seed 15, it19--100 | 192 | 29 | 4,737 | **3.31%** | 27/82 |
| seed 16, it19--100 | 210 | 17 | 2,735 | **2.01%** | 16/82 |

Other gammas generally contribute 15--18% each. The low-gamma condition is
therefore trained five to eight times less than intended. The trainer exits
once its total valid-rollout quota and both labels are present; it does not
require a valid rollout for every gamma. Per-rollout window counts and random
window batching introduce another imbalance because long trajectories
contribute more targets.

This explains why gamma conditioning trends and gamma=.1 coverage cannot be
expected to converge reliably from rotation alone.

## 5. SOCP margin and target-safety blind spot

The revised margin is the mean fitted-polytope level-set slack, but the lower
half remains dominated by numerical ties:

- 51--63% (seed 15) and 51--65% (seed 16) of windows have margin <= 1e-12.
- **100% of selected frontier windows have margin <= 1e-12.** The safety axis
  is therefore primarily a zero/tie test, not a robust ordering within the
  tight half.
- About 2.2--2.5% of all stored plans and 2.8--3.0% of frontier plans have a
  negative planned-window certificate (`margin=-5` for failed certification).

This is possible because the gate certifies the **executed receding-horizon
path**, while each stored target is the full planned `U` at a replanning step.
Only `U[0]` contributes to the executed path. The stored full target is checked
for task-space and progress, but not required to have a nonnegative planned
window certificate. Over 82 updates, the CFM learner repeatedly sees a small
but nonzero set of uncertified full-sequence targets.

This does not by itself explain the large SR oscillation, but it contradicts the
strong interpretation that every learned 10-action sequence was independently
SOCP-verified and contributes to residual CR.

## 6. Resume discontinuity at iteration 100

A saved checkpoint contains model weights/configuration/iteration only. A new
run reconstructs Adam and resets the query buffer, GP factorization, RNG stream,
and cumulative coverage. It is therefore a weight warm start, not a stateful
training resume.

At absolute iteration 101 the current recipe also crosses `early_until=100`, so
it changes from one update step per iteration to two. Its first gather has no GP
buffer, hence sigma is identically one. All windows pass the sigma plane and the
frontier becomes margin AND progress:

| resume | it101 frontier | it101 gamma=.5 SR50/CR | it102 | it103 |
|---|---:|---:|---:|---:|
| seed 17 | 997/3251 = 30.7% | .60/.00 | .22/.04 | .06/.02 |
| seed 18 | 997/3251 = 30.7% | .68/.02 | .62/.00 | .60/.00 |

This combines a cold uncertainty estimator, a 2.8x frontier-fraction jump, lost
optimizer state, and doubled gradient exposure at the exact boundary. Continuing
these arms would not test the intended stationary q=.5 recipe cleanly.

## 7. Constructive recommendations

### Immediate checkpoint policy

1. Preserve seed15 iteration 100 as the rollback point.
2. Do not promote any seed16, seed17, or seed18 artifact.
3. Gate every candidate with all-gamma faithful evaluation (M>=100), plus the
   late-context first-action/full-plan consistency diagnostic. Do not select on
   gamma=.5 SR50 or online gather SR alone.
4. Treat the current seed15 M=100 table as an intermediate result, not T2 final.

### Minimum changes needed for a meaningful continuation

1. Make a resume stateful: serialize/restore Adam, q-buffer (or GP buffer), RNG,
   and coverage. If that is not implemented, prime the GP with gather-only
   iterations before any gradient and avoid changing inner-step count on the
   first resumed update.
2. Keep one gradient step per iteration until the causal action diagnostic is
   stable; the current two-step post-100 phase doubles the failure rate before
   any measurement gate can intervene.
3. Require a valid per-gamma quota over a bounded window, or use a bounded
   gamma-balanced replay/batch. Gamma=.1 needs substantially more attempts;
   a total-rollout quota cannot solve this.
4. Balance by staircase/first-crossing mode, not only rollout ID. Report both
   union coverage and mode entropy/dominant share.
5. Preserve the three-axis AND rule, but prevent its high-progress plane from
   eliminating all origin branch decisions. Within the GOAL's allowed sweep,
   keep a high easy share and test a lower early quantile with a fixed absolute
   schedule. Verify directly that near-origin frontier usage rises before a
   long run.
6. Add a causal temporal-consistency guard: either weight the first action in
   the flow objective, constrain first-action direction against the verified
   plan displacement, or train a shifted/receding consistency target. Full-plan
   validity alone is not preserved under repeated `U[0]` execution.
7. At minimum, exclude planned windows whose own SOCP certificate is negative.
   This is stricter than, and does not weaken, the existing Valid2 trajectory
   gate; whether it is permitted under the locked experiment must be stated.

### Go/no-go telemetry for the next short arm

Before spending another 100-iteration unit, require all of:

- no cold-GP gradient step;
- all seven gammas represented in the bounded training buffer;
- gamma=.1 window share materially above the current 2--3%;
- near-origin representation measured separately in easy and frontier batches;
- mode dominant share below the current 60--73%;
- late-context `mean(a0_x-a0_y)` nonnegative and stable under several fixed
  noise panels;
- faithful all-gamma SR/CR measured every short checkpoint, with collapse
  stopping active before iteration 600.

These checks target the observed mechanisms. More raw iterations under the
unchanged resume state will primarily reproduce the oscillation and mode
collapse rather than bridge the remaining GOAL.md gap.
