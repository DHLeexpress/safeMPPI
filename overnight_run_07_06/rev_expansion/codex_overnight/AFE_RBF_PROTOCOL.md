# Single-arm AFE-RBF protocol

## Object and safety semantics

One query object is a context-plan pair `(c_t, U_t)` where `U_t` contains ten
predicted actions.  The deterministic verifier checks the whole predicted
window before execution.  A full-window positive enters cumulative `D+`; a
certified goal-reaching prefix may be executed but does not relabel its suffix.
Only the first action is executed.  If no queried plan is execution-admissible,
the episode terminates with `NO_VERIFIED_POSITIVE`.  There is no expert,
fallback, curriculum, proximal term, or progress condition in the safety label.

## What one round means

One round is not one rollout.  It is a synchronous batch of

```
7 gamma values x R closed-loop replicas per gamma
```

where each replica runs until goal, NVP, collision/OOB (which certified
execution should prevent), or `T=300`.  At every active control step the model
produces `K=64` plans and acquisition sends `B=8` plans to the full verifier.
Thus the hard query-object ceiling is `7 * R * 300 * 8`; actual counts are lower
because episodes terminate.  All active contexts share one GPU proposal call at
each synchronous control tick.  Full-verifier jobs run in a persistent spawned
CPU process pool.

## Acquisition memory versus learning memory

The two memories are intentionally different.

* Acquisition: exact RBF-GP on at most 512 full-window positives from the
  immediately preceding round.  Selection is gamma-balanced, random without
  replacement, and deterministic from the run seed.  The selected plans are
  re-embedded using the current `phi_s` before the GP is fitted.  The GP is then
  frozen for the whole round, making parallel replicas independent of arbitrary
  completion order.  Proposal and acquisition RNG streams are keyed by
  `(purpose, gamma, replica, control step)`; controller-evaluation streams do
  not depend on the training round.  Acquisition draws `B=8` plans
  sequentially.  At draw `b`,
  each pending candidate is scored by its RBF posterior variance conditioned on
  the GP buffer and only the first `b-1` selected locations.  A Schur-complement
  update suppresses a selected plan's near duplicates; the other unqueried
  members of `K` are never treated as observations.  This is the `B<K`
  budget-consistent adaptation of the public peptide implementation's
  batch-conditional covariance.
* Learning: uniform replay with replacement from every full-window positive in
  cumulative `D+`.  Each round takes 250 CFM gradient steps, batch 128, Adam
  learning rate `1e-4`, with all encoder/trunk/head parameters trainable.

The GP posterior standard deviation is novelty relative to its acquisition
buffer.  It is not a probability of validity and it does not certify safety.

## RBF calibration

Before expansion, sample exactly 50 plans from the pretrained model across the
seven initial gamma contexts.  L2-normalize their `phi_s` embeddings and set the
RBF length scale to their mean off-diagonal pairwise distance.  The temperature
calibration is separate and pretrained-only: uniform `B`-budget calibration
rollouts provide exactly 512 gamma-balanced full-window positives for an
operational-size GP.  An independent uniform-acquisition rollout archive
provides the context distribution.  At every such context, a beta-neutral
random pending order produces the eight sequential score vectors of lengths
64 through 57.  Solve once for beta using the predeclared median normalized
stage-normalized `ESS/M_remaining=0.375` target over those vectors.  The first
archive supplies the round-1 GP, but neither archive enters `D+`, training,
audit, or reported controller evaluation.  Every round-0 verifier query,
positive, SOCP solve, and wall-clock component is recorded as calibration
budget.  Hold length scale and beta fixed throughout expansion; there is no
per-round recalibration.  Realized first-step `ESS/K` and median sequential
`ESS/M_remaining` are reported separately.

## Scope and assumptions

The RBF and positive-only choices follow the task-specific therapeutic-peptide
implementation described in the AFE appendix; they are not requirements of the
main linear-kernel AFE formulation.  The previous-round cap and frozen
within-round GP are control-specific computational assumptions.  The
gamma-balanced random cap is stratified compression, not an unbiased estimate
of the pooled trajectory distribution and not a generalization guarantee.  Exact RBF
inference is cubic in the cap, which is why the acquisition archive is not the
full cumulative `D+` archive.  The operational-size calibration is required
because a temperature fitted to a 39-point start-state GP does not preserve the
declared acquisition scale under a 512-point rollout-context GP; it is fixed
before adaptation rather than tuned from pilot outcomes.

## First pilot

The first preflight uses the promoted Codex checkpoint and scene
`codex_radius04_v1`: start `(0.5,0.5)`, goal `(4.5,4.5)`, and exactly the sixteen
interior obstacle radii changed from 0.2 to 0.4.  Boundary walls and eight plugs
remain radius 0.2.  Run 5 rounds, 2 replicas/gamma/round, `M_eval=2`, and 16
verifier workers.  This is a runtime/behavior preflight, not paper evidence.
The separate pilot true evaluation uses `M=20` and labels its finite-sample
intervals accordingly.  A longer run is authorized only after the preflight
passes provenance, runtime, non-flat acquisition, and artifact checks.
