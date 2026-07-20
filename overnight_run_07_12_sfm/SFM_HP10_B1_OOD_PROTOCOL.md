# SFM Hp10 + B1: corrected ID/OOD protocol

## What changed after `103476d`

The original result files did not serialize their crowd environment.  A source
audit established that the `103476d` expansion, M20 screening, and M100
confirmation all used 20 pedestrians at 1.0--1.5 m/s.  They were therefore a
velocity-OOD test relative to the training data (20 pedestrians at 0.5--1.0
m/s), not the requested ID test.

Every new command is fail-closed on an explicit scene profile:

| profile | pedestrians | speed [m/s] | interpretation |
|---|---:|---:|---|
| `training` | 20 | 0.5--1.0 | demonstration distribution |
| `id` | 10 | 0.5--1.0 | requested easier benchmark; lower density than training |
| `legacy_velocity_ood` | 20 | 1.0--1.5 | exact `103476d` environment |
| `requested_ood` | 30 | 1.0--1.5 | density-plus-velocity OOD |

This distinction is stored in every new metrics JSON.

## Model and expansion semantics

The Hp10 policy uses the ten most recent nominal-polytope (H_P) grids and a
frozen visual encoder.  At each alive context, B1:

1. draws (K=16) windows from the raw conditional flow;
2. uses an RBF-GP posterior uncertainty to query (B=4) windows;
3. stores every resolved query in (D), and every full-horizon positive in
   (D^+);
4. executes only a queried plan satisfying the full-H verifier and
   (H_P(x_{t+1})\ge(1-\gamma)H_P(x_t));
5. terminates that replica on no verified positive (NVP);
6. replays the most recent (W=2) rounds with equal mass over
   gamma -> (round, episode) -> context -> positive query.

Arm A ranks admissible queries by maximum one-step nominal margin.  Arms B--D
rank the same admissible queried sequences by the frozen SafeMPPI proposal cost:

\[
j^\star=\arg\min_{j:\,y_j=1,\,
H_P(x_{t+1}^j)\ge(1-\gamma)H_P(x_t)} J_{\rm SafeMPPI}(U^j).
\]

Thus the cost never admits an unverified plan and never generates an expert
proposal.  B, C, and D differ only in negative-gradient coefficient
\(\alpha\in\{0,10^{-3},10^{-2}\}\).

The authenticated preflight uses exactly 50 gamma/scenario-balanced pretrained
embeddings:

- \(\ell_0=0.48421653441442203\)
- selected multiplier (0.5), \(\ell=0.24210826720721101\)
- cap 256, \(\lambda=0.01\)
- calibrated \(\beta=0.11756989408559083\), normalized ESS 0.5
- uplift 0.06455287337303162, condition 540.5854, effective rank 30.6369

These are preflight values, not a guarantee that sequential realized ESS remains
exactly 0.5 throughout a rollout.

## Honest selection of the existing checkpoint

Arm A round 10 remains the selected checkpoint because it won the declared,
frozen M20 screening rule.  D looked slightly better only on the later disjoint
M100 confirmation.  Replacing A by D after reading confirmation would leak the
confirmation bank.  D can be reported as a post-hoc diagnostic, not relabeled as
the selected method.

## Evaluation contract

Raw evaluation means temperature one, NFE 8, one generated window per context,
and execution of its first action.  It uses no RBF uncertainty, verifier,
SafeMPPI selector, or fallback.  Kazuki is separately named because it adds goal
and safety guidance plus MPPI refinement.

`sfm_b1_deploy_driver.py` performs:

- matched M100/gamma deployment of Hp10 r0, selected A-r10 raw, and default
  Kazuki on `id` and `requested_ood`;
- raw M50/gamma evaluation of every A checkpoint, rounds 0--20, on fixed banks;
- separate ID/OOD gallery PNG/MP4 with radius-0.2 pedestrian disks;
- a diagnostic-only paired OOD rollout for the margin and SafeMPPI-cost
  selectors, followed by two fixed-frame 3x3 certificate figures.

The query figure colors are fixed: gray (K), orange queried (B), green
SOCP-positive, red rejected, and thick blue executed first action.  Blue nominal
and green verifier polytopes contain exactly the ten gamma-dependent horizon
sets.  At \(\gamma=1\), the ten sets genuinely coincide and are labeled as such.

The paired query diagnostic never enters (D,D^+), the GP, or model training.
It chooses encounter snapshots by a declared minimum-distance rule over both
selectors, rather than visual curation.

## Known blind spots

- A fitted H=10 verifier proves the queried window, not infinite-horizon
  viability under moving pedestrians.
- The constant-velocity pedestrian prediction is a modeling assumption.
- The requested `id` profile has lower density than the demonstration data and
  is therefore an easier matched-speed benchmark, not literally the full
  training distribution.
- The RBF embedding can still become insensitive to behaviorally different
  windows; uncertainty uplift and realized ESS must be monitored.
- One gallery episode is explanatory evidence only.  Claims must use fixed-bank
  M100 metrics and confidence intervals.
- `goal_coef=safe_coef=0` does not make Kazuki identical to raw flow because its
  MPPI refinement and warm start remain active.

## Entry points

- `sfm_scene.py`: scene and profile contracts
- `sfm_hp_history.py`: Hp10 construction
- `grid_policy_sfm.py`: flow policy and frozen visual encoder
- `sfm_metrics2.py`: full-H moving-pedestrian verifier
- `sfm_b1_rbf.py`: RBF-GP uncertainty and adaptive ESS calibration
- `sfm_b1_store.py`: round-sharded (D,D^+,D^-) and exact hierarchical replay
- `sfm_b1_cost.py`: one-step (H_P) gate and execution selectors
- `sfm_b1_expand.py`: B1 gathering and update
- `sfm_b1_eval.py`: raw evaluation
- `sfm_b1_benchmark.py`: matched deployment and checkpoint curves
- `sfm_b1_query_diagnostic.py`: paired closed-loop selector diagnostics
- `sfm_b1_viz.py`: gallery, query paths, and ten-level polytopes
- `sfm_b1_deploy_driver.py`: authenticated two-GPU delivery driver
