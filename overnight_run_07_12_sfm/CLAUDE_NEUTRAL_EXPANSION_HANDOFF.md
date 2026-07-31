# Claude handoff: Hp10 neutral Safe Flow Expansion

## Start here

Claude must begin by running this command on Helios and showing its complete
stdout before editing or launching anything:

```bash
cd /home/dohyun/projects/<CLAUDE_PRIVATE_WORKTREE>
/home/dohyun/miniforge3/envs/cfm_mppi/bin/python \
  overnight_run_07_12_sfm/show_claude_neutral_handoff.py
```

The first line is the authenticated **Neutral continuation MP4**:

```text
/data3/research1/sfm_neutral_claude_handoff_assets/neutral_continuation.mp4
```

The command only reads and authenticates existing artifacts. It does not run an
evaluation. The cached reference delivery is
`/data3/research1/sfm_neutral_gamma_temp_0e441d6/DELIVERY_COMPLETE.json`.

## Immutable shared inputs

- Pretrained checkpoint:
  `/home/dohyun/projects/sfm_hp10_b1_runs/103476d/pretrained_hp10.pt`
- Checkpoint SHA-256:
  `1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215`
- ID demonstration dataset:
  `/home/dohyun/projects/cfm_mppi/overnight_run_07_12_sfm/dataset_id_v01`
- Pretraining report:
  `/home/dohyun/projects/sfm_hp10_b1_runs/103476d/pretrain/pretraining_report.json`
- Existing neutral baseline run, for read-only comparison:
  `/data3/research1/sfm_neutral_multiround_round50_a36dfe7/lr1em5_s01`
- New Claude outputs must be under a new root such as
  `/data3/research1/claude_sfm_neutral_<frozen_sha7>/`. Never reuse or modify a
  shared result root.

## ID SafeMPPI pretraining contract

The demonstration distribution is 20 pedestrians with desired speeds sampled
in 0.5--1.0 m/s. Demonstrations are successful-only trajectories at all seven
gamma values `(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)`.

`sfm_b1_expert.py` defines the faithful data expert: horizon 10, 2048 MPPI
rollouts, MPPI temperature 0.1, velocity-aware K=16 nominal polytope,
`centroid_gain=.2`, `centroid_smooth=.25`, `centroid_eps=.15`,
`predict_gain=.25`, and `smooth_weight=.12`. Temperature 0.1 belongs only to
the demonstration expert; it is not silently used by raw learned-policy
evaluation.

`stage3_pretrain_sfm.py` constructs the ten-step Hp history, splits complete
trajectories 90/10, and trains from scratch with objective mass
`gamma -> successful trajectory -> window`. Its defaults are 120 epochs,
batch 256, AdamW lr `3e-4`, weight decay `1e-4`, five warmup epochs, cosine
decay, and seed `20260720`. Checkpoint promotion uses ID validation followed by
a fixed trajectory-disjoint ID raw temperature-1 gate; no OOD result selects
the pretrained checkpoint.

## Canonical neutral expansion recipe

The entry point is `sfm_b1_neutral_multiround.py`. The handoff baseline is the
explicit `lr1em5_s01` recipe, not the script's historical smoke defaults:

- OOD collector: 40 pedestrians, 1.0--2.0 m/s.
- One macro-round: two new scenario seeds x all seven gamma values, synchronously.
- `T=180`, `H=10`, `K=16`, `B=4`, NFE 8, generation temperature 1.
- Penultimate noised representation at `s=.9`; each stored plan retains its
  original Gaussian base `x0`.
- RBF-GP: previous round's executed full-H positives only, gamma-balanced cap
  512, `ell=.24210826720721101`, lambda `1e-2`, adaptive ESS target `.5`.
  The GP is fixed within the macro-round and rebuilt once at the next round.
- Exact moving-pedestrian full-H verifier labels every queried/repaired plan.
- If ordinary queried candidates are admissible, execute the full-H positive
  passing the nominal-Hp gate with maximum one-step margin.
- If guided repair is triggered and all four guided candidates are exact
  negatives, execute one by the same selector as an offline continuation. It
  retains audit truth `y=0` and is stored only in `D0`; it never enters `D+` or
  the GP. Collision, success, timeout, and the three-step trap stop remain.
- Update order is whole `D+` and then isolated whole `D0`. Every record is used
  exactly once per inner pass. Each population contributes one Adam step per
  pass. Canonical baseline: batch 128, lr `1e-5`, one inner pass, alpha 0,
  frozen visual encoder, no expert/prox/anchor/curriculum/rollback.
- Every round records fixed-context/fixed-latent before/after probes. Same-lineage
  M2 is a fit diagnostic, never a scientific evaluation.

Claude may add private experimental arms, but must keep a control with these
exact semantics and isolate each algorithmic change. Do not relabel `D0` as a
safety positive, leak it into the GP, or change the locked pretrained/Kazuki
baselines.

## Evaluation and target

The four primary metrics are collision rate (lower), window-level Validity
(higher), successful minimum clearance (higher), and successful time-to-goal
(lower, while retaining liveness). Also report SR and timeout.

Desired gamma family:

- lower gamma: no higher CR, higher successful clearance, and longer time;
- higher gamma: higher Validity and faster progress.

Trend eligibility uses adjacent-pair tolerances `rate=.1`, `clearance=.02 m`,
and `time=1 s`; each family must satisfy at least 75% of adjacent pairs.

Canonical raw evaluation is temperature 1. If temperature tuning is explored,
it must be named explicitly: select a per-gamma schedule on a separate
calibration bank, freeze it before confirmation, and run a fresh disjoint bank.
Never choose temperature after reading confirmation. Report temperature-1 raw
results alongside a tuned deployment result when making a scientific claim.

The current authenticated fresh disjoint M50 references (350 trajectories per
method, `ep0=485000`) are printed by the start command. They are approximately:

| method | SR | CR | timeout | Validity | clearance [m] | time [s] |
|---|---:|---:|---:|---:|---:|---:|
| pretrained | .660 | .337 | .003 | .607 | .130 | 8.53 |
| locked Kazuki | .840 | .157 | .003 | .325 | .184 | 4.32 |

These are cached deployment results, not values to recompute during handoff.

## Files that define the mechanism

- `sfm_b1_expert.py`: faithful ID SafeMPPI demonstration expert.
- `stage3_pretrain_sfm.py`: from-scratch Hp10 pretraining and ID-only promotion.
- `grid_policy_sfm.py`, `sfm_hp_history.py`: policy and ten-grid history.
- `sfm_b1_neutral_multiround.py`: canonical macro-round and two-population update.
- `sfm_b1_kazuki_repair_audit.py`: ordinary/guided collection and `D+`/`D0` storage.
- `sfm_b1_rbf.py`, `sfm_b1_store.py`: RBF uncertainty and replay accounting.
- `sfm_metrics2.py`: exact full-H moving-pedestrian certificate implementation.
- `sfm_b1_offline_eval.py`: fixed raw evaluation and four-metric plot contract.
- `run_sfm_neutral_gamma_temperature.py`: separated calibration/confirmation.

Blind spots to keep visible: same-lineage probes overstate generalization;
temperature schedules can overfit a bank; `D0` is behaviorally useful but
verifier-negative; exact certification and closed-loop collision avoidance are
not the same objective; and longer training has not shown monotonic improvement.
