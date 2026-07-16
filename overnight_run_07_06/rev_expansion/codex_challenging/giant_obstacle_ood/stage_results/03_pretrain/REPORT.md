# Stage 3 — endpoint-free ID pretraining

## Decision

**Awaiting approval, with one disclosed gamma=1 caveat.** The selected policy is the fresh A32
endpoint-free model trained only on the approved Stage-2B ordinary-stadium demonstrations. At the fixed
ID pair `(0.5,0.5) -> (4.5,4.5)`, its faithful receding-horizon M=16 evaluation has **93.8% success,
4.5% collision, and 57/48 successful R-first/U-first trajectories**. Six of seven per-gamma gates pass.
Gamma 1.0 remains the only failure: 75% success and 25% collision, although its 12 successful samples are
exactly balanced 6/6.

No giant-obstacle data were used, and Stage 4 has not started.

## Selected recipe

- Original endpoint-free context: `low5 + E(H_P)`, 37 dimensions; no raw start/goal, GRU, or boundary
  adapter.
- Original A32 backbone: representation 32, trunk `[160, 96]`, H=10.
- Fresh initialization; fully unfrozen visual encoder.
- 8,064 optimization windows and 2,688 trajectory-held-out monitoring windows. Every gamma/signature
  stratum contributes three training trajectories and one held-out trajectory.
- Exact x/y reflection augmentation and reflection-field penalty weight 1.0. This was the lever that
  removed the earlier mode bias.
- Best held-out objective 0.96553 at epoch 478; encoder effective rank 6.28 with all 32 token dimensions
  active.
- Plain, unguided rollout sampler; temperature 0.1, NFE 12, H-exec 1, reach 0.2 m, T=300.

The lower temperature is not a safety filter or candidate selector. It narrows the learned source-noise
spread. Temperature 0.01 eliminated collisions but collapsed to one mode, while 0.1 retained both modes.

## ID rollout gate

| gamma | SR | CR | successful R/U | mean time (success) | mean min clearance (success) | gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.1 | 100.0% | 0.0% | 10 / 6 | 16.14 s | 0.080 m | pass |
| 0.2 | 100.0% | 0.0% | 8 / 8 | 15.38 s | 0.083 m | pass |
| 0.3 | 100.0% | 0.0% | 10 / 6 | 14.69 s | 0.079 m | pass |
| 0.4 | 100.0% | 0.0% | 9 / 7 | 14.93 s | 0.057 m | pass |
| 0.5 | 93.8% | 6.2% | 8 / 7 | 14.47 s | 0.061 m | pass |
| 0.7 | 87.5% | 0.0% | 6 / 8 | 13.73 s | 0.038 m | pass |
| 1.0 | 75.0% | 25.0% | 6 / 6 | 12.13 s | 0.023 m | **fail** |

The panel shows the intended diagonal prior and both geometric route families without the previous
up-only bias. The gamma=1 errors are small-margin contacts around the most aggressive paths; there are no
out-of-bounds failures or timeouts at gamma=1.

## Recipe-selection controls

The following bounded controls were evaluated and rejected rather than promoted:

- A20, including an all-data refit: representation collapse and high collision.
- A48: no validation or rollout improvement.
- Global minibatch OT coupling: invalid for effectively unique state conditions and only 1.8% rollout
  success.
- Route-bit/noise coupling: 16.1% rollout success.
- Reflection weight 5: over-regularized the CFM field and only 3.6% success.
- Final all-data A32 refit at weight 1: 90.2% success and 8.0% collision, but U-biased (15/86), so the
  held-out-selected checkpoint was retained.

## Artifacts and audit

- Selected checkpoint: `data/pretrained_id_balanced_a32.pt`
  (`sha256 a5c8280f593fbf6ef6129dbe632f740ba3282067726d8a1d5bc7039cc7aaa236`)
- Rollout panel: `viz/id_rollouts_all_gamma.png`
- Training curves: `viz/training_curves.png`
- Exact rollout tensors: `data/selected_id_rollouts_m16.npz`
- Per-gamma table: `tables/id_rollout_metrics.csv`
- Metrics and training provenance: `logs/selected_id_metrics.json`,
  `logs/selected_training_summary.json`, `logs/selected_split_audit.json`
- Independent audit: `logs/independent_audit.json` — `PASS_WITH_GAMMA1_CAVEAT`; all seven Stage-2B source
  hashes verified, held-out leakage absent, architecture and evaluator settings verified.

## Approval gate

Approval means accepting this diagonal controller, including the gamma=1 caveat, as the frozen Stage-4
pretrained baseline. If stricter gamma=1 ID performance is required, Stage 3 should remain open; Stage 4
must not start implicitly.
