# Canonical 100-round AFE estimator study

This study has one AFE training arm and changes only the uncertainty estimator:

- `rbf`: exact RBF-GP acquisition. Round 1 uses exactly 512 verifier-positive
  pretrained calibration samples. Later rounds use at most 512 full-window
  positives from the immediately preceding round, selected gamma-balanced and
  without replacement. The GP is rebuilt in the current representation and is
  frozen during the next round.
- `ensemble`: five 32-100-100-1 MLP verifier-label regressors. Round 1 queries
  uniformly. Later rounds use ensemble disagreement. All successful verifier
  queries, positive and negative, are cumulative, re-embedded in the current
  representation, and the ensemble is refit from scratch after each round.

Both estimators share the same control and learning protocol: seven gamma
values, two synchronous closed-loop replicas per gamma, `T=300`, `K=64`,
`B=8`, deterministic full verifier before execution, maximum-progress verified
plan execution, and NVP termination without fallback. Positive full-window
queries alone enter the cumulative `D+` flow replay. Each round performs 250
Adam CFM steps with batch 128 and learning rate `1e-4`. There is no proximal
term, curriculum, anchor replay, or expert fallback.

For two replicas, the hard per-round upper bounds are 14 episodes, 268,800
generated plans, and 33,600 verifier queries. Reaching the goal, NVP, or timeout
ends an episode, so realized counts are logged and normally smaller.

Run the two estimators sequentially on one GPU:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<physical-index> \
PYTHON=/home/dohyun/miniforge3/envs/cfm_mppi/bin/python \
./run_afe_estimators_100.sh \
  <codex_radius1_v1-or-codex_radius03_v1> \
  <promoted-checkpoint> <checkpoint-sha256> <new-output-root> 2 2 16
```

Each estimator writes its own authenticated report, delivery manifest, and
video. Trainer artifacts remain available at every round. Videos render rounds
1 through 10, then 20, 30, ..., 100, for exactly 19 frames.
