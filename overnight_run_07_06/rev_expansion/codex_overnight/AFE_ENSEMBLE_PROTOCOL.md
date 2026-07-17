# AFE Deep-Ensemble Protocol

This is one AFE arm. It changes the uncertainty estimator only; the scene,
pretrained policy, deterministic verifier, expert-free execution, positive-only
CFM replay, and evaluation protocol are shared with the validated RBF run.

## Two memories

- Uncertainty data \(\mathcal D_n^u\): every successful full-verifier query,
  including \(y=0\) and \(y=1\). SOCP errors enter neither dataset.
- Flow data \(\mathcal D_n^+\): the cumulative subset with full-window
  verifier label \(y=1\).

The ensemble cannot be trained on positives only: that would give it a constant
target and remove the validity boundary it is meant to learn.
If an ordinary round nevertheless observes only one class, the implementation
matches the reference code and fits the resulting constant standardized target;
the recorded class count exposes that degeneracy rather than rebalancing it.

## Reference-faithful estimator

At the end of each round, after the CFM update:

1. Re-embed all of \(\mathcal D_n^u\) using the updated current representation
   \(\phi_s^{(n)}\).
2. Standardize the binary verifier labels globally.
3. Reinitialize five independent MLPs, each `32 -> 100 -> 100 -> 1`, with
   ReLU and 10% dropout.
4. Give each member an independent 90% random subsample without replacement.
5. Fit by full-batch MSE with Adam at \(10^{-3}\), for at most 1000 steps with
   the public implementation's 30-step early-stop rule.
6. Freeze the ensemble during the next parallel gathering round and use the
   population standard deviation of its five raw predictions as \(\sigma\).

This follows the public AFE molecule/protein implementation. The paper calls
the 90% datasets “bootstrap” samples, while the public code uses `randperm`, so
this reproduction follows the code's without-replacement behavior.

## Acquisition

Round 1 is a fully logged uniform \(B=8\)-without-replacement bootstrap because
\(\mathcal D_0^u\) is empty. It is part of the ordinary expansion budget, not a
hidden archive or curriculum. After the round-1 ensemble fit, new unverified
candidate pools at the stored bootstrap contexts calibrate one fixed \(\beta\)
to normalized ESS 0.375. No additional verifier calls are used.
Calibration uses beta-neutral random removal orders, whereas actual gathering
uses Gibbs-biased removal. Thus 0.375 is a round-1 calibration target, not a
guarantee on realized later-stage ESS. Realized ESS is logged every round.
Because verifier-label class balance can change and labels are re-standardized,
the fixed score scale may drift; \(\beta\) is intentionally not retuned.

From round 2 onward, each \(K=64\) pool is scored once and queried according to

\[
\pi_j \propto \exp\!\left(
\frac{\sigma_{\mathrm{ens}}(U_j,c)-\max_k\sigma_{\mathrm{ens}}(U_k,c)}{\beta}
\right)
\]

without replacement. A frozen ensemble has no GP posterior conditioning after
an unlabeled candidate is selected. Only that index is removed; no diversity
penalty or synthetic covariance update is added.

## Fixed control and learning choices

- Seven safety levels per round, synchronous replicas, \(K=64\), \(B=8\).
- The deterministic full verifier runs before execution.
- Execute the first action of the verified plan with maximum progress.
- If no queried plan is execution-admissible, terminate with NVP.
- No expert, fallback, curriculum, anchor replay, proximal loss, or progress
  training label.
- CFM uses uniform cumulative \(\mathcal D_n^+\) replay, batch 128, learning
  rate \(10^{-4}\), and 250 steps per round.

## Complexity and scope

For \(N=|\mathcal D_n^u|\), the exact GP's cubic factorization and quadratic
kernel storage disappear. Dataset storage and re-embedding remain \(O(N)\), and
the reference full-batch ensemble refit is linear in \(N\) per optimizer step.
Candidate-time uncertainty is independent of \(N\).
If the cumulative query count grows linearly over \(R\) rounds, repeated
full-buffer refits have \(O(R^2)\) aggregate data-processing cost. The arm
removes cubic GP factorization, not all dependence on a cumulative archive.

The ensemble standard deviation is an empirical epistemic score. This arm does
not inherit the exact GP posterior, confidence interval, or information-gain
guarantees from the kernel theory.

Each post-fit ensemble is saved as `ensemble_roundN.pt`; the round-`N+1`
visualization records that exact acquisition checkpoint. The initial unfit
round-0 state is saved as well. This first launcher is a five-round mechanism
pilot (report and expansion video), not the separate M=100 bare-policy true
evaluation pipeline.

The RBF and ensemble pilots necessarily differ in their cold start: RBF uses a
pretrained positive GP seed and can tilt round 1; the ensemble needs labels and
therefore uses its ordinary round-1 queries as an all-label bootstrap. This is
an estimator-specific initialization, not evidence from a perfectly paired
round-1 comparison.
