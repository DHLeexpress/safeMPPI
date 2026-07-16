# Safe Flow Expansion vector-field diagnosis

Date: 2026-07-10 PDT  
Scope: `GridHPFlowPolicy`, CFM training/sampling, P2 gather/label/update/resume, and selected P2 checkpoints.
This audit is read-only with respect to production code. Mizuta/Kazuki is not involved and must remain
benchmark-only.

## Bottom line

The base CFM equation is conventional and internally consistent: controls are normalized by `u_max`,
Cond-OT interpolation uses `x_tau=(1-tau)x0+tau*x1`, and deployment integrates the learned velocity field.
The dominant failures are around the field, not in that equation itself:

1. The CFM targets are not the objects actually certified by Valid2. The executed receding-horizon path is
   certified, but the policy is trained on all ten actions of each proposal even though only action zero was
   executed. Some stored targets are exactly verifier-infeasible. The gather also silently stops at the old
   0.45-m reach radius and does not enforce executed-trajectory progress.
2. The purported SOCP-margin frontier axis is numerically degenerate for a fitted certificate. Nearly every
   frontier point has certificate slack at the active constraint, approximately zero; membership is mostly
   floating-point tie-breaking, plus a small set of truly infeasible windows.
3. Checkpoint resume is not a training-state resume. Adam, the GP query buffer, RNG state, and the LwF teacher
   are recreated. The first resumed update therefore sees sigma=1 for every point and immediately damages an
   otherwise strong origin policy. A direct one-step A/B proves this is causal.
4. Diagnostics overwrite the global Torch RNG. Per-iteration probes force the next gather to reuse the same
   latent stream, making nominally different seeds share byte-identical gathers and suppressing exploration.
5. Global quantiles and unstratified batching severely starve gamma=0.1. The learned field is consequently
   losing gamma sensitivity while its origin control distribution loses rank.

Continuing the current recipe without correcting these mechanisms is unlikely to produce simultaneous
SR=100%, CR=0%, expert-beating speed/safety, and coverage near 16. It is self-training a narrowing field on
misaligned targets, not faithfully expanding toward verified safe-progress control sequences.

## Authoritative evidence

### 1. The stored H=10 CFM target is not what the verifier certifies

The rollout samples an H=10 control window but executes only `U[0]` (`grid_rollout.py:121-160`). The gather
then checks `GM.socp_ok(out["path"])` on the sequence of executed first actions
(`grid_expand_fixed.py:410-419`). It stores every full proposal `U` after only task-space and progress checks
(`grid_expand_fixed.py:427-446`). The per-window cheap label placed in `rec[4]` is not used, and that label
does not contain SOCP anyway (`grid_metrics2.py:85-93`).

Across all 82 snapshots / 143,122 accepted windows of the successful seed-15 unit:

- 2.24% of all stored H=10 targets have clipped exact certificate margin `-5`, meaning the raw verifier
  result was infeasible (`-inf`).
- 2.80% of frontier targets are hard-infeasible.
- At gamma=0.1 the hard-infeasible rate is **9.92%**; at gamma=0.2 it is **4.94%**.

Thus a collision-free/certified executed path does not establish that the velocity field is being fit toward
certified endpoint control sequences.

### 2. The gather is using the wrong terminal radius and only half of actual Valid2

`GR.fm_deploy` defaults to `reach=GM.REACH`, and `GM.REACH` is 0.45
(`grid_rollout.py:103-104`, `grid_metrics.py:25`). `_gather_fresh` does not pass `cfg.reach`, even though the
trainer's final evaluator uses `cfg.reach=0.1`.

For all 1,148 accepted executed trajectories in seed 15:

- 0.0% end within 0.1 m of the goal.
- 99.04% reach the legacy 0.45-m disk.
- Only 52.44% pass the actual sliding-window executed-path approach criterion in
  `GM2.traj_valid2`; seed 16 is 53.57%.

The trainer checks executed-path SOCP, but planned-window progress. Under replanning, ten planned progressing
actions do not imply that the sequence of first actions progresses. Therefore the operational gate is not
the unchanged Valid2 definition stated in the goal. The DR pretraining trajectories also stop at 0.4 m, so
neither demo replay nor current expansion directly supervises the strict final approach.

### 3. The fitted-polytope minimum slack is not a usable ranking margin

The variable verifier fits faces to contain the candidate trajectory. At least one constraint is normally
active, so `min_t(H_P-alpha_t)` is zero up to numerical precision. In seed 15:

- Every per-iteration q=0.5 margin plane has absolute value below `9.8e-16`; the median is `3.6e-16`.
- 53.02% of all accepted windows have `|margin| <= 1e-8`.
- **97.20% of frontier windows** have `|margin| <= 1e-8`; another 2.80% are hard-infeasible.
- The realized frontier fraction averages 11.1%, but ranges from 8.6% to 32.0%. Cold starts select 32%,
  not the intended approximately 12.5%, because sigma is tied at one.

The same result appears in seed 16 (97.00% zero-tie frontier, 3.00% hard-infeasible). The current low-margin
plane therefore does not order valid trajectories by meaningful certificate robustness.

### 4. Resume discontinuity is causal, not merely correlated

`HP.save_hp` saves only model weights/config plus scalar metadata (`grid_hp_expt.py:83-87`). On every run,
the trainer creates a new Adam and empty `GPUncertainty` (`grid_expand_fixed.py:565-566`), `qbuf=None`
(`:603`), and a new teacher copied from the resumed student (`:569-573`). Optimizer moments, GP memory,
query buffer, training RNG, and the original fixed teacher are absent.

Direct same-checkpoint/seed A/B from the iteration-100 incumbent:

| Resume treatment | First trained iteration | sigma plane | frontier pool | gamma=.5 SR50 / CR50 |
|---|---:|---:|---:|---:|
| Empty GP, update immediately | 101 | 1.000 | 30.7% | 0.64 / 0.02 |
| One gather-only GP prime, then update | 102 | 0.484 | 10.1% | **0.92 / 0.00** |
| Incumbent before either branch | 100 | n/a | n/a | 0.94 / 0.00 |

Artifacts: `analysis/runs/resume_no_prime_s17/` and `analysis/runs/resume_prime1_s17/`.

The unprimed update changes total parameters by only 0.091%, yet changes the origin field by 2.87% and drops
the deterministic gamma=.5 SR50 by 30 percentage points. The issue is the first cold selection/update, not
a large or obviously anomalous gradient.

### 5. Evaluation probes mutate the training random stream

`sr_cr_eval.eval_policy` calls `torch.manual_seed(seed0+i)` inside every rollout (`sr_cr_eval.py:43-45`).
The baseline measure therefore erases the CLI training seed before iteration one. With `probe_cov=1`, every
iteration's probe leaves the global RNG at seed 49, so the next gather begins from the same Torch latent
stream. NumPy minibatch selection remains seed-dependent, which is why two branches diverge only after the
shared gather.

Concrete reproduction:

- `finalunit...s15/viz_db/it19.pt` and `finalunit...s16/viz_db/it19.pt` have the identical SHA-256
  `5056633146cd1200e71529aa851515df05bcf13169001af6edc030b95c71b883`.
- `resume100_to200_s17/viz_db/it101.pt` and the seed-18 equivalent are also byte-identical, SHA-256
  `fec698c5e74a9930e02013993f88d054dafa6af88b8c96c95a9150f5273eda8c`.

This both invalidates intended seed independence and repeatedly queries a narrow family of latent candidates,
directly opposing coverage expansion.

### 6. Gamma=0.1 is starved by global planes and aggregate sampling

`gamma_ready` is computed but not part of the gather stopping condition (`grid_expand_fixed.py:400-458`).
Planes are global across gamma, and class draws are not gamma-stratified. Aggregated seed-15 snapshots show:

| gamma | share of all accepted windows | share of frontier windows |
|---:|---:|---:|
| 0.1 | **3.31%** | **1.06%** |
| 0.2 | 17.82% | 12.03% |
| 0.3 | 15.80% | 18.86% |
| 0.4 | 15.34% | 20.41% |
| 0.5 | 14.91% | 19.02% |
| 0.7 | 15.87% | 14.88% |
| 1.0 | 16.96% | 13.73% |

Balanced conditioning would be 14.29% per gamma. The conservative condition is approximately 13.4 times
underrepresented in the frontier. This is especially damaging because gamma=0.1 also has the largest rate of
uncertified planned tails.

### 7. The true learned vector field is narrowing and forgetting

`analysis/vector_field_probe.py` evaluates fixed latent draws and fixed balanced demo contexts, eliminating
Monte-Carlo differences. Key changes from the untouched pretrained model to seed-15 iteration 100:

| Diagnostic | Pretrained | Seed-15 it100 | Direction |
|---|---:|---:|---|
| balanced-demo fixed CFM MSE | 1.005 | 1.085 | 7.9% worse |
| balanced-demo velocity cosine | 0.586 | 0.543 | worse |
| tau=.75 CFM MSE | 1.267 | 1.390 | 9.7% worse |
| tau=.75 velocity cosine | 0.453 | 0.365 | substantially worse |
| origin field norm, gamma=.5 | 4.93 | 6.03 | 22% larger |
| origin local x-Lipschitz mean, gamma=.5 | 1.07 | 1.30 | 21% larger |
| origin sampled-control effective rank, gamma=.5 | 11.77 | **7.42** | mode narrowing |
| `||v(g=1)-v(g=.1)||` | 0.893 | **0.735** | 18% less conditioning |

Only 1.22% relative trunk-parameter drift produces 29.1% mean origin-field drift because origin is outside the
demo start distribution (minimum demo-start norm is 1.26 m and `|x-y|>=0.5`). The two seed-15/seed-16
iteration-100 models differ by only 0.58% in parameters and 5.9% in their origin fields, yet their paired M=25
aggregate outcomes are approximately 0.903/0 versus 0.53/0.09 SR/CR.

The update is doing its local objective: on iteration-100 expansion targets, fixed CFM MSE improves from
0.742 under the base model to 0.644 under seed-15 iteration 100. At the same time, balanced-demo loss worsens.
This is classic self-training distribution shift: the field fits its selected self-samples while losing the
broader conditional field and reducing deployed rank.

The LwF term cannot adequately stop this:

- Its teacher is reset to the current checkpoint at each unit, so its reference ratchets.
- It is evaluated only on off-diagonal demo contexts, not the OOD origin that determines the reported metric.
- At the first optimizer step student and teacher are identical, so the LwF gradient is exactly zero.
- Only the encoder is gradient-clipped; trunk/head field parameters are not.

### 8. Sigma tilt is weak and not aligned with the joint event

`analysis/uncertainty_joint_probe.py` reconstructs a representative 384-point qbuf from the last ten accepted
snapshots and uses the production phi_s/RBF/ell=0.2/beta=0.3 score at the origin. Across gamma:

- Weight ESS is 91.6%-93.8% of 128 candidates: selection is close to uniform.
- Sigma-weighted probability of exact-certificate-plus-progress is essentially unchanged from uniform. For
  gamma=.1 it decreases from 0.609 to 0.593; for gamma=.5 it is 0.587 versus 0.587.
- Sigma/progress Pearson correlation ranges from -0.08 to +0.15.

The phi representation also clusters as training proceeds. At gamma=.5, mean cosine between random normalized
phi candidates rises from 0.890 to 0.926 and feature-coordinate standard deviation falls from 0.0523 to
0.0463. Same-context variation due to controls shrinks, while context variation becomes relatively larger
(context/control angular-distance ratio 0.43 -> 0.70). Because phi is the trainable field trunk itself, the
uncertainty geometry co-adapts and becomes less discriminative for alternate controls.

Finally, there is no trajectory-coherent exploration variable: each receding step independently resamples a
window, `broad=0`, and no `style_rho`/target is used. Positive self-training can only amplify modes already
sampled by the current field; it has no reliable mechanism to create a coherent new staircase.

### 9. Smaller but real numerical mismatch

Gather uses NFE=6 while evaluation uses NFE=8. On identical origin latent draws, seed-15 iteration 100's
sampled H=10 windows differ by 5.11% in relative L2 between NFE 6 and 8 (up from 4.39% for the base model).
With a narrowing, higher-Lipschitz field near tight corridors, training one numerical sampler and reporting
another adds avoidable deployment error.

## Ranked constructive fixes

These preserve the user's fixed quantile schedule, beta=0.3, no demo backfill, and unchanged Valid2 definition.
They correct implementation fidelity before another recipe sweep.

### P0: make “verified safe-progress target” literally true

1. Pass `reach=cfg.reach` into exploration deployment so gather and evaluation both use 0.1 m.
2. Require `GM2.traj_valid2(out["path"], env, gamma)` for the executed rollout. This uses the existing
   unchanged Valid2, rather than weakening or replacing it.
3. Before placing a planned H=10 window in the CFM pool, require its exact certificate boolean in addition to
   its existing task-space/progress tests. Do not train on the unexecuted nine-action tail unless that tail is
   itself certified.
4. Log strict reach rate, and hard-assert executed Valid2 plus planned-window exact-cert validity. A serious run
   should show 100% for the latter two; any class-empty condition triggers more gathering, never demo fill.

### P0: replace the degenerate scalar while retaining the same verifier

5. Keep the binary fitted-polytope verifier unchanged, but do not use its active-constraint minimum slack as
   a continuous rank. Use an SOCP robustness radius, for example the maximum additional robot-radius inflation
   delta for which `certify_window` remains feasible, found by a small fixed bisection. Low delta is genuinely
   low certificate robustness and remains paper-legible. Compute the low-margin quantile only among already
   certified windows.

### P0: balance the conditional problem the final table asks for

6. Make valid-rollout quotas explicit per gamma and include `gamma_ready` in the stop gate. Compute the three
   quantile planes within each gamma, then draw an equal easy/frontier quota per gamma (e.g. 8 fresh examples
   per gamma in a 56-fresh batch). This retains the AND quantile rule while preventing the global gamma=0.1
   starvation observed above.

### P0: make resume and diagnostics state-preserving

7. Evaluate inside `torch.random.fork_rng` (and preserve/restore NumPy state) or pass local generators. Probes
   must never modify the training RNG. Record the actual training seed and test that first gathers differ
   across seeds.
8. Save and restore Adam state, qbuf raw tensors, Torch/CUDA/NumPy RNG states, cumulative coverage, and the
   fixed teacher identity. For legacy checkpoints without qbuf, force at least one gather-only prime before
   any gradient step. The completed A/B shows this alone changes SR50 from .64/.02 to .92/0.

### P1: bound OOD field movement rather than trusting batch loss

9. Use one inner step until an all-gamma safety gate is stable; lower field LR or add trunk/head gradient
   clipping. The first-step change is currently behaviorally too large despite ordinary loss/RMS values.
10. Use a fixed original teacher, not a teacher copied anew at every resume. Add a small field trust-region on
    fixed origin-neighborhood contexts for every gamma and fixed latent/tau points. This anchors the metric
    domain while still allowing verified targets to expand it. Monitor fixed-demo CFM, origin field drift,
    gamma sensitivity, and sampled-control effective rank; rollback an update that violates the trust gate.
11. Keep gather/eval NFE equal.

### P1: make exploration joint and trajectory-coherent

12. Within each state, apply the unchanged cheap/exact safety-progress gates first, then use sigma to choose
    among eligible candidates. Current sigma-only selection does not enrich the joint event.
13. Decouple uncertainty features from the moving field trunk: use a frozen teacher phi or a fixed/whitened
    control-residual feature. Preserve qbuf across resumes. Track candidate sigma range and ESS; beta=0.3 is
    only meaningful if the ranking has nontrivial spread.
14. Add a rollout-level coherent proposal latent (a fixed `style_rho` or targeted uncovered staircase for the
    whole rollout), while retaining exact Valid2 and AND-frontier acceptance. This creates new coherent modes;
    independent per-step tail reweighting does not. This is an expansion proposal for our method only, never
    Mizuta.

## Minimal verification sequence before the next long unit

1. One-iteration smoke from the incumbent: strict reach is used, executed Valid2=100%, planned-target exact
   certificate=100%, every gamma represented, margin robustness nondegenerate, and sigma not identically one.
2. Two independent 5-10 iteration branches: first gathers must differ by hash; origin SR/CR must remain stable;
   balanced-demo CFM and gamma sensitivity must not regress materially.
3. Only then run the fixed 100-iteration unit. Select checkpoints by all-gamma M>=100, not gamma=.5 alone.
4. Continue Mizuta/Kazuki as an untouched benchmark; no flow expansion is warranted or used there.

## Reproducible diagnostic artifacts

- `analysis/vector_field_probe.py` / `analysis/vector_field_probe.json`
- `analysis/vector_field_one_update.json`
- `analysis/vector_field_resume_prime_compare.json`
- `analysis/uncertainty_joint_probe.py` / `analysis/uncertainty_joint_probe.json`
- `analysis/audit_frontier_verifier.py` (independent snapshot audit)
- `analysis/runs/resume_no_prime_s17/`
- `analysis/runs/resume_prime1_s17/`
