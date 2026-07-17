# AFE2 dual-scene protocol

This file is the canonical source of truth for the matched Claude-grid and
Codex-radius-1 studies. Historical results and proposed sweeps in
`AFE2_HANDOFF.md` and
`../codex_challenging/afe_restart/AFE2_RADIUS1_HANDOFF.md` are not run
instructions. Both scenes must use the same commit, each from its own clean
worktree; the scene profile and eligible pretrained checkpoint are the only
task-specific inputs.

## Scientific target

Test whether uncertainty-directed, deterministic-verifier-positive replay can
expand the intrinsic safety of a conditional flow model in an unseen obstacle
scene, while keeping safety and progress separate. The comparison is between
the same two update rules on:

- `claude_grid_v1`: the original plugged grid, start `(0.3,0.3)`, goal
  `(4.7,4.7)`;
- `codex_radius1_v1`: the matched grid in which the four central disks are
  replaced by one disk `(2.5,2.5,1.0)`, start `(0.5,0.5)`, goal `(4.5,4.5)`.

Calling the central obstacle *unseen* additionally requires the training/demo
scene manifest to show that this geometry was absent. Checkpoint identity or a
filename alone is not evidence of obstacle OOD.

## Shared expansion algorithm

One object is preserved throughout:

```text
generated plan == uncertainty-scored plan == queried plan == archived plan
full-H positive archived plan == replayed plan
executed action == first action of an execution-admissible queried plan
```

At round `n`, the current flow and its trainable 32-dimensional representation
are `theta_n` and `phi_s^(n)`, with `s=0.9`. The raw completed-verifier-query
archive `D_n` is cumulative. At the round boundary, re-embed every archived
`(c_i,U_i)` under the current representation, normalize
`z_i=phi_s^(n)(c_i,U_i)/||phi_s^(n)(c_i,U_i)||`, and rebuild

```text
A_n = I + lambda^-1 sum_i z_i z_i^T,       lambda = 10,
sigma_n(c,U)^2 = z^T A_n^-1 z.
```

Hold `theta_n` and `phi_s^(n)` fixed during that round's gathering. At every
receding-horizon step:

1. Draw `K=64` length-`H=10` plans from the untilted current flow.
2. Compute one pre-query vector of `sigma` values and
   `pi_j proportional to exp((sigma_j-max sigma)/beta)`.
3. Draw `B=8` indices from `pi` without replacement in one batch. This protocol
   does not re-score the remaining pool after each query.
4. Run the deterministic task-box plus SOCP verifier before execution. Every
   completed positive or negative query enters `D_n` and updates `A_n` once;
   `socp_error` enters neither.
5. Among execution-admissible queries, select maximum progress, execute only
   its first action, and replan. If none is admissible, terminate with
   `NO_VERIFIED_POSITIVE`; never call or imitate an expert fallback.

Progress, execution cost, SOCP margin, full-window safety, and execution
admissibility are separate fields. Sigma chooses verifier queries only. It is
not a safety probability and does not weight replay.

Both profiles deliberately retain the inherited `GM.in_taskspace` convention,
which accepts coordinates in `[-0.12,5.12]`. Report that tolerance; do not
describe the verifier as enforcing the exact `[0,5]` box.

### Absorbing-goal correction

Let `s_H` denote full-window certification, `G` the unchanged radius-0.15 goal
set, and `tau_G` the first predicted goal hitting time. The execution gate is

```text
e = s_H OR (tau_G <= H AND the prefix 1:tau_G is certified).
```

This corrects a stopping-time mismatch: actions after the task has already
terminated are not executed and therefore must not create a near-goal NVP.
The goal is absorbing and the safety claim ends at the first goal hit. A
prefix-rescued full-H negative remains negative and never enters `D_n+`; only
`s_H=1` plans are training positives. Log full positives, execution positives,
prefix rechecks, selected rescues, SOCP solve count, and verifier time
separately. Thus `B=8` is the query-object budget, not necessarily the final
number of solver calls.

This is a semantic correction, not a learned workaround. Enlarging the goal
radius, accepting an uncertified prefix, or relabeling the suffix would change
the task or the training label and is forbidden.

## Acquisition-temperature calibration

Calibrate once before training, independently for each `(scene, checkpoint)`.
Use beta-neutral round-0 pools from the complete seven-gamma sweep and solve
continuously, by deterministic log-bisection, for

```text
median_pool ESS(pi)/K = 3B/K = 0.375.
```

The factor `3` is a fixed acquisition-design assumption, not a theoretical
consequence of AFE or a safety guarantee. It is frozen before either arm runs.
The tolerance is `1e-4`. Persist the exact sigma-pool digest and solver witness;
fail on flat pools or an unbracketed/unattained root. Fix the resulting `beta`
for all ten rounds and share it between the two arms. Do not choose beta from
the absolute sigma magnitude, add a candidate after seeing a failure, use a
nearest-grid fallback, or tune it independently per arm.

## Two locked update arms

Both arms start from the same checkpoint and share the scene, calibration,
gathering/evaluation indices, `K/B`, verifier, terminal gate, archive rule,
execution rule, batch size, and ten-round schedule. Their trajectories diverge
after their updates, which is expected.

| arm | update on uniform cumulative `D_n+` |
|---|---|
| `prox` control | batch 128; CFM plus `||theta-theta_n||^2/(2 eta)`; Adam lr `2e-5`; `eta=0.01`; stop at relative functional change `0.03` or 40 steps |
| `afe` | batch 128; ordinary CFM; Adam lr `1e-4`; exactly 250 steps; no proximal term |

The visual encoder, flow trunk, and head remain trainable. There is no
curriculum, expert/demo replay, anchor, easy/frontier split, uncertainty-weighted
replay, automatic rollback, or checkpoint selection by the best-looking round.

Training is ten rounds, one episode for every
`gamma in {0.1,0.2,0.3,0.4,0.5,0.7,1.0}` in every round, `T=300`, seed 910.
The complete seven-gamma video is part of expansion monitoring, not merely a
four-gamma result gallery.

## Why the primary uncertainty kernel is linear

The locked study keeps the normalized current-representation linear kernel
above. It is the AFE linear construction and permits an exact cumulative
`32 x 32` state while the representation evolves and the archive grows. It
also makes the claimed uncertainty object explicit: coverage in the chosen
representation, not posterior uncertainty of the deterministic verifier.

Do not replace it with RBF in these two primary runs. Exact cumulative RBF
requires a growing `N x N` Gram matrix (about 28.8 GB in float64 at 60,000
queries, before factorization); sparse RBF or random features add a lengthscale,
approximation size, landmarks/features, and seeds. More importantly, RBF
cannot recover distinctions already erased by `phi_s^(n)`. Switching kernels
after observing low effective rank would therefore confound the scene
comparison.

RBF is a legitimate later, preregistered ablation only: fix its representation,
lengthscale rule (for example, median pairwise distance on the beta-calibration
pools), approximation budget and seed before outcomes; rerun beta calibration;
and report it separately from the locked linear result.

## Diagnosis, not symptom treatment

| observation | diagnosis/measurement in this protocol | excluded ad-hoc response |
|---|---|---|
| NVP clusters after an otherwise safe approach to the goal | compare full-H rejection with first-goal-hit prefix certification; use the absorbing-goal execution semantics above | enlarge the goal, move walls, inject a fallback, accept an uncertified prefix, or train on rescued negatives |
| strict gamma has low verifier-positive mass | report it as the expected safety-feasibility limitation of the fixed multi-step certificate | gamma curriculum, extra strict-gamma episodes, recovery starts, or a looser verifier |
| sigma spread/uplift vanishes | report all-pool versus selected-pool statistics, centered feature rank, and representation drift | hand-lower beta, concatenate raw actions after seeing results, reset `A`, switch to RBF, or change `K/B` |
| prox is nearly frozen and AFE oscillates | preserve and compare the two declared arms | middle learning rates, looser prox, fewer AFE steps, more rounds, or automatic rollback |
| selected trajectories lose diversity | measure route/feature coverage | pi-sampled execution or a diversity quota |

Changing the context process `rho(c)` can be a scientifically useful future
experiment, but it changes the expansion distribution and is not part of this
matched replication.

## Three-layer monitoring

Monitoring must not alter acquisition, replay, stopping, or checkpoint choice.

1. **Verifier and execution:** per gamma query count, full-H positive count,
   execution-positive count, prefix rechecks/rescues, SOCP errors/solves/time,
   margin, NVP, SR, CR, obstacle collision, OOB, true minimum clearance, and
   time-to-goal.
2. **Acquisition and representation:** all-K and selected-B sigma quantiles,
   IQR/span/uplift, ESS and entropy; `A` spectrum/effective rank; centered
   feature spectrum/effective rank; fixed-probe cosine drift; CFM loss,
   module gradient norms, and relative parameter changes. Uncentered
   `Z^T Z` rank alone is evidence, not proof, of mode collapse.
3. **Intrinsic model evaluation:** keep tilted query acceptance distinct from
   untilted `V_safe` (task-space and SOCP) and `V_full` (also the fixed
   approach criterion). Audit samples never enter `D_n` or `A_n`.
   Report per-gamma bare-policy SR/CR, validity, clearance, and successful
   time-to-goal with intervals and fixed indices.

The true evaluation uses `M=100` fixed-index rollouts for
`gamma in {0.1,0.3,0.5,1.0}` at every checkpoint. Its gallery displays ten
seed-fixed, outcome-stratified members of those same cells and labels that
selection; it is not curated evidence. Compare expert, purely pretrained bare
flow, round-10 AFE bare flow, and the gamma-blind Kazuki baseline. Use Wilson
intervals for binary rates and bootstrap intervals for continuous summaries,
with the explicit limitation that one training seed does not quantify
across-training-run uncertainty.

## Checkpoint and artifact eligibility

- `claude_grid_v1`: require the explicitly supplied 64-character file SHA-256,
  `0eede103cc7c24ce23d2cd0e83aa3a64fdeb1a1f644c24973c5aa33a242499f4`,
  model-state SHA-256
  `5af84097e47976e92669690073f81634edf5bebbc3cb139e641dcf1924331336`,
  the audited 32-D architecture including trunk `89->160->96->32`, and the
  legacy metadata contract (`data=druni_`, `per_gamma_cap=0`). This exact-
  artifact compatibility gate does not upgrade the checkpoint into a modern
  promoted model.
- `codex_radius1_v1`: require a fresh Stage-3 checkpoint that passes
  `require_promoted_fresh_pretrain`; record both file and stable model-state
  hashes. Two documented promoted file/model-hash pairs are
  `bfbb925a8499205a4639b33b8fe819ae4527fa8cafcabcc8722dd9bedea21efb` /
  `c988ba1e3edb9a7cca1cb117796b1d101b4d644a8624a08326326b86dd7a3275`
  and
  `36cb9d6651d8aa86791ad6639be987f0da8f44d76b97fe9245a419f765ce0b08` /
  `59cf4b6f7c13cb3ca535bff27fd587cdd9e19a65c7a6e307c6371fa5d715037b`.
  Select the intended replica from its Stage-3 manifest; do not infer it from
  this list or substitute the old giant-obstacle checkpoint.

Every run records the clean source commit, scene fingerprint, checkpoint
contract and hashes, calibration-pool digest, named seed streams, and CUDA
visibility/logical device/physical device identity. A valid training delivery
requires both arm `COMPLETE.json` files, the matched-pair manifest, report and
videos, and top-level `DELIVERY_COMPLETE.json`. A valid true evaluation must be
bound to that completed pair's `afe_s910` checkpoints and have its own hashed
delivery manifest; the presence of `final.pt` or a PNG is insufficient.

Existing `paper_results/true_eval_{gallery,curves}.{png,pdf}` files predate
this sealed raw-cell contract and are historical illustrations, not evidence
for the present protocol. See `paper_results/HISTORICAL_TRUE_EVAL.md`.

## Clean, separate execution

After the final source commit is on `origin/master`, use its exact SHA in two
new detached worktrees. Do not pull into either agent's editing folder.

```bash
git -C /path/to/safeMPPI fetch origin master
git -C /path/to/safeMPPI worktree add --detach /path/to/afe2-claude FINAL_SHA
git -C /path/to/safeMPPI worktree add --detach /path/to/afe2-codex FINAL_SHA
git -C /path/to/afe2-claude status --porcelain
git -C /path/to/afe2-codex status --porcelain
```

Both status commands must be empty and both `rev-parse HEAD` values must equal
`FINAL_SHA`. Use new output roots. The GPU shown inside each process is
logically `cuda:0`; physical assignment comes from `CUDA_VISIBLE_DEVICES`.

Claude outline, physical GPU 3:

```bash
cd /path/to/afe2-claude/overnight_run_07_06/rev_expansion/codex_overnight
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
./run_afe2_pair.sh claude_grid_v1 /absolute/claude_checkpoint.pt FULL_FILE_SHA256 /new/claude_pair
./run_true_eval.sh claude_grid_v1 /new/claude_pair /new/claude_true_eval
```

Codex outline, physical GPU 1:

```bash
cd /path/to/afe2-codex/overnight_run_07_06/rev_expansion/codex_overnight
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
./run_afe2_pair.sh codex_radius1_v1 /absolute/promoted_checkpoint.pt FULL_FILE_SHA256 /new/codex_pair
./run_true_eval.sh codex_radius1_v1 /new/codex_pair /new/codex_true_eval
```

The launchers run calibration, prox, and AFE sequentially and fail closed.
Agents should not edit code, substitute a checkpoint, relax a gate, resume into
a stale directory, or choose a new beta when a run stops. They should report
the exact command, source/checkpoint hashes, GPU provenance, and failing
artifact, then wait for a protocol-level decision.

## Claim boundary

Pre-execution deterministic certification plus verify-or-terminate gives a
conditional execution-safety statement only through the first goal hit, under
the implemented dynamics, task-box convention, and SOCP verifier assumptions.
Empirical `M=100` validity and success rates are estimates, not a theorem that
the learned generator is safe, performant, or successful on unknown contexts.
The experiment tests expansion of verified mass; it does not establish an EBM
density-floor guarantee or a general probabilistic safety guarantee.
