# Challenging start+goal expansion progress

## 2026-07-14 — Stage 1: off-diagonal seed pools

### Step 1.1 — Implement the deterministic walled-scene seed generator

**CMD**

```bash
python gen_uniform_data.py --out seeds.png
```

**RESULT**

Created `seeds.png` from the fixed-jitter 32×32 grid after applying all eight wall plugs. The final pool
contains 566 points: 282 blue upper-off-diagonal starts (49.8%) and 284 red lower-off-diagonal goals
(50.2%). The rendered scene contains all 72 obstacles (64 original + 8 plugs).

**DECISION**

Keep all 566 reference points. The two-point asymmetry comes from applying the reference's fixed jitter
before its `|y-x| >= 1` test; dropping points merely to force identical pool sizes would no longer
reproduce the cited distribution. Episode sampling in stage 2 will choose a pool first, so balanced
trajectory counts do not depend on the pool-size difference.

### Step 1.2 — Validate the geometric invariants

**CMD**

```bash
python - <<'PY'
import numpy as np
import gen_uniform_data as g

env = g.make_walled_env(8)
blue, red = g.start_goal_pools(env)
points = np.concatenate((blue, red))
obs = env.obstacles.detach().cpu().numpy()
clearance = (np.linalg.norm(points[:, None] - obs[None, :, :2], axis=2)
             - obs[None, :, 2] - float(env.r_robot))
print(len(blue), len(red), len(points), len(obs))
print(np.all(blue[:, 1] > blue[:, 0]), np.all(red[:, 1] < red[:, 0]))
print(np.abs(points[:, 1] - points[:, 0]).min(), clearance.min())
PY
python -m py_compile gen_uniform_data.py
```

**RESULT**

The pool counts are `282 284 566`, the scene has 72 obstacles, both half-plane checks are true, the
minimum diagonal offset is 1.0188 m, the minimum obstacle clearance is 0.0511 m, and compilation passes.

**DECISION**

Stage 1 meets its numeric and visual acceptance criteria. Stop here for the required permission gate
before generating SafeMPPI demonstrations in stage 2.

## 2026-07-14 — Artifact organization and Stage 2A fixed-pair approval preview

### Step 2A.1 — Separate stage outputs and lock visualization semantics

**CMD**

```bash
python gen_uniform_data.py
```

**RESULT**

Created `stage_results/<stage>/{viz,logs,data}` as the result layout. Stage 1 now has its visualization at
`stage_results/01_seeds/viz/seeds.png` and a machine-readable summary at
`stage_results/01_seeds/logs/seed_summary.json`. Added `viz_style.py`, matching Image #1 with seven
discrete plasma samples over `[0.02, 0.90]` for gamma and reserving viridis exclusively for
sigma/uncertainty.

**DECISION**

Use the shared palette module for every later trajectory, scatter, curriculum, and internals figure. Do
not use viridis for gamma or plasma for sigma.

### Step 2A.2 — Generate the slow synchronized fixed-pair expert GIF

**CMD**

```bash
OMP_NUM_THREADS=8 python fixed_pair_expert_gif.py
```

**RESULT**

Selected exact approved-pool points `start=(0.3168483, 4.6734347)` and
`goal=(4.6653547, 0.3146670)`. All seven SafeMPPI experts reached within 0.2 m, remained collision-free
and in task space, and produced these `(steps, minimum clearance)` results:

```text
gamma 0.1: (134, 0.207 m)    gamma 0.2: (119, 0.085 m)
gamma 0.3: ( 97, 0.076 m)    gamma 0.4: ( 94, 0.085 m)
gamma 0.5: ( 92, 0.086 m)    gamma 0.7: (100, 0.020 m)
gamma 1.0: ( 96, 0.094 m)
```

The slow GIF is 1180×656, 35 frames at 500 ms/frame with a 2.5 s final hold. Every unique displayed
window passed `verifier_polytope.certify_window`: respectively `35/35, 31/31, 25/25, 25/25, 24/24,
26/26, 25/25` from gamma 0.1 through 1.0. The animation shows all gamma values simultaneously, with the
moving nominal SafeMPPI polytope/level sets, the executed H-step window, and the fitted verifier boundary
in green.

Artifacts:

```text
stage_results/02_demos/viz/fixed_pair_expert_polytopes.gif
stage_results/02_demos/viz/fixed_pair_expert_polytopes_final.png
stage_results/02_demos/data/fixed_pair_preview_paths.npz
stage_results/02_demos/logs/fixed_pair_preview.json
```

**DECISION**

This is the requested fixed-pair approval preview only. Do not launch the 300-random-pair × 7-gamma
training-data generation until the user approves this GIF.

## 2026-07-14 — Stage 2: 300-pair goal-aware SafeMPPI demonstrations

### Step 2.1 — Freeze one random-pair manifest and implement goal-aware windows

**CMD**

```bash
python gen_sg_data.py manifest \
  --out stage_results/02_demos/data/random_pairs_300.npz \
  --pairs 300 --seed 20260714
```

**RESULT**

Created 300 unique Cartesian-product pairs from the Stage 1 pools and reused those exact pairs at every
gamma. The manifest contains 177/282 distinct upper-pool starts and 183/284 distinct lower-pool goals;
the maximum marginal reuse is 5 starts and 4 goals. `gen_sg_data.py` passes each episode goal into both
SafeMPPI planning and window construction. Each merged dataset stores `grid`, goal-aware `low5`, `hist`,
`U`, `starts`, `goals`, `window_starts`, `window_goals`, and pair indices; the low5 relative-goal feature
is therefore never computed against the old fixed `(5,5)` goal.

**DECISION**

Use one immutable pair manifest across all seven gamma values. This keeps gamma comparisons paired and
prevents a different random-pair draw from masquerading as a gamma effect.

### Step 2.2 — Saturate physical GPU 2 and generate all demonstrations

**CMD**

```bash
# CUDA MPS daemon bound to physical GPU 2; workers see its remapped cuda:0.
python run_stage2_gpu2.py \
  --gpu 2 --client-visible-device 0 \
  --pairs 300 --shards-per-gamma 2 --max-retries 1
```

**RESULT**

Used CUDA MPS with 14 concurrent workers (two shards for each of seven gamma values). Physical GPU 2
held at 99–100% utilization during the production phase, peaked at 14,039 MiB, and completed the worker
phase in 136.0 seconds; the slowest individual worker took 128.5 seconds. The isolated MPS daemon was
stopped after completion, and all temporary worker tensors were removed after successful merging.

All 2,100/2,100 trajectories reached their episode goals, stayed in bounds, and were collision-free:

```text
gamma   trajectories   windows   mean steps   minimum clearance
 0.1        300         24,944      83.15            0.0587 m
 0.2        300         19,222      64.07            0.0311 m
 0.3        300         17,935      59.78            0.0162 m
 0.4        300         17,717      59.06            0.0152 m
 0.5        300         17,291      57.64            0.0091 m
 0.7        300         17,543      58.48            0.0048 m
 1.0        300         17,738      59.13            0.0028 m
total      2,100       132,390
```

The seven merged `.pt` files occupy 1,659,739,636 bytes (1.546 GiB) and are stored under
`stage_results/02_demos/data/`, alongside compact per-gamma trajectory archives. Per-gamma summaries and
the 14 worker logs are under `stage_results/02_demos/logs/`.

**DECISION**

Keep two MPS shards per gamma as the production recipe. A one-worker benchmark under-filled GPU 2, while
14 ordinary CUDA contexts incurred context-switch overhead; CUDA MPS provided full utilization without
changing the generated schema or pair manifest.

### Step 2.3 — Render the all-gamma mass overlay and independently audit the data

**CMD**

```bash
python plot_sg_demo_overlay.py
python validate_sg_data.py \
  --physical-gpu 2 --cuda-mps --worker-seconds 136.0 \
  --gpu-utilization-max 100 --gpu-memory-max-mib 14039
python -m py_compile gen_uniform_data.py fixed_pair_expert_gif.py viz_style.py \
  gen_sg_data.py plot_sg_demo_overlay.py run_stage2_gpu2.py validate_sg_data.py
```

**RESULT**

Rendered `stage_results/02_demos/viz/demo_300_pairs_all_gamma.png`: one large 2,100-trajectory overlay,
a discrete top gamma colorbar, and seven per-gamma panels in the style of the supplied reference. Gamma
uses the shared truncated-plasma palette; sigma/uncertainty remains reserved for viridis.

The independent validation is `PASS`. It checked tensor shapes and finiteness, exact gamma channels,
manifest equality, per-window start/goal identity, goal-aware low5 reconstruction, executed-control
alignment, collision clearance, workspace bounds, and final goal distance for every gamma. Maximum
reconstructed-position error was `4.77e-7`. The machine-readable audit is
`stage_results/02_demos/logs/stage2_validation.json`, and all source files compile.

**DECISION**

Stage 2 is complete and accepted internally: 300 paired demonstrations per gamma, goal-aware training
windows, the requested mass overlay, and an independent audit are all present. Pause at the required
stage gate and wait for user permission before Stage 3 pretraining.

## 2026-07-14 — Stage 3: from-scratch start+goal policy pretraining

### Step 3.1 — Add explicit raw episode endpoints to the H_P policy context

**CMD**

```bash
python grid_hp_expt.py
python -m py_compile grid_hp_expt.py pretrain_sg.py sg_rollout.py \
  eval_pretrained_sg.py validate_stage3.py
```

**RESULT**

Implemented the local `GridHPFlowPolicy` as

```text
context = low5(current state, episode goal, gamma)
        + [start_x, start_y, goal_x, goal_y]       # raw world-frame metres
        + E(H_P)[32]                              # trainable 32×32 CNN/AAP visual token
```

The context dimension is therefore 41 instead of 37, and the first flow-trunk layer has 93 inputs
(`U[20] + context[41] + Fourier-time[32]`). The model has 56,056 parameters, including 34,048 in the
unfrozen visual encoder. The checkpoint schema is `sg-hp-v1-raw-endpoints`; loading an old fixed-goal
checkpoint through this local loader is rejected. The forward/backward test and source compilation pass.

**DECISION**

Keep the original goal-relative `low5` because it describes the current control problem, but append raw
episode start and goal coordinates as independent, constant conditions. This preserves the useful local
feature while making endpoint identity explicit for arbitrary-pair reasoning and later reverse-direction
deployment.

### Step 3.2 — Train from scratch on all seven goal-aware datasets

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python pretrain_sg.py
```

**RESULT**

Loaded all 132,390 Stage 2 windows onto physical GPU 2. The deterministic split is grouped by complete
pair index across every gamma: 270 pairs / 118,866 windows for training and 30 pairs / 13,524 windows for
validation. Thus no later window from a validation trajectory can leak into training. The policy was
initialized from scratch; no fixed-goal checkpoint, demo anchor, or W&B service was used.

The 120-epoch AdamW/cosine run used BF16 autocast, batch size 2,048, learning rate `3e-4`, and a fully
trainable encoder. GPU 2 reached 99% utilization during the measured run. Training finished in 46.8 s;
the best grouped-pair validation CFM loss was `1.1917606` at epoch 111, down from `1.63965` at epoch 0.
The saved file is the epoch-111 state, not the final-epoch state.

The encoder did not collapse: all 32 token dimensions have standard deviation above `1e-3`, mean token
feature standard deviation is `0.9881`, covariance effective rank is `4.38`, and encoder gradient norms
remain nonzero through the end of training. Shuffling only the raw endpoint coordinates raises matched
validation CFM loss by 0.37%, a modest but measurable dependence in addition to the strong relative-goal
condition.

Artifacts:

```text
pretrained_sg_walls8.pt
stage_results/03_pretrain/data/pretrained_sg_walls8.pt
stage_results/03_pretrain/data/pair_split.npz
stage_results/03_pretrain/logs/pretrain_history.csv
stage_results/03_pretrain/logs/pretrain_summary.json
stage_results/03_pretrain/logs/pretrain_stdout.log
stage_results/03_pretrain/viz/pretrain_curves.png
```

**DECISION**

Use the best grouped-pair checkpoint as the Stage 4/5 initialization. Grouped validation is stricter than
the reference's random-window split and gives an honest test of trajectory-pair generalization.

### Step 3.3 — Test a pair with two unseen endpoint marginals and audit the checkpoint

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python eval_pretrained_sg.py --seeds-per-gamma 5
python validate_stage3.py
```

**RESULT**

Predeclared `start=(0.3372544, 1.6730419)` (blue-pool index 7) and
`goal=(3.9420285, 0.48492995)` (red-pool index 65). Neither endpoint marginal—and therefore not the exact
pair—appears anywhere in the 300-pair Stage 2 manifest. Across five plain, unguided receding-horizon
rollouts at every gamma, 9/35 reached within 0.2 m while remaining collision-free and in bounds:

```text
gamma        0.1   0.2   0.3   0.4   0.5   0.7   1.0
safe reach   1/5   2/5   2/5   0/5   0/5   3/5   1/5
```

This satisfies the Stage 3 held-out-pair reach sanity check. Failures remain visible in the rollout figure
and are intentionally retained as the honest pre-expansion baseline for Stage 4.

The independent `validate_stage3.py` audit is `PASS`. It verifies byte-identical checkpoint mirrors
(SHA-256 `2665b744871cd89684c8a61b5fdb5f017656e2658756f4fb3496e35cd1763e13`), finite weights, the
41-dimensional context and exact raw-endpoint placement, complete/non-overlapping 270/30 split, all
per-gamma window counts, best-epoch reconciliation, non-collapsed visual tokens, absent held-out
marginals, and reach/safety criteria for all nine successful paths.

Artifacts:

```text
stage_results/03_pretrain/viz/heldout_pair_rollouts_all_gamma.png
stage_results/03_pretrain/data/heldout_pair_rollouts.npz
stage_results/03_pretrain/logs/heldout_pair_eval.json
stage_results/03_pretrain/logs/stage3_validation.json
```

**DECISION**

Stage 3 is complete: the required from-scratch, explicit start+goal-conditioned checkpoint exists and
reaches a stricter fully unseen-endpoint pair. Pause at the required gate before Stage 4 canonical
origin→goal deployment; do not interpret the 9/35 sanity rate as the final post-expansion performance.

## 2026-07-14 — Stage 3 revision: restore the original endpoint-free model

> The user rejected the raw-start/raw-goal model above. This section supersedes that Stage 3 decision.
> The rejected checkpoint and its logs/figures remain only under
> `stage_results/03_pretrain/rejected_raw_endpoints/` for provenance.

### Step 3R.1 — Remove the four raw endpoint inputs without discarding learned weights

**CMD**

```bash
python grid_hp_expt.py
python - <<'PY'
from pathlib import Path
import grid_hp_expt as HP
from pretrain_sg import migrate_raw_endpoint_checkpoint

policy = HP.GridHPFlowPolicy()
print(migrate_raw_endpoint_checkpoint(
    policy,
    Path("stage_results/03_pretrain/rejected_raw_endpoints/data/pretrained_raw_endpoints.pt"),
))
PY
```

**RESULT**

Restored the original policy context exactly:

```text
context = low5(relative goal, velocity, gamma) + E(H_P)[32]
```

There is no raw start or absolute goal input. Context width is 37, the flow trunk input is
`U[20] + context[37] + Fourier-time[32] = 89`, and the model has 55,544 parameters. The active loader
rejects any checkpoint whose config still says `raw_start_goal=true`, and `ctx_from` no longer accepts a
fourth endpoint argument.

Migrated the rejected checkpoint surgically. Its old first-layer layout was
`U[20] + low5[5] + endpoints[4] + E(H_P)[32] + time[32]` (93 columns). Columns `[25:29]` were removed,
yielding 89 columns; the audit confirms every other tensor was copied bit-for-bit. The removed endpoint
columns had L2 norm `1.50090`.

**DECISION**

Use only the original `low5 + E(H_P)` structure going forward. Preserve the rejected run for auditability,
but overwrite both active `pretrained_sg_walls8.pt` mirrors with the endpoint-free fine-tuned checkpoint.

### Step 3R.2 — Fine-tune the migrated endpoint-free policy

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib python pretrain_sg.py
```

**RESULT**

Fine-tuned on the unchanged leakage-free split: 270 pairs / 118,866 windows for training and 30 complete
pairs / 13,524 windows for validation. The 160-epoch run used batch size 1,024, AdamW, initial learning
rate `1e-4`, BF16, cosine decay, and an unfrozen visual encoder. It finished on GPU 2 in 69.3 seconds.

The endpoint-deletion initialization had validation CFM `1.2052906`. The best fine-tuned value is
`1.1736705` at epoch 157—better than both the migrated initialization and the rejected raw-endpoint
model's `1.1917606`. All 32 visual-token dimensions remain active; effective rank is `4.37`. Shuffling
only `low5`'s relative-goal components increases matched validation loss by 4.93%, versus only 0.37% for
shuffling the rejected model's raw endpoints.

**DECISION**

Keep epoch 157 as the active pre-expansion checkpoint. The improved loss and stronger relative-goal
sensitivity support the simpler original structure requested by the user.

### Step 3R.3 — Deploy on both the unseen pair and canonical expansion target

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib \
  python eval_pretrained_sg.py --seeds-per-gamma 8
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib python validate_stage3.py
```

**RESULT**

Ran 56 plain, unguided rollouts on each requested scene (8 seeds × 7 gamma values).

The fully unseen-marginal pair remains
`(0.3372544, 1.6730419) → (3.9420285, 0.48492995)` with reach 0.2 m. It now achieves 16/56 safe reaches,
with at least one success at every gamma:

```text
gamma        0.1   0.2   0.3   0.4   0.5   0.7   1.0
safe reach   2/8   2/8   3/8   1/8   2/8   2/8   4/8
```

On the exact expansion target, the protocol is start `(0.05,0.05)`, goal `(5,5)`, and reach 0.15 m.
The start offset provides `0.054951 m` clearance from the origin plugs; the goal stays at `(5,5)` and
the 0.15 m reach condition stops valid trajectories before the goal plugs. The pre-expansion policy gets
0/56 safe reaches. The closest rollout ends 0.324 m from the goal, but it has already collided, so it is
correctly counted as failure. In total, 54/56 canonical rollouts collide and four also leave task space.
This is the measured reverse-direction distribution gap that safe flow expansion must repair.

Artifacts:

```text
stage_results/03_pretrain/viz/heldout_pair_rollouts_all_gamma.png
stage_results/03_pretrain/viz/canonical_target_rollouts_all_gamma.png
stage_results/03_pretrain/viz/pretrain_curves.png
stage_results/03_pretrain/data/heldout_pair_rollouts.npz
stage_results/03_pretrain/data/canonical_target_rollouts.npz
stage_results/03_pretrain/logs/deployment_eval.json
stage_results/03_pretrain/logs/stage3_validation.json
```

The revised independent audit is `PASS`: active checkpoint mirrors are identical (SHA-256
`5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2`), differ from the rejected raw
checkpoint, use context 37/trunk 89, reject endpoint arguments, reproduce the exact column deletion,
retain the complete pair split, improve validation, and reconcile both deployment path archives with
their collision/bounds/reach criteria.

**DECISION**

The requested endpoint-free Stage 3 revision and canonical baseline deployment are complete. Keep the
0/56 canonical result as the honest pre-expansion baseline and pause for user review before Stage 5 safe
flow expansion.

## 2026-07-14 — Stages 4–6 bounded sanity run

The user authorized a short Stage 4–6 launch only, explicitly before any big expansion campaign. The
questions were whether unfreezing the encoder, changing β, and a very small rejected-sample α expose a
promising direction. All long-run/default-budget claims remain deferred.

### Step 4.1 — Formalize the canonical baseline

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python reference/eval_ae.py policy-worker \
  --ckpt pretrained_sg_walls8.pt --wall-plugs 8 --start-eps .05 --reach .15 \
  --M 6 --gamma <each of seven gamma values>
python make_stage456_reports.py
```

**RESULT**

The formal Stage 4 panel uses 6 faithful, unguided rollouts per γ (42 total) on the exact 8-plug scene.
It gets `0/42` successes and `CR=100%`. This agrees with the larger Stage 3 baseline and is retained
without relabeling close-but-colliding endpoints.

Artifacts:

```text
stage_results/04_canonical/REPORT.md
stage_results/04_canonical/logs/stage4_summary.json
stage_results/04_canonical/data/pretrained_m6/table.md
stage_results/04_canonical/viz/canonical_pretrained_m6.png
```

**DECISION**

Stage 4 is complete. Use this fixed-seed M=6 panel for sanity comparisons and the Stage 3 M=8 panel as
the larger pre-expansion record.

### Step 5.1 — Restore the optional signed rejected-sample path and validate it

**CMD**

```bash
python reference/analysis/test_hardtail_trainer.py \
  --json stage_results/05_sanity/logs/test_hardtail_trainer.json
python reference/analysis/test_signed_negative.py
```

**RESULT**

The hard-tail trainer had removed rejected samples entirely, so α previously could not influence a run.
Added bounded, fresh rejected replay with the paper objective
`L = L_positive - alpha * L_negative`, separate negative minibatches, and telemetry for positive/negative
loss, negative pool size, and raw field/encoder negative-gradient RMS. `alpha=0` keeps the established
accepted-only signature and objective. The parser now accepts small positive β values as well.

The existing corrected-trainer suite is `20/20 PASS`; the new signed-loss identity/gradient gate is also
`PASS` at `alpha=5e-4`. The copied reference path bootstrap was corrected so it loads this endpoint-free
model and the corrected `codex_overnight/grid_metrics2.py`, not the stale rev-expansion module.

**DECISION**

The α pilot is now real and auditable. Keep gradient clipping and functional-step rollback on because
the signed objective is unbounded below.

### Step 5.2 — Unseeded cold-start probes and three ablation brothers

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python reference/grid_expand_hardtail.py ... \
  --emergent-gamma --min-modes-per-gamma 0 --legacy-prime-iters 0 \
  --beta <.05|.1|.2|.4> --iters 1 --gather-attempt-cap <8|12>
# Repeat once with each of --ablate-progress, --ablate-curriculum, --ablate-socp.
```

**RESULT**

The exact method found `0` valid rollouts in `44` queries across β `.05/.10/.20/.40`; therefore encoder
freezing and α cannot act at all from the raw pretrained checkpoint. `−Progress` and `−Curriculum` also
found zero. Only `−SOCP` supplied data (`1/3` rollouts and 46 progressive windows), proving the cold-start
bottleneck is exact certification/support rather than β, labeling, or optimizer topology.

At the original `lr=2e-5`, that first `−SOCP` update exceeded the functional-step bound and rolled back.
At `lr=5e-6`, three-iteration diagnostic arms updated stably. The unfrozen encoder carried nonzero
gradient RMS; β `.1` changed pool size/composition; `alpha=5e-4` had the intended tiny loss contribution
when rejected rows existed. These `−SOCP` results are mechanism diagnostics, not safe performance.

**DECISION**

Do not launch a long unseeded run: it would spend its budget collecting no positive data.

### Step 5.3 — One-time certified seed and exact “hope” comparison

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python canonical_seed_finetune.py --regenerate \
  --outdir stage_results/05_sanity/runs/canonical_seed_unfrozen \
  --steps 25 --lr 5e-6 --enc-lr-mult .3
CUDA_VISIBLE_DEVICES=2 python reference/grid_expand_hardtail.py ... \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --no-freeze --enc-lr-mult .3 --beta .2 --alpha <0|.0005> --iters 3
```

**RESULT**

Generated one exact-valid canonical SafeMPPI trajectory per γ: 819 windows total, every one-time seed
trajectory checked by unchanged `traj_valid2`. The seed was used for 25 CFM steps and then discarded;
there is no persistent demo fraction or LwF anchor. This is explicitly a diagnostic branch and does not
overwrite the active pretrained model.

The first support signal appears only for the unfrozen seed checkpoint at β `.2`: `1/5` queried rollouts
is exact-valid (82 windows). The frozen twin and the unfrozen β `.1` probe remain at zero. From the same
unfrozen β `.2` checkpoint:

```text
alpha=0       exact-valid rollouts by iter: 1, 1, 0   (2 total)
alpha=5e-4    exact-valid rollouts by iter: 1, 1, 1   (3 total)
```

At iteration 3, tiny α retains 72 certified windows from one rollout, a 512-row negative pool, raw
negative CFM `1.143`, online `SR=29.4%`, and online `CR=41.2%`; α=0 has no valid data that iteration and
online `CR=63.6%`. This is a small stochastic comparison, not a causal claim.

Crucially, independent faithful M=6/γ deployment remains `0/42, CR=100%` for the one-time seed, α=0,
and tiny-α checkpoints. The hope is persistent exact-query support, not a deployment win.

Artifacts:

```text
stage_results/05_sanity/SANITY_REPORT.md
stage_results/05_sanity/logs/stage5_sanity_summary.json
stage_results/05_sanity/viz/stage5_sanity_dashboard.png
stage_results/05_sanity/data/canonical_seed_windows.pt
stage_results/05_sanity/runs/hope_exact_b02_a0/
stage_results/05_sanity/runs/hope_exact_b02_a00005/
```

The dashboard uses plasma only for γ and viridis only for σ; the certified trajectory lines and the
σ-fill/γ-edge scatter make those semantics explicit.

**DECISION**

Candidate for the future big dive: one-time exact seed → unfrozen encoder at `0.3×` field LR → β `.2` →
α `5e-4`, with early stops on lost exact support or worsening faithful CR. Do not claim a win yet.

### Step 6.1 — SafeMPPI expert and reduced Kazuki rough sweep

**CMD**

```bash
python reference/eval_ae.py expert-worker --wall-plugs 8 --start-eps .05 \
  --reach .15 --M 6 --gamma <each gamma>
CUDA_VISIBLE_DEVICES=2 python reference/kazuki_baseline.py \
  --ckpt pretrained_sg_walls8.pt --wall-plugs 8 --start-eps .05 --reach .15 \
  --w-safe <sweep> --coll-w <sweep> --goal-w <sweep> --goal-coef <sweep> \
  --n-sample 100 --n-elite 5 --n-copy 50
```

**RESULT**

SafeMPPI gets `42/42` successes and zero collisions; mean successful clearance is `0.22–0.27 m`. Thus
the canonical geometry is feasible and certifiable.

The reduced Kazuki rough-sweep winner is `w_safe=.3, coll_w=20, goal_w=2, goal_coef=.5`. Its fixed M=3/γ
panel is `0/21, CR=100%`, but a separate γ `.5` M=6 sensitivity panel contains one clean success
(`SR=16.7%, CR=66.7%`). This finds the requested nonzero behavior while exposing severe seed instability.
The final-fidelity `200/10/200` Kazuki run is intentionally deferred.

Artifacts:

```text
stage_results/06_baselines/REPORT.md
stage_results/06_baselines/logs/stage6_summary.json
stage_results/06_baselines/data/comparison_table.md
stage_results/06_baselines/viz/stage6_rollout_comparison.png
```

**DECISION**

Stages 4–6 sanity are complete. SafeMPPI is the real target, Kazuki is beatable but needs the full-fidelity
rerun, and no big expansion has been launched.

### Step 6.2 — Independent Stage 4–6 audit

**CMD**

```bash
python validate_stage456.py
```

**RESULT**

`PASS`. The audit verifies the active pretrained SHA-256 is still
`5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2`, the endpoint-free schema, all
trainer gates, Stage 4 metrics, the `0/44` cold start, all seven seed γ values, exact-support sequences,
faithful Stage 5 failures, expert/Kazuki results, and nonempty figures/reports. Machine-readable output:
`stage_results/05_sanity/logs/stage456_validation.json`.

**DECISION**

Pause here for review as requested. The next action, if approved, is the big dive using the candidate
seeded/unfrozen/β=.2/tiny-α recipe with fixed frequent faithful evaluation gates.

### Step 5.4 — certification-bootstrap amendment and matched v4 sanity arms

**CMD**

```bash
# Trainer amendment: direct exact H=10 demo file, demo-only cold-start update,
# and irreversible fraction decay after the first exact-certified rollout.
python reference/analysis/test_hardtail_trainer.py
python reference/analysis/test_signed_negative.py

CUDA_VISIBLE_DEVICES=2 python reference/grid_expand_hardtail.py \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --iters 6 --no-freeze --enc-lr-mult .3 --beta .2 --alpha .0005 \
  --demo-file stage_results/05_sanity/data/canonical_seed_windows.pt \
  --demo-frac .50 --demo-bootstrap --demo-decay-on-valid --demo-frac-after-valid .05 \
  --emergent-gamma --min-modes-per-gamma 0 --seed 5010 ...
# Repeated with exactly one of --ablate-socp / --ablate-progress / --ablate-curriculum.
```

**RESULT**

The default-off amendment preserves the historical trainer (20/20 gates PASS) and the tiny signed-negative
test passes. All four final streams find an exact rollout at iteration 1, so the requested demo share drops
immediately from `8/16` rows to `1/16`. Certified-stream totals over six iterations:

```text
Ours            2 rollouts / 171 windows / 2 viz_db frames
−SOCP           7 rollouts / 323 windows / 6 viz_db frames
−Progress       4 rollouts / 653 windows / 4 viz_db frames
−Curriculum     2 rollouts / 201 windows / 2 viz_db frames
```

Faithful M=6×7 deployment remains `0/42, CR=100%` for every short arm. This is therefore a certification-
support sanity result, not a performance claim.

**DECISION**

The early exact H=10 distillation + 10× decay behaves as requested and gives the visualization stream enough
certified data. Keep this as the big-dive candidate, but do not launch the big run yet.

### Step 6.3 — low-guidance Kazuki replacement

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python reference/kazuki_baseline.py \
  --ckpt pretrained_sg_walls8.pt --wall-plugs 8 --start-eps .05 --reach .15 \
  --w-safe .02 --coll-w 2 --goal-w 2 --goal-coef .1 \
  --n-sample 100 --n-elite 5 --n-copy 50 --M 6 --gamma-ctx <gamma>
```

**RESULT**

The weak-guidance paths traverse the map and expose the pretrained-policy behavior instead of looking like a
safety filter. Across M=6×7 it has `3/42` successes and `39/42` collisions; successful gammas are `.4/.7/1.0`.

**DECISION**

Use this configuration in the sanity rollout/scatter/table. It is visually informative and honestly unsafe.

### Step 8.1 — exact v4 sanity visualization package

**CMD**

```bash
python reference/paper_results/scatter_v4.py
python reference/paper_results/rollouts_v4.py
python reference/paper_results/internals_v4.py
python reference/analysis/make_table.py
python reference/video_curriculum_fixed.py \
  --run stage_results/05_sanity/runs/final_v4_ours \
  --out stage_results/05_sanity/viz/curriculum_it6.mp4 \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --iters 0,1,2,3,4,5,6
python validate_v4_sanity.py
```

**RESULT**

`PASS`. The package now follows the reference v4 grammar exactly: 2×4 rollout order with all three NO
brothers; 1×2 marker/method and truncated-plasma/γ scatter; 2×3 internals; IEEE+Markdown table; viridis-σ
curriculum movie. The MP4 is H.264 `2028×1014`, 2 fps, 14 frames, 7 s. The previous ad-hoc Stage 5 dashboard
was overwritten by `internals_v4`.

Artifacts:

```text
stage_results/05_sanity/viz/curriculum_it6.mp4
stage_results/05_sanity/viz/internals_v4.png
stage_results/05_sanity/viz/rollouts_v4.png
stage_results/05_sanity/viz/scatter_v4.png
stage_results/05_sanity/data/table_v4.md
stage_results/05_sanity/data/table_v4.tex
stage_results/05_sanity/logs/v4_sanity_validation.json
```

**DECISION**

Pause for visual/metric review before any big dive. Active `pretrained_sg_walls8.pt` remains byte-identical.

### Step 5.5 — corrected reached-trajectory predicates and retained distillation

**SEMANTICS**

```text
Ours:          H10 ∧ task-space ∧ reach ∧ progress ∧ SOCP; easy/frontier split
−SOCP:         H10 ∧ task-space ∧ reach ∧ progress;        easy/frontier split
−Progress:     H10 ∧ task-space ∧ reach ∧ SOCP;            easy/frontier split
−Curriculum:   H10 ∧ task-space ∧ reach ∧ progress ∧ SOCP; one class
```

The trainer now implements and audits these predicates directly. The omitted predicate is not evaluated in
its ablation. A new regression gate brings the suite to `21/21 PASS`; signed-negative replay still passes.

The distillation source is explicitly the canonical OOD SafeMPPI expert
`stage_results/05_sanity/data/canonical_seed_windows.pt`: `(0.05,0.05)→(5,5)`, all seven γ values, 819 exact
H=10 windows. It is not the upper-left→lower-right pretraining distribution.

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python reference/grid_expand_hardtail.py \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --iters 6 --seed 5010 --no-freeze --enc-lr-mult .3 --beta .2 --alpha .0005 \
  --batch 16 --demo-file stage_results/05_sanity/data/canonical_seed_windows.pt \
  --demo-frac .50 --demo-bootstrap --demo-decay-on-valid --demo-frac-after-valid .10 \
  --wall-plugs 8 --start-eps .05 --reach .15 --emergent-gamma --min-modes-per-gamma 0 ...
# Matched runs add exactly one of --ablate-socp / --ablate-progress / --ablate-curriculum.
```

GPU 2 reached 93–99% utilization while the four arms ran concurrently. Every arm latched at iteration 1.
With the intentionally unchanged batch 16, `.10` retains two OOD-expert rows after the latch; empty gather
iterations take a two-row demo-only update. The rejected `.01` interpretation would have rounded to zero.

**RESULT**

```text
Ours            1 reached rollout  / 106 windows / 1 viz_db frame
−SOCP           6 reached rollouts / 460 windows / 6 viz_db frames
−Progress       3 reached rollouts / 407 windows / 3 viz_db frames
−Curriculum     2 reached rollouts / 222 windows / 2 viz_db frames
```

All four faithful M=6×7 evaluations remain `0/42, CR=100%`. The reduced acceptance behaves correctly, but
six iterations do not produce a deployable controller.

### Step 8.2 — corrected v4 overwrite

The exact v4 filenames were overwritten from the `final_v6_*` streams. The rollout gallery’s pretraining-data
panel now includes both start seeds and goal seeds. `validate_v4_sanity.py` passes and additionally checks the
acceptance masks, goal termination, OOD expert metadata, effective 8→2 demo rows, 21/21 trainer tests, and
video encoding.

Artifacts remain:

```text
stage_results/05_sanity/viz/curriculum_it6.mp4
stage_results/05_sanity/viz/internals_v4.png
stage_results/05_sanity/viz/rollouts_v4.png
stage_results/05_sanity/viz/scatter_v4.png
stage_results/05_sanity/data/table_v4.md
stage_results/05_sanity/data/table_v4.tex
stage_results/05_sanity/logs/v4_sanity_validation.json
```

**DECISION**

Pause for review. No big dive was launched, and `pretrained_sg_walls8.pt` remains byte-identical.

### Step 5.6 — twenty-iteration wait, retained 0.25 demo share, and controlled −Curriculum replay

**CMD**

```bash
CUDA_VISIBLE_DEVICES=2 python reference/grid_expand_hardtail.py \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --iters 20 --seed 5010 --no-freeze --enc-lr-mult .3 --beta .2 --alpha .0005 \
  --rollouts-per-iter 2 --gather-attempt-cap 12 --batch 16 \
  --demo-file stage_results/05_sanity/data/canonical_seed_windows.pt \
  --demo-frac .50 --demo-bootstrap --demo-decay-on-valid --demo-frac-after-valid .25 \
  --wall-plugs 8 --start-eps .05 --reach .15 --emergent-gamma --min-modes-per-gamma 0 ...
# −SOCP and −Progress add their single named flag.
# −Curriculum additionally receives the full arm's accepted-window budget and exact replay directory.
```

**RESULT**

Six iterations were too short: the full arm found new batches at iterations 8, 9, 11, 12, and 20. Over
20 iterations its accepted-window sequence is:

```text
106, 0, 73, 0, 95, 0, 0, 59, 85, 0, 77, 51, 0, 0, 0, 0, 0, 0, 0, 55
```

The total is 601 windows from eight reached rollouts. The controlled −Curriculum arm replays those exact
accepted tensors and the same bounded rejected tensors (3,367 rows over the eight nonempty updates), then
removes only the easy/frontier split. Tensor equality and the per-iteration counts are independently checked.
The 0.50→0.25 latch retains four exact OOD-expert rows in every batch-16 update.

The final streams are:

```text
Ours            8 reached rollouts  /  601 windows /  8 nonempty updates
−SOCP          19 reached rollouts  / 1542 windows / 19 nonempty updates
−Progress      11 reached rollouts  / 1416 windows / 11 nonempty updates
−Curriculum     8 reached rollouts  /  601 windows /  8 nonempty updates (controlled replay)
```

Faithful M=6×7 deployment is `0/42, 42 collisions` for Ours, −Progress, and −Curriculum. −SOCP gets the
only learned-arm success (`1/42`) but collides on the other 41 episodes. The run therefore confirms that
SOCP certification limits the positive stream, while also showing that simply removing SOCP is unsafe.

The hard-tail suite is now `22/22 PASS`; the signed-negative gradient/loss test remains PASS. The active
checkpoint SHA-256 remains
`5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2`.

**DECISION**

There is no deployment-ready “hope” configuration in this bounded run. Retaining more expert data and waiting
20 iterations improves evidence quality, not faithful control. Do not launch the big dive.

### Step 8.3 — final v7 exact-style overwrite and audit

**CMD**

```bash
python reference/paper_results/scatter_v4.py
python reference/paper_results/rollouts_v4.py
python reference/paper_results/internals_v4.py
python reference/analysis/make_table.py --out-prefix stage_results/05_sanity/data/table_v4
python reference/video_curriculum_fixed.py \
  --run stage_results/05_sanity/runs/final_v7_ours \
  --out stage_results/05_sanity/viz/curriculum_it20.mp4 \
  --ckpt stage_results/05_sanity/runs/canonical_seed_unfrozen/final.pt \
  --iters 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
python validate_v4_sanity.py
```

**RESULT**

The exact v4 filenames were overwritten from `final_v7_*`. The 2×4 rollout gallery contains all three NO
brothers and plots both pretraining start and goal seeds. Gamma remains truncated-plasma; curriculum σ remains
viridis. The slow curriculum MP4 is H.264 2028×1014, 2 fps, 42 frames, and 21 seconds.

`validate_v4_sanity.py` is PASS. It checks the 20 predicate audits, effective 8→4 demo schedule, M=6 rows,
bit-exact positive/negative curriculum control, 22/22 trainer gates, signed-negative test, baselines, artifact
sizes, and video encoding.

**DECISION**

The corrected bounded sanity package is complete. Pause for user review; no big expansion was started.

### Step 5.7 — window-native aggregation handoff and matched No-brother sanity

`WINDOW_LEVEL_GATHER.md` was read end-to-end and its central correction was isolated in a new module,
`reference/window_expand_hardtail.py`; the handed-off `reference/grid_expand_hardtail.py` remains byte-identical
to the `codex_overnight` source (SHA-256
`941f4890cb7f1fead7635c91b7075eb50d2f9999547f9bb6ff325bcdb8991a11`). The new module makes coherent
executed H=10 windows the aggregation unit. Whole-rollout valid2, later collision, and goal reach are audit
signals only and cannot veto a locally accepted sibling window.

The masks are explicit and independently tested:

```text
Full / -Curriculum: task-space AND progress AND SOCP
-SOCP:              task-space AND progress AND positive geometric clearance (SOCP not called)
-Progress:          task-space AND SOCP (progress not called)
```

The new regression suite is `9/9 PASS`. It also catches and fixes the handed-off single-class batch bug:
`-Curriculum` now draws `16+0` instead of silently returning an empty batch. Full writes its per-iteration
accepted-window budget and the controlled `-Curriculum` arm consumes it exactly.

The six-iteration cleared-stadium sanity used `(0.3,0.3)->(4.7,4.7)`, all seven gamma values, wall plugs 8,
beta `.2`, mix `.4/.6`, no demo/LwF, and no emergent/recovery/hard/targeted mechanisms. Full accepted
`3378` locally valid windows from `42` contributing rollouts even though all `42/42` failed whole-path
valid2. The old trajectory gate would have accepted none. `-SOCP` accepted `3492` windows with zero SOCP
evaluations and rejected 13 locally colliding windows; `-Progress` accepted `5003` with zero progress
evaluations. Controlled `-Curriculum` matched Full exactly at
`437, 695, 554, 585, 410, 697` windows and trained with a true `16+0` batch. Every arm had six finite,
nonzero updates and zero rollbacks.

Matched faithful M=6×7 deployment remains preliminary and unsuccessful: SR is zero for every arm; CR is
`1.000 / .976 / 1.000 / .976` for Full / -SOCP / -Progress / -Curriculum. This validates the gather fix,
not policy performance. The handoff prose lists GP buffer 500, while the cited archived
`faithful_g47/recipe.json` records GP/query buffers 200/200; this sanity used the archived 200/200 lineage.

Artifacts:

```text
stage_results/05_window_native/REPORT.md
stage_results/05_window_native/viz/prelim_rollouts.png
stage_results/05_window_native/viz/prelim_training.png
stage_results/05_window_native/logs/prelim_summary.json
stage_results/05_window_native/data/eval_m6/
```

**DECISION**

Pause at the approval gate. Do not launch the 50-iteration dive. The giant-obstacle Stage 2B build remains
paused, and this legacy cleared-stadium sanity must not be presented as a giant-obstacle Stage 5 result.

### Giant Stage 5 — temperature probe, stable window-native expansion, and exact Stage-6 package

The approved radius-1.2 giant scene was finally run end-to-end in its own folder. The actual OOD SafeMPPI
expert supplied 2,688 complete H=10 rows: 64 equally spaced windows from each of 42 successful trajectories,
exactly 384 rows per gamma, no padding, and zero dynamics reconstruction residual. This—not the ID diagonal
dataset—is the expansion demo anchor.

Matched frozen-policy deployment and one-iteration gathering compared temperatures 0.1, 0.5, and 1.0:

```text
temperature       0.1      0.5      1.0
mean |Δu|        0.153    0.495    0.915
valid2 windows     249      222      480
all-gamma/classes  yes      yes      yes
```

Temperature 0.5 was selected as the non-collapsed compromise. The first expansion calibration at LR `1e-5`
was rejected because Full and −Progress repeatedly crossed the 0.025 functional-step ceiling. It is retained
under `runs/temp0.5_lr1e-5_rejected/` and was never promoted. The corrected suite used LR `5e-6`, an unfrozen
encoder at 0.3× LR, beta 0.2, batch 16, and actual-expert demo mass 8/16 through iteration 10 then 4/16 for
iterations 11–20.

The stable four-arm mechanics are clean:

```text
arm             accepted H10 windows   max functional step   rollbacks
Full                    5511                   0.0165              0
−SOCP                   5856                   0.0170              0
−Progress               6769                   0.0169              0
−Curriculum             5511                   0.0185              0
```

Full made 6,093 SOCP and 7,758 progress evaluations. −SOCP made exactly zero SOCP calls and used positive
geometric clearance; −Progress made exactly zero progress calls and retained 7,524 SOCP evaluations.
−Curriculum used the same Full accepted-window count at every iteration and a true single-class batch.
All 5,511 Full windows came from 140 parent rollouts that failed whole-trajectory valid2, directly confirming
that whole-trajectory status is audit-only under the requested window-native semantics.

The behavioral result is negative. Checkpoints 5/10/15/20/final and deployment temperatures 0.1/0.5/1.0
were screened. The promoted visualization checkpoint is iteration 10 (SHA-256
`0c2f2713fa319cbebc04b44b78c38b77fa3648d8a5553b8cb6e7397cda4da5c3`) because it had the largest
boundary-following arc, but matched M=6×7 deployment is still `0/42, CR=1.0` at every temperature. Ours
increases mean boundary arc from the pretrained 0.206 to 0.343 rad and improves mean endpoint distance from
4.029 to 3.938 m, but all 42 trajectories still collide (37 giant, 5 other); there is therefore no successful
multi-mode diversity. Expert remains `42/42`, two modes; low-guidance CFM-MPPI remains `0/42`, CR=0 with 42
local-minimum timeouts.

Requested exact-style artifacts:

```text
giant_obstacle_ood/stage_results/06_exact_reports/viz/rollouts_v4.png
giant_obstacle_ood/stage_results/06_exact_reports/viz/internals_v4.png
giant_obstacle_ood/stage_results/06_exact_reports/viz/scatter_v4.png
giant_obstacle_ood/stage_results/06_exact_reports/viz/curriculum_it20.mp4
giant_obstacle_ood/stage_results/06_exact_reports/viz/temperature_sweep.png
giant_obstacle_ood/stage_results/06_exact_reports/logs/independent_audit.json
```

Gamma is truncated-plasma in rollout/scatter views and sigma is viridis in the curriculum view. The rollout
gallery contains the pretraining start and goal seed, Expert, Pretrained, low-guidance CFM-MPPI, all three No
brothers, and Ours. The MP4 is H.264 2028×1014, 2 fps, 42 frames, and 21 seconds. The independent mechanics
and artifact audit is PASS; its scientific outcome field is intentionally ALERT because Full has no success.

**DECISION**

Pause for user review. The valid2 curriculum and all controls are faithful and stable, but the bounded run did
not provide behavioral hope. Do not launch a long expansion from this recipe without a new approved change.

### Giant Stage 5 continuation — balanced pretraining panel and T=1 expansion hope gate

**CMD**

- Expanded only on the target gamma set `{0.1, 0.5, 1.0}` with rollout/gather temperature `1.0`, NFE 8,
  window-native H10 validity, beta `0.2`, and the actual balanced OOD expert anchor. The selected continuation
  resumed iteration 5, reset Adam, used LR `1e-7`, LwF `4.0`, and completed iterations 6–10.
- Evaluated every checkpoint candidate with a separate matched deployment gate at temperature `0.5`, M=20 per
  gamma, 300 steps, seed 92500, and a persistent symmetric route bit. The gate required nonzero successes for
  all three displayed gammas.
- Re-ran `-SOCP`, `-Progress`, and controlled `-Curriculum` from the same iteration-5 state and evaluated their
  iteration-10 checkpoints with the identical T=0.5/M=20 protocol. Re-rendered `rollouts_v4.png/.pdf` from
  these new archives.
- Replaced the pretraining-data preview with every approved Stage-2B U/R-balanced path for the displayed
  gammas: 24 paths per gamma, 72 total. No ordering-dependent prefix selection remains.

**RESULT**

- Promoted bounded-sanity checkpoint: `runs/from_target3_it5_preserve/full/ckpt_10.pt`, SHA-256
  `ee72ee43c0ee669f131740083bb746bf41cc66007ff343b452958da11105e6c3`.
- The requested displayed-gamma gate passed: gamma 0.1 = `1/20`, gamma 0.5 = `3/20`, gamma 1.0 = `3/20`.
  Across all seven evaluation gammas the checkpoint reached `16/140` (SR 0.1143), collided `101/140`
  (CR 0.7214), and successful paths included both lower-right (10) and upper-left (6) giant-obstacle detours.
- Full learning health is PASS: gather T=1.0 / eval T=0.5, 5,022 accepted local valid2 windows, all 5,022
  from whole-invalid parent trajectories, all taskspace/progress/SOCP predicates enabled, and zero rollback,
  readiness, invalid-loss, or missing-update failures.
- The controlled `-Curriculum` arm consumed Full's exact iteration budgets
  `1245/1029/652/1366/730` as a true easy-only single pool. `-SOCP` made no SOCP requirement and
  `-Progress` made no progress requirement; neither altered the other arm predicates.
- Matched sibling SR at T=0.5 is `10/140` (-SOCP), `8/140` (-Progress), and `15/140` (-Curriculum).
  Thus the brothers are mechanically faithful, but the desired strong ablation separation is not established.
- The pretraining-panel audit is PASS: each displayed gamma has 24 real trajectories and maximum U/R
  mirror-count residual 0. The revised rollout panel includes the real balanced data, Expert, Pretrained,
  CFM-MPPI*, all three freshly evaluated brothers, and Ours.

**DECISION**

Promote iteration 10 only as the requested visualization/hope checkpoint. The all-three-gamma nonzero-SR
gate is satisfied and learning is stable, but collision rate remains high, gamma 0.4 is still zero-SR in the
seven-gamma audit, and route diversity is not robust per gamma. Do not describe this bounded sanity as the
final scientific win or as evidence for the expected ablation ordering.
