# P2 frontier / verifier / resume audit

Date: 2026-07-10. Scope: `grid_expand_fixed.py`, local
`grid_metrics2.py`, the imported verifier/rollout/GP stack, and saved P2
`viz_db` data. No Mizuta code or production trainer was modified.

## Bottom line

The dominant instability is not a beta or learning-rate tuning problem. The
current run violates four semantic invariants that should hold before another
long sweep is interpreted:

1. Gathering silently uses the legacy 0.45 m terminal radius, not the required
   0.1 m radius.
2. An executed path is accepted on SOCP alone; it is not required to pass the
   unchanged Valid2 progress test. Its stored planned windows are also not
   individually SOCP-gated.
3. Resume restores only model weights. It resets the query buffer/GP, Adam,
   RNG, coverage, and LwF anchor. The first resumed iteration therefore has
   constant uncertainty and is not a three-axis frontier iteration.
4. Measurement/probe evaluation mutates the training RNG. It erases the
   command-line torch seed before iteration 1 and resets the next exploration
   stream after every probe.

The global, gamma-pooled quantiles then amplify the problem: gamma 0.1 is often
absent, and when it is present almost none of its windows are frontier. This is
incompatible with the all-gamma objective.

## Reproducible evidence

Run:

```bash
LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib OMP_NUM_THREADS=16 \
python analysis/audit_frontier_verifier.py --recompute 64 \
  results/p2/finalunit_q50_k14_s15_from_it18/viz_db/it100.pt \
  results/p2/resume100_to200_s17/viz_db/it101.pt \
  results/p2/resume100_to200_s17/viz_db/it102.pt
```

The label recomputation has zero mismatches, and a deterministic subset of raw
verifier calls matches the saved margins exactly. Thus the following is data
evidence, not an inference from plotting.

| snapshot | strict reach `<0.1` | legacy reach `<0.45` | executed Valid2 | planned cert-infeasible | `|slack|<=1e-8` | frontier | gamma 0.1 windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| it100 | 0/14 | 14/14 | 11/14 (78.6%) | 2.60% | 53.2% | 8.87% | 180 (1 rollout) |
| resumed it101 | 0/28 | 28/28 | 20/28 (71.4%) | 0.86% | 57.7% | 30.67% | 0 |
| resumed it102 | 0/28 | 28/28 | 20/28 (71.4%) | 1.81% | 53.5% | 10.77% | 0 |

All accepted it100 terminal distances are 0.378--0.444 m; resumed it101 spans
0.380--0.448 m. This exactly matches the imported `GM.REACH=0.45` default.

At resumed it101, all 3,251 sigma values are exactly 1.0 (`std=0`). The AND
sigma predicate therefore passes every sample and the nominal 3-axis rule
reduces to a correlated 2-axis rule, yielding 30.7% frontier. This follows
directly from `GPUncertainty.sigma`: an empty buffer returns constant unit
uncertainty.

The two nominally independent seed-17 and seed-18 resumes have bit-identical
it101 gathered data: all 3,251 `low5`, `U`, gamma, sigma, margin, progress,
rollout IDs, window indices, and all 28 paths match exactly. Their `U` SHA-256
prefix is `3f44bac680977905`. The batches/losses later differ because NumPy batch
sampling retains the command seed; torch rollout sampling does not.

Gamma-conditioned imbalance is severe even outside cold start:

- it100: gamma 0.1 supplies 9.7% of windows but only 1.1% of those are frontier;
  most other gammas have 7.6--12.9% frontier.
- resumed it102: gamma 0.1 supplies zero windows. The trainer computes
  `gamma_ready=False` but does not use it in its stop condition.
- global raw progress is condition-dependent: it100 mean progress is 0.406 at
  gamma 0.1 versus 0.596 at gamma 0.5. A global high-progress plane therefore
  structurally suppresses the conservative condition.

## Root causes, in fix order

### P0 — wrong terminal radius during gather

`grid_expand_fixed.py:410-413` calls `GR.fm_deploy` without
`reach=cfg.reach`. `fm_deploy` therefore uses `grid_metrics.REACH=0.45`, while
the required evaluation and `CurConfig.reach` use 0.1. The gather consequently
declares all stored paths reached and stops before collecting the exact
terminal behavior needed by the strict metric. Coverage tracking also invokes
`staircase_id` with its 0.45 default.

Required change: pass `reach=cfg.reach` to every training deployment and pass
the same reach explicitly to coverage extraction.

### P0 — Valid2 is claimed but not executed

The recipe records `valid2_unchanged=true`, but the whole-path gate at
`grid_expand_fixed.py:418` is only `GM.socp_ok`. Planned-window taskspace and
progress are checked later, but this is not equivalent to
`GM2.traj_valid2(executed_path)`. Saved data proves the difference: 21--29% of
accepted executed paths fail the local unchanged Valid2 check.

There is a second mismatch. The path certificate verifies only controls that
were actually executed (the first action of each receding-horizon proposal).
The CFM target is the full proposed 10-control `U`. No exact certificate gate
is applied to that full target. In saved snapshots, 0.9--2.6% of the targets
are certificate-infeasible; the code maps their `-inf` slack to `-5` and trains
them. At it100, 3.66% of frontier targets are infeasible.

Required change: require `GM2.traj_valid2` for the executed trajectory and
require an exact `certify_window` pass for every planned `U` before it enters
either class. Never convert an infeasible/NaN certificate into a numeric
frontier score.

### P0 — probes overwrite the exploration RNG

`sr_cr_eval.eval_policy:44` calls `torch.manual_seed(seed0+i)` inside its loop.
`run_expand_cur` calls `_measure` before iteration 1, so both seed-17 and
seed-18 runs leave the baseline at seed 24. With `--probe-cov 1`, every
iteration then leaves the global RNG at seed 49. This controls candidate noise,
systematic resampling, CFM noise, and random query-buffer subsampling.

Required change: run every diagnostic/evaluation under a torch/NumPy RNG
preservation context. Keep common deterministic evaluation seeds, but restore
the training states on exit. Record the training seed in `recipe.json`.

### P0 — model-only resume is not an algorithmic resume

At `run_expand_cur:565-607`, Adam, GP, `qbuf`, coverage, pile, and teacher are
always newly constructed. Checkpoints saved by `HP.save_hp` contain only model
weights plus scalar metadata. Effects:

- the first resumed uncertainty is identically one;
- Adam moments and bias correction restart at each 100-iteration unit;
- the LwF teacher becomes the latest adapted model rather than the original
  fixed anchor, allowing anchor drift across units;
- rejected regions and discovered coverage are forgotten;
- a resumed 200-step schedule is not equivalent to an uninterrupted 200-step
  schedule.

Required change: save/restore optimizer state, query buffer, GP reconstruction
inputs, NumPy/CPU/CUDA RNG states, covered IDs, optional pile, original teacher
state, and history. The GP Cholesky itself need not be serialized; rebuild it
from restored `qbuf` under the current policy. Deployment-only `safe_best.pt`
can remain small, but every resumable checkpoint needs a companion full state.

### P1 — the query buffer remembers only accepted samples

`qbuf` is updated at `grid_expand_fixed.py:446`, after the trajectory SOCP
gate and per-window filters. Selected windows from rejected trajectories never
enter the query memory. They therefore remain maximally novel and can be
selected repeatedly, consuming attempts without moving toward safe/high-
progress data. This especially hurts strict gamma 0.1.

Move `_to_t(out["recs"])` and a subsampled `qbuf` update immediately after the
nonempty-record check. These selected proposals have been queried and received
a verifier outcome even when rejected. Do not add all unselected candidates,
because they have no exact verifier observation.

### P1 — the current certificate “margin” is numerically degenerate

The axis is the minimum `check_certificate` residual after fitting the
polytope to the same trajectory. At least 53% of saved feasible windows have
absolute residual <=1e-8, and the median planes are around `3e-16`. The lower
half is consequently dominated by active constraints and floating-point sign,
not a robust ordering of certificate tightness.

The literal nondegenerate SOCP decision margin is the fitted real-face
`Face.m`. On 1,000 deterministic it102 windows, `min(real Face.m)` spans
0.013--0.596, has 997 distinct values at 1e-8 resolution, and has Spearman
correlation 0.586 with geometric clearance. It remains verifier-native rather
than reverting to raw clearance. The minimum feasible angular interval width
is another robustness diagnostic, but is not literally the optimized margin.

Recommended frontier axis: after binary certification, use
`min(f.m for f in faces if f.kind == "real")` (with `R_eff` fallback when no
real face is sensed). Keep the residual and interval width as diagnostics.

### P1 — global planes and batches starve conditions

Quantiles are computed over all gammas and positions. Gamma changes the
certificate decay and policy speed, so raw margin/progress distributions are
not exchangeable. `_gather_fresh` computes `gamma_ready` but the break only
requires total valid count and two global classes. The update sampler is also
class-balanced but not gamma-balanced.

Use the same absolute quantile schedule separately within each gamma. Require a
minimum accepted-rollout quota per gamma (or a rolling block quota) and split
the easy/frontier batch quotas across gamma before rollout/window sampling.
This does not change the user-fixed AND rule; it makes it true for every
condition.

### P2 — gathered-data utilization and stop semantics

`target_e` and `target_f` are passed to `_gather_fresh` but never used. One
sample in each class is enough to stop after `K`. Updates draw with replacement
even when thousands of unique windows exist. For resumed it102, 3,434 accepted
windows were gathered but two updates can draw at most 112 fresh examples
(3.3%), with replacement. `--strat-rid` was not enabled in the final-unit
recipe.

At minimum, enforce enough unique class samples for the requested batch, draw
without replacement when possible, and stratify first by gamma, then rollout.
A bounded shuffled pass over accepted windows is preferable to increasing the
learning rate or beta.

## Proposed patch (production file intentionally not edited here)

The following is the minimal semantic patch, shown as a design diff rather
than a directly applied patch:

```diff
--- a/grid_expand_fixed.py
+++ b/grid_expand_fixed.py
@@ _gather_fresh(...)
         out = GR.fm_deploy(...,
             record=True, verify_fn=GM2.window_label_cheap,
+            reach=cfg.reach,
             device=device)
         if not out["recs"]:
             continue
+        # Query memory includes selected proposals regardless of acceptance.
+        G, L, H, U = GE._to_t(out["recs"])
+        qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3],
+                       cap=cfg.qbuf_cap)
-        if not GM.socp_ok(out["path"], env, float(g)):
+        if not GM2.traj_valid2(out["path"], env, float(g),
+                               check_socp=True):
             continue
-        G, L, H, U = GE._to_t(out["recs"])
         keep, wp, wm = [], [], []
         for i, r in enumerate(out["recs"]):
             p_i, pts, d = _window_progress(r[1], r[3], env)
             if not GM.in_taskspace(pts) or not GM2.approach_ok(d):
                 continue
             if p_i < min(cfg.valid_prog_floor, 0.5 * d[0]):
                 continue
+            cert_ok, cert_residual, face_margin = \
+                GM2.window_socp_stats(GX2.state_from_low5(r[1]),
+                                      r[3], env, float(g))
+            if not cert_ok or not np.isfinite(face_margin):
+                continue
             keep.append(i); wp.append(p_i)
+            wm.append(face_margin)
         ...
+        fresh["socp_margin"] = np.asarray(all_wm)
         ...
-        if valid >= K_eff and classes_ready:
+        if (valid >= K_eff and classes_ready and gamma_ready
+                and len(easy_idx) >= target_e
+                and len(frontier_idx) >= target_f):
             break
@@ label_fresh(...)
-    margin = np.array([GM2.window_socp_margin(...) for ...])
-    margin = np.nan_to_num(np.clip(margin, -5, 5), ...)
+    margin = np.asarray(fresh["socp_margin"], dtype=float)
+    if not np.isfinite(margin).all():
+        raise RuntimeError("non-finite margin survived binary certificate gate")
+    # Compute planes within gamma; retain raw per-gamma planes for viz/audit.
+    front, planes = per_gamma_and_planes(sigma, margin, prog,
+                                         fresh["gamma"], q)
@@ _measure / _cov_probe callers
+    with preserve_training_rng(device):
+        rows, agg, paths = SR.eval_policy(... deterministic eval seeds ...)
@@ checkpoint save/resume
+    save_train_state(model, optimizer, qbuf, covered, pile, teacher,
+                     np_rng, torch_rng, cuda_rng, history, absolute_iter)
+    restore_train_state_if_present(...)
@@ coverage
-    sid = GM.staircase_id(out["path"])
+    sid = GM.staircase_id(out["path"], reach=cfg.reach)
```

Suggested local `grid_metrics2.py` API:

```python
def window_socp_stats(state, U, env, gamma, R=2.5, n_theta=180):
    seg = GR.window_positions(np.asarray(state, float), U, env.dt)
    path = np.vstack([np.asarray(state, float)[:2], seg])
    ok, faces, _raw, R_eff = VP.certify_window(
        path, env.obstacles.detach().cpu().numpy(), float(env.r_robot),
        float(gamma), R=R, n_theta=n_theta)
    alpha = (1.0 - float(gamma)) ** np.arange(len(path))
    ok2, residual, _ = VP.check_certificate(
        faces, path - path[0], alpha, include_start=False)
    if bool(ok) != bool(ok2):
        raise RuntimeError("certificate/check mismatch")
    real_m = [float(f.m) for f in faces if f.kind == "real" and f.feasible]
    face_margin = min(real_m) if real_m else float(R_eff)
    return bool(ok and ok2), float(residual), float(face_margin)
```

For RNG isolation, a context must preserve CPU, visible CUDA, and NumPy state;
`torch.random.fork_rng` alone does not preserve NumPy:

```python
@contextmanager
def preserve_training_rng(device):
    np_state = np.random.get_state()
    devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(np_state)
```

## Regression gates before a serious rerun

1. **Strict terminal:** every path marked reached by gather has final distance
   `<0.1`; a saved snapshot contains terminal windows below 0.1.
2. **Exact Valid2:** every accepted executed path passes
   `GM2.traj_valid2(..., check_socp=True)`.
3. **Exact planned certificate:** every kept full `U` passes
   `window_socp_stats.ok`; no non-finite margin is labelable.
4. **RNG isolation:** CPU/CUDA/NumPy RNG states are byte-identical before and
   after `_measure` and `_cov_probe`. Seed-17 and seed-18 iteration-1 gathered
   controls must not be bit-identical.
5. **Resume equivalence:** a two-iteration uninterrupted smoke run and a
   one-iteration + full-state-resume smoke run match model, qbuf, optimizer,
   labels, and RNG (within deterministic GPU tolerance). First resumed sigma
   must not be constant one.
6. **All-gamma sampling:** each training block has nonzero accepted and
   frontier samples for all seven gammas; log both rollout and batch counts.
7. **Margin quality:** face-margin plane is finite and nondegenerate; warn if
   more than 5% of values tie at the plane.
8. **Class/batch sufficiency:** gathering cannot report ready unless it can
   fill the requested unique easy/frontier batch quotas.
9. **Small M smoke only:** after the gates pass, run a 3--5 iteration smoke and
   inspect the snapshot; only then launch 100 iterations. Final selection still
   requires the GOAL's all-gamma M>=100 evaluation.

The current post-it100 resumes should not be used as evidence for or against
the fixed recipe: their first step is a GP cold start, their training seed is
overwritten, and none of their accepted trajectories reaches the required
terminal radius.
