# Window-level gathering — how & where (handoff to codex, 2026-07-14)

**One-line answer:** the switch lives in **`grid_expand_hardtail.py`**, function **`_gather_fresh()`**
(≈ line 953), and is turned on by the CLI flag **`--window-level`** (→ `cfg.window_level`). Validity is
harvested **per sliding control-window**, not per whole trajectory.

Reference copies of the two files that matter are in `reference/grid_expand_hardtail.py` and
`reference/grid_metrics2.py` (refreshed 2026-07-14 to include this code).

---

## Why this exists (the problem it solves)

Old behaviour = **trajectory-level**: a rollout was gathered **only if the WHOLE executed trajectory
passed `valid2`** (`traj_ok`). At low γ (0.1/0.2) from a pretrained policy, almost no *whole* trajectory
is valid2, so the buffer starved → the update was skipped → we had to bolt on ad-hoc rescues
(`--emergent-gamma`, `--recovery-frac`, strip sub-quotas). All of that is **no longer needed**.

New behaviour = **window-level** (`--window-level`): we **harvest every individual control window that is
locally valid2**, even if the rollout it belongs to *eventually collides*. Valid **samples** become
plentiful → no starvation, **no ad-hoc plumbing**. The pretrained is "confused" for the first few iters
(a locally-valid window can sit inside an eventually-colliding rollout), but SR rises as the field is
corrected.

---

## Exactly where, in `_gather_fresh()`

### 1. The switch that turns off the whole-trajectory discard — line ~1072
```python
# WINDOW-LEVEL: validity is per-WINDOW, not per-trajectory. Harvest every locally valid2 window
# (taskspace ∧ progress ∧ SOCP, checked in the per-window loop below) even from trajectories that
# later collide. This makes valid SAMPLES plentiful -> no low-gamma starvation, no emergent-gamma /
# recovery needed.
if not traj_ok and not getattr(cfg, "window_level", False):
    continue                       # <-- OLD path: whole rollout dropped if any window fails
```
`traj_ok` (lines 1060-1066) is still computed for auditing (`GM2.traj_valid2(path, env, γ)`), but with
`--window-level` set, a failing `traj_ok` **no longer discards the rollout** — control falls through to
the per-window loop.

### 2. The per-window harvest loop — the REAL gate, lines ~1088-1110
This loop was *always* per-window (it is how windows become training targets). `--window-level` simply
stops the upstream filter from throwing away good windows that came from colliding rollouts.
```python
for i in range(U.shape[0]):
    p_i, pts, d = _window_progress(L[i].numpy(), U[i].numpy(), env)
    if not GM.in_taskspace(pts):                 # (1) window stays in the [0,5.12] task box
        continue
    if not getattr(cfg, "ablate_progress", False):
        if not GM2.approach_ok(d):               # (2) net-progress >= 0.10  (valid2 progress cond.)
            continue
        if p_i < min(cfg.valid_prog_floor, 0.5 * d[0]):   # reject safe-STATIONARY windows
            continue
    plan_ok, face_margin, residual = GM2.window_socp_stats(   # (3) window is SOCP-certifiable at γ
        GX2.state_from_low5(L[i].numpy()), U[i].numpy(), env, float(g))
    if not plan_ok:                              # never train on an infeasible planned target
        continue
    keep.append(i); wp.append(p_i); wm.append(face_margin); wr.append(residual)
```
**A window is kept iff:** `in_taskspace` ∧ `approach_ok` (net-progress ≥ 0.10) ∧ not-stationary ∧
`window_socp_stats.plan_ok` (SOCP-feasible at that γ). Kept windows (their `grid/low5/hist/U`) are the
training targets appended at line ~1114.

### 3. Flag wiring
- CLI: `--window-level` (argparse ~line 2131) → `cfg.window_level = bool(args.window_level)` (~line 2198).
- Also gather-relevant here: `--goal-xy GX GY` sets `env.goal` **and** `GM2.GOAL_XY` (the goal all the
  progress/valid2 math is measured against — must be set in BOTH places), `--start-eps`, `--wall-plugs`.

---

## The valid2 primitives it calls (`grid_metrics2.py`)

| function | line | what it enforces |
|---|---|---|
| `approach_ok(dists)` | ~35 | net progress `d0 - dH >= min(0.10, 0.5·d0)`; `DELTA_PROG = 0.10`. Safety is SOCP's job, so the old bounded-retreat / per-step-tol stages are dropped. |
| `traj_valid2(path,env,γ)` | ~60 | whole-traj validity = **every** sliding window (H=10, stride=2) passes taskspace ∧ approach ∧ SOCP. This is the `traj_ok` audit signal (now non-blocking under window-level). |
| `window_socp_stats(state,U,env,γ)` | ~107 | per-window verifier certificate. **γ-dependent**: `alpha_t=(1-γ)^t` (line ~123) — higher γ looser (may hug the boundary), lower γ stricter. Returns `(ok, face_margin, cert_residual)`. |
| `window_label_cheap(state,U,env,γ)` | ~85 | cheap in-rollout buffer label (taskspace ∧ approach, **NO SOCP** — SOCP runs once per window in the harvest loop). Passed as `verify_fn` to `GR.fm_deploy`. |

`GM2.GOAL_XY` (line 24) is the goal every distance is measured against — `--goal-xy` overwrites it.

---

## How to run it (the faithful recipe, no ad-hoc mechanisms)

```bash
python grid_expand_hardtail.py --window-level \
  --goal-xy 4.7 4.7 --start-eps 0.3 --wall-plugs 8 \
  --rollouts-per-iter 14 --beta 0.2 \
  --mix-start 0.4 0.6 --mix-end 0.4 0.6 --quantile-schedule 0:0.30 \
  --demo-frac 0 --lwf 0 \
  --recovery-frac 0 --hard-quota 0 --targeted-frac 0 --min-modes-per-gamma 0 \
  --gp-buf 500 \
  --gammas 0.1 0.2 0.3 0.4 0.5 0.7 1.0 --iters 50 \
  --init <pretrained.pt> --out <run_dir>
```

**Do NOT pass `--emergent-gamma`.** Window-level supersedes it: because valid *windows* are plentiful at
every γ, there is no zero-certified-γ to rescue. Same for `--recovery-frac` / `--hard-quota` — leave them
at 0. That is the whole point: window-level = valid samples without any of the rescue plumbing.

**Verify it trained** (probe.jsonl): `functional_step > 0`, `batch_e > 0 AND batch_f > 0`, finite `loss`
every iter. Under window-level, low γ (0.1/0.2) should yield hundreds of valid windows per iter instead of
the old `e0/f0 (no fresh)` starvation.

---

## Result on the faithful run (`results/p2/faithful_g47`, it50, goal (4.7,4.7), start-eps 0.3)

- Trained cleanly with **no emergent-γ / recovery / strip quotas** — window-level alone fed the batch.
- Pooled **SR 0.93, CR 0.06, clearance 0.283**; γ1.0 → **SR 1.0 / CR 0.0 / clearance 0.281** (safer than
  the expert's 0.262). **Safest method at every γ** (beats both the expert and the tuned CFM-MPPI on
  clearance). a-d 11/28 (blocked only by time — the safety↔speed tradeoff — and SR<1 at low-mid γ).
- Honest caveat: σ of easy/frontier windows stayed **flat** (~0.46 / 0.55) across all 50 iters — the FIFO
  novelty buffer tracks the policy, so uncertainty does not decay on its own.
