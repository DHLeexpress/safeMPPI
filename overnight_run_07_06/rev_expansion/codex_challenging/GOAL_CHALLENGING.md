# CODEX — START HERE: start+goal-conditioned Safe Flow Expansion on the WALLED scene (from scratch)

> **Work ONLY in `rev_expansion/codex_challenging/`.** Copy the exact reference code cited below (all paths
> are relative to `rev_expansion/codex_overnight/` unless noted) into this folder and adapt. Use **GPU by
> `nvidia-smi`** (grab an empty one; Claude uses 3). No wandb/push. Append a `PROGRESS.md` here
> (CMD / RESULT / DECISION per step). **PAUSE and ask the user for permission between every numbered stage.**

## The new task (harder than codex_overnight)
The generative policy now must navigate **arbitrary
start→goal pairs** (Check whether our generative model is conditioned on the start and goal I think it is encoded in rel vec and state (current))on the **8-plug walled scene**, learned entirely **from scratch** (no reuse of the fixed-
goal `pretrained_a32uni.pt`). The demo distribution: **start ∈ upper off-diagonal (blue, y>x)**, **goal ∈
lower off-diagonal (red, y<x)** — so the SafeMPPI expert moves **top-left → bottom-right** on average
(Image #10: `|y-x|>=1`, obstacle-free, 566 pts/γ, 50/50 blue-above / red-below). We then **deploy** on the
canonical **origin→(5,5)** task (opposite direction) — the generalization is the point: the visual encoder
captures the scene, the policy reasons about *this* start & goal, not a memorized (5,5).

**Win condition (stage 7):** beat **Kazuki** (for sure) and the **demo expert** (bonus: more coverage) on
metrics **a–e** at deployment, and the three ablation "brothers" must degrade as specified below.

## Recipe DEFAULTS for this task (differ from codex_overnight — set these, then sweep)
```
--wall-plugs 8 --start-eps 0.05 --reach 0.2        # walled scene, cleared corners (see below)
--demo-frac 0.0 --lwf-eta 0.0                       # NO anchor by default (isolates the mechanism +
                                                     #   makes the ablation brothers show cleanly)
--rollouts-per-iter 14                              # (not 28)
--mix-start 0.4 0.6 --mix-end 0.4 0.6               # frontier-heavy
--quantile-schedule 0:0.30 --beta 0.2 --gp-buf 200 --qbuf-cap 200 --gp-buf==qbuf-cap
--lr 2e-5 --inner-steps 2 --field-grad-clip 1.0
# NOT emergent-gamma by default. BUT KNOW IT EXISTS: `--emergent-gamma` gathers every certified window
#   uniformly and does NOT block the update on gammas with zero certified windows yet (from a fresh
#   policy gamma 0.1/0.2 are 0% SOCP-valid). Turn it on if the round-robin gather starves at low gamma.
```
Reference trainer (copy, don't reinvent): `grid_expand_hardtail.py`. It already supports every flag above
plus `--gammas`, `--ablate-socp/-progress/-curriculum`, full-state resume (`--ckpt ckpt_N.pt` +
`--start-iter N` + `--resume-allow-recipe-drift`), and the 22-gate harness `analysis/test_hardtail_trainer.py`.

---

## STAGES (pause + ask permission after each)

### (1) Seeds — off-diagonal start/goal grid  [Image #10]
Copy `rev_expansion/gen_uniform_data.py::uniform_starts()` (32×32 grid, `|y-x|>=1`, obstacle-free, ±0.02
fixed jitter). **Split it:** `blue = pts[y>x]` (start pool), `red = pts[y<x]` (goal pool), 50/50. Apply the
8-plug walls first (`grid_expand_hardtail._apply_wall_plugs(env, 8)` or `eval_ae._WALL_PLUGS8`) so the free-
space test uses the walled obstacles. **Deliverable:** reproduce Image #10 (blue above / red below,
`|y-x|<1` excluded, walls drawn) → `codex_challenging/seeds.png`. **PAUSE.**

### (2) Demo generation — SafeMPPI on start→goal pairs (nominal-polytope viz)
Copy `overnight_run_07_06/gen_dr_data.py` (already patched: `--wall-plugs`, `--start-eps`, `--canonical-frac`,
`--reach`) and `rollout_from`. **Change:** sample `start∈blue`, `goal∈red` **per trajectory** (currently
goal is fixed `env.goal`); pass the sampled goal into the SafeMPPI `ad.plan(..., goal_t=sampled_goal, ...)`
and into `windows_from(states, controls, env, gamma)` (make it goal-aware — the window's `low5` must use the
*episode* goal, not (5,5)). Save `dataset/w8sg_windows_g{γ}.pt` (schema: grid/low5/hist/U/starts/**goals**).
150–300 starts/γ, 7 γ, CPU-parallel per γ. **Viz the nominal polytope** along a few demo rollouts on the
walled scene: reuse `analysis/socp_sanity.py::poly_from_faces` + `verifier_polytope.certify_window` (GREEN
= certified). Confirm the expert stays inside its polytope top-left→bottom-right. **PAUSE.**

### (3) Pretrain the start+goal policy (from scratch, walled)
Copy `overnight_run_07_06/pretrain_repr.py` + `stage3_pretrain.py` (they build `grid_hp_expt.GridHPFlowPolicy`
with the 32×32 CNN/AAP visual encoder + repr head). **Two more raw conditions:** `low5(state, goal, γ)`
already carries the *relative-goal* vector `(goal-pos)/R_GOAL` (`grid_feats.py:62`), so goal-conditioning
exists — but for arbitrary goals **append the raw `start_xy` and `goal_xy` (4 nums) to the policy's context
vector** (extend `featurize`/the policy ctx dim) so it can spatially reason, and **train on the w8sg demos**
(varied goal). Frozen-vs-unfrozen encoder: start frozen=False here (from scratch), watch for collapse.
Save `codex_challenging/pretrained_sg_walls8.pt`. **Sanity:** the pretrained reaches a *held-out* start→goal
pair. **PAUSE.**

### (4) Deploy the pretrained on the canonical task
Deploy `pretrained_sg_walls8.pt` from **start (0.05,0.05)** to **goal (5,5) minus ε** (a slightly-cleared
goal — the goal-corner plugs (5.2,5.0)&(5.0,5.2) sit ON (5,5), so use reach 0.15; `fm_deploy(..., reach=0.15)`,
`grid_rollout.py:167`). Report per-γ SR/CR + a single-rollout viz (`analysis/one_rollout_viz.py --mode single`).
This is the pre-expansion baseline. **PAUSE.**

### (5) Safe Flow Expansion + the three ablation "brothers"
Run the expansion from `pretrained_sg_walls8.pt` with the **defaults above** (demo 0 / lwf 0 / rollouts 14).
Then the **brothers**, each = the SAME recipe with exactly ONE flag (methodologically clean; verified in
codex_overnight that mixing anchor-removal + flag = confound — here anchor is already 0, so it's clean):
- `--ablate-socp`     → **−SOCP** (H10 + taskspace + goal termination + progress; no SOCP check)
- `--ablate-progress` → **−Progress** (H10 + taskspace + goal termination + SOCP; no progress check)
- `--ablate-curriculum` → **−Curriculum** (unchanged full acceptance; single class, no σ easy/frontier split)
Code paths: `grid_expand_hardtail.py:1060` (socp/progress in the gather), `:323` (curriculum single-class).

**Approved bounded-sanity bootstrap amendment:** use the actual canonical OOD SafeMPPI expert
`stage_results/05_sanity/data/canonical_seed_windows.pt` (lower-left `(0.05,0.05)` → upper-right `(5,5)`,
all seven γ, 819 exact H10 windows), not the upper-left→lower-right pretraining distribution. Request
`demo_frac=.50` until the first accepted rollout, then `.25`; keep batch 16 for the sanity run, retaining
four post-latch expert rows. (`.01` would round to zero and is too drastic.) For the controlled
`−Curriculum` sanity arm, replay the full arm's exact accepted and rejected tensors at the same iterations;
remove only the easy/frontier split. This amendment does not authorize the big dive.
**Expected per-γ outcome (this is the paper claim — verify it):**
- **−SOCP → strictly LOWER min/avg clearance than ours** at every γ.
- **−Progress → LOWER time-to-goal than ours** (drops the progress gate → shortcuts, at a safety cost).
- **−Curriculum → mixed/unsatisfying on BOTH** clearance and time (unstable).
Metrics via `analysis/report_at.py` (a–d vs expert) + `analysis/per_gamma_valid.py` (valid2 rate) +
`analysis/faithful_taxonomy.py` (reach/collision/oob). **PAUSE.**

### (6) Demo expert + Kazuki baselines (rough sweep) — measure THEIR metrics on THIS scene
- **Demo expert:** `eval_ae.py expert-worker --wall-plugs 8 --start-eps 0.05 --reach 0.15 --M 100` per γ →
  `results/expert_challenging`. (Walls squeeze the expert: expect clr ~0.23–0.27, g0.1 SR<1.)
- **Kazuki:** `kazuki_baseline.py` (already patched: `--reach`, `--wall-plugs`, `--start-eps`). **Rough
  sweep** `w_safe × coll_w × goal_w` (start from `--coll-w 20 --goal-w 2 --goal-coef 0.5 --beta-mppi 20
  --reach 0.15`) to get a **decent, non-zero** result. **Expected: mode-collapse / guidance dominates**
  (its MPPI struggles in the dense 72-obstacle walled scene; on codex_overnight even the tuned w_safe.9 hit
  SR 0 / all-timeout — you must find a config with SOME success). Eval with `eval_ae.py saved-worker`.
  **PAUSE.**

### (7) Iterate (5)→(7) until you WIN
Extend the expansion (resume `--ckpt ckpt_N.pt --start-iter N`) in it20 blocks; optionally greedily sweep
β / mix / gp_buf (see `analysis/greedy_sched_driver.sh` for the pattern) and/or turn on `--emergent-gamma`
if low γ starve. **Win = beat Kazuki (certain) and the demo expert on a–e; bonus = higher coverage.**
Re-measure the baselines' metrics whenever the scene/eval changes. **PAUSE at each block.**

### (8) Final vizs (include the brothers)
Copy and re-point (they read `results/.../row_g*.json` + `paths_g*.npz`):
- **scatter** `paper_results/scatter_v4.py` (1×2 SR-CR & clearance-time phase planes; marker=method,
  color=γ **plasma_trunc** — NEVER viridis, that's σ's; series = Expert / **Our approach** (bold) /
  Pretrained / CFM-MPPI\* + **add −SOCP/−Progress/−Curriculum series**).
- **table** `analysis/make_table.py` (IEEE .tex+.md; rows demo / Kazuki / ours / **3 brothers**).
- **rollouts** `paper_results/rollouts_v4.py` (2×4: pretrained-data start+goal seed clouds+expert / expert / pretrained+fail
  inset / Kazuki+stuck inset / **the 3 brothers showing mode-collapse or unsafe** / Ours bold balanced
  modes). γ=plasma.
- **curriculum video** `video_curriculum_fixed.py` (σ=viridis; per-iter viz_db).
- **internals** `paper_results/internals_v4.py` (per-γ certified-window share = the curriculum; pools;
  fstep/loss; valid rate; σ by class).

## Metrics a–e (authoritative, from `audit_p2_goals.py`), per γ ∈ {.1,.2,.3,.4,.5,.7,1.0}
**a** SR==1 · **b** CR==0 · **c** clearance > expert (safer) · **d** time < expert (faster) · **e** coverage
> expert (distinct staircase modes; bonus). Ours must beat Kazuki on all, and the demo expert on a–e.
`traj_valid2 = in_taskspace ∧ approach_ok(net-progress≥0.10) ∧ socp_ok(γ)` (`grid_metrics2.py:60`); the
training gather additionally requires that the executed rollout terminates inside the configured goal-reach
radius in every arm. SOCP is γ-dependent (`alpha_t=(1-γ)^t`, higher γ looser). Eval reach 0.15
(goal-corner plugs block 0.1).

## Hard-won lessons from codex_overnight (read `../codex_overnight/` + its memory)
1. **Coverage collapses to one mode** (U-first only) under the frontier curriculum while the expert is
   balanced — watch mode balance (`GM.staircase_id`), it's the 'e' risk.
2. **Kazuki deploy reach was hard-coded 0.1** → SR 0 by construction on the walled goal; the `--reach` fix
   is already in `kazuki_baseline.py`.
3. **gp_buf must == qbuf_cap** or σ jitters ~15%.
4. **Goal-corner plugs put the goal ON an obstacle** → reach 0.15 (not 0.1); origin sits on the corner
   plugs → `--start-eps 0.05`.
5. **The three ablations were "too good"** in codex_overnight only because a demo+LwF anchor carried the
   safety — with **demo 0 / lwf 0 here they will show the expected degradation.** That's why this task
   sets the anchor to zero.
