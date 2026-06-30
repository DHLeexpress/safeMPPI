# NOTE.md — experiment log, datasets, references, pipeline

This is the running record of **what we actually learned** building Stage 2 (SafeMPPI γ-tunable behavior) and the
crowd dataset — the trial-and-error behind `GOAL.md`. We are in **experiment mode**: the working files
(`safemppi.py`, `polytope_v2.py`, `mppi/sweep.py`) carry experimental options; the pre-experiment baseline is
archived in `experiments/baseline/`.

## Experiment log (trial-and-error)

**1. Step-2: γ barely changes the executed trajectory on real crowds.** Visualizing single-integrator SafeMPPI on
UCY (`step2_safemppi_grid.py`) and measuring it: across γ∈{0.1..1.0} the *level-set geometry* changes (slope =
1−γ), but the *trajectory* is nearly γ-invariant (Spearman |corr(γ, clearance)| ≈ 0.04–0.10). So γ was not tuning
conservativeness.

**2. MPPI options — isolating γ backfired.** To isolate γ we turned **`use_guidance=False`, `use_aniso_cov=False`,
`safety_margin=0`** (the Step-2 plot config). This made γ *more* inert, because the guidance path carries
`margin_eff = safety_margin + margin_gain·(1−γ)` — i.e. the **margin-gain coupling is γ's only working lever**, and
guidance-off deletes it. Lesson: "isolating" γ to the DCBF rejection removes the mechanism that actually makes γ
shape behavior.

**3. Temperature was a red herring (but exposed a real bug).** Hypothesis: low MPPI temperature would let γ move
the mean. On inspection the adapter used **`action = controls[argmin(costs)]`** — pure greedy, the `temperature`
field was **never used**. We wired in the true MPPI weighting `w_m = softmax(−S_m/temp)` (= Williams/Mizuta
`exp(−S/temp)`). A temperature × noise study then showed temperature does **not** create γ-separation: higher noise
adds *random* trajectory spread (0.04→0.46) but `corr(γ,clearance) ≈ 0` at every temperature. Noise diversifies;
it does not make the diversity γ-structured.

**4. Why γ is inert (root cause).** γ enters only through the DCBF *rejection*, which removes the high-cost
(aggressive) rollouts — but those already carry near-zero MPPI weight. The weighted mean is dominated by the
goal-seeking rollouts that are feasible at *every* γ, so the action is γ-invariant. γ and the cost-driven behavior
are **decoupled**, unless the goal-optimal path itself is tight to obstacles. Single-integrator compounds this (no
momentum ⇒ the recursive DCBF rarely binds).

**5. The three γ-coupling levers (the fix space).** (a) guidance/PSF projects the nominal into the safe set;
(b) **margin-gain** `ρ_eff = r + margin + margin_gain·(1−γ)` (γ changes obstacle inflation); (c) **(1−γ)-scaled
cost** `J = J_goal + w·(1−γ)·Σ relu(−h)²` (γ in the objective, works even when the rejection never binds).

**6. Suggestion 1 — velocity-predictive margin (success fix).** Moving pedestrians caused ~38% collisions (SafeMPPI
is perfect on static obstacles). We push each obstacle face back by the predicted closing:
`b = ‖o−c‖ − ρ − κ·τ·max(0, closing speed)` (`κ` = `predict_gain`, sweepable; in `polytope_v2` arg + an obstacle
inflation in the rollout). Result: success **63% → 73%** (κ=2 best); κ≥4 over-conservative (reach drops). 99% not
reached — reactive SI vs. moving crowds is fundamentally hard (constant-velocity prediction is imperfect).

**7. Suggestion 2 — escape-bias (the γ-lever that works).** Bias the sampling mean toward the obstacle-escape (open)
direction = **negative gradient of the repulsive log-barrier** `d̂ = −Σ(1/clrⱼ)mⱼ / ‖·‖`; magnitude `‖g‖` is the
urgency (small in free space, large when tight — solves the "far-centroid" problem); for SI it is directly the
escape velocity (no extra gradient). `β = clip(escape_gain·(1−γ)·‖g‖, 0, 1)`, `u_mean = (1−β)u_nom + β·u_max·d̂`.
This finally makes γ move the trajectory: **corr(γ,clearance) 0 → +0.34** (sign then flipped to (1−γ) for the
conventional "high γ = aggressive = tight"). Caveats: at escape=1.0 success crashes to 17% (too strong → abandons
the goal) — needs a **sweet-spot gain**; and it only acts in the guidance-off path. **This is the next Stage-2 task.**

**8. Dataset journey.** `train80_ego.pt` is **ego-only** ETH/BIWI snippets (no crowd) — SafeMPPI had nothing to
avoid. The trajdata GitHub repo alone is **not enough** (it's a loader; still needs raw data). We downloaded the raw
ETH/UCY trajectories (Trajectron++ mirror), built **ego+crowd** scenes (`build_crowd_scenes.py`), cleaned + packaged
them in Mizuta's eval80 format (`package_crowd_dataset.py`). See the recipe below.

**9. ep15 diagnostic — the all-rejection is a u_max problem, fixed by an adaptive velocity-CBF
(`ep15_diagnostic_v1_adaptive.py`).** UCY ep15 = goal 8.2 m through a **43-pedestrian** crowd. Root failure: under
the **full H-step recursive DCBF check (real SafeMPPI)** with u_max=2, **all 128 samples are rejected at every step**
⇒ fallback goal-seeking ⇒ drives through a pedestrian (min_clear ≈ −0.6). Findings:
- **sensing / barrier_topk / predict_gain do NOT fix the all-rejection** (still 100% rejected) — empirically tested.
- `check_first_control_only=True` *does* (0% rejected) but it is **NOT SafeMPPI** — it drops the recursive H-step
  guarantee. **Rejected by design.**
- **Root cause = u_max** (user's insight): high u_max makes H-step rollouts travel far and approach obstacles too
  fast ⇒ the recursive DTCBF rejects everything. Under the full check, **u_max=2→100% rej, u_max=0.5→0% rej**. Fixed
  u_max=0.5 is too slow to reach 8 m, so use a **proximity-adaptive DIRECTIONAL velocity-CBF cap**: bound only the
  *approach* velocity to the nearest obstacle, `ẋ·n ≤ u_max·clip(clr/react, min_frac, 1)`, leaving lateral (dodge)
  speed at full u_max (`_approach_cap`, applied per rollout step). The cap **slope `u_max/react ≤ γ/dt`** is the
  feasibility condition — so the cap is *linked to γ*; use `min_frac=0` (stop-at-contact CBF).
- A **total-speed** cap instead *collides* (min_clear −0.2) — slowing lets **moving peds walk into the robot**; the
  **directional** cap keeps the fast lateral dodge ⇒ collision-free.
- **Proximity-adaptive variance blow-open** (`sigma_expand_gain`): widen the sampling covariance when an obstacle is
  near, so samples explore **backward/lateral** escapes instead of the "go-straight" nominal ⇒ better dodges
  (clearance +0.47→+0.58).
- **Result: the adaptive directional cap alone SOLVES ep15** — collision-free + reaches at **all γ** (min_clear
  +0.32..+0.47), **recursive check intact**. predict/control_weight/sensing keep success; **escape is now optional**
  (it over-conservatizes ⇒ no-reach at γ=0.1).
- *Caveat (unsolved):* sample rejection stays high (~85–90%, accept 0 at the densest steps) — **a property of the
  affine single-nearest barrier**, not of u_max. Velocity capping cannot fix it: an exact velocity-CBF projection
  cap (force `h_new ≥ (1−γ)h_old` per step) did **not** lower rejection (the rejection's `h_new` uses the *moved*
  obstacle position) and *worsened* collisions, so it was reverted. The heuristic directional cap is kept because it
  guarantees the **executed** action is safe regardless of sample rejection. The real lever for ≥1 feasible
  sample/step is a **smoother multi-obstacle barrier** (soft-min over obstacles instead of hard single-nearest) —
  a separate architectural change (also touches the verifier). Even 1 obstacle rejects ~98% at high noise, so it is
  the barrier's non-smoothness + wide sampling, confirmed empirically.

**Verdict table (ep15):** baseline (u_max=2 fixed) COLLIDE at every γ; **row 0 (u_max→adaptive directional cap +
variance): OK at γ=0.1, 0.3, 0.5**, and all later rows stay OK. ~~escape_gain=0.10 via check_first~~ (superseded).

**10. Param sweep on UCY+SDD (50 eps/pair, target ≥90% success & 0 collision).** The ep15-winner family
(**guidance off + escape-bias 0.05 + short sensing 2.0 + small predict_gain 0.4 + control_weight 0.10**, SI) is the
breakthrough: **0 collision on both datasets at γ≤0.7** (vs the production family which had no 0-collision pair,
20–40% collision):
| γ | UCY succ / col | SDD succ / col |
|---|---|---|
| 0.5 | 80% / **0%** | **92% / 0%** |
| 0.7 | 82% / **0%** | 90% / **0%** |
| 1.0 | 50% / 50% ✗ | 68% / 32% ✗ |
**SDD clears ≥90%+0-collision; UCY tops out at 82%+0-collision** (~18% genuinely-hard no-reach; reducing escape or
raising γ recovers reach only by re-introducing collisions — a Pareto wall). Key proofs: (a) the escape-bias
(mean-steering) provides the 0-collision safety **because the DCBF rejection saturates** (all samples rejected) in
dense crowds; (b) **γ=1.0 removes the DTCBF margin and collapses to 50% collision — the rejection is essential.**
num_samples / control_weight do not move the UCY reach ceiling. **Selected data-gen config:** the family above with
γ∈[0.1,0.7]. The UCY reach gap is the intended conservativeness that **safe-flow-expansion is meant to resolve** —
so this 0-collision generator is the right Stage-2 baseline to pretrain + expand from.

**11. Polytope-based redesign — the rejection barrier was wrong; mean/sigma steer off the polytope
(`polytope_explainer.py`).** Two corrections superseding items 9–10:
- **Rejection = the NOMINAL polytope level sets**, not `affine_barrier_h`. `use_polytope_barrier=True` builds the
  robot-centered `polytope_v2` once at x0 (K-gon ∩ per-obstacle tangent faces, **all** nearby obstacles, smooth, with
  the `predict_gain` velocity retreat) and rejects on `H_P(x_{i+1}) ≥ (1−γ)H_P(x_i)`, `H_P(x)=min_k (b_k−a_k·x)/margin_k`.
  Validated: **open space 0% reject, off-path obstacle 6%, on-path obstacle 81%** (the straight nominal drives in).
  The old single-nearest affine barrier re-picked the nearest each step and jumped → that was the real "accept-0".
- **μ from the polytope centroid, σ from the polytope size** (replaces escape-bias + the directional cap, both
  deleted). Mean: bias toward the free-space centroid via `u += centroid_gain·u_max·(Bᵀd̂)/‖·‖`, `d̂ = −Σ a_k/margin_k`
  (analytic-center gradient) — generalizes to any system via the input matrix B; symmetric polytope (open) ⇒ d̂≈0 ⇒
  no bias. σ: `σ·(1+sigma_volume_gain·clip(1−size/R,0,1))` ⇒ wide when trapped (small polytope), clipped. **With
  centroid+σ steering the on-path reject drops 81%→9% (117/128 accepted)** — the mean is redirected into the opening.
  ep15 (robot inside safety-margin-inflated peds, size≈0) stays degenerate/hard — a genuine pathological-density case.
- **Config purge:** deleted `escape_gain`, `umax_react_dist/min_frac`, `sigma_expand_gain`, `check_first_control_only`;
  added `centroid_gain`, `sigma_volume_gain`, `use_polytope_barrier`, `polytope_nbase`, `predict_gain`. The
  untested "overnight" block (`use_ho_barrier`, guidance, aniso, sets-backup, adaptive_gamma, filter, proposal_mix)
  is slated for deletion next.
- **Default viz update:** accept/reject **trajectories** with the **green accepted vivid and drawn on top of the
  reds** (previously buried). Use the `polytope_explainer.py` small-multiples panel (zoomed polytope + level sets +
  accept/reject + centroid arrow on top; control-space mean+cov on the bottom) to sanity-check each part BEFORE any
  full grid GIF.

**12. MPPI done right: nominal=0 + WARM-START, safety_margin=0, polytope mean/cov (breakthrough).** Corrections from
re-reading `safeGPC/algs/mppi.py`:
- **We do NOT refine a goal-seeking nominal** (Mizuta does; `_nominal_control = to_goal/(H·dt)` clamped to u_max
  SATURATES for all H steps and is anti-MPPI). `use_goal_nominal=False` => base nominal = **0** (cold seed).
- **WARM-START is essential** (`warm_start=True`): cold nominal=0 *every* step makes the executed action RANDOM (the
  rollout that ends goal-ward got there by the sum of 10 noisy steps, so its FIRST control is random → robot
  oscillates at the start). The fix = carry the **reward-weighted sequence** `mean_new = Σ w·controls`
  (`w = softmax(−J/temp)`, rejected samples weight 0) forward (shifted one step) as the nominal. The executed action
  is `mean_new[0]` (the weighted mean), **NOT argmin** — safeGPC's `execute()` argmin/`usingMin` was the greedy
  mistake. Then the cold seed evolves into the goal-directed solution and the robot reaches.
- **safety_margin = 0** (the constant offset). KEEP the **per-obstacle predict_gain** velocity inflation — the
  differential hyperplane retreat per obstacle (the dashed-circle viz) is the part that's liked; the constant margin
  only collapsed the polytope to ~0 in dense crowds.
- **mean/cov shift over the first K steps (NOT full H)**, weight `w0=clip(centroid_gain·trapped,0,1)`,
  `trapped=(R−size)/(size+eps)` ("1/volume"-like, eps=stability), decayed (smoothness); σ scaled by the same
  `trapped`, capped (`sigma_max_mult`). The viz draws the **sampling mean** (steered nominal[0]) — which aligns with
  the centroid arrow for SI — not the executed/safe-fallback control.
- **Mean/cov FINE-TUNE (54 configs cg×K×sv×noise, 50 UCY eps each) CHOSEN config:
  `centroid_gain=0.1, centroid_horizon=3, sigma_volume_gain=1.0, noise_sigma=0.5`
  → 78% success, 82% near (final dist<1.5), 2% collision, 86% acceptance.** On 50 *diverse* episodes **0 collision is
  not reachable** — the floor is 2% (a single episode, **ep30**, crowd 9, a moving pedestrian walks into the robot;
  more conservative cg=0.15 only trades success without removing it). Pareto: lowest-collision = this config (2%/78%/
  86% acc); highest-success = cg=0.05,K=3,sv=1.5 (82% succ / 6% col). More steering buys acceptance, costs success —
  a light nudge wins because the polytope rejection + warm-start do the work. Full config: nominal=0, warm_start,
  margin=0, temperature=0.3, sensing=3.0, predict_gain=0.4, H=10. (A 15-eps sub-sample looked 0-collision; 50 eps is
  the honest number.)
- **FULL eval-set sweep (this config × γ, ALL 300 episodes/dataset, UCY+SDD, `full_sweep.py`):**
  | γ | SDD succ/col/acc | UCY succ/col/acc |
  |---|---|---|
  | 0.1 | 73 / **2** / 49 | 50 / **2** / 41 |
  | 0.3 | 90 / 3 / 89 | 75 / 6 / 81 |
  | 0.5 | 91 / 4 / 92 | 77 / 6 / 85 |
  | 0.7 | 92 / 3 / 94 | 75 / 8 / 86 |
  | 1.0 | 92 / 4 / 94 | 78 / 8 / 88 |
  **SDD 90–92% succ / 3–4% col; UCY 75–78% succ / 6–8% col** (γ≥0.3). **γ is a clean conservativeness knob** (γ=0.1
  → 2% collision but low reach/acceptance; γ↑ → reach+acceptance up, collision up a few %) — proper DTCBF semantics.
  Far better than the production family (20–40% col); for the FM data-gen the success-only filter drops the residual
  collisions. Table saved `figures/full_sweep_table.json`.
- **PARAM fine-tune on BOTH UCY+SDD (`param_finetune.py`, 54 configs cg×sv×control_weight×K, 20-ep spread each ×
  γ{0.1,0.5,1.0}) REFINED config: `centroid_gain=0.1, sigma_volume_gain=0.5, control_weight=0.03, K=3`**
  → 82% succ / 2% col / 77% acc overall; **SDD 100%/0% at γ≥0.5, 85%/0% at γ=0.1; UCY 55–80% / 0–10% col.** Key
  refinement: **sv=0.5 (not 1.0)** — every top config has sv=0.5 (less covariance blow-up ⇒ fewer collisions);
  control_weight=0.03 (default) is fine. Success-leader: cg=0.05/sv=0.5/cw=0.15/K=3 → 85%/3%. **SDD is essentially
  solved; UCY is the hard set.** This (sv=0.5) is now the default in `polytope_explainer.py` / `polytope_grid.py`.

**13. Stage-2 retrospective — my mistakes and what we figured out.**

*Mistakes I made (and the user caught), in order:*
1. **`check_first_control_only=True`** as the "all-rejection fix" — it drops the recursive H-step guarantee, so it is
   **not SafeMPPI**. Reverted.
2. **Wrong rejection barrier** — I used the **affine single-nearest** barrier (re-picks the nearest obstacle every
   step ⇒ jumpy/non-smooth ⇒ the "accept-0" all-rejection). It should have been the **nominal polytope level sets**
   from the start (smooth, all nearby obstacles, built once at x0).
3. **Directional velocity-CBF cap** — single-nearest, SI-specific, didn't account for all obstacles; didn't align
   with the polytope idea. A dead-end (the exact-DCBF variant even worsened collisions).
4. **safety_margin too large** (0.3–0.5) — the constant offset collapsed the polytope to ~0 in dense crowds. Should
   be **0**, with the per-obstacle `predict_gain` velocity inflation doing the differential per-hyperplane retreat.
5. **Reused Mizuta's goal-seeking nominal** (`to_goal/(H·dt)` clamped to u_max) — it **saturates** at full speed for
   all H steps ⇒ every rollout is a goal-beam ⇒ anti-MPPI. **We have no nominal control.**
6. **No warm-start** — cold nominal=0 *each* step makes the executed action random (the goal-ward rollout got there by
   the sum of 10 noisy steps, so its first control is random) ⇒ the robot oscillates.
7. **Drew the executed (safe-fallback) control instead of the sampling mean** in the viz ⇒ the "centroid arrow ≠ mean
   arrow" confusion (for SI they must align).
8. **Steered only step 0** (not the first K steps) ⇒ the rollout *paths* still shot to the goal.
9. **Blocked-blend** over-committed to the centroid ⇒ tanked reach; the light *additive* blend was right.
10. **Reported 0-collision on a 15-episode sub-sample** that did not hold at 50 eps (the honest floor is ~2%).

*What we figured out (the validated design):*
- All-rejection root cause = full-H recursive check + **saturated goal nominal** + **jumpy single-nearest barrier**.
  Fix = **nominal=0 + warm-start + polytope-level-set rejection**.
- **MPPI done right (safeGPC `algs/mppi.py` parity):** nominal=0 cold seed, warm-start the reward-weighted sequence
  `mean_new=Σw·controls`, execute the **weighted mean** (not greedy argmin), goal in the cost.
- **Mean/cov from the polytope:** mean = blend warm-start with the free-space **centroid** over the first **K** steps
  (weight ∝ trapped ≈ 1/volume, decayed); σ scaled by polytope size. **safety_margin=0** + per-obstacle predict_gain.
- **Final config** `cg=0.1, sv=0.5, cw=0.03, K=3` ⇒ **SDD 90–100% / 0–4% col, UCY 75–80% / 6–8% col**; γ = clean DTCBF
  conservativeness knob. **SDD essentially solved; UCY is the hard set.** sv=0.5 beat 1.0 (less covariance blow-up).

## Datasets & authoritative references (the "solid" papers)

### ETH / BIWI Walking Pedestrians (subsets: ETH, HOTEL)
- **Paper:** S. Pellegrini, A. Ess, K. Schindler, L. Van Gool, *"You'll Never Walk Alone: Modeling Social Behavior
  for Multi-Target Tracking,"* **ICCV 2009**, pp. 261–268.
  DOI: `10.1109/ICCV.2009.5459260`. (RG copy: researchgate.net/publication/221111966)
- **Dataset (official):** ETH Zürich Computer Vision Lab — BIWI Walking Pedestrians.
  https://vision.ee.ethz.ch/datasets.html  (annotations TGZ ~397 KB; bird's-eye, manually annotated, 2.5 fps).
- Recording: 2.5 fps (dt_native = 0.4 s); ~750 pedestrian tracks (≈360 `seq_eth` + ≈390 `seq_hotel`).
- **This is the source of `dataset/train80_ego.pt`** (Mizuta ego-extracted + interpolated to dt=0.1, windowed
  1–8 s → 273,989 ego-only snippets; the crowd was discarded).

### UCY (subsets: ZARA01/02/03, UNIV = STUDENTS001/003, uni_examples)
- **Paper:** A. Lerner, Y. Chrysanthou, D. Lischinski, *"Crowds by Example,"* **Computer Graphics Forum** 26(3):
  655–664, **2007** (Eurographics).
  DOI: `10.1111/j.1467-8659.2007.01089.x`. Wiley: onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2007.01089.x
  (paywalled); Eurographics DL: diglib.eg.org/items/1cf9547e-9e21-4cd1-b327-cf98d1abc8d3.
- **Dataset (official):** UCY graphics lab — https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data.

### Raw trajectory files we actually use (free, no registration)
- Trajectron++ mirror (world-frame `frame ped x y`, all subsets), downloaded to `dataset/eth_ucy_raw/`:
  https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data
- (General multi-dataset loader, for SDD/nuScenes/Waymo later: **NVlabs/trajdata**, https://github.com/NVlabs/trajdata,
  paper arXiv:2307.13924 — note it still requires these raw files / per-dataset downloads.)

> PDFs could not be auto-downloaded (UCY behind Wiley; no stable open ETH URL) — use the DOIs / official pages above.

## Episode counts from our build (`cfm_mppi/data/build_crowd_scenes.py`, 8 s / 80-step windows, stride 4 s)
| subset | dataset | episodes | mean crowd |
|---|---|---|---|
| students001 | UCY (univ) | 1558 | 24 |
| students003 | UCY (univ) | 1140 | 24 |
| crowds_zara02 | UCY | 652 | 14 |
| crowds_zara01/03 | UCY | 289 / 289 | 10–11 |
| biwi_hotel | ETH | 163 | 14 |
| biwi_eth | ETH | 55 | 16 |
| uni_examples | UCY | 96 | 6 |

**UCY students (univ) is the largest** (2,698 episodes) > zara (1,230) > ETH (218).

## Plan (current)
- **Train** the γ-conditioned FM policy on **UCY students** (largest, densest) crowd scenes → SafeMPPI γ-rollouts.
- **Test (OOD)** on **ETH**, plus a held-out **UCY students** split (in-distribution check).
- Pipeline: `build_crowd_scenes.py` (ego + surrounding pedestrians) → offline SafeMPPI γ-sweep
  (`generate_guided_dataset.py`, SI) → `train_safe_cfm.py`. Eval/visualization separately on UCY/SDD `eval80`.

## Dataset RECIPE (packaged, Mizuta eval80 format) — `dataset/crowd/`
Built by: `build_crowd_scenes.py` (raw → ego+crowd episodes) → `package_crowd_dataset.py` (clean + Mizuta format).

**File format (exactly like `eval80_ego/obs`):**
- `<split>_ego.pt`  : tensor `[N, 6, 80]` — channels `[x, y, vx, vy, sin(heading), cos(heading)]`, dt=0.1, 8 s.
- `<split>_obs.pkl` : `list[N]` of tensor `[1, N_ped, 6, 80]` — same 6 channels per surrounding pedestrian;
  **NaN where a pedestrian is absent at a step** (downstream filters NaN, like `eval80_obs`).
