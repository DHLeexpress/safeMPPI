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

**14. Bimodal mixture proposal + exact centroid + double-integrator (`ep16_study.py`, `di_gap.py`, `di_grid.py`,
`design/MEANCOV_STEERING.md`).** Redesign of the mean/cov geometric steering:
- **d_centroid = the EXACT polygon centroid** (`scipy HalfspaceIntersection` → shoelace area-centroid), not the
  analytic-center gradient. **Proposal = a BIMODAL Gaussian mixture over ALL H steps**: Mode A `N(warm, Σ)` goal-ward
  + Mode B `N(warm + u_max·B⁺d̂, Σ_aniso)` opening-ward, fraction `p=clip(centroid_gain·trapped,0,1)`. Replaces the
  K-step nominal blend (which pulled the 3rd step into the 2nd via warm-start). **Smoothness = a temporal low-pass on
  p** across plan steps. `Σ_aniso` = anisotropic ellipsoid (wide ∥ opening). B⁺ vs Bᵀ is the same direction for SI/DI
  (matters only for unicycle) — **safety comes from the clever bimodal samples**, not the B⁺ detail.
- *Executed (navy) ≠ centroid (orange)* because we bias the SAMPLING; the executed is the **reward-weighted** mean
  (cost-driven), = centroid only in open space.
- **Sensing × rollout-count:** more rollouts → higher success (sensing 2.0: 80%→95% for 128→512 on a 20-ep subset);
  sensing should match the reachable range `u_max·H·dt≈2`. But the **full 300-ep headline (sensing=2.0, ns=512):
  SDD 89–93%/4–7%, UCY 78–81%/6–13% col** — more collision at high γ than the old sensing=3.0/ns128 config; the
  20-ep subset was optimistic. Keep sensing≈3.0 for lower collision.
- **Double-integrator works** (the bimodal steering generalizes via B⁺; polytope rejection on rolled-out positions).
  DI gap demo threads a 2-obstacle gap + detours a single obstacle. **DI fine-tune (UCY+SDD): best
  cg=0.1/sv=1.0/aniso=2.5/sens=2.0 → 88% succ / 8% col / 60% acc** — reaches well but **collides more than SI**
  (momentum + the position barrier doesn't see braking). Clean fix = a **velocity-aware (higher-order) polytope
  barrier** (next step).
- **Mechanism detail:** the EXACT centroid `C=(1/6A)Σ(vᵢ+vᵢ₊₁)(xᵢyᵢ₊₁−xᵢ₊₁yᵢ)` from the polygon vertices
  (`HalfspaceIntersection`); `d̂=(C−robot)/‖·‖` points into the opening/gap (verified on synthetic polytopes). The
  **smoothness** worry (user's "ramp-up / reinforce 1,2,3") is solved instead by the **consistent all-step mixture**
  (no step-0-heavy blend) + the temporal low-pass on `p`; the K-step blend was the culprit (warm-start pulled the 3rd
  step into the 2nd). The **cost provides the goal** (nominal=0 + warm-start; progress + terminal_goal pull to the
  goal; the centroid is only for safety/exploration).
- **B⁺ theory:** least-norm control for a position direction is `Δc=B⁺d̂`; `B⁺(SI)=(1/dt)I`, `B⁺(DI)=pinv([0.5dt²I;
  dtI])` — same DIRECTION (isotropic position block) so SI/DI ≈ `d̂`; B⁺ matters only for non-isotropic systems
  (unicycle). Cov maps the same: `Σ_u=B⁺ Σ_x B⁺ᵀ`.
- **DI eval grid (4 eps, `di_grid.py`):** the mixture ADAPTS — ep90 open (size≈1.96 ⇒ `p=0`, single goal-ward mode),
  ep150 dense (size 0.3–0.5 ⇒ `p=0.24–0.45`, opening mode active), ep16/47 between. The executed accel sits inside the
  ACCEPTED cloud (not on the centroid arrow) — the reward-weighting picks the goal-ward *safe* sample.
- **Files:** adapter `cfm_mppi/safegpc_adapter/safemppi.py`; theory `design/MEANCOV_STEERING.md`; SI viz
  `overnight_run_2026-06-28/{ep16_study.py→figures/ep16_study.gif, polytope_grid.py, polytope_explainer.py}`; DI viz
  `{di_gap.py→di_gap.gif, di_grid.py→di_grid.gif}`; sweeps `{full_sweep.py, param_finetune.py, param_finetune_di.py}`.

**15. 3-mode mixture (Mode C: always-on random/braking backup) — the DI collision fix (NO higher-order barrier).**
Two observed DI collision causes: (1) `predict_gain` too sensitive — in some frames it inflates an obstacle past the
robot → the polytope **degenerates** (a face vanishes) → the exact-centroid fails → `p_t=0` → bimodal steering OFF;
(2) **no sampling backup** — the polytope is fine but no rollout survives the rejection → fall back to warm-start MPPI.
Fix = a 3rd proposal mode, `z ~ Categorical(p_a, p_b, p_c)`:
- **Mode A** = warm isotropic (goal-ward), fraction `p_a = 1−p_b−p_c`.
- **Mode B** = centroid/opening anisotropic, `warm + u_max·B⁺d̂`, fraction `p_b = p_t = clip(cg·trapped,0,1)`.
- **Mode C** = always-on backup, fraction `p_c = random_backup_frac`. **Half BRAKING** (`u = clamp(−v/dt)`, full
  deceleration ⇒ robot brakes/backs off ⇒ displacement shrinks ⇒ H preserved ⇒ ACCEPTED) **+ half random-360°**
  (`warm − u_max·d_i`, even directions + per-plan offset, exploration). Fires EVERY frame (incl. open p_t=0 and
  degenerate-polytope frames). Verified always-on (t=0 open A=497/B=0/C=15; trapped A=417/B=80/C=15).
- **FINDING — "always ≥1 accepted" is NOT achievable for DI by sampling alone.** Iterating p_c∈{0.03..0.3} never
  reaches 0 all-rejected frames (best ~22/960 ≈ 2%, all ep16/ep90). Two reasons: (i) random-at-u_max samples move too
  far ⇒ rejected, AND they crowd out the accepted near-warm Mode-A samples (so higher p_c is WORSE, not better);
  (ii) even max BRAKING, a DI robot drifts forward `½·dt·v` (relative-degree-2 momentum), so a cornered robot near a
  moving obstacle has NO feasible control under the position-only level set ⇒ those ~2-4% frames hit the
  **safe-fallback (= execute the safest = max braking)**, which is correct. A clean guarantee would need the
  velocity-aware barrier (excluded). Braking is kept as the principled backup that IS accepted in feasible-but-tight
  frames; `p_c` is now an OAT sweep param to measure its effect on collision.
- `p_t` smoothing = receding-horizon ITERATION low-pass (`self._p_prev`), NOT horizon-step. `d̂_ctrl` = least-norm
  control for a centroid-direction displacement (B⁺d̂, magnitude u_max), not "fastest direction."
- Code: `safemppi.py` (new config `random_backup_frac`; `plan` 3-mode split + braking/random Mode C + `sample_mode`
  info). Viz: `di_grid.py` (samples colored by mode A=blue/B=green/**C=magenta** large+on-top, accepted=o / rejected=✕,
  navy ✗=executed). OAT sweep: `param_oat_di.py` (**11 params** incl. `random_backup_frac`{0,0.05,0.1,0.2}; UCY+SDD
  50 eps × γ{0.1,0.5,1.0}, each param one-at-a-time around the di_grid center; γ=0.1 extra steps).
- **OAT RESULT (50 eps/dataset × γ): combined-best DI = `predict_gain=0.6, temp=0.1, sensing=3.0, ns=512,
  centroid_smooth=0.5, centroid_gain=0.3, random_backup_frac=0.0` → 88% succ / 12% col / 60% acc.** Key trends:
  **`predict_gain` 0.0→0.6 cuts collision 17%→12%** (more velocity inflation HELPS — the OPPOSITE of the
  "too-sensitive degeneration" hypothesis; reason-1 not borne out); `temperature` lower (0.1) better; `centroid_smooth`
  0.5 > 0.0; `sensing` 2.5–3.0 (acc rises to ~60); `ns=512 > 256`; `centroid_gain` 0.2–0.3. **`random_backup_frac=0.0`
  is best — Mode C does NOT improve the aggregate** (random-u_max crowds out accepted near-warm samples; the braking
  half is REDUNDANT with the safe-fallback, which already executes max braking on all-rejected frames). So the
  production default is Mode C OFF; it stays in the code as an option / for the chart. `sigma_volume_gain`,
  `sigma_aniso`, `noise`, `centroid_eps` ~flat. **DI collision floor ≈ 12%** (the genuinely-hard moving-crowd frames;
  `predict_gain`+`temperature` are the only real levers; a velocity-aware barrier would be needed to go lower).
- **RANDOM-SEARCH RESULT (`param_random_di.py`, 160 configs = best ± 1-5 changed params; the FULL grid ≈295k configs
  ≈380 GPU-days is infeasible). The combined search BEAT the OAT best and contradicts its p_c conclusion:**
  **BALANCED DI = `centroid_gain=0.2, sigma_volume_gain=0.0, sigma_aniso=2.5, sensing=3.0, num_samples=512,
  temperature=0.1, noise=0.3, predict_gain=0.6, centroid_smooth=0.5, centroid_eps=0.15, random_backup_frac=0.2`
  → 91–92% succ / 7% col / ~60% acc** (search-set 91/7, held-out-50 92/7; base 88–91 / 9–12). **KEY INTERACTION OAT
  MISSED: Mode C (`p_c=0.2`) HELPS when paired with low `noise=0.3` + `sigma_volume_gain=0.0`** (tighter sampling +
  backup) — OAT varied `p_c` ALONE at the center (noise=0.5/sv=1.0) and wrongly concluded `p_c=0` best, so **Mode C is
  NOT useless — it helps in the right combination** (the reason the combined search was worth 5 GPU-hours). New lever:
  **`noise=0.3`** (tighter ⇒ fewer collisions); `predict_gain=0.6` / `ns=512` / `temp=0.1` confirmed everywhere.
  ⇒ **Recommended DI config = BALANCED above.**
- **(sensing, horizon) MATCHING — the DI reach is QUADRATIC, and "fully active" HURTS (`di_compare_sh.py`, 100
  eps/dataset).** SI reach `= u_max·H·dt` (linear ⇒ H=5R); **DI reach `= ½·u_max·(H·dt)²`** (quadratic ⇒ **H=10√R**;
  u_max=2,dt=0.1). The current `sensing=3/H=10` reaches only **1.0 m** ⇒ outer ⅔ of the polytope never binds
  ("partially active"). Matching H up to reach the sensing radius (`s2.0/H14`, `s2.5/H16`, `s3.0/H17`) makes the DTCBF
  fully active (acc 60%→43%) but the momentum-driven longer rollouts OVERSHOOT ⇒ over-conservative ⇒ **success
  crashes 92%→64-72%** (col falls to 4% but not worth −20% succ). Shrinking sensing to match the short reach
  (`s1.0/H10`) is catastrophic (myopic ⇒ 34% succ / 19% col). **Result (100 eps/ds): `s3.0/H10` (current) = 92/7/60
  WINS**; `s3.0/H17` = 72/4/43. So **for DI you do NOT want the DTCBF fully active** — the executed first step + the
  safe-fallback give safety, and under-reaching avoids over-conservatism. Same lesson as the SI H=15 test, now with
  the correct quadratic relation. **FINAL DI config: BALANCED @ sensing=3, H=10.**
- **H=15 PURPOSE-TUNED search (`param_random_h15.py`, 50 configs @ fixed sensing=3,H=15) — the long horizon CAN cut
  collision, but as a Pareto TRADEOFF, not a win.** The earlier H=15 failure (78%) was unfair (it reused the
  H=10-tuned config). Given its OWN tuning, the best H=15 config **r3** (`noise=0.7, centroid_smooth=0.0,
  random_backup_frac=0.0, cg=0.3, sv=1.0, ns=512, temp=0.1, predict=0.6, eps=0.05`) = **89% succ / 5% col / 51% acc**
  (100 eps/ds) vs H=10 BALANCED **92 / 7 / 60**. So H=15 is SAFER (5 vs 7 col) and STABLE (89/5 on BOTH 50- and
  100-ep sets, while H=10 varies 88/12↔92/7) but lower success + much lower acceptance (more conservative). The
  H=15-optimal config is the OPPOSITE of H=10's: **high noise=0.7 + Mode C OFF + no smoothing** — the long horizon
  itself supplies the safety, so it wants MORE exploration, not a backup. **Decision (user): keep H=10 BALANCED
  (92/7/60)** as the all-around winner; r3 stays on record as the safety-priority alternative. `di_grid.gif` unchanged.

**16. Importance sampling — GEOMETRIC (Mode B → Mode-4) + BACKUP (Mode C) attempts to keep ≥1 accepted sample/step.**
The DTCBF rejection can reject ALL N samples in a tight frame ⇒ the safe-fallback fires (execute the safest rollout).
Two paradigms to keep ≥1 accepted EVERY step (avoid the fallback entirely):
- **BACKUP (Mode C)** — always-on random/braking samples (`p_c=random_backup_frac`): ½ braking `clamp(-v/dt)` (the
  reliably-accepted half) + ½ random-360. Cheap insurance, but random-at-u_max rarely survives and braking is
  REDUNDANT with the safe-fallback ⇒ `p_c=0` best in isolation (helped only when paired with low `noise` in the
  interaction search). It does NOT guarantee acceptance (cornered DI momentum is infeasible — item 15).
- **GEOMETRIC (Mode-4 polytope-AREA importance sampling)** — `polytope_area_sampling`: instead of Mode B pointing only
  at the centroid, sample random rays INSIDE the velocity-retreated polytope (`_polytope_ray_controls`: random θ,
  radius `√U·r_max(θ)`, magnitude to reach the target over H ⇒ the Mode-B rollouts SPAN the whole safe set and land
  inside ⇒ accepted by construction). If the polytope is a half-disk, the rays span its actual radius+θ. **Smoke
  (UCY ep16, DI): standard centroid Mode-B hits min-acc/step = 0 (fallback); area sampling hits ≥3–21 (≥1 EVERY step)
  and reaches faster.** This is the principled fix — the samples ARE the safe set, not Gaussian guesses around it.
- **Urgency modes:** mode 1 `ρ=(R−size)/(size+ε)` (current, magnitude) vs **mode 4 `ρ=max(0,size_{k-1}−size_k)`**
  (SHRINK RATE — fires only when an obstacle actually starts closing the polytope, "sensitive at onset";
  `urgency_size_diff` flag, `self._size_prev` state). `p_b=clip(c_g·ρ,0,1)`, temporally low-passed (`centroid_smooth`).
- Sweep `area_sweep.py`: predict_gain × centroid_smooth × centroid_eps × centroid_gain × mode{1,4} on the
  `di_grid_current_best.gif` episodes (UCY 16,47,90,150) × γ. PRIORITY = worst min-accepted-per-step ≥ 1 over ALL
  (ep,γ,step); then collision-free success on extended duration. **Recommend `predict_gain=0` first** (less face
  retreat ⇒ bigger polytope ⇒ easier acceptance + avoids the degenerate-polytope p=0 frames). No `random_backup_frac`
  / `sigma_aniso` (polytope sampling supplies Mode-B diversity); a base `noise_sigma` is kept ONLY for goal-seeking
  (nominal=0 + no shrink ⇒ no movement otherwise).
- **SWEEP RESULT (108 cfgs, 4 ep × γ):** the PRIORITY (≥1 accepted EVERY step, all ep×γ×step) is met by **ONLY
  `predict_gain=0` + mode 4** — 8/108 (pg0.2→0/36, pg0.4→0/36; mode1→0/54, mode4→8/54). Confirms: face retreat
  (pg>0) shrinks/degenerates the polytope ⇒ all-rejected frames; the shrink-rate urgency (mode 4) is what holds the
  priority. **Extended 200-step check (best `pg=0,mode4,cg=0.1`): 9/12 collision-free + reach (ep16/47/90 ALL γ);
  ep150 COLLIDES at all 3 γ** (min-acc/step still ≥1 — the priority holds, but the accepted samples are safe vs the
  STATIC polytope, not the dense MOVING crowd). ⇒ **≥1-accepted ⇒ collision-free holds for static/sparse but NOT
  dense-moving**; ep150 needs a small `predict_gain` (tension: anticipate motion vs degenerate polytope) or a
  velocity-aware polytope. Next: UCY+SDD prevalence to see how often the static-safe-set guarantee suffices.
- **predict_gain PROBE {0.0, 0.05, 0.1} (area, mode 4, cg=0.1, 4 ep × γ × 200 steps): NO sweet spot.** pg=0.0 holds
  the priority (worst-acc=1) but ep150 collides (3/12); **pg=0.05 AND pg=0.1 BOTH break the priority (worst-acc=0)
  AND ep150 still collides (3/12 unchanged)** — worst of both. So `predict_gain` is effectively binary here: 0
  (priority, myopic) vs >0 (degenerate ⇒ fallback); even 0.1 m of retreat degenerates the polytope at the worst frame
  in a dense crowd without covering ep150's fast peds. ⇒ **ep150 needs a VELOCITY-AWARE polytope/barrier, not a
  predict_gain tweak.** Final stance: `predict=0 + mode-4 + cg=0.1` is the priority-preserving config (9/12
  collision-free; ep150 the structural exception). UCY+SDD prevalence run pending user request.
- **`urgency_floor` PARAMETER (`p_b=clip(c_g·ρ, urgency_floor, 1)`) — fixes the mode-4 deactivation, NOT the geometric
  trap.** User insight: mode-4 `ρ=size_diff→0` once trapped ⇒ `p_b→0` ⇒ Mode B switches off ⇒ no escape. Fix = a
  floor so `p_b ≥ urgency_floor` always (Mode B always points into the remaining polytope). Added `urgency_floor`
  config; `di_grid_mode4.gif` (area+mode4+predict=0+cg=0.1+**floor=0.02**) confirms **p=0.02 every frame, Mode B always
  active**, ep16/47/90 reach. BUT **ep150 still STUCK** — the GIF shows its polytope is a dead-end WEDGE pointing into
  the obstacle cluster AWAY from the goal: the floor makes the executed enter the wedge, but the wedge IS the local
  minimum. So the floor fixes the SAMPLING-deactivation bug (real) but not the GEOMETRIC trap (raising floor→0.1 still
  stuck, Mode-B acc 13). Escaping ep150 needs a NON-local lever: back out of the pocket (go away from the goal), a
  longer horizon to see past it, or temp↑/anti-stall when `size` small + no progress. `di_grid_current_best.gif` kept.

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
