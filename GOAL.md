# GOAL.md — Safe Flow Expansion for γ-conditioned generative path planning

> **→ See `NOTE.md`** for the running **experiment log (trial-and-error)**, the MPPI option findings
> (guidance/temperature/γ-coupling), the dataset recipe, and dataset references. GOAL.md = the plan; NOTE.md = what
> we actually learned while doing it. Baseline (pre-experiment) code is archived under `experiments/baseline/`.

> **DEFAULT MPPI VISUALIZATION** (style of `results/benchmark_videos/di_gamma.mp4`): always show the
> **accepted (green) / rejected (red) ROLLOUT TRAJECTORIES** (not just endpoints), mark **rejected endpoints with
> ✗**, show the **accept/reject COUNT** box, over the nested `{H≥(1−γ)^i}` level sets + the executed path (red),
> robot-centered zoom. Reference impl: `overnight_run_2026-06-28/ep15_diagnostic.py::draw_cell`.

## Overarching goal
Use the **scarce, conservative SafeMPPI dataset** to train a flow-matching (FM) policy, then **expand & verify** it
with polytope-verifier ideas, to **quantitatively prove that exploration coverage increases while the generative
policy stays safe — conditioned on the safety parameter γ.** SafeMPPI's key idea: `Dohyun_ICRA2026_final/root.tex`.

**Why this matters (the tension we exploit):** in narrow gaps / compact pedestrian crowds a plain FM policy is
*too conservative* — it creeps slowly toward the goal and never actively detours or threads the gap. A γ-conditioned
verifier that certifies *"this trajectory satisfies the recursive DTCBF constraints for some polytope with parameter
γ ⇒ intrinsically safe for upcoming steps"* lets us expand the policy into the less-conservative, gap-threading
behaviors **without losing the safety guarantee**.

**Key SafeMPPI facts (`Dohyun_ICRA2026_final/root.tex`):** affine barrier
`h^aff(x_{k+i}) = nᵀ(x_{k+i}−p)/nᵀ(x_k−p)`; safety spec `h^aff(x_{k+i}) ≥ (1−γ)^i`; per-sample **rejection** of any
rollout violating it (∞ cost); constrain-then-average ⇒ the MPPI mean is provably safe (Props 1–2). **γ:** low γ →
conservative / wide margins, high γ → aggressive / tight gaps.

## Pipeline (5 pillars)
1. **Scene context** — finite-sensing Mizuta pedestrian scenes.
2. **Verified polytope loop** — nominal polytope + SafeMPPI γ-rejection propagation.
3. **Certified planning** — efficient verifier polytope (DTCBF certificate, *less conservative* than nominal).
4. **Pretrain FM policy** — γ-conditioned, on the scarce conservative SafeMPPI data (Mizuta backbone).
5. **Safe flow expansion** — ActiveFlowExpansion (Eq 9/10 + Alg 1) grows coverage under the verifier; prove coverage↑.

## Detailed steps (ordered) — with the best current code per step

### 0) Abstract demo + baseline: Eq (9)(10) + Algorithm 1 in path planning
A clean demo that **explicitly** runs ActiveFlowExpansion on a robotics path-planning toy, abstractly showing that
expansion helps the FM **policy** cover more *validated* behaviors (toward the overarching goal). Set up a
**baseline** (no-expansion / naive sampling) to compare **coverage** and **validity**.
- Algorithm 1 / Eq 9 / Eq 10: `overnight_run_today/src/safeflow.py` (`run_safeflow`=Alg 1, `active_exploration`=Eq 9,
  `update_flow`=signed grad) · `overnight_run_today/src/uncertainty.py` (`GPUncertainty.sigma`=Eq 10) ·
  `overnight_run_today/src/flow_policy.py` (`FlowPolicy.{cfm_loss,sample,phi_s}`) · verifier
  `overnight_run_today/src/dtcbf.py` (`verify`).
- Current toy demos: `overnight_run_2026-06-28/track2a_chessboard_actflow.py` (chessboard, Eq 9/10),
  `track2b_grid_coverage.py` (conditional FM, 7×7 grid path coverage), `track2b_inference_trajectories.py`,
  `lattice_paths.py`.
- **TO BUILD (see Note 3):** rebuild Step 0 around `safeflow.run_safeflow` on a single-integrator path-planning toy
  (single obstacle / narrow gap / left-right dilemma) with an explicit baseline + coverage/validity curves.

### 1) Scene context (Pillar 1) — finite sensing
Mizuta UCY setup but with a **finite sensing range** (only obstacles within R are detected).
- Loading: `cfm_mppi/evaluation/render_validation_comparison.py` (`_load_validation_scene`, `_frame_obstacles`);
  data `dataset/eval80_{ego,obs}_ucy.{pt,pkl}`. Finite sensing already honored by polytope_v2 (clearance ≤ R).
  Demo: `overnight_run_2026-06-28/track1_polytope_v2_mizuta.py`.

### 2) Verified polytope loop (Pillar 2) — nominal polytope + SafeMPPI γ-rejection  **[VALIDATED ✅]**
Single-integrator robot. SafeMPPI weight = per-sample rejection vs the nominal polytope (reject rollout m if ∃i:
`H_P(x^m_{k+i}) < (1−γ)^i`), γ-dependent; the executed action is the reward-weighted average of the survivors.

**KEY UPDATES (Stage 2, validated — full detail + the trial-and-error in NOTE.md items 11–13):**
- **MPPI done right (safeGPC `algs/mppi.py` parity):** nominal control = **0** (cold seed) + **WARM-START** the
  reward-weighted sequence `mean_new = Σ w·controls` (`w = softmax(−J/temp)`, rejected weight 0); executed action =
  the **weighted mean**, NOT the greedy argmin. We do **not** refine a goal-seeking nominal (that's Mizuta); the goal
  lives in the cost (progress + terminal). Cold nominal=0 *without* warm-start makes the first action random ⇒ robot
  oscillates — warm-start is essential.
- **Rejection = the nominal POLYTOPE level sets** (`use_polytope_barrier=True`), NOT the affine single-nearest barrier
  (jumpy/non-smooth ⇒ the "accept-0" all-rejection). Polytope built once at x0: smooth, all nearby obstacles.
- **Mean/cov from the polytope = a BIMODAL Gaussian MIXTURE over ALL H steps** (replaces the K-step nominal blend,
  which awkwardly pulled the 3rd step into the 2nd via warm-start): each control δu ~ a mix of **Mode A** `N(warm, Σ)`
  (goal-ward) + **Mode B** `N(warm + u_max·B⁺d̂, Σ_aniso)` (opening-ward), fraction `p=clip(centroid_gain·trapped,0,1)`,
  `trapped=(R−size)/(size+ε)`. **d̂ = direction to the EXACT polygon centroid** (scipy HalfspaceIntersection + shoelace
  area-centroid), NOT the analytic-center gradient. **Smoothness = a temporal low-pass on `p`** across plan steps.
  `Σ_aniso` = anisotropic ellipsoid, wide ∥ opening. `B⁺`≈`Bᵀ` direction for SI/DI (matters only for unicycle) —
  **safety comes from the clever bimodal samples**, not the B⁺ detail. *Executed (navy) ≠ centroid (orange)*: we bias
  the SAMPLING; the executed is the reward-weighted (goal-driven) mean, =centroid only in open space.
  **safety_margin = 0** (keep the per-obstacle `predict_gain` velocity inflation; the constant offset collapsed the
  polytope to ~0 in dense crowds).
- **FINAL SI config:** `centroid_gain=0.1, sigma_volume_gain=0.5, control_weight=0.03, centroid_smooth=0.5,
  sigma_aniso=2.0, predict_gain=0.4, temperature=0.3, H=10`. **300-ep/dataset × γ:** sensing=3.0/ns=128 → SDD
  90–92%/3–4% col, UCY 75–78%/6–8%; sensing=2.0/ns=512 → SDD 89–93%/4–7%, UCY 78–81%/6–13% (more reach but more
  high-γ collision). **γ = clean DTCBF conservativeness knob; SDD essentially solved, UCY the hard set.** Keep
  sensing≈3.0 for lower collision.
- **Double-integrator works** (same bimodal steering via `B⁺`; polytope rejection on the rolled-out positions). DI
  fine-tune (UCY+SDD): `cg=0.1/sv=1.0/aniso=2.5/sens=2.0` → **88% succ / 8% col / 60% acc** — reaches well but
  collides more (momentum + the position-only barrier doesn't see braking). **Next fix = a velocity-aware
  (higher-order) polytope barrier.**
- Code: `cfm_mppi/safegpc_adapter/safemppi.py` (`SafeMPPIAdapter.plan` — bimodal mixture + warm-start + exact-centroid
  steering + polytope rejection + safe fallback; `_polygon_centroid`, `_polytope_proposal`) · `polytope_v2.py`.
  Viz/sweeps in `overnight_run_2026-06-28/`: `polytope_explainer.py`, `polytope_grid.py`, `ep16_study.py`,
  `di_gap.py`, `di_grid.py`, `full_sweep.py`, `param_finetune.py`, `param_finetune_di.py`. Theory:
  `design/MEANCOV_STEERING.md`.

### 3) Certified planning (Pillar 3) — efficient verifier polytope (less conservative)
Verifier answers: *"does there EXIST a polytope with safety parameter γ such that this trajectory satisfies the
recursive DTCBF constraints ⇒ intrinsically safe for upcoming steps?"* — crucially **less conservative** than the
nominal polytope (certifies gap-threading the nominal rejects). For single-integrator there is no braking
constraint, so gap-threading is genuinely certifiable.
- **HORIZON: the verifier horizon ≡ the MPPI horizon (= H = 10).** The verifier certifies exactly the H-step
  trajectory the planner produces; the two are always equal.
- **Geometric optimization (robot-centered level sets):** robot at center `c`; polytope `P={x:aₖ·(x−c)≤bₖ}`, `bₖ>0`
  ⇒ robot interior, `H_P(c)=1`; barrier `H_P(x)=minₖ[1−aₖ·(x−c)/bₖ]`; level set `{H_P≥ℓ}` = `P` scaled by `(1−ℓ)`
  about `c`. **The slope of the nested level sets (decay rate `1−γ`) is the decision variable tied to γ.** Find
  `(A,b>0)` + minimal `γ≤γ_max` s.t. (i) safety: each obstacle excluded by some face, (ii) recursive DTCBF
  `H_P(x_{i+1})≥(1−γ)H_P(x_i)`. Per-face this is **LINEAR in `b` (and γ)** ⇒ an **LP feasibility**; minimizing γ gives
  `req_γ`. `req_γ≤γ_max` ⇒ certified (return `b*`); else **report INFEASIBILITY** (binding obstacle/step). Less
  conservative because the verifier *chooses* `b`/normals to fit the trajectory vs the nominal's fixed tangents.
- Current best (**UNDER CONSTRUCTION**): `overnight_run_today/src/dtcbf.py` (`verify` — sound closed-form DTCBF
  certificate) + new `verifier_experiment/` (LP verifier + `validate_polytope_v2` + `design/VERIFIER_GEOMETRY.md`).
  (Note: the deleted v2 rectangle verifier was UNSOUND for double-integrator braking; single-integrator removes it.)

### 4) Pretrain the FM policy (Pillar 4) — γ-conditioned, on scarce conservative data
Use **Mizuta's model as the backbone** (do NOT re-derive the flow architecture). **Do NOT adapt the part that injects
noise in the middle of the flow** (keep the CondOT path / mid-flow noise mechanism intact). Condition on **start,
goal, and γ**; train on the scarce conservative SafeMPPI dataset.

- **γ is different here:** in Mizuta's model γ is a **safety-GUIDANCE** signal (it scales reward/CBF guidance), **not
  an explicit DTCBF constraint** the way SafeMPPI's γ is. So the γ we condition the policy on tunes conservativeness
  via guidance, and must be reconciled with the SafeMPPI/verifier DTCBF γ — they are conceptually linked but enter
  through different mechanisms.
- **Data / inputs — needs an EXTRA simulation step (don't just watch ego data):** `dataset/train80_ego.pt`
  (`[273989, 9, 80]`, raw Mizuta ego trajectories) is *expert ego motion*, not the network's conditioning inputs. To
  build training pairs without confusing the inputs, add a step that **simulates each scene by DEPLOYING SafeMPPI**
  (Step 2 machinery) and records the exact conditioning tensors + the resulting safe control sequence as the target:
  `start, goal, ego_current, ego_history, action_history, nearest_obstacle_history, gamma (guidance role),
  safety_margin → safe control sequence`. **First task:** pin down the relation between `train80_ego.pt` and the
  polytope-based SafeMPPI rollout (which channels of the 9 map to start/goal/ego state; where obstacles come from).
- **CROWD DATA (critical, provenance CONFIRMED):** `train80_ego` (273,989) is Mizuta's **ego-only sliding-window
  snippets of the ETH/BIWI dataset** (Pellegrini et al., *You'll Never Walk Alone*, ICCV 2009; his paper: "276,874
  trajectories 1–8 s") — downloaded from his Drive, **no in-repo builder, crowd discarded** ⇒ it cannot supply
  SafeMPPI obstacles. Phase-4 real-crowd source = **NVlabs/trajdata** (`github.com/NVlabs/trajdata`; Mizuta himself
  acknowledges trajdata, so this reproduces his pipeline): `pip install trajdata` (ETH/UCY + SDD need no
  registration), `UnifiedDataset(desired_data=["eupeds_eth", "eupeds_zara1", …], centric="scene", desired_dt=0.1,
  only_types=[PEDESTRIAN], history_sec=(3.9,3.9), future_sec=(4.0,4.0), standardize_data=False)` → each scene's
  co-present agents are the moving obstacles. Build ego+crowd episodes (ego=one pedestrian, others=obstacles, 80
  steps → ≫300 episodes) via **new** `cfm_mppi/data/build_crowd_scenes.py` → patched `generate_guided_dataset.py`.
  (The exact ETH crowd is also recoverable from raw BIWI `obsmat.txt` by frame overlap.) **Phase-2 plot/eval uses
  UCY/SDD (`eval80_obs_{ucy,sdd}.pkl`) which already include the real crowd — no recovery needed.**
- Model (current best, **already γ-conditioned**): `cfm_mppi/models/contextual_transformer.py`
  (`ContextualTransformerModel`) via `cfm_mppi/models/context_encoder.py` (`ContextEncoder` stacks `[γ, safety_margin]`).
- Training: `cfm_mppi/training/train_safe_cfm.py` (`main`), `train_loop_safe_cfm.py` (`safe_cfm_loss`); flow path
  `cfm_mppi/flow_matching/path/affine.py` (`CondOTProbPath`); data `cfm_mppi/data/canonical_dataset.py`
  (`CanonicalDataset`). Inspect train-loss scheme + W&B (Note 2).

> **Side note — what the conditioning variable `c` actually is (Algorithm 1 writes one vague `c`; code splits it):**
> - **Mizuta / UnifiedGenRefine** (`UnifiedGenRefine_arXiv-2508.01192v3`, Alg 1): the base transformer call is
>   essentially **start + goal + flow-time only** — `model(noisy_action_seq[B,2,T], noise_level/τ, start=[B,2]
>   (≈0, ego-relative), goal=[B,2] (relative, ×1/10))`; see `cfm_mppi/evaluation/eval_utils.py:17` and the model call
>   at `eval_utils.py:52`. Everything else enters **outside** the transformer: `control_history[1,2,H≤10]` is clamped
>   into the **prefix of z_τ** (not a token); `obs_positions/obs_velocities` feed the **reward-guidance CBF gradient**;
>   `safe_margin_coefs` (len 5 → `[B,1,1]`) and `goal_margin_coef` **scale guidance**, not model conditioning. So
>   Mizuta's `c` ≈ start+goal+time, with history via the z_τ prefix and obstacles via guidance/MPPI.
> - **`ContextualTransformerModel`** (our pick): conditions on **7 explicit context tokens** `[B,7,256]` —
>   `start, goal, ego_current, ego_history(mean-pool), action_history(mean-pool), nearest_obstacle_history(mean-pool),
>   [γ, safety_margin]` — prepended to the `[B,T,256]` action tokens → `[B, 7+T, 256]`; see
>   `cfm_mppi/models/context_encoder.py:64` and `cfm_mppi/models/contextual_transformer.py:81`.
> - **Bottom line:** Mizuta-style generation ≈ "start & goal only" (obstacles/history handled around the net);
>   `ContextualTransformerModel` truly conditions on state/action/obstacle history + γ + safety_margin as tokens.
>   Choose deliberately which inputs we feed when training on SafeMPPI-simulated data.

### 5) Safe flow expansion (Pillar 5) — prove coverage↑ + safety, conditioned on γ
Run ActiveFlowExpansion on the pretrained γ-conditioned policy, gated by the Step-3 verifier, conditioned on γ.
Quantitatively prove **coverage ↑ while validity (safety) is preserved** vs the Step-0 baseline.

**Key ActiveFlowExpansion facts (`ActiveFlowExpansion_arXiv-2606.08802v1`):**
- **Eq 9 (active exploration):** `x_{t+1} ~ argmax_q E[σ_t(φ_s(x))] − β·KL(q‖p_θ)`.
- **Eq 10 (GP posterior variance):** `σ²(x) = k(x,x) − k(x,X)(K+λI)⁻¹k(X,x)` over the noised-flow representation φ_s.
- **Algorithm 1:** refit σ → explore (Eq 9) → query verifier → update buffer → UpdateFlow (`∇L⁺ − α∇L⁻`).

- Loop: `overnight_run_today/src/safeflow.py` (`run_safeflow`); σ `uncertainty.py`; verifier `dtcbf.py`. Adapt to
  drive the Step-4 `ContextualTransformerModel` + the Step-3 verifier polytope, threaded with γ.

## Notes (frequently edited)
1. **GPUs:** default to **2 GPUs** in our cluster; if the rest are free, run full.
2. **Training (Steps 4 & 5):** inspect the **train-loss scheme** and use **W&B** to fine-tune the FM policy.
3. **Step 0 status:** the current demo is the discrete-flow-matching example and is **not satisfying** — it does not
   explicitly formulate Eq (9)(10) + Algorithm 1, rests on self-made assumptions, and is *not actually discrete*.
   Redo Step 0 as an explicit Algorithm-1 robotics path-planning demo with a baseline.
