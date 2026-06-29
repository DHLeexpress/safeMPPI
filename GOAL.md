# GOAL.md — Safe Flow Expansion for γ-conditioned generative path planning

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

### 2) Verified polytope loop (Pillar 2) — nominal polytope + SafeMPPI γ-rejection
Visualize the nominal polytope + nested `{H ≥ (1−γ)^i}` level sets as SafeMPPI control propagates over pedestrian
eval data; **single-integrator** robot (matches Mizuta). SafeMPPI weight = **per-sample rejection** vs the polytope
(reject rollout m if ∃i: `h^aff(x^m_{k+i}) < (1−γ)^i`) — γ-dependent.
- Nominal polytope (current best): `cfm_mppi/safegpc_adapter/polytope_v2.py` (`build_polytope_v2` — robot-centered
  K-gon disk + obstacle tangent faces, no head bias). **Can be improved → `polytope_v3.py`** (less-conservative /
  max-volume refinement).
- SafeMPPI: `cfm_mppi/safegpc_adapter/safemppi.py` (`SafeMPPIAdapter.plan`) · `barrier.py`
  (`affine_barrier_h_ho_all`) · `gamma_schedule.py` (adaptive γ) · `mirror_sampler.py`.
- Level-set viz: `track1_polytope_v2_mizuta.py` (`_norm_barrier`), `cfm_mppi/evaluation/visualize_mirror_episode.py`.
- Single-integrator: `cfm_mppi/mppi/utils.py` (`singleintegrator_dynamics`).

### 3) Certified planning (Pillar 3) — efficient verifier polytope (less conservative)
Verifier answers: *"does there EXIST a polytope with safety parameter γ such that this trajectory satisfies the
recursive DTCBF constraints ⇒ intrinsically safe for upcoming steps?"* — crucially **less conservative** than the
nominal polytope (certifies gap-threading the nominal rejects). For single-integrator there is no braking
constraint, so gap-threading is genuinely certifiable.
- Current best (**UNDER CONSTRUCTION**): `overnight_run_today/src/dtcbf.py` (`verify` — sound 2-D closed-form DTCBF
  certificate sweeping normal angles; `build_candidate_polytope` — conservative baseline). No dedicated
  verifier-polytope module yet → build an **efficient `verifier_polytope`** (note: the deleted v2 rectangle verifier
  was UNSOUND for double-integrator braking; single-integrator removes that pitfall).

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
