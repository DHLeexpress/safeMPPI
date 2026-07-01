# Polytope SafeMPPI — mean/covariance geometric steering (code map + theory)

The proposal is **clever sampling, not a hand-set control**: nominal=0, a bimodal Gaussian mixture biased by the
polytope, then the DTCBF rejection + reward-weighting decide the executed action. Safety comes from the samples
being clever enough (explore the opening) + the recursive rejection.

## The MPPI loop — where each piece lives in `cfm_mppi/safegpc_adapter/safemppi.py`
All inside `SafeMPPIAdapter.plan`:
1. **Nominal = 0 + WARM-START** (`else:` branch after the guidance check). No goal-seeking nominal (Mizuta refines
   one; we don't, `use_goal_nominal=False`). Cold seed = 0; `warm_start` carries the previous reward-weighted
   sequence shifted one step (`self._u_prev`) so the executed 1st action stops being random.
2. **Polytope at x0** — `_polytope_proposal`: `build_polytope_v2` → faces `A x ≤ b`, robot center `c`,
   `margins=b−A@c`, `size=min(margins)` (the "trapped" indicator), and the **exact centroid `C`**.
3. **Bimodal mixture proposal** (the clever sampling) — see below.
4. **Rollout + polytope-level-set rejection** — `use_polytope_barrier`: reject sample if `H_P(x_{i+1}) <
   (1−γ)·H_P(x_i)`, with `H_P(x)=min_k (b_k−a_k·x)/margin_k` (`_polytope_H`; =1 at robot, 0 on a face). Built once at
   x0 ⇒ smooth, all nearby obstacles (NOT the old affine single-nearest barrier).
5. **Reward-weighted executed action** — `action = Σ softmax(−J/temp)·controls[:,0]` (NOT greedy argmin — that was a
   safeGPC `execute()` mistake). The full weighted sequence is stored as the next warm-start.
6. **Safe fallback** — if every sample is rejected, execute the SAFEST rollout (highest barrier `min_h`), not the
   goal-seeking one.

## The mean/cov geometric steering
- **(4) EXACT centroid `C`** — `_polygon_centroid`: `scipy.spatial.HalfspaceIntersection(A, b, robot)` → polygon
  vertices → shoelace area-centroid `C=(1/6A)Σ(vᵢ+vᵢ₊₁)(xᵢyᵢ₊₁−xᵢ₊₁yᵢ)`. `d_centroid = (C − robot)/‖·‖`.
  *(The old `−Σ aₖ/marginₖ` is the analytic-center GRADIENT — a valid direction but NOT the centroid position;
  replaced by the exact area-centroid, which is exactly computable as the user noted.)*
- **(5) Control mapping via B⁺** — the least-norm control to move the position toward `d_centroid` is
  `Δc = B⁺·d_centroid` (`B` from `_linear_matrices`). **SI** `B⁺=(1/dt)I`, **DI** `B⁺=pinv([0.5dt²I; dtI])` — the
  same DIRECTION for both (isotropic position block) so for SI/DI it ≈ `d_centroid`; B⁺ only matters for
  non-isotropic systems (unicycle). The covariance maps the same way: `Σ_u = B⁺ Σ_x B⁺ᵀ`.
- **(3)+(2) 3-MODE categorical mixture over ALL H steps** — `z ~ Categorical(p_a, p_b, p_c)`:
  - **Mode A** `N(warm[t], Σ_iso)` — goal-ward (warm-start), fraction `p_a = 1−p_b−p_c`.
  - **Mode B** `N(warm[t] + u_max·d̂_ctrl, Σ_aniso)` — opening-ward (toward centroid `C`), fraction `p_b = p_t`.
  - **Mode C** — ALWAYS-ON backup, fraction `p_c = random_backup_frac` (fixed): ½ braking `clamp(−v/dt)` (the
    reliably-accepted backup) + ½ even-spaced random-360° (escape local minima / fast objects). `safemppi.py:680–693`.
  - `p_t = clip(centroid_gain·ρ, 0, 1)`, **urgency `ρ`** has two modes: **mode 1** `ρ=(R−size)/(size+ε)` (magnitude,
    `urgency_size_diff=False`) · **mode 4** `ρ=max(0,size_{k-1}−size_k)` (SHRINK RATE — fires at the onset of an
    obstacle closing in; `urgency_size_diff=True`, `self._size_prev`).
  - **Smoothness** = temporal low-pass on `p_b` across plan steps (`centroid_smooth`, `self._p_prev`):
    `p ← (1−s)·p + s·p_prev` (matters most for DI = acceleration). Consistent ALL-step mixture (no step-0 jerk).
  - `Σ_aniso` = anisotropic ellipsoid (wide ∥ d̂_ctrl, narrow ⟂), ratio `sigma_aniso`.
- **(3b) GEOMETRIC importance sampling (`polytope_area_sampling`)** — the experimental Mode B: instead of pointing only
  at the centroid, draw random rays INSIDE the velocity-retreated polytope (`_polytope_ray_controls`: random θ, radius
  `√U·r_max(θ)`, magnitude to reach the target over H). The Mode-B rollouts **span the whole safe set and land inside
  ⇒ accepted by construction** ⇒ keeps ≥1 accepted/step (no fallback). Disables Mode C + `sigma_aniso`; a base
  `noise_sigma` is kept ONLY for Mode-A goal-seeking. See NOTE.md item 16.
- **Why executed (navy ✗) ≠ centroid (orange):** we bias the SAMPLING, not the control. The **executed = the
  reward-weighted mean** (cost = goal + clearance), pulled to the goal-ward *safe* samples; = centroid only in open
  space. The sampling mean = `warm[0] + p_b·u_max·d̂`.

## Config knobs (`SafeMPPIConfig`)
`centroid_gain` (Mode-B `p_b` gain), `centroid_smooth` (temporal low-pass), `sigma_volume_gain` (σ↑ when trapped),
`sigma_aniso` (ellipsoid anisotropy), `centroid_eps` (stability), `random_backup_frac` (Mode-C `p_c`),
`urgency_size_diff` (mode 1 vs 4), `polytope_area_sampling` (geometric importance sampling), `use_polytope_barrier`,
`predict_gain` (per-obstacle velocity inflation; `safety_margin=0`), `warm_start`, `use_goal_nominal=False`.
**Final DI config (BALANCED, 100-eps): `centroid_gain=0.2, sigma_volume_gain=0.0, sigma_aniso=2.5, sensing=3.0,
num_samples=512, temperature=0.1, noise_sigma=0.3, predict_gain=0.6, centroid_smooth=0.5, centroid_eps=0.15,
random_backup_frac=0.2, H=10` → 92% succ / 7% col / 60% acc.** SI (50-eps): `cg=0.1, sv=0.5, predict=0.4, sensing=3.0,
temp=0.3`.
