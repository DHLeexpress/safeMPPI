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
- **(3)+(2) BIMODAL mixture over ALL H steps** — each rollout's control δu is drawn from a 2-Gaussian mixture:
  - **Mode A** `N(warm[t], Σ_iso)` — goal-ward (warm-start), fraction `1−p`.
  - **Mode B** `N(warm[t] + u_max·d̂_ctrl, Σ_aniso)` — opening-ward (toward `C`), fraction `p`.
  - `p = clip(centroid_gain·trapped, 0, 1)`, `trapped=(R−size)/(size+ε)` ("1/volume"-like; 0 in open space).
  - **Smoothness regularization** = temporal low-pass on `p` across plan steps (`centroid_smooth`, `self._p_prev`):
    `p ← (1−s)·p + s·p_prev` — so `trapped` jumps don't jerk the control (matters most for DI = acceleration). The
    consistent ALL-step mixture (vs the old step-0-heavy K-blend, which pulled the 3rd step into the 2nd via
    warm-start) removes the discontinuity.
  - `Σ_aniso` = anisotropic ellipsoid (wide ∥ d̂_ctrl to explore the opening, narrow ⟂), ratio `sigma_aniso`.
- **Why executed (navy ✗) ≠ centroid (orange):** we bias the SAMPLING, not the control. The **executed = the
  reward-weighted mean** (cost = goal + clearance), so it is pulled to the goal-ward *safe* samples; it equals the
  centroid only when the centroid IS the goal-ward safe direction (open space). The drawn "sampling mean" =
  `warm[0] + p·u_max·d̂`.

## Config knobs (`SafeMPPIConfig`)
`centroid_gain` (Mode-B fraction gain), `centroid_smooth` (temporal low-pass), `sigma_volume_gain` (σ↑ when
trapped), `sigma_aniso` (ellipsoid anisotropy), `centroid_eps` (stability), `use_polytope_barrier`,
`predict_gain` (per-obstacle velocity inflation — keep; `safety_margin=0` constant offset), `warm_start`,
`use_goal_nominal=False`, `num_samples` (≥256 helps acceptance — see the sensing×rollout analysis).
Validated config (SI, 50-eps fine-tune): `centroid_gain=0.1, sigma_volume_gain=0.5, control_weight=0.03,
predict_gain=0.4, sensing=3.0, temperature=0.3, H=10`.
