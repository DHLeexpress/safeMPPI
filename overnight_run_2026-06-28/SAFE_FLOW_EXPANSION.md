# Safe Flow Expansion — explicit formulation

The objects, the verifier, and the loop, written so they map 1:1 onto the code. Scene-agnostic (single,
gap, clutter all use the same definitions).

## Objects
- **Scene / context** `c = (x0, goal, O)` with obstacles `O = {(o_j, r_j)}`.
- **Design** `U = (u_0,…,u_{T-1}) ∈ R^{T×2}` (control sequence); the rollout `ξ(U;x0)={x_i}` is the
  double-integrator trajectory; `p_i` its position.
- **FM policy** `q_θ(U | c)` — conditional flow-matching (lightened Mizuta backbone); sampled by Euler ODE
  from noise (no MPPI, no rejection).
- **Sensing ball** `B(p, R) = {q : ‖q−p‖ ≤ R}` (robot-centered).
- **Barrier of a polytope** `P`: `H_P` normalized so `H_P(ref)=1`, `H_P=0` on `∂P`; nested level sets
  `{H_P ≥ (1−γ)^i}`.

## Verifier (the gate)  `V_γ(U) ∈ {0,1}`
`V_γ(U) = 1` iff the trajectory is **collision-free**, **goal-reaching**, and **per-step certifiable**:
for (almost) every step `k`, taking the forward segment `S_k = {p_i ∈ ξ : k ≤ i, p_i ∈ B(p_k,R)}`,

> **∃** a polytope `P*` inside `B(p_k,R)`, with faces tangent to / separating the obstacles, such that every
> segment state satisfies the DTCBF ruler `H_{P*}(p_i) ≥ (1−γ)^i · H_{P*}(p_k)`.

(Current instantiation: `P*` searched as an oriented rectangle, outermost-state-seeded, warm-started across
steps — `rectangle_verifier.verify_trajectory`. The existence question is what matters; the rectangle is one
parameterization and is the candidate for replacement by a max-volume convex region.)

`Ω*_γ(c) = { U : V_γ(U)=1 }` — the **verified-safe** (less-conservative) design set. It is strictly larger /
less conservative than the set induced by the single deterministic candidate polytope.

## The loop  (Safe Flow Expansion)
```
input:  seed policy θ_0 (conservative; CFM-fit to wide-berth, polytope-respecting demos)
        verifier V_γ ,  rounds T ,  buffer D_0 = {verified-safe conservative demos}
for t = 0 … T-1:
    (1) PROPOSE   candidates  U^(1..N) ~ q_{θ_t}(·|c)  (exploratory temp)  ∪  broad surrounding proposal B(c)
                  -- covers the constrained space so q may deviate from the prior (finite-β Eq.9 spirit)
    (2) GATE      y^(i) = V_γ(U^(i))            -- per-inference-step rectangle verifier, warm-started
    (3) BUFFER    D_{t+1} = D_t ∪ { U^(i) : y^(i)=1 }
    (4) UPDATEFLOW θ_{t+1} = argmin_θ  E_{U ~ D_{t+1}}  ‖ v_θ(U_τ, τ, c) − (U − noise) ‖²   (CFM on verified-safe)
return θ_T
```
Three things co-evolve: `θ_t` (policy), the candidate pool (its own samples), the buffer `D_t`. The verifier is
the only safety authority; only `V_γ`-certified samples ever train the policy.

## What success means (measured, scene-agnostic)
- **diversity ↑**: number of distinct homotopy classes generated (sign vector of which side of each obstacle the
  path passes) + Vendi score on the trajectory descriptor (lateral profile at fixed longitudinal slices).
- **conservativeness ↓**: median min-clearance of generated valid trajectories decreases (paths hug closer),
  while still `V_γ`-certified.
- vs. the **conservative polytope-induced** policy: expanded policy covers more homotopy classes and lower
  clearance — i.e. it learns the less-conservative, multi-modal verified-safe set `Ω*_γ`.

## Known strain points (motivating the backup-strategy plan)
- The candidate/verifier polytope is currently an axis-rectangle that can exceed the sensing ball and pre-commits
  a side; in clutter (many obstacles) a single rectangle is a poor fit for the local free space.
- A fully *curvy* generable policy may not be certifiable by any single convex polytope per step.
- Direction: replace the rectangle with a **max-volume convex region inside the ball** whose faces are defined by
  obstacle tangencies (IRIS / FIRI / convex-decomposition via an external library), points strictly inside the
  ball (not required to touch it). Planned separately.
