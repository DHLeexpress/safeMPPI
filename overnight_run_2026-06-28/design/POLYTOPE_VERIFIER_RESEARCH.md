# Polytope/Verifier research study (multi-agent workflow output)

> 8 agents, ~325k tokens. Report below, then the adversarial review that corrects it.

# === DESIGN REPORT ===

Based on the research dossier and the actual current implementation (`polytope.py`, `rectangle_verifier.py`), here is the design report.

---

# DESIGN REPORT — Safe-Region Module & Verifier for Flow-Matching + DTCBF (cfm_mppi / SafeFlow)

## 1. TL;DR recommendation

Replace the single axis-aligned/oriented rectangle with a **three-layer stack** whose hard guarantee rests on the bottom layer, not the geometry:

1. **Region primitive = FIRI max-volume convex polytope**, seeded at the robot, **clamped to the sensing ball** (ball added as 8–16 half-spaces). Carry its **max-volume inscribed ellipsoid (MVIE)** `E={Bu+d:‖u‖≤1}` alongside the `{Ap≤b}` polytope. This single change fixes problem (a) and is the least-conservative *single* convex region, real-time in 2-D (FIRI's analytic max-area ellipse) and 3-D (SOCP). FIRI: https://arxiv.org/abs/2403.02977

2. **Verifier = predictive-safety-filter existence LP** over the forward segment (not an orientation search): "does there exist a control sequence keeping each forward state in its active region, satisfying the DTCBF ruler, ending in a control-invariant set?" Because the double integrator is affine and `h` is affine/quadratic, this is an LP/QP/SOCP, warm-started receding-horizon. For sample-gating it degrades to a batched feasibility *evaluation* (microseconds) — exactly your current rejection mask. Wabersich–Zeilinger PSF: https://arxiv.org/abs/1812.05506

3. **Backup = sequence of overlapping convex regions (safe corridor) + minimal-norm QP projection.** A head-on obstacle and a curvy multimodal policy are **theorem-level impossible** for any single convex set containing the robot (problems b, c); the union-of-convex corridor is the only class that breaks this. The unconditional hard guarantee is the per-step QP projection onto the active region's half-spaces with a control-invariant terminal set — it always returns a safe control (braking is always feasible).

Carry the DTCBF on the **MVIE ellipsoid barrier** `h(x)=1−‖B⁻¹(p−d)‖²`, whose super-level sets `{h≥(1−γ)^i}` are *exactly* the nested shrunken ellipsoids the ruler asks for. Use HOCBF with **separate γ for position vs. velocity** (relative degree 2), not one γ=0.5.

Net: corridor = where you may go (multimodality); MVIE/polytope = least-conservative certified region per segment; DTCBF/QP = the per-step hard enforcer.

## 2. Region-class decision (trade-off table)

n∈{2,3}, m = obstacle half-planes in the ball, T = horizon, K = orientation samples.

| Class | Conservatism | Curvy / multimodal? | DTCBF-ruler fit | Warm-start | 2-D / 3-D complexity | Real-time | Hard guarantee |
|---|---|---|---|---|---|---|---|
| **(i) Rectangle** (current) | worst (wastes corners vs. ball & angled obstacles) | **no** (b,c fail) | piecewise-linear, kinked, poor on velocity sets | orientation grid — **non-convex** outer loop | K·GP; 3-D empty-box **NP-hard-ish** | 2-D ok / 3-D poor | ok but corners exceed ball ⇒ certifies unsensed space (bug a) |
| **(ii) MVIE ellipsoid** | best *smooth*; John sandwich `E⊆F⊆√n·E` ⇒ ≤√2 (2-D)/√3 (3-D) linear loss | no | **best** (concentric scaled level sets = the ruler, literally) | excellent, **rotation-free** (deform B,d) | O(m) 2-D analytic / SOCP 3-D | yes | strong; never exceeds ball |
| **(iii) FIRI polytope** | best *single* convex (max volume, hugs gaps, seed-contained) | no (b fails) alone | exact `h=min_j(b_j−a_jᵀp)`, piecewise-linear (+MVIE inside) | good (monotone inflation reuse) | O(d) linear; 2-D analytic / 3-D SOCP | **yes (FIRI)** | strong; FIRI guarantees robot ∈ region |
| **(iv) Corridor / GCS tube** | **best overall** (union ≈ true free space) | **yes** (a,b,c) | active-region barrier + overlap handoff | excellent (region + window reuse) | O(T·m) regions; GCS MICP offline | online corridor **yes** / GCS offline | **strongest** (+ tube tightening) |

John ellipsoid bound: https://en.wikipedia.org/wiki/John_ellipsoid · MVIE SDP: https://web.cvxr.com/cvx/examples/cvxbook/Ch08_geometric_probs/html/max_vol_ellip_in_polyhedra.html · empty-box hardness: https://arxiv.org/abs/1803.00849

**Decision:** the rectangle (i) is dominated on both conservatism and 3-D tractability — demote to a Chebyshev seed only. (ii) and (iii) are each *necessary components* but neither passes a head-on obstacle. **Choose (iv)**, built from FIRI polytopes (iii) carrying MVIE barriers (ii). It is simultaneously least-conservative, the only multimodal-capable class, and strongest-guarantee — the only class meeting all of (a)–(e).

## 3. Recommended safe-region MODULE

**Representation.** Keep the existing `Polytope` dataclass interface (`A,b,ref`, `margins`, `contains`, `barrier`) — it is consumer-agnostic and correct. Add fields `B,d` (MVIE) and a method `ellipsoid_barrier(p)=1−‖B⁻¹(p−d)‖²`. Emit H-polytope `{Ap≤b}` ∩ ball, plus its inscribed MVIE. Keep `{Ap≤b}` for the feasibility certificate (least-conservative free region), use the MVIE for the smooth DTCBF and as a whitening map `p=d+Bu` to push flow proposals inside (raises the ~1% accept rate noted in `POLYTOPE_IDEA.md`).

**Computation (FIRI, with fallbacks):**
1. **Seed:** Chebyshev-center LP `max r s.t. aᵢᵀx_c + r‖aᵢ‖ ≤ bᵢ` (pure LP, sub-ms; how IRIS/FIRI initialize; warm-start carrier). https://en.wikipedia.org/wiki/Chebyshev_center
2. **Inflate:** FIRI alternation — Restrictive Inflation (one separating hyperplane per obstacle at the closest point on the current ellipsoid, forced to contain the robot seed; this is what `build_nominal_polytope` already does for the SFC case) + MVIE. FIRI's 2-D path is the first **linear-time analytic** max-area ellipse (no SDP); 3-D is an SOCP.
3. **Single obstacle ⇒ one half-space** (clean fallback, already present).

**Ball clamp (fixes a):** append an inner polygonal approximation of the radius-R ball as K≈8–16 half-spaces `aₖ=(cosθₖ,sinθₖ), bₖ=aₖ·p_robot+R`. FIRI seed-containment then guarantees robot ∈ region ⊆ ball — the region can never certify unsensed space.

**External libraries (pick by latency budget):**
- **FIRI / GCOPTER** (recommended, real-time C++, has `firi` + SFC headers): https://github.com/ZJU-FAST-Lab/GCOPTER · lab: https://github.com/ZJU-FAST-Lab
- **Drake** (one-stop Python `pydrake`, maintained, 2-D/3-D): `HPolyhedron`, `Hyperellipsoid`, `Iris`, and `HPolyhedron.MaximumVolumeInscribedEllipsoid()` (the exact MVIE SDP). https://github.com/RobotLocomotion/drake · docs https://drake.mit.edu/pydrake/pydrake.geometry.optimization.html
- **Liu DecompUtil** (header-only C++, fastest, point-cloud-native, native 3-D, drop-in for your tangent-hyperplane code): https://github.com/sikang/DecompUtil
- **iris-distro:** https://github.com/rdeits/iris-distro
- **Prototyping (pure Python):** `cvxpy` `cp.log_det`+`cp.SOC` for MVIE, `scipy.optimize.linprog` for the Chebyshev LP — adequate at n=2,3 with few half-planes; swap to FIRI for production.

Recommendation: **FIRI (via GCOPTER) for production**, `build_nominal_polytope` kept as the zero-dependency fallback, `pydrake` MVIE for the smooth barrier. All three emit the same `{A,b,B,d}`, so the verifier/backup are untouched.

## 4. Recommended VERIFIER

**Existence check (reframed from orientation search to a predictive-safety-filter LP).** Given x₀=(p₀,v₀), active regions `Fᵢ={Aᵢp≤bᵢ}`, rate γ, horizon T:

```
find  U=(u_0..u_{T-1}), states x_1..x_T   (double-integrator rollout, linear in U)
s.t.  A_i p_i ≤ b_i                       (containment in active region)
      h(x_{i+1}) ≥ (1-γ) h(x_i)           (DTCBF ruler — linear in U if h affine)
      u ∈ [u_min,u_max]                   (actuation)
      x_T ∈ C_inv                         (terminal control-invariant set)
```

Affine dynamics + affine/quadratic `h` ⇒ **LP** (affine h) or small **QP/SOCP** (ellipsoidal h). Two modes:
- **Sample-certification (gates the FM update):** fix U = flow sample, *evaluate* the inequalities — pure feasibility, batched in torch, microseconds. This is your existing rejection mask in `safemppi.plan`; reuse it verbatim, just feed FIRI faces instead of rectangle faces.
- **Region-existence (replaces the orientation loop):** solve the LP for the FIRI region around x₀; feasibility ⇔ "a certified region exists this step," with witness U.

**DTCBF tie-in.** Carry the ruler on the **MVIE ellipsoid barrier** `h(x)=1−‖B⁻¹(p−d)‖²`: `{h≥(1−γ)^i}` are concentric ellipsoids scaled by √(0.5^i), so for γ=0.5 the ruler is an exact conic check, smooth and well-conditioned. Affine polytope barrier `h_j=(b_j−a_jᵀp)−η·a_jᵀv` (already in `barrier.py:affine_barrier_h_ho_all`) remains the cheap path. **Relative-degree-2:** use HOCBF with **separate position/velocity decay rates**, not one γ — a single γ on a rel-deg-2 system is a documented conservatism source (https://arxiv.org/pdf/2503.15014). Promote your `gamma_schedule.py` to a **per-segment γ decision variable** in `[γ_min,γ_max]`; RT-CBF gives a necessary-and-sufficient invariance guarantee under bounded γ-adaptation (https://arxiv.org/abs/2303.12966).

**Terminal control-invariant set C_inv.** For the linear DI with box actuation, the maximal control-invariant set inside a polytope is itself polyhedral (iterate the one-step pre-image offline; conservatively, a braking set: states drivable to v=0 within the region under u_max). Anchoring "reach C_inv within T" is provably **less conservative than the monotone (1−γ)^i ruler**, which over-constrains by forbidding legitimate dip-and-recover (https://arxiv.org/html/2605.05575). Keep the ruler as the cheap default; add C_inv as the terminal constraint on hard maps.

**Per-step + warm-start.** Per inference step: build region(s) at x₀ → batched sample-certify the U-batch → gate Safe-Flow-Expansion → run QP projection if a hard guarantee on the applied action is needed. Warm-starts that kill the orientation-search cost:
- Region: carry `(A,b,B,d)` + Chebyshev seed across steps; FIRI re-inflation converges in 2–3 alternations (obstacles move little). Reuse your existing clearance-sorted obstacle ordering as the hyperplane-generation order.
- Verifier LP: warm-start OSQP/Clarabel with the previous witness U shifted one step (receding-horizon; 1–2 active-set changes). OSQP https://osqp.org · Clarabel https://github.com/oxfordcontrol/Clarabel.rs
- γ: carry per-segment γ.

This replaces "warm-started over orientations" (non-convex outer loop) with "warm-started convex region + warm-started convex LP" — both convex/monotone, both real-time.

**3-D.** Dimension-agnostic: FIRI SOCP MVIE; `DecompUtil` `EllipsoidDecomp3D`; Drake IRIS/MVIE 3-D; ball-clamp → polyhedral sphere inner-approx; barrier identical with B∈ℝ³ˣ³. The DI state becomes (p∈ℝ³,v∈ℝ³); `safemppi._step`/`_linear_matrices` add z rows trivially; the LP/QP grows from 2 to 3 spatial dims — negligible.

## 5. BACKUP STRATEGY (curvy/multimodal policies not single-convex-certifiable)

Three nested fallbacks; escalate only as needed. **(b)/(c) are theorem-level — the guarantee must rest here, not on §3/§4.**

**5.1 Sequential convex regions / safe corridor (primary backup, fixes b,c).** Do not require one region to contain the whole forward segment. Build a chain of overlapping FIRI polytopes along the sample's own waypoints `F_0,F_1,…` (SFC-style, DecompUtil), and let the **active region switch along the horizon**: forward state `x_i` is certified against `F_{r(i)}` (the region containing p_i with max margin), with consecutive segments required to share a non-empty **overlap** `F_{r(i)}∩F_{r(i)+1}≠∅` (one LP) so the hand-off is provably safe. The ruler is enforced w.r.t. the active region. A head-on obstacle is now passable because the *union* is non-convex; different homotopy classes = different region sequences. This formalizes what `_sets_backup_controls` is reaching for. Liu SFC: https://ieeexplore.ieee.org/document/7839930/

**Multimodal curation (offline):** run **GCS** (Drake `GcsTrajectoryOptimization`) offline to enumerate homotopy classes natively and curate the Safe-Flow-Expansion *training distribution* — train toward all certifiable modes instead of rejecting them. Not per-step. GCS: https://arxiv.org/pdf/2101.11565 · https://github.com/RobotLocomotion/gcs-science-robotics

**5.2 Safe tube (robustness certificate, hard closed-loop guarantee).** Wrap a tube so the *realized* closed-loop state stays in the corridor: for the linear DI use a **tube-MPC RPI tube** (offline RPI set, online QP) and tighten each region via Minkowski difference, requiring `(nominal ⊕ tube) ⊆ F_{r(i)}`. For nonlinear 3-D, a **trajectory-agnostic RCCM tube** (one offline SOS synthesis valid around any curvy U). This converts "plan in free space" into a guarantee under model error/disturbance. RCCM: https://arxiv.org/abs/2109.04453 · https://github.com/boranzhao/robust_ccm_tube

**5.3 Per-step QP projection (terminal hard guarantee, preserves expressiveness).** `safemppi.safety_filter_action` already projects onto active HO-DCBF half-spaces. Upgrade: (i) **minimal-norm QP** (OSQP) onto the active region's half-spaces instead of Jacobi sweeps; (ii) **C_inv terminal** so it never reports infeasible when braking exists; (iii) **path-consistent intervention** (PACS-style) — prefer slowing *along* the policy's intended path over steering off it, changing speed/timing not homotopy, so multimodality survives and the corrected control stays in-distribution for the flow. PACS: https://arxiv.org/html/2511.06385

**Escalation per step:** (a) single FIRI region certify → else (b) corridor/tube certify → else (c) minimal-norm QP onto active region + C_inv. (c) always returns a provably safe control, so the guarantee is **independent of whether the flow sample was certifiable**.

## 6. Migration path from the current rectangle

Current state: `rectangle_verifier.py` does orientation search (`certify`/`certify_fast`, K=8–24 angles, warm-started theta) building `build_rectangle_polytope`; `polytope.py` `build_nominal_polytope` already builds the SFC tangent-hyperplane polytope. The `Polytope` interface is already correct and consumer-agnostic.

1. **Extend `Polytope` (`polytope.py`):** add `B,d` fields and `ellipsoid_barrier(p)=1−‖B⁻¹(p−d)‖²`. Keep `barrier` (LSE) for the affine path. Non-breaking.
2. **New `region.py`:** `build_firi_region(pos, obstacles, R, seed) -> Polytope(A,b,B,d)`; backends `firi|drake|sfc|cvxpy`; Chebyshev-LP seed; ball-clamp. `sfc` backend = today's `build_nominal_polytope` + an MVIE solve, so the zero-dependency path works immediately while FIRI/Drake are wired in.
3. **New `verifier.py`:** `certify_samples(U_batch, regions, gamma, x0) -> mask` (batched torch; lift the ratio math straight out of `certify_fast`/the `safemppi.plan` rejection loop — it already computes `req = max(1−H[i+1]/H[i])` vs. γ); `exists_region(x0, region, gamma, T, C_inv) -> (feasible, U_witness)` (OSQP/cvxpy LP, warm-started). **Delete the orientation loop** — the rectangle's worst feature — replaced by the convex region build + convex LP.
4. **New `corridor.py`:** `build_corridor(waypoints, obstacles) -> [Polytope]` (FIRI/DecompUtil chain) + overlap-LP check + active-region selector `r(i)`.
5. **Reuse + upgrade `safemppi.safety_filter_action`:** Jacobi → OSQP minimal-norm QP + C_inv terminal + path-consistent objective (§5.3).
6. **Offline:** Drake GCS to curate the multimodal training set.
7. **Deprecate** `rectangle_verifier.py` / `polytope_v1.py` to a `legacy/` shim once `region.py`+`verifier.py` pass parity (same accept-rate floor, higher certified volume). Keep the rectangle as a Chebyshev seed only.

Because every backend emits the same `{A,b,B,d}` and the verifier already reduces to your existing batched ratio test, this is incremental: each numbered step is independently shippable and testable against the current rectangle accept-rate.

## 7. Open risks

- **FIRI is a local maximizer** (seed-dependent), not a global largest region; a bad seed can yield a thin polytope. Mitigation: Chebyshev seed + warm-start from the previous step; fall back to the SFC cut if MVIE volume drops below a threshold.
- **Discrete-time CBF validity.** The continuous-time barrier implemented in discrete time needs a margin to stay valid (https://arxiv.org/pdf/2404.12329). Validate the `(1−γ)^i` ruler with a per-step margin tied to `u_max·Δt²`.
- **Corridor region-assignment `r(i)` is greedy** and can thrash near overlaps; the overlap-LP guarantees safety but not optimal mode selection. GCS offline mitigates by curating training modes; monitor switch frequency.
- **C_inv computation cost / shape.** Exact maximal control-invariant set iteration can be expensive; the braking-set inner approximation is safe but conservative. Acceptable as a terminal anchor; revisit if accept-rate suffers.
- **GCS MICP latency** is offline-only; do not put it in the per-step loop. Risk is staleness if obstacles move faster than the curation cadence.
- **Tube width vs. real disturbance.** RPI/RCCM tubes assume bounded disturbance; an underestimated bound breaks the hard guarantee. Size the tube from measured residuals (consider a learned residual model, https://arxiv.org/pdf/2311.05802) with a safety factor.
- **MVIE conditioning** as B→singular in thin corridors makes `B⁻¹` blow up; clamp the smallest semi-axis and fall back to the affine polytope barrier there.
- **3-D real-time budget** is asserted from UAV literature, not measured in cfm_mppi; benchmark FIRI 3-D + LP at target rate before committing the SOCP path.

Key URLs: FIRI https://arxiv.org/abs/2403.02977 · GCOPTER https://github.com/ZJU-FAST-Lab/GCOPTER · DecompUtil https://github.com/sikang/DecompUtil · Drake https://github.com/RobotLocomotion/drake (opt docs https://drake.mit.edu/pydrake/pydrake.geometry.optimization.html) · IRIS https://groups.csail.mit.edu/robotics-center/public_papers/Deits14.pdf · GCS https://arxiv.org/pdf/2101.11565 · MVIE/John https://en.wikipedia.org/wiki/John_ellipsoid · PSF https://arxiv.org/abs/1812.05506 · MPC-MCI https://arxiv.org/html/2605.05575 · RT-CBF https://arxiv.org/abs/2303.12966 · DTCBF practice https://arxiv.org/pdf/2404.12329 · RCCM tube https://arxiv.org/abs/2109.04453 · PACS https://arxiv.org/html/2511.06385 · Safe Flow Matching https://arxiv.org/html/2504.08661v2 · OSQP https://osqp.org

Relevant files: `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/polytope.py`, `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/rectangle_verifier.py`, `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/barrier.py`, `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/safemppi.py`, `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/gamma_schedule.py`.


# === ADVERSARIAL REVIEW (corrections) ===

## Adversarial Review — SafeFlow Design Report vs. actual `cfm_mppi` code

I read `polytope.py`, `rectangle_verifier.py`, `barrier.py`, `safemppi.py`, `gamma_schedule.py`. The report is well-organized and the ball-clamp + corridor instincts are correct, but several of its load-bearing guarantee claims do not survive contact with the implementation or with the math. Attack by axis, then a concrete fix list.

---

### (a) Does it GUARANTEE forward invariance? No — it asserts a guarantee it never constructs.

The report says the hard guarantee "rests on the bottom layer" (per-step QP projection onto the active region with a control-invariant terminal set, "braking is always feasible"). Every clause of that is currently unbacked:

1. **C_inv does not exist.** §4 and §5.3 lean the entire guarantee on a terminal control-invariant set, but §4 itself downgrades it to "iterate the pre-image offline; conservatively a braking set… add on hard maps," and §7 lists its cost as an open risk. Nothing in the codebase computes it, and the QP in §5.3 is never constrained to reach it. Without a constructed, verified C_inv and a QP that targets it, there is no recursive feasibility, hence no forward invariance. The guarantee is named, not proven.

2. **"Braking is always feasible" is false for the actual plant.** The system is an input-bounded double integrator (`_step`, `u_max=2.0`). Stopping distance is `v²/(2·u_max)`. If the robot enters a state with clearance < stopping distance — exactly the regime where the filter is invoked — *no* admissible control avoids collision. "Always returns a safe control" only holds if the state is already inside the braking-invariant set, which is precisely the invariant the report assumes rather than enforces. The argument is circular: invariance is guaranteed by the filter, the filter is feasible only under invariance.

3. **The actual enforcement is not a guarantee, even empirically-as-coded.** In `plan()` (safemppi.py:638-642), if every sample violates the DTCBF, `costs = raw_costs + 1e4*infeasible` and `argmin` returns the *least-bad infeasible* control. `safety_filter_action` is the only hard path, and it is gated behind `filter_output AND use_ho_barrier` (line 644) — both default `False`. Worse, the filter explicitly *cannot* certify: the code comment at safemppi.py:334-335 says clamping to `u_max` "may reintroduce a deficit," and it returns `filter_feasible=False` in that case. So today's "rejection mask" the report proposes to reuse "verbatim" provides no invariance.

4. **The barrier the report builds on is not invariant for this plant.** Two compounding gaps:
   - The MVIE barrier `h(p)=1−‖B⁻¹(p−d)‖²` is **position-only on a relative-degree-2 system**. Its super-level sets are not forward-invariant under the DI without a HOCBF lift. The report flags rel-deg-2 for γ but then claims the ellipsoid level sets are "exactly the nested… ruler the system asks for" — conflating geometric nesting with dynamic invariance. They are not the same.
   - The legacy barrier actually used (`affine_barrier_h_ho_all`) freezes the separating normal and `nearest0` at **x₀** (barrier.py:152-157). It is a tangent half-space linearized at the initial state, held fixed across the whole rollout. The `(1−γ)^i` ruler is enforced against a *stale* plane, not the true obstacle set. For curved/lateral motion this can certify states that are unsafe w.r.t. the real circle.

5. **Discrete-/inter-sample gap.** The ruler is checked only at `dt=0.1` nodes. Continuous DI trajectories can dip below `h=0` between nodes. §7 mentions this but the design adds no required inter-sample margin (e.g. `u_max·dt²` tube) as a *constraint* — it stays a "validate later" note.

6. **One thing it does fix:** the ball-clamp is a real bug fix — `rectangle_verifier._extents_analytic` line 111 literally admits "corners may slightly exceed R," i.e. the current verifier certifies unsensed space. The K-face inner-ball clamp + seed-containment legitimately closes that. Credit where due.

**Verdict (a): empirical only.** As specified it gives no forward-invariance certificate; the certificate it claims requires C_inv construction + input-bounded DTCBF-validity verification + HOCBF lift + inter-sample margin, none of which are present.

---

### (b) "Less conservative" — partly hand-wavy, and one claim is self-contradictory.

- **The √n John bound is cited misleadingly.** `E ⊆ F ⊆ √n·E` bounds the MVIE against *its own enclosing polytope*. The conservatism that matters is region vs. *true free space*, and a single convex region in a non-convex / dynamic scene can be arbitrarily (down to zero-volume) conservative — the report itself concedes single-convex fails head-on. So "≤√2/√3 linear loss" is a true-but-irrelevant number dressed as the headline conservatism guarantee.
- **C_inv "provably less conservative than the ruler" contradicts the recommendation to keep both.** Terminal-set anchoring is less conservative than the monotone ruler *only if you replace the ruler with a terminal-only constraint*. §4 says "keep the ruler as the cheap default; add C_inv as the terminal constraint." Ruler ∧ C_inv is strictly *more* constraints than ruler alone → strictly more conservative. The dip-and-recover benefit is forfeited the moment you keep the monotone ruler. Internally inconsistent.
- **The barrier-level-set math is simply wrong.** For `h=1−‖u‖²`, `{h ≥ (1−γ)^i}` is `{‖u‖ ≤ √(1−(1−γ)^i)}`, whose radius **grows toward 1** as i increases — the admissible set expands to the boundary, the ruler becomes *vacuous*, not a "shrinking nested ellipsoid." The report's "concentric ellipsoids scaled by √(0.5^i)" is the wrong functional form (it would be `√(1−0.5^i)`) and the wrong direction. This undercuts the central "the ruler is literally the ellipsoid level sets" selling point.
- **"Raises the ~1% accept rate" via whitening** is asserted with no analysis. A 1% accept rate (from `POLYTOPE_IDEA.md`) is itself a red flag for the whole sample-gating premise; pushing the mean into one ellipsoid does nothing for a *multimodal* flow whose mass straddles two homotopy classes.
- **Separate position/velocity γ** is plausible but unsupported by the cited mechanism: `gamma_schedule.gamma_distance_velocity` returns a single scalar γ, not two decay rates. The recommendation isn't realizable from the referenced code.

---

### (c) Real-time / 3-D — overstated; the report conflates the cheap and expensive paths.

- **Per-sample corridor construction is not microseconds.** §4 says sample-certification "degrades to evaluating inequalities… microseconds." But §5.1's primary backup builds a *chain of FIRI polytopes along each sample's own waypoints*. For a 128-sample batch (`num_samples=128`) that is 128 × (chain of FIRI inflations), in 3-D each an SOCP+MVIE per alternation. That is region *construction*, not inequality *evaluation* — orders of magnitude away from the quoted µs, and it is on the per-step hot path the moment the single-region certify fails (i.e. exactly the hard cases). The two latency regimes are silently merged.
- **FIRI "analytic, no SDP" is 2-D only.** §2/§3 repeatedly carry the 2-D analytic ellipse as the speed argument, then assert 3-D is "negligible." 3-D MVIE is an SOCP per alternation per obstacle; at crowd/`max_obstacles` scale this is the dominant cost, not the "LP grows 2→3 dims" term they call negligible.
- **3-D real-time is admittedly unmeasured** (§7: "asserted from UAV literature, not measured in cfm_mppi"). So the "yes, 3-D feasible" cells in the §2 table are claims, not results.
- **GCS-offline mismatch with a dynamic scene.** The code is built around moving obstacles (`obstacle_velocities` threaded through `plan`, `_guide_nominal`, `safety_filter_action`). Offline GCS homotopy enumeration assumes a *fixed* environment; curated modes are stale at runtime. §7 notes the staleness risk but still makes GCS a recommended component — for this setting it's a poor fit unless obstacle motion is negligible over the curation cadence.

---

### (d) Backup soundness — the layer the guarantee "must rest on" is the weakest.

- **Corridor overlap ≠ feasible dynamic handoff.** §5.1 requires `F_{r(i)} ∩ F_{r(i)+1} ≠ ∅` (a position-set overlap LP). For a relative-degree-2 plant, position overlap does not imply the *state* (position+velocity) can transition between segments under `u∈[u_min,u_max]`. A non-empty spatial overlap with the robot arriving too fast is not a safe hand-off. The handoff certificate must be over the state set, not the position set.
- **Branches are candidates, not certificates.** The existing `_sets_backup_controls` (Gramian directions + away/tangent push + unicycle reverse) feeds extra sequences through the *same* rejection; if all violate, `plan` still returns least-bad. The report inherits this — escalation step (c) is only a guarantee if step (c)'s QP is itself proven feasible (see (a.2)/(a.1)). It is not.
- **Greedy `r(i)` thrashing** is acknowledged but unmitigated for the *control* layer; GCS curation (offline, static) doesn't fix runtime thrash with moving obstacles.
- **PACS path-consistency** is a preference on the correction direction — fine, but it has zero bearing on the safety guarantee and shouldn't be listed as part of the "hard enforcer."

---

### (e) Library fit — several misfits on license, maintenance, and dimensionality/overkill.

- **iris-distro (rdeits):** defaults to **Mosek** (commercial license) for its SDP, and is effectively superseded/unmaintained in favor of Drake's IRIS. Concrete misfit on both license and maintenance — drop it.
- **RCCM tube (boranzhao/robust_ccm_tube):** dimensionality/complexity *overkill and misfit*. The plant is a **linear double integrator** — for a linear system the exact tube is a polytopic **RPI set** (trivial, offline, no SOS). Recommending nonlinear contraction-metric SOS synthesis (needs an SOS solver, system-specific metric search, offline) for a 2-D/3-D DI is using UAV-nonlinear machinery where a 10-line RPI computation suffices. Also "trajectory-agnostic tube valid around any curvy U" overstates CCM.
- **GCOPTER/FIRI and DecompUtil:** both are **C++** research code (ZJU-FAST-Lab / sikang). "Drop-in for your tangent-hyperplane code" is wrong — they need pybind/ctypes bridges into a torch/numpy per-step loop; not drop-in for this Python codebase. Maintenance on both is sporadic/research-grade; DecompUtil in particular has had little activity and historically ambiguous licensing — verify the LICENSE before vendoring.
- **Drake/pydrake:** BSD-3 and pip-installable (acceptable license), but IRIS and `MaximumVolumeInscribedEllipsoid` are **SDP-based and designed for offline configuration-space decomposition**, not per-step replanning at 10+ Hz with moving obstacles. Fine as an offline/prototype backend; misfit as the per-step production path the report sometimes implies.
- **General dimensionality point:** the whole imported stack (FIRI, GCOPTER, GCS, RCCM, HOCBF zoo) is calibrated for nonlinear/high-D UAVs. For a 2-D/3-D *linear* DI, exact tools exist (polytopic invariant sets, linear tube-MPC, explicit braking set). The report over-imports complexity that the plant doesn't justify.
- Correct fits: OSQP/Clarabel (apt), cvxpy for prototyping (apt, slow at batch scale).

---

### Additional concrete technical errors found

- **Verifier LP is not an LP/SOCP with the ellipsoid barrier.** §4 claims `h(x_{i+1}) ≥ (1−γ)h(x_i)` is "LP (affine h) or small QP/SOCP (ellipsoidal h)." With `h=1−‖B⁻¹(p−d)‖²` (concave quadratic), a *lower bound* on a concave function is a **nonconvex** quadratic constraint — not SOCP-representable. The convex-LP/QP claim, and therefore the "warm-started convex" real-time argument, fails for the very ellipsoid barrier the report makes central.
- **Executed guarantee is step-0 only.** Only `controls[best,0]` is applied and the planner replans (`check_first_control_only` path, line 612-616). The multi-step `(1−γ)^i` ruler is cost-shaping for *selection*, not an enforced multi-step constraint on the executed trajectory. Cross-replan invariance (the actual closed-loop property) is never established.

---

## Concrete required fixes (ordered)

1. **Construct C_inv before claiming any guarantee.** Compute the explicit braking/RPI invariant set for the input-bounded DI per active region (offline pre-image iteration or, since linear, RPI). Make the §5.3 QP *target* it. Without this, drop all "hard guarantee" language.
2. **Verify DTCBF validity under input bounds.** Prove (or numerically certify on a grid) that for all states with `h≥0` there exists `u∈[u_min,u_max]` with `h(x⁺)≥(1−γ)h(x)`. Restrict the certified set to where this holds; reject initial states outside it (don't pretend braking is "always feasible").
3. **Use a HOCBF lift for the ellipsoid/region barrier.** A position-only `h` on the DI is not invariant. Define the rel-deg-2 barrier (or carry velocity in the state-space region) so the ruler corresponds to a true invariant set.
4. **Fix the barrier-level-set math.** `{1−‖u‖²≥(1−γ)^i}` gives radius `√(1−(1−γ)^i)` (growing/vacuous), not `√(0.5^i)` shrinking. Re-derive the ruler so the safety margin actually contracts, or abandon the "ellipsoid level sets = ruler" framing.
5. **Add an inter-sample margin as a constraint**, not a footnote: tighten each region by the DI continuous-overshoot bound (≈`u_max·dt²/2` + velocity term) so node-wise satisfaction implies continuous safety.
6. **Make the corridor handoff a state-space (position+velocity) feasibility check, not a position-overlap LP.** Verify a control exists driving `x` from segment i to a safe state in segment i+1 under actuation limits.
7. **Replace the LP claim for the ellipsoid barrier** with the truth: either keep the affine (frozen-normal) barrier for a real LP, or accept a nonconvex QCQP / re-linearize the ellipsoid constraint per step. Don't advertise convex real-time for a nonconvex constraint.
8. **Re-justify or drop "less conservative."** Either commit to terminal-only (drop the monotone ruler) to get the dip-and-recover gain, or stop claiming C_inv reduces conservatism while keeping both. Replace the √n John bound headline with a region-vs-free-space metric.
9. **Separate the cost regimes in the real-time argument.** Per-sample corridor *construction* is not the µs evaluation path; either restrict corridor building to the chosen mode (1 build/step, not 128) or precompute a shared decomposition and only *assign* samples to regions on the hot path.
10. **Quantify the accept-rate claim.** Show whitening actually moves accept rate off ~1% on a multimodal flow before relying on it; for genuinely bimodal mass, a single ellipsoid won't.
11. **Library substitutions:** drop iris-distro (Mosek/unmaintained); replace RCCM with a linear RPI tube for the DI; treat Drake/GCOPTER/DecompUtil as offline/prototype backends and budget the C++↔Python binding work (don't call them "drop-in"); verify DecompUtil's license before vendoring.
12. **Make the hard path default-on and actually feasible.** Today `filter_output`/`use_ho_barrier` default `False` and the filter can return `filter_feasible=False` after clamping. The "guarantee independent of whether the sample was certifiable" requires this path on, QP-exact (not Jacobi sweeps), and provably feasible via fixes 1-2.
13. **Resolve the dynamic-obstacle vs. offline-GCS mismatch.** The code threads obstacle velocities everywhere; either justify GCS staleness quantitatively against obstacle speed × cadence, or drop GCS for this setting.

Net: the **ball-clamp**, **corridor (as a structure)**, **OSQP minimal-norm projection**, and **demoting the orientation search** are sound and worth doing. But the report currently *claims* forward invariance it does not build, rests "always-feasible braking" on an unverified invariant, advertises convex real-time for a nonconvex constraint, contains a wrong barrier-level-set derivation, conflates per-sample region construction with µs evaluation, and imports nonlinear/UAV libraries (RCCM, iris-distro, GCS) that misfit a linear, dynamic, 2-3 D double integrator. Fixes 1-7 are mandatory before any "guarantee" wording is defensible.

Relevant code: `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/safemppi.py` (plan least-bad fallback lines 638-642; gated/feasible-uncertain filter lines 334-346, 644-652; frozen-at-x0 barrier consumed at 598-611), `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/barrier.py:152-171` (x₀-linearized half-space), `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/rectangle_verifier.py:111` (corners-exceed-R bug the ball-clamp fixes), `/home/dohyun/projects/cfm_mppi/cfm_mppi/safegpc_adapter/gamma_schedule.py:13-24` (single scalar γ, not per-axis).
