# Polytope module + verifier — reconciled decision

Synthesis of the research report + adversarial review (`POLYTOPE_VERIFIER_RESEARCH.md`). The report's
*architecture* is right; the review correctly shows several of its *guarantee* claims are not yet real. This
doc is the corrected, actionable version.

## The architecture (3 layers — separation of concerns)
1. **Region = where you may go.** NOT a single rectangle. A **max-volume convex polytope** (IRIS/FIRI-style),
   **clamped inside the sensing ball** (ball as 8–16 half-spaces → fixes the "rectangle exceeds R" bug), carrying
   its **max-volume inscribed ellipsoid (MVIE)**. For multimodal / curvy paths use a **sequence of overlapping
   convex regions (safe corridor)** — a single convex set provably *cannot* pass a head-on obstacle or follow a
   curvy multimodal path (this is your msg-1 "not a rectangle / curvy may be unfittable" intuition, confirmed).
2. **Verifier = does a safe control exist.** Replace the non-convex orientation search with a **convex feasibility
   program** (predictive-safety-filter LP/QP): ∃ U s.t. each state stays in its active region, respects the DTCBF
   ruler, obeys actuation, and ends in a control-invariant set. For sample-gating it collapses to a **batched
   feasibility evaluation** (your current rejection mask) — keep that for speed.
3. **Enforcer = the hard guarantee.** A per-step **minimal-norm QP projection** (OSQP) onto the active region's
   half-spaces with a **terminal control-invariant (braking) set** — returns a provably safe control regardless of
   whether the flow sample was certifiable.

## What the review CORRECTED (do not skip — the "guarantee" is not real until these are done)
1. **C_inv must actually be constructed.** The guarantee rests on a terminal control-invariant / braking set; the
   report only *names* it. For the linear DI this is exact and cheap (polytopic RPI / one-step pre-image iteration,
   or a braking set `states drivable to v=0 within the region under u_max`). The QP must *target* it.
2. **"Braking is always feasible" is FALSE** for an input-bounded DI: stopping distance `v²/(2u_max)`. Must verify
   DTCBF feasibility under `u∈[u_min,u_max]` and restrict the certified set to where it holds — don't enter states
   with clearance < stopping distance.
3. **Position-only barrier is not invariant** (DI is relative-degree-2). Use a **HOCBF lift** (or carry velocity in
   the region/state), not a position ellipsoid alone.
4. **The report's ellipsoid level-set math is wrong.** For `h=1−‖B⁻¹(p−d)‖²`, `{h≥(1−γ)^i}` has radius
   `√(1−(1−γ)^i)` which *grows* (ruler becomes vacuous) — not a shrinking nested ellipsoid. → keep the **affine
   polytope barrier** (a real LP; `barrier.py::affine_barrier_h_ho_all` already has the form) or derive a correct
   HOCBF; don't advertise convex real-time for the (nonconvex) ellipsoid lower-bound constraint.
5. **Inter-sample margin** must be a *constraint*, not a footnote: tighten each region by the DI continuous
   overshoot (`≈ u_max·dt²/2` + velocity term) so node-wise checks imply continuous safety.
6. **Corridor hand-off must be a state-space (position+velocity) feasibility check**, not a position-overlap LP —
   spatial overlap with too much speed is not a safe transition.
7. **Real-time honesty:** per-sample *corridor construction* is NOT the µs evaluation path. Build regions once per
   step for the chosen mode (not per sample); only *assign* samples to existing regions on the hot path.

## Libraries (corrected)
- **Prototype in pure Python first**: `scipy.optimize.linprog` (Chebyshev seed), `cvxpy` log-det (MVIE), and the
  existing `build_nominal_polytope` (SFC cuts) — adequate at n=2,3 with few half-planes. **OSQP** for the projection QP.
- **Drake / pydrake** (BSD, pip): `HPolyhedron`, `MaximumVolumeInscribedEllipsoid`, `IRIS`, `GcsTrajectoryOptimization`
  — good *offline / prototype* backend (SDP-based, not the per-step hot path).
- **GCOPTER/FIRI** (C++) + **DecompUtil** (C++): fast production region builders, but need pybind/ctypes bridges
  (NOT "drop-in"); verify DecompUtil license before vendoring.
- **DROP `iris-distro`** (defaults to Mosek/commercial, unmaintained). **DROP RCCM** for the tube — the plant is a
  *linear* DI, so use an exact **polytopic RPI set / linear tube-MPC**, not nonlinear contraction-metric SOS.
- GCS is **offline & assumes a static scene**; our obstacles move (velocities threaded everywhere) → only use GCS to
  curate the multimodal *training distribution*, never in the per-step loop.

## Migration from the current rectangle (incremental, each step shippable)
1. **Now (real bug fix):** clamp the region to the sensing ball (K half-spaces) — closes the "certifies unsensed
   space" bug in `rectangle_verifier._extents_analytic`.
2. Extend `Polytope` with `B,d` (MVIE) + `ellipsoid_barrier`; new `region.py` (`build_region` with backends
   `sfc|cvxpy|drake|firi`, Chebyshev seed, ball clamp) — `sfc` = today's `build_nominal_polytope` + MVIE works with
   zero new deps.
3. New `verifier.py`: `certify_samples` (batched ratio test, lifted from `certify_fast`) + `exists_safe_control`
   (convex LP/QP, warm-started) — **delete the orientation search**.
4. New `corridor.py`: chain of overlapping regions + **state-space** hand-off check + active-region selector.
5. Upgrade `safemppi.safety_filter_action`: Jacobi → **OSQP minimal-norm QP** + **C_inv terminal** + path-consistent
   objective; make the hard path default-on and provably feasible (fixes 1–2 above).

## Bottom line
**Best performance + least conservatism + safety = (corridor of max-volume convex regions inside the ball) +
(convex feasibility verifier) + (QP projection onto a constructed braking-invariant set).** The geometry buys
expressiveness/low-conservatism; the *hard guarantee comes only from the constructed C_inv + verified
input-bounded DTCBF + HOCBF lift + inter-sample margin* — not from the region shape. Prototype in pure Python,
import C++ region builders only when real-time demands it.
