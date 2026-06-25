# Convex-polytope feasible set + learn-to-sample-within (Dohyun's idea, 2026-06-24)

## The reframing that makes the method honest
Measured fact: under a Gaussian proposal the affine-DCBF accept rate is ~1.2%
(median 0%, 48/80 steps with ZERO accepted samples) — MPPI averaging is never
really exercised; an output-projection filter carries the result. So the central
algorithmic problem is NOT "reject + patch the output", it is:

  **learn a proposal q_θ whose sampled rollouts land INSIDE the convex feasible
  polytope**, so rejection is rare, averaging is meaningful & multimodal, and the
  (convex) safety guarantee is actually used.

## The geometric object (Dohyun)
At each step, instead of ONE tangent half-space to the nearest obstacle, build the
convex polytope that separates the robot from ALL pedestrians:
  P = ⋂_{j=1}^N { p : a_jᵀ(p − p_o,j) ≥ 0 }   (one separating hyperplane per obstacle).
Lift to (p,v) with the HOCBF term per face: a_jᵀ(p − p_o,j) + η a_jᵀ(v − v_o,j).
P is convex (intersection of half-spaces); single obstacle ⇒ P = one half-space =
the current method (clean fallback).

## A single barrier on NESTED convex level sets (h=0 on ∂P, h=1 at robot)
Smooth polytope barrier via log-sum-exp soft-min of the per-face margins h_j:
  H(x) = −(1/β) log Σ_j exp(−β h_j(x))   → min_j h_j(x) as β→∞.
H=0 on ∂P (some face active), H>0 inside; normalize H(robot)=1. The level sets
{H ≥ c} are nested shrunken polytopes. The (1−γ)^i schedule generalizes to
  H(x_{k+i}) ≥ (1−γ)^i,
so "accepted" = the rollout stays in the nested polytope sequence. The flow model's
job is to put its mass on controls satisfying this — i.e. sample within the polytope.

## Why LARGEST polytope matters → existing geometric toolkit (research)
A thin separating polytope is as hard to sample as one hyperplane. We want the
MAXIMUM convex obstacle-free region around the robot so the feasible set is wide
and the proposal can actually live in it. This is a solved problem class:
  - **IRIS** (Deits & Tedrake 2015): alternate (i) separating hyperplanes between
    obstacles and a seed ellipsoid, (ii) max-volume inscribed ellipsoid (SDP).
    Produces a large convex free polytope in ~ms. <-- directly our construction.
  - **Max-volume inscribed ellipsoid (MVE / John)** and **Chebyshev center** (LP):
    certify/seed the largest inscribed ball/ellipsoid of P.
  - **Safe Flight Corridors** (Liu et al. 2017): convex polyhedral free-space.
  (Exact formulations + real-time costs: see RESEARCH agent output, pending.)

## Learning to sample inside a polytope (research)
The proposal must be supported on (or biased into) {A p ≤ b}. Candidate machinery:
reflected / mirror flow-matching & diffusion on convex domains, Dikin-walk / barrier
Langevin, constrained flow matching. Train q_θ on guided rollouts that stay in P,
conditioned on the polytope faces (a_j, b_j) and γ; reward-weight by MPPI weight;
Expert-Iterate. Acceptance becomes high by construction (the user's "your learned
flow matching model handles the acceptance").

## Contribution restated (stronger + honest)
1. Affine HOCBF half-space per obstacle (Reviewer-6 fix; PROOF_HOCBF.md).
2. Their intersection = convex polytope feasible set; averaging stays inside (proof
   corollary). Build the LARGEST such polytope (IRIS/MVE) so it is wide.
3. A single smooth polytope barrier H with nested convex level sets ((1−γ)^i
   schedule); single-obstacle fallback = one hyperplane.
4. Learn q_θ to sample WITHIN the polytope (reflected/constrained flow), so MPPI
   averaging is genuinely exercised (high accept), multimodal, and certified — no
   output-projection band-aid. Safety stays proposal-agnostic (Props 3–4).

## Research synthesis — methods, equations, citations (2026-06-24)

### Construction of the (largest) convex obstacle-free polytope
- **SFC** (what our `build_nominal_polytope` is): inflate an ellipsoid along the
  nominal direction, drop a tangent separating hyperplane at the closest obstacle
  point p*, normal `a=(CCᵀ)⁻¹(p*−d)`, `b=aᵀp*`; shrink, remove cut obstacles,
  repeat. No SDP. *S. Liu et al., "Planning Dynamically Feasible Trajectories ...
  Safe Flight Corridors in 3-D ...," IEEE RA-L 2(3):1688–1695, 2017.* Code:
  github.com/sikang/DecompUtil. (Per-segment timing not publicly documented; full
  replan ~50–300 ms — measure ours.)
- **IRIS** (max-volume, gold standard): alternate (a) separating hyperplanes
  `a_j=2C⁻¹C⁻ᵀ(x*−d)` and (b) max-volume inscribed ellipsoid SDP. Monotone volume,
  2–8 iters. *R. Deits & R. Tedrake, "Computing Large Convex Regions of
  Obstacle-Free Space Through SDP," WAFR 2014 / STAR 107:109–124, 2015.* SDP per
  step is borderline real-time.
- **FIRI (USE THIS for max region, real-time)**: IRIS alternation in workspace
  2D/3D with the FIRST linear-complexity ANALYTIC max-area inscribed ellipse — no
  SDP, designed for online quadrotors. *Q. Wang et al., "Fast Iterative Region
  Inflation ...," arXiv:2403.02977, 2024.* Also IRIS-ZO/NP2 (arXiv:2410.12649).

### Seed / sampling center
- **Chebyshev center (LP, sub-ms)**: `max r s.t. a_iᵀx_c + r‖a_i‖ ≤ b_i` — natural
  MPPI mean / deep interior seed.
- **Max-Volume Inscribed Ellipsoid (MVE, convex/SDP)**: `max log det B s.t.
  ‖B a_i‖ + a_iᵀd ≤ b_i` — whitening transform to push the sampling cloud into P.
  John: MVE×n ⊇ P (×√n if symmetric); n=2 ⇒ factor 2. *Boyd & Vandenberghe,
  Convex Optimization, §4.3.1, §8.4.2.*

### THE single barrier we already use (validated)
Our `Polytope.barrier` = `−(1/κ) log Σ_i exp(−κ·margin_i)` is EXACTLY the
log-sum-exp composed-CBF: `h(x)=−(1/κ)log Σ_i e^{κ(a_iᵀx−b_i)}`, `{h≥0}⊆P`,
`∇h=−Σ w_i a_i` (softmax weights), reduces to one half-space at m=1, O(mn)
real-time. *T. Molnar & A. Ames, "Composing CBFs for Complex Safety Specs,"
IEEE L-CSS 7:3615–3620, 2023 (arXiv:2309.06647); Wu et al. arXiv:2502.16293, 2025
(polytopic CBF); Lindemann & Dimarogonas L-CSS 2019.* ⇒ our barrier is the
right object; nested level sets {h ≥ (1−γ)^i} give the DCBF schedule.

### THE fix for the 1.2%-accept problem: sample INSIDE P by construction
- **Mirror-map + flow matching (RECOMMENDED, feasible by construction)**: use the
  log-barrier mirror `φ(x)=−Σ_i log(b_i−a_iᵀx)`; run ordinary FM in the
  unconstrained dual `y`, decode `x=∇φ*(y) ∈ P` for EVERY y ⇒ **accept rate 100%,
  no rejection/projection/reflection**, smooth dual ODE distills to few NFE
  (real-time). *G.-H. Liu et al., "Mirror Diffusion Models ...," NeurIPS 2023
  (arXiv:2310.01236); Y. Guan et al., "Mirror Flow Matching ... on Convex Domains,"
  arXiv:2510.08929, 2025 (polytopes/simplex/PSD, Wasserstein bounds).*
- Alternatives: Reflected Flow Matching (Xie et al., ICML 2024, arXiv:2405.16577 —
  closed-form on boxes, per-facet cost on polytopes); Projected diffusion
  (Christopher et al., arXiv:2402.03559); Barrier/Dikin-metric FM = open gap (most
  novel). Offline ground-truth feasible samples: hit-and-run / Dikin walk.

### Recommended pipeline (citable)
1. Build P per step: tangent half-spaces (pruned) — FIRI for a maximal region,
   our nominal-box-cut as cheap fallback; single obstacle ⇒ one half-space.
2. Chebyshev-center LP ⇒ MPPI mean; (optional) MVE to whiten the proposal into P.
3. LSE polytope CBF (= our `barrier`) for the certificate / cost; nested level sets.
4. Train q_θ via log-barrier MIRROR-MAP flow matching ⇒ samples feasible by
   construction (kills the 1.2% accept problem), distill to few NFE. Safety stays
   proposal-agnostic (Props 3–4).

## Immediate validation experiments (later, not now)
- Plot accept-rate vs method: Gaussian (~1%) vs polytope-constrained proposal (→high).
- Show averaging is exercised (|A|≫1) and multimodal (left/right corridors).
- Ablate single-hyperplane vs IRIS-polytope feasible set.
