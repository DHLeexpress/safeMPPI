# Affine Higher-Order DCBF — statement and proof (for the paper)

## Setup
Discrete-time double integrator, state x=(p,v)∈R^{2d}, acceleration input u∈R^d,
sampling time Δt:
  p_{k+1} = p_k + Δt·v_k + (Δt²/2)·u_k,   v_{k+1} = v_k + Δt·u_k.        (1)
Obstacle: disk (center c, radius ρ). Let p_o be the boundary point nearest the
current robot position and n the outward unit normal (from obstacle toward robot).
Position margin (signed distance to the supporting tangent plane):
  h_0(x) = nᵀ(p − p_o).         h_0 ≥ 0 ⇔ robot on the safe side of the plane.   (2)

## The relative-degree obstacle (why h_0 alone fails)
First difference of (2) along (1):
  Δh_0 = h_0(x_{k+1}) − h_0(x_k) = Δt·nᵀv_k + (Δt²/2)·nᵀu_k.                      (3)
The input enters only at O(Δt²): with bounded |u|≤u_max the achievable one-step
change of h_0 from u is ≤ (Δt²/2)u_max — negligible. So the discrete CBF condition
h_0(x_{k+1}) ≥ (1−γ)h_0(x_k) is, for a robot already moving toward the obstacle
(nᵀv_k<0), infeasible for ALL admissible u — the planner must reject every sample.
This is exactly Reviewer 6's "(12) is not a valid DCBF": h_0 ignores the dynamics.

## The affine higher-order barrier
Lift the relative degree by adding the (affine) velocity term:
  h_ho(x) = nᵀ(p − p_o) + η·nᵀv = nᵀ(p − p_o + η v),     η>0.                   (4)
This is the first HOCBF cascade ψ_1 with linear class-K α_1(h_0)=(1/η)h_0:
  ψ_1 = ḣ_0 + α_1(h_0) = nᵀv + (1/η)h_0  ⇒  η·ψ_1 = h_0 + η nᵀv = h_ho.          (5)
(Continuous-time form; the discrete cascade of Xiong et al. 2023 gives the same.)

**Lemma 1 (validity / unit relative degree).** Along (1),
  h_ho(x_{k+1}) = h_ho(x_k) + Δt·nᵀv_k + (ηΔt + Δt²/2)·nᵀu_k.                     (6)
Hence ∂h_ho(x_{k+1})/∂u_k = (ηΔt + Δt²/2)·n = O(Δt): the input enters at first
order, so for any x_k there exists a bounded u_k achieving
h_ho(x_{k+1}) ≥ (1−γ)h_ho(x_k). Thus h_ho is a valid discrete CBF for (1). ∎

*Choice of η.* η is the velocity look-ahead (braking time-constant). The inner set
{h_0≥0}∩{ψ_1≥0} is controlled-invariant when η is large enough to brake within the
margin: with |u|≤u_max a sufficient condition is η ≥ v_max/u_max along n (so the
allowed closing speed (1/η)h_0 is brakeable). η too small ⇒ inner set empty
(over-rejection); η too large ⇒ ψ_2 infeasibility at the boundary (Nguyen–Sreenath
2016; Xiao–Belta 2019, Remark 2 for the h_0=0, ḣ_0<0 trap).

**Convexity.** h_ho is affine in x=(p,v) ⇒ C := {x : h_ho(x) ≥ 0} is a half-space
of R^{2d}, hence convex. The time-varying safe sets
C_i := {x : h_ho(x) ≥ (1−γ)^i h_ho(x_k)} are half-spaces too (convex).

## Averaging safety (the MPPI guarantee)
**Proposition (constrain-then-average is safe).** Consider M control-sequence
samples for the LTI system (1). Let A be the ACCEPTED set — samples whose rollout
satisfies h_ho(x^{(m)}_{k+i}) ≥ (1−γ)^i h_ho(x_k) for all i. Let the applied plan be
the importance-weighted average x*_{k+i} = Σ_{m∈A} ω_m x^{(m)}_{k+i}, ω_m≥0,
Σω_m=1. If A ≠ ∅ then x*_{k+i} ∈ C_i for all i.
*Proof.* For LTI (1), x_{k+i} is affine in the control sequence:
x_{k+i}=A^i x_k + Σ_{j<i} A^{i−1−j} B u_{k+j}. So x*_{k+i}=Σ_{m∈A} ω_m x^{(m)}_{k+i}
is a convex combination of the accepted rollout states. Each accepted state lies in
the half-space C_i, which is convex; a convex combination of points in a convex set
remains in it ⇒ x*_{k+i} ∈ C_i. ∎

**Corollary (multi-obstacle).** With one half-space per obstacle, the feasible set
is the intersection ∩_j C_{i,j} — a convex polytope. The same argument gives
x*_{k+i} ∈ ∩_j C_{i,j}: averaging stays in the polytope. (This is the geometric
object §POLYTOPE builds on.)

## Honest limitation discovered empirically (must be stated in the paper)
The Proposition is VACUOUS when A=∅. Measured on UCY ep.110, γ=0.2, 512
samples/step: mean accept rate 1.2%, median 0.0%, 48/80 steps with zero accepted
samples. A Gaussian proposal (even mean-shifted + anisotropic) almost never lands
in the thin feasible polytope, so the averaging guarantee is rarely exercised and
the controller degenerates to an output-projection safety filter. ⇒ The real
algorithmic task is to LEARN a proposal q_θ whose samples lie inside the polytope
(see POLYTOPE note), so that averaging is meaningful, multimodal, and certified.
