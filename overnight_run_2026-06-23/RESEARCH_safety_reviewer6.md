# Safety / Reviewer-6 — margin ⇒ velocity-tracking ⇒ safety (research synthesis, verified)

## The objection (narrowly correct)
Double integrator x=(p,v), input u=accel, |u|≤a_max. Position half-space
h0(p)=nᵀ(p−p_o). u enters only at ḧ0=nᵀu (ḣ0=nᵀv), so h0 has RELATIVE DEGREE 2 ⇒
h0 alone is NOT a valid CBF. Reviewer 6 is right on that narrow point.

## Our h_ho is the first HOCBF level (confirmed exactly)
With ψ0=h0, linear class-K α1(s)=s/η:
  ψ1 = ḣ0 + α1(h0) = nᵀv + (1/η)h0  ⇒  η·ψ1 = h0 + η nᵀv = h_ho.   ✓
So h_ho = η·ψ1. BUT ψ1≥0 is only the first cascade level — NOT a bounded-input
safety certificate by itself: enforcing the next level ψ2≥0 demands nᵀu growing
with closing speed nᵀv (deceleration unbounded in v), which bounded a_max cannot
supply near the boundary. So a LINEAR position+velocity half-space cannot be made
controlled-invariant under bounded acceleration. (Xiao-Belta TAC 2022;
Nguyen-Sreenath ACC 2016; Agrawal-Sreenath discrete RSS 2017.)

## THE valid fix — braking-distance barrier (your margin claim, made exact)
**Chen, Janković, Santillo, Ames, "Backup CBFs," CDC 2021 (arXiv:2104.11332), Eq.4:**
  h_brake(p,v) = nᵀ(p−p_o) − (1/(2 a_max))·((nᵀv)_+)²    (C¹ at v=0)
is a VALID CBF: {h_brake≥0} = states that can brake to v=0 without crossing — a
controlled-invariant safe set under bounded a_max (the backup CBF of the max-braking
policy). THE IDENTITY (the collaborator's claim, exact): with position margin
m = nᵀ(p−p_o),
  h_brake ≥ 0  ⟺  (nᵀv)_+ ≤ sqrt(2 a_max m).
=> a polytope face with margin m PERMITS closing speed up to sqrt(2 a_max m). The
robot tracks a safe velocity bounded by the margin. margin ⇒ velocity budget ⇒ safety.

## Equivalent: backstepping CBF (velocity-tracking-error margin)
**Taylor, Ong, Molnár, Ames, "Safe Backstepping with CBFs," CDC 2022 (arXiv:2204.00653):**
  h(p,v) = h0(p) − (1/(2µ))‖v − k0(p)‖²,   k0 = safe velocity command.
Enforcing h≥0 forces h0 ≥ (1/2µ)‖v−k0‖² ≥ 0: the position margin EQUALS the squared
velocity-tracking error. Theorem 4 certifies h as a valid CBF for the full
relative-degree-2 system (needs strict reduced condition). This is literally
"margin ⇒ velocity tracking ⇒ safety," and a clean two-stage form for the paper.

## Robustness to constant-velocity pedestrian prediction error (margin absorbs it)
**Kolathaya & Ames, ISSf, L-CSS 2019 (arXiv:1803.03035):** enforce h ≥ δ with
δ ≥ γ(‖disturbance‖) ⇒ the true (disturbed) constraint holds. For mispredicted
pedestrian motion ‖p_o^true − p_o^pred‖ ≤ ε, inflate the half-space by ε
(r → r+ε): h_true ≥ h_pred − ε, so predicted-safe ⇒ true-safe. (Also Janković
Automatica 2018 robust CBF; Kim-Diagne-Krstić arXiv:2412.03678 moving obstacle.)

## Dual relative degree — citation CORRECTION
**Bahati, Cosner, Cohen, Bena, Ames, "CBF Synthesis for Nonlinear Systems with
Dual Relative Degree," arXiv:2504.00397 (2025):** "dual relative degree" is a
MULTI-INPUT notion (different inputs act at different orders, e.g. unicycle/quadrotor),
NOT a single-input rel-deg-2 fix. Construction h = h0 − (1/µ)V (same margin-via-
tracking idea as backstepping). Cite as RELATED CONTEXT only, with this caveat —
our double integrator does NOT have dual relative degree in their sense.

## One-line answer to Reviewer 6 (hand to the reviewer)
h0 is indeed not a CBF (rel. deg. 2); h_ho = h0+η nᵀv is the first HOCBF level η·ψ1
(α1(s)=s/η); the position polytope-with-MARGIN becomes a valid relative-degree-2
certificate exactly when the inner set is controlled-invariant under bounded a_max
— i.e. the braking-distance barrier nᵀ(p−p_o) ≥ (nᵀv)_+²/(2a_max) (Chen et al. 2021),
equivalently the velocity-tracking margin h0−(1/2µ)‖v−k0‖² (Taylor et al. 2022).
Margin m = budget for closing speed sqrt(2 a_max m); the same margin absorbs
constant-velocity prediction error via ISSf (Kolathaya-Ames 2019).

## Minimal citation set: Taylor CDC2022 + Chen CDC2021 (load-bearing) +
Xiao-Belta TAC2022 / Nguyen-Sreenath ACC2016 (HOCBF framing) + Kolathaya-Ames
L-CSS2019 (robust margin). Bahati 2025 = related context only.
