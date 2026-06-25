# Reward-Tilted Feasible Proposal — research synthesis + concrete recipe (2026-06-24)

## Target (what q_θ should learn)
q_θ(U | o, γ) ≈ p(U|o,γ) ∝ 1[U∈F(o,γ)] · exp(−S(U)/λ).
This is the KL-regularized reward-max optimum (Gibbs / control-as-inference):
  p* = argmax_p E_p[R] − λ·KL(p‖p0),  R=−S.
=> our MPPI weights w_m ∝ exp(−S(U_m)/λ) ARE the reward tilt. We want to AMORTIZE
the tilt into a conditional generator. (Uehara et al. tutorial, arXiv:2501.09685.)

## THE training recipe (use this) — Energy/Reward-Weighted Conditional Flow Matching
**EFM** (Zhang, Zhang, Gu, arXiv:2503.04975): minimize the standard conditional FM
loss but reweight each endpoint sample by the self-normalized tilt weight
  L(θ) = E_{t, U~accepted, x~p_t(·|U)} [ w(U) · ‖ v_θ(t,x | o,γ) − u_t(x|U) ‖² ],
  w(U) = exp(−S(U)/λ) / mean_batch[exp(−S/λ)].
PROVEN: ∇L = ∇(unweighted energy-guided FM), so the terminal marginal is exactly
∝ p0·exp(−S/λ) — the tilt. (= AWR/RWR weighted-MLE lifted to a flow over control
sequences; weights are our MPPI weights.)

## Anti-mode-collapse (must add) — W2/KL trust region
**ORW-CFM-W2** (Fan et al., arXiv:2502.06061): reward-weighting ALONE collapses to
a Dirac at the reward-argmax (Lemma). Fix: add a Wasserstein-2 / KL trust region to
the base flow:  L += α‖v_θ − v_ref‖².  Preserves multimodality (left/right modes).
Also heed over-optimization critique (Dandapanthula & Boffi, arXiv:2606.02884:
finite-particle reward guidance over-optimizes; use reward damping).

## Feasibility-by-construction (keep safety separate)
Bake F into the architecture: a differentiable polytope-projection final layer (or
our mirror-map decode) so EVERY q_θ output is feasible. Then the tilt is ONLY for
performance (reward concentration). Clean separation = our Props 3-4. (SBSRL
arXiv:2605.19469 = the safe-sampling-RL ref, but its safety is over model
uncertainty; our F is a known convex polytope, so projection is the right tool.)

## Real-time amortization (few-NFE)
Distill the trained flow to 1-4 NFE: Consistency Models (arXiv:2303.01469),
Rectified Flow/Reflow (2209.03003), Consistency Policy (2405.07503). Keep the
projection layer AFTER the one-step generator so distilled outputs stay feasible.

## Temperature λ
λ = tilt sharpness (MPPI free-energy temperature, Williams et al. arXiv:1509.01149).
λ→0 greedy/mode-seeking (collapse risk), λ→∞ uniform-over-feasible. Anneal λ from
high (coverage) to low (sharpen); pair low λ with the W2 trust region.

## Concrete plan for our q_θ(U|o,γ)
1. DATA: run mirror-MPPI on many scenes; per step store (o, γ, accepted samples
   {U_m}, weights w_m=exp(−S_m/λ)). Keep top-weighted + a diversity tail.
2. TRAIN: reward-weighted conditional FM loss (EFM) + W2 trust region to the base
   mirror proposal; condition on (o, γ).
3. FEASIBILITY: mirror-map / polytope-projection layer on the output.
4. INFER: sample q_θ -> (optional MPPI refine) -> apply; distill to 1-NFE drifting.
5. EVAL: accept-rate, success, collision, MULTIMODALITY vs Gaussian/Mizuta.

## Citations
EFM 2503.04975 · ORW-CFM-W2 2502.06061 · Adjoint Matching 2409.08861 (memoryless
schedule needed for exact tilt) · Uehara tutorial 2501.09685 · RWR (Peters-Schaal
ICML07) · AWR 1910.00177 · AWAC 2006.09359 · MPO 1806.06920 · control-as-inference
1805.00909 · Flow Matching 2210.02747 · Diffuser 2205.09991 · Feynman-Kac steering
2501.06848 · Consistency 2303.01469 · your ids: reward-tilt critique 2606.02884,
safe-sampling-RL 2605.19469.
