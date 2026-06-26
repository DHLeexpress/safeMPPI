# Unified Framework — fresh restart (2026-06-24)

## Thesis
A LEARNED, reward/safety-TILTED sampling distribution (offline) + a convex-polytope
DTCBF SAMPLE REJECTION ruler (online) → robust to OOD pedestrian datasets, beating
Mizuta. Offline: "given these conditions, sample these perturbations" (trained on
dataset A). Online: the DTCBF rejects unsafe samples on dataset B regardless. =>
benefits from offline + online both.

## Audit corrections (honesty — from safeMPPI_safety_verification_report.pdf)
- I WAS WRONG earlier: for a FIXED affine face, the braking-distance safe set
  {(p,v): nᵀ(p−p_o) ≥ (nᵀv)₊²/(2a_max)} IS CONVEX (epigraph of a convex parabola).
  Non-convexity comes from circular geometry / nearest-obstacle switching / unions,
  NOT from the fixed-face quadratic itself. So the velocity-aware (braking) margin
  is COMPATIBLE with convex averaging.
- The current CODE uses a LINEAR affine velocity margin b_j ← b_j − η(nᵀv_rel)₊
  (≈ m ≥ ηc), NOT the exact quadratic braking m ≥ c²/(2a_max). HOCBF + final output
  filter are OFF by default. So code ≠ the strongest certificate.
- Hard safety for general nonlinear = NO. Valid only for the restricted class
  (double-integrator / feedback-linearizable rel-deg-2 normal channel, bounded
  a_max, known relative dynamics, inflated margins, enforced final filter).

## Design (the 6 points)
1. NO HOCBF. Keep MPPI general (affine position half-spaces / DTCBF on a convex
   polytope). Drop the η·ψ₁ HOCBF construction.
2. Max-brake intervention / PSF output projection: shelve for FUTURE.
3. Convex polytope recipe (simple, piecewise-LINEAR, not smooth): per MPPI step k,
   outer boundary = intersection of the TANGENT half-spaces (supporting hyperplanes)
   to the NEAR obstacles (near = within a sensing radius / those whose tangent
   actually binds). The nested LEVEL SETS are the (1−γ)^i shrinkings, i=0..N, and
   #level sets = #MPPI horizon steps N. (Current smooth log-sum-exp viz must be
   replaced by the piecewise-linear polytope + N level sets matching the figure.)
4. The polytope is a RULER for sample rejection (DTCBF: reject rollout if
   h(x_{k+i}) < (1−γ)^i). The random perturbations (centered on nominal control) come
   from a flow-matching/DRIFTING model that is REWARD/SAFETY-TILTED so the samples
   are conservative/performative per γ — i.e. tweak the samples by reward/safety
   (analogue of Mizuta's gradient term, but learned + tilted).
5. WHY generative + drifting: produce, in ONE batched shot, many multi-step
   perturbation sequences that SURVIVE the DTCBF rejection — so the generator must be
   reward/safety-tilted (mass on the feasible high-reward region). Drifting = fast
   1-NFE parallel generation. Multimodal collapse is a risk → W2 trust region +
   γ-conditioning + guidance.
6. WHY γ (the real meaning): NOT "multimodal trajectories" for its own sake. By
   enforcing the DTCBF with parameter γ we learn CLASSES of trajectories encoded by
   γ (conservative ↔ performative). The nominal control = a generative policy (like
   Mizuta) but with one extra condition, γ. For each discrete γ we iteratively fit
   the distribution; later close the loop (offline↔online).

## Safety scope (P.S. — do NOT overclaim)
- LTI / double integrator: state is affine in the control sequence, so the averaged
  trajectory is a CONVEX combination of the (DTCBF-feasible, convex-set) rollout
  states => SAFE FOR FREE after averaging. This is the rigorous, restricted claim.
- General nonlinear: averaging can leave the feasible set (non-convex) => do NOT
  claim hard safety; report empirical.
- Possible proof to chase: a SAMPLING-TIME condition that guarantees safety AFTER
  averaging beyond LTI (e.g. all samples in a common convex set in a lifted/feedback-
  linearized coordinate). Open.
- Reviewer 6 answer: h0 (position) is rel-deg-2, not a CBF; the hard certificate for
  the bounded-accel rel-deg-2 channel is the braking margin m ≥ c²/(2a_max) (Chen et
  al. CDC2021), a velocity-tightened CONVEX face — usable inside the convex polytope;
  + ISSf margin inflation for prediction error (Kolathaya-Ames 2019). NOT HOCBF, NOT
  general nonlinear.

## What changes in the build
- Replace smooth log-sum-exp polytope viz with piecewise-linear tangent polytope +
  N=horizon level sets ((1−γ)^i).
- Drop HOCBF (η·ψ₁) and output PSF from the core method (PSF = future).
- The generative proposal = reward/safety-tilted drifting conditioned on γ, trained
  to produce samples that survive DTCBF rejection; offline train + online reject.
- Evaluate OOD (train dist A, test dist B) vs Mizuta — the headline claim.
