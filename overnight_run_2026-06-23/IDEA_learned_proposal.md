# Idea: Learned γ-conditioned Feasible-Set Proposal for Safe MPPI

**Origin:** Dohyun's idea (2026-06-23) — learn the sampling distribution (the
half-space to recover) with flow matching from success cases, and iterate.
**Verdict:** valid research-level contribution; complementary to the γ-knob core,
does NOT replace the safety certificate. Slot in as a 3rd contribution.

## Problem it solves
Gaussian-around-nominal proposal puts ~7% of mass in the γ-feasible polytope
(observed infeasibility_rate ≈ 0.93) => mass rejection, freeze, low diversity.

## Formulation
At MPC step, context o = (state, nearest-obstacle geometry, pedestrian vels),
safety knob γ. Feasible control-sequence set
  F(o,γ) = { U : h_j(x_{k+i}(U)) ≥ (1-γ)^i h_j(x_k), ∀ active j, ∀ i }.
Target = reward-tilted feasible distribution
  p(U | o, γ) ∝ 1[U ∈ F(o,γ)] · exp(-S(U)/λ).
Learn proposal q_θ(U | o, γ) by conditional flow matching on ACCEPTED, low-cost
samples from guided MPPI (the "success cases").

## Inference (one MPC step) — guarantee preserved
1. Draw M candidates: (1-ρ)M from q_θ(·|o,γ), ρM Gaussian-around-nominal (coverage).
2. Apply the SAME hard DCBF rejection (+ PSF projection backstop).
3. Importance-weighted average over accepted samples (constrain-then-average ⇒
   provably in the convex feasible set; Props 1-2 unchanged).
4. Apply first control (optionally PSF-projected => hard per-step certificate).

The certificate is steps 2-4; q_θ only changes step 1. γ conditions q_θ AND the
threshold => knob fully retained and tunable.

## Refinement loop (the "reiterate") — CEM/DAgger-flavored
- Round 0: q_θ ← flow matching on accepted samples from Gaussian-proposal guided MPPI.
- Round t≥1: run MPPI with proposal q_θ over scenarios; collect newly accepted
  low-cost samples; append to buffer; retrain q_θ. Single q_θ for all γ (conditioned).
- Stop on accept-rate / success plateau.
- Anti-collapse: keep Gaussian mixing ρ>0, γ-conditioning, entropy/diversity reg.

## Positioning (carve novelty honestly)
- = flow generalization of CEM (refit expressive flow to FEASIBLE elites, not a Gaussian to elites).
- vs Biased-MPPI (Trevisan & Alonso-Mora RA-L'24), Stein-MPPI (Honda ICRA'23):
  those bias with controllers/SVGD; ours is a learned γ-conditioned feasible-set
  generator carrying a hard CBF constrain-then-average certificate.
- vs Mizuta (CFM→mode-selective MPPI): ours learns the PROPOSAL to match the
  γ-feasible reward-tilted law, retains the averaging-safety proof + tunable γ.

## Risks / caveats
- Hybrid cost: CFM inference inside MPPI loop. Mitigate: 1-NFE drifting generator as proposal.
- Mode collapse / distribution shift over rounds => mixing + diversity reg.
- Need ≥1 feasible sample for the average; PSF projection is the backstop (already built).

## Self-improvement via Expert Iteration (Dohyun, 2026-06-23)
MPPI+certificate = a policy-IMPROVEMENT operator (reject+reweight+project => a
better, CERTIFIED distribution). The one-step drifting model AMORTIZES it. Iterate.
KEY: by Props 3-4 every iteration is safe by construction => self-improve with zero
safety risk (the proposal quality never affects the guarantee).

Also: since safety is in the CERTIFICATE, the proposal need NOT be γ-conditioned —
train q_θ once on ego data (dataset/train80_ego.pt, 273989 traj, SAME source as
Mizuta, via build_canonical_from_mizuta); put the entire γ-knob in the certificate
(threshold (1−γ)^i + projection) => tunable safety with no retraining.

### Criteria for which samples update the vector field (ranked)
1. Reward-weighted (not success-binary): weight by the MPPI weight itself
   ω_m=exp(−(S_m−β)/λ) over the certificate-FEASIBLE set (RWR/AWR; self-consistent
   with MPPI). "Successful-only" is the degenerate hard-threshold special case.
2. Advantage gate: reinforce only S_m < baseline(o) (CRR/AWAC) — monotone improvement.
3. DAgger at failure states: at stall/collision/large-projection states, inject the
   guided-MPPI / projected expert action as high-weight label (fixes the
   stall-then-collide distribution shift directly).
4. Diversity quota: cluster accepted samples by pass-side (homotopy), reweight per
   mode + entropy bonus + keep exploratory Gaussian mix (anti-collapse, keeps multimodality).

### One-step drifting loss (batched), iteration k
  run MPPI(q_θ^k)+certificate -> {U_m,S_m}, feasible A
  ω_m = softmax_{m∈A}(−(S_m−β)/λ) · 1[A_m>0] · mode_balance(m)
  L(θ) = Σ_m ω_m ‖ g_θ(ε_m,o) − U_m ‖²      # g_θ = one-step drifting generator
  + DAgger targets at failure states ; iterate to accept-rate/success plateau (held-out).

### Risks
- Reward-hacking the certificate (emit junk rescued by projection): penalize
  correction magnitude / train on the projected action.
- Compounding shift / mode collapse: criteria 3+4, exploration mix, entropy floor.
- Stop on held-out plateau, not training scenarios.

## Cheapest first experiment (round-0, near-free)
Use the γ-conditioned CFM already being trained on guided data as the MPPI
PROPOSAL (method `cfm_proposal_mppi`): sample M sequences from CFM, run DCBF
rejection + weighted average + PSF. Measure: accept-rate (expect ≫7%), success,
collision, multimodality vs Gaussian-proposal guided MPPI and vs Mizuta.
Implement AFTER sweep+headline so the core γ-knob result is undisturbed.
