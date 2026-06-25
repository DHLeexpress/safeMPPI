# Current Status — Safe Sampling-Based Control for Crowd Navigation (2026-06-24)

Honest, comprehensive snapshot of where the project stands and the path forward.

## 1. Problem & goal
ICRA-2026 submission #5447 ("Generative Robotic Control with Tunable Safety")
was rejected. Resubmission. Core aim: make a SAMPLING-BASED safe controller
(MPPI) work in MOVING-pedestrian navigation (UCY/SDD), with a tunable safety knob
γ, and a learned generative policy — positioned against the competitor
Mizuta CFM-MPPI (arXiv:2508.01192). Double integrator, dt=0.1, receding horizon.

## 2. The journey & what we built (components)
1. **Diagnosis**: last year's MPPI-DCBF (reject samples violating an affine
   half-space) FREEZES in moving crowds — it rejects ~all samples (accept ≈1%),
   so the robot stalls (0% goal-reach on the failure episode). Confirmed
   objectively.
2. **Affine higher-order DCBF** h_ho = h0 + η·nᵀv (= η·ψ₁ of a discrete HOCBF):
   a half-space in (p,v), valid for relative-degree-2, keeps convex-averaging
   safety. Partially answers Reviewer 6.
3. **Guided Safe MPPI**: multi-obstacle half-space barrier + PSF mean-projection
   guidance + anisotropic covariance + output projection filter.
4. **Theory (Props 1–4)**: convex-combination-of-feasible-samples-is-safe
   (1–2); SAFETY IS PROPOSAL-AGNOSTIC — the certificate (reject+project) does not
   depend on the sampler, so distribution learning and safety are decoupled (3–4).
5. **Learned proposal**: cfm_proposal_mppi, guided_drifting; the certificate makes
   ANY policy (incl. Mizuta) safe + γ-tunable.
6. **Mirror-map proposal (current frontier)**: log-barrier mirror map →
   FEASIBLE-BY-CONSTRUCTION samples → accept rate 1.2% → ~96%. MPPI averaging is
   genuinely exercised. Tunable convex polytope with (1−γ)ⁱ level-sets (visualized).
7. **Visualizations (locked in)**: convex polytope level-sets around the robot
   (compact at γ=0.1, fanned at γ=0.5, robot = deepest), with the proposal sample
   cloud trapped inside; single-episode + 3-dense-in-a-row videos.

## 3. What WORKS (validated, held-out, CIs + paired tests)
- **Guided Safe MPPI vs last-year rejection**: collision 50%→7% (UCY), 18%→1%
  (SDD); success +25–35 pts; all p<1e-13. The freeze/averaging pathology is solved.
- **vs Mizuta**: UCY 75%/7% (Mizuta 98%/1%), SDD 91%/1% (Mizuta 100%/0%);
  ~2× faster (45 vs 100 ms); clearance ≈ parity; tunable γ Pareto front; provable
  per-step safety Mizuta lacks. We do NOT beat Mizuta on raw success.
- **Mirror-map**: accept rate 1.2%→~96% (the core fix); feasible sampling; tunable
  polytope; nav tuned to ~83%/8% (sweep best balance), 0%-collision conservative
  Pareto point available.
- Mizuta wrapped in our certificate: keeps ~96% success, adds γ-knob + higher
  clearance (Flavor A) — a clean Props 3–4 demo.

## 4. What DOESN'T / open issues (honest)
- **Raw success below Mizuta** everywhere (reactive-planner ceiling vs a 274k
  real-human-trajectory anticipatory policy). Densest crowds (20+ peds packed) are
  near-impassable for a reactive planner (~40% collision on the densest 5).
- **The learned proposal is NOT yet reward-tilted/trained**: mirror_mppi currently
  samples Gaussian-in-dual + decode (un-learned). The CFM proposal trained on
  guided/SFM data matched the Gaussian baseline (bounded by its teacher).
- **Reviewer 6 ("not a valid DCBF") only partially addressed**: we have the HOCBF
  η·ψ₁; we still need the margin → velocity-tracking → safety formalization
  (backstepping CBF / dual relative degree) to fully close it.
- Collision diagnostic: residual collisions are 2/3 constant-velocity prediction
  error + 1/3 cornering (H=∅), not sampler infeasibility.

## 5. Contributions we can defend
1. Affine HO-DCBF making MPPI averaging provably safe (Props 1–2); Reviewer-6 lever.
2. Proposal-agnostic safety certificate (Props 3–4) — decouple learning from safety.
3. **Mirror-map feasible-by-construction sampling** (accept 1%→96%) — the principled
   cure for the averaging pathology; the sampling distribution lives in the polytope.
4. Tunable γ convex-polytope safety with (1−γ)ⁱ level-sets (the visualization).

## 6. CURRENT FRONTIER (what we proceed on now, max effort)
A. **Viz upgrade**: show the ROLLOUT samples (full proposal trajectories trapped in
   the polytope) and how the convex polytope is CONSTRUCTED (hyperplanes per
   pedestrian), not just next-step dots. (Dohyun's request.)
B. **Reward-tilted learned proposal**: the MPPI weights exp(−S/λ) ARE a reward
   tilting; train a CONDITIONAL flow / one-step generator q_θ(U|o,γ) to sample the
   reward-tilted FEASIBLE target p ∝ 1[U∈F]·exp(−S/λ) (mirror map for feasibility +
   reward tilting for performance). Refs: reward-tilted distributions
   (arXiv:2606.02884; emergent-mind topic), safe sampling-based RL (arXiv:2605.19469),
   classical RWR/AWR backbone. Amortize to few-NFE for real-time.
C. **Safety / Reviewer 6 via margin → velocity-tracking**: Dohyun's claim — a
   polytope with a little MARGIN certifies safety through velocity tracking. Connect
   to backstepping CBF (Taylor et al., CDC 2022) and dual relative degree
   (arXiv:2504.00397): the position half-space + margin + a tracked velocity bound
   (braking distance nᵀ(p−p_o) ≥ (nᵀv)₊²/(2 a_max)) becomes a valid CBF for the
   relative-degree-2 double integrator, robust to pedestrian-prediction error
   (input-to-state safety). This is the rigorous fix to "not a valid DCBF".

## 7. Plan
1. (now) Upgrade the viz to show rollout samples + polytope construction.
2. (now) Fold the two research syntheses (reward tilting; backstepping/dual-rel-deg)
   into THEORY.md + a method section.
3. Train the reward-tilted conditional flow proposal on MPPI-accepted, reward-weighted
   samples; evaluate accept-rate, success, multimodality; amortize to few-NFE.
4. Formalize + state the margin/velocity-tracking safety proposition (Reviewer 6).
5. Final held-out comparison vs Mizuta with the learned reward-tilted proposal.
