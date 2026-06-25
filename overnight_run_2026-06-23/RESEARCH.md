# Safe-MPPI Research Synthesis (for ICRA resubmission)

Mechanism-level synthesis from web research + the local competitor PDF
(Mizuta & Leung, arXiv:2508.01192v3). Priority: topics 4/5/6 (our fix), then
1–3 (baselines), then 7 (crowd baselines). Used to position and harden the
theory in `THEORY.md`.

## Topic 4 — Higher-Order / exponential discrete CBF (relative degree 2)
**Our `h_ho = h0 + eta*(v - v_obs)·n` is exactly `eta*psi_1`**, the first cascade
term of an HOCBF with linear class-K `alpha_1(h0) = (1/eta) h0`:
```
psi_0 = h0(p);  psi_1 = h0_dot + (1/eta) h0 = n·(v-v_obs) + (1/eta) h0
```
`{psi_1>=0}` is the (position,velocity) half-space; control (accel) enters at
`psi_2 = psi_1_dot + alpha_2(psi_1) >= 0`. **Frame our barrier as psi_1, not an
ad-hoc additive barrier.** Invariance is for the intersection
`C1∩C2 = {h0>=0}∩{psi_1>=0}`; per-rollout we should also check that next-step
psi_1 stays >=0 (the discrete psi_2 layer).
- Discrete exp-CBF (Agrawal–Sreenath): `h(x_{k+1}) >= (1-gamma) h(x_k)` =>
  `h(x_k) >= (1-gamma)^k h(x_0)` — **literally our rejection threshold.**
- eta selection = class-K `1/k_1` / pole placement / braking time-constant;
  with `|u|<=u_max`, feasibility ~ `(1/eta)h0 >= sqrt(2 u_max h0)`.
- Pitfalls to cite: eta too large => arrive too fast, psi_2 infeasible under
  bounded accel; eta too small => inner set tiny => mass rejection (our symptom);
  IC trap (Xiao–Belta Rmk 2): h0=0 & h0_dot<0 => psi_1<0 for any eta.
- Cites: Nguyen–Sreenath ACC2016; Xiao–Belta CDC2019 arXiv:1903.04706 / TAC2022;
  Agrawal–Sreenath RSS2017; Xiong et al. T-Cyb 2023.

## Topic 5 — Moving-obstacle / relative-velocity CBF
`h = ||p_r-p_o(t)||^2 - R^2` => `h_dot = 2(p_r-p_o)·(v_r-v_o)`; explicit time
term `∂h/∂t = -2(p_r-p_o)·v_o`. So the **relative-velocity term is literature
standard** and matching the pedestrian (`v_r->v_o`) zeroes the closing term =>
robot may walk alongside (this is our mass-rejection fix, rigorously).
- Cites: Xu–Tabuada–Grizzle–Ames IFAC2015; Lindemann–Dimarogonas LCSS2019
  (time-varying CBF with `∂b/∂t`); Ames et al. ECC2019 arXiv:1903.11199;
  Dai et al. arXiv:2309.17226 (ICRA2024, moving-obstacle TV-CBF); Collision-Cone
  CBF Tayal et al. arXiv:2403.07043 / 2209.11524; unknown-velocity robust
  variant Kim–Diagne–Krstic arXiv:2412.03678 (ACC2025).

## Topic 6 — Predictive safety filter as projection (CONFIRMS our guidance)
Projecting the MPPI **mean** onto the feasible half-space before sampling does
**not** weaken the per-sample rejection guarantee: the guarantee is a property of
the accept/reject predicate applied to every realized sample; moving the mean
only raises acceptance (variance reduction / warm start). Hilbert projection =>
unique; single half-space closed form `u* = u_nom + max(0, b - a^T u_nom)/||a||^2 a`.
**Caveat:** closed form exact only for ONE half-space; multi-obstacle needs the
QP `min||u-mu||^2 s.t. A u<=b` (we use a one-sweep Jacobi approx; guarantee still
holds because rejection is the certificate).
- Cites: Wabersich–Zeilinger Automatica2021 arXiv:1812.05506; Ames et al.
  TAC2017 arXiv:1609.06408 (CBF-QP = projection); Data-Driven Safety Filters
  arXiv:2311.13824. Closest MPPI+CBF: Rabiee–Hoagg arXiv:2410.02154
  (per-sample composite-CBF filter, no rejection); Tao/Kang arXiv:2111.06974
  (CBF shifts the sampling distribution — direct precedent for mean-projection).

## Topic 1 — CC-MPPI (covariance-controlled MPPI)
Appends state-feedback `K_k y_k` to samples; gain from a convex SDP covariance-
steering with terminal-covariance LMI. **Obstacles do not enter the SDP** (only
the rollout cost). Our anisotropic (tangent-wide/normal-narrow) covariance is a
geometry-aware, obstacle-coupled instance — cite CC-MPPI as precedent, then
differentiate. Cite: Yin–Zhang–Theodorou–Tsiotras ICRA2022 arXiv:2109.12147.

## Topic 2 — Shield-MPPI (key contrast)
CBF as (1) augmented running cost and (2) **post-hoc gradient repair of the
single averaged control** (heuristic, no guarantee). **Positioning (verbatim):**
Shield-MPPI = *average-then-repair* (heuristic); ours = *constrain-then-average*
(convex combination of feasible samples stays feasible). Honest caveat: our
guarantee is w.r.t. the linear half-space approximation. Cite: Yin–Dawson–Fan–
Tsiotras arXiv:2302.11719 (RA-L 2023).

## Topic 3 — CS-MPPI / covariance steering, constrained linear
Chance-constraint tightening `a^T mu_k + Phi^{-1}(1-delta) sqrt(a^T Sigma_k a) <= b`
=> mean inside polytope shrunk by std-dev projected on the constraint normal.
**Rigorous backing for narrow-along-normal covariance.** Cite: Okamoto–Tsiotras
arXiv:1905.13296; RA-L companion arXiv:1809.03380.

## Topic 7 — Crowd baselines + the direct competitor
Social-Force (Helbing–Molnar 1995), RVO/ORCA (van den Berg 2008/2011), MPC-CBF
(Zeng–Zhang–Sreenath ACC2021 arXiv:2007.11718), DWA, CADRL (arXiv:1609.07845),
SARL (ICRA2019), Social-GAN (CVPR2018), Diffuser (arXiv:2205.09991), MID
(CVPR2022).

**Direct competitor — Mizuta & Leung, "Unified Generation-Refinement Planning:
Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation,"
arXiv:2508.01192v3** (UW+NVIDIA; project cfm-mppi.github.io). They identify the
same averaging pathology ("combining distinct modes through a single cost-
weighted average ... lies in an infeasible region between modes") and fix it by
**mode-selective MPPI** (top-K* CFM modes, refine each with parallel MPPI, pick
best) + a **soft CBF reward** `r_safe = gamma*min{0, h_dot+alpha(h)}` on the
generator. **Our differentiation:** hard per-sample HOCBF half-space with a
provable average-safety guarantee (constrain-then-average), vs their heuristic
mode separation + soft generative reward (no hard guarantee). Their suite
(ETH/UCY, SDD, SFM 20-agent; baselines MPPI/Diff-MPPI/CFM/CFM-MPPI) is our
ready-made evaluation. Neighbors to cite: CoBL-Diffusion (Mizuta–Leung IROS),
Stein-Variational MPPI (Honda ICRA2023), Biased-MPPI (Trevisan–Alonso-Mora
RA-L2024), Contingency-MPPI (Jung L4DC2025).

### Caveats
Double-check arXiv:2109.12147 author list and arXiv:2508.01192 venue before
camera-ready; a few primary PDFs were confirmed via authoritative restatements.
