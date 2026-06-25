# Guided Safe MPPI — Theoretical Contribution

**Goal.** Fix the moving-obstacle failure of MPPI-DCBF (rejection collapses the
sample distribution → robot freezes in a moving crowd) while *keeping* the
convex-averaging safety guarantee, *and* answer Reviewer 6's objection that the
position-only affine barrier (draft Eq. (12)/`eq:aff_barrier`) is not a valid
DCBF for relative-degree > 1 systems.

This file is the proof side of "our method is superior." The statistics side is
in `RESULTS.md`.

---

## 0. Notation recap

Discrete-time control-affine system
$$x_{k+1}=f(x_k)+g(x_k)\,u_k,\qquad x=(p,v),\ p\in\mathbb R^{d},\ v\in\mathbb R^{d}.$$
MPPI samples $u^{(m)}=\bar u+\varepsilon^{(m)}$, weights
$\omega_m\propto e^{-S^{(m)}/\lambda}$ (zero weight = rejected), and applies the
weighted average $u^\star=\sum_m\omega_m u^{(m)}$, $\sum_m\omega_m=1$,
$\omega_m\ge 0$. Under affine dynamics the **averaged state is the convex
combination of the rollout states**, $x^\star_{k+i}=\sum_m\omega_m x^{(m)}_{k+i}$
(one step for control-affine, full horizon for LTI). This is the lever the whole
safety argument hangs on.

The draft's affine barrier is the supporting half-space at the closest boundary
point $p_o$ with outward normal $n$:
$$h^{\mathrm{aff}}(x;x_0)=\frac{n^\top(p-p_o)}{n^\top(p_0-p_o)}.$$

---

## 1. Why the current method fails on moving obstacles (precise diagnosis)

Three coupled defects, all confirmed by the episode-110 metrics
(`safemppi_gamma`: success 0 %, goal never reached, clearance frozen at 0.015):

**(D1) Relative degree.** The barrier is on **position only**, but the input
(acceleration / thrust) has relative degree 2 to position. The one-step decay
condition $h(x_{k+1})\ge(1-\gamma)h(x_k)$ can be **infeasible for every $u_k$**
when the robot already carries velocity toward the obstacle (braking distance >
margin). MPPI then rejects *all* samples → no valid average → freeze. This is
exactly R6's "(12) is not a valid DCBF" — a valid DCBF must encode the dynamics.

**(D2) Static geometry.** $p_o,n$ are computed from the obstacle's *current*
position. For a pedestrian moving with velocity $v_o$, the half-space lags the
obstacle by $i\,\Delta t\,v_o$ over the horizon.

**(D3) Mean–constraint misalignment (the freeze).** When the robot walks
*alongside* a pedestrian toward a shared goal, the warm-start nominal $\bar u$
points roughly along the (lagged) normal's unsafe side. Gaussian samples
$\bar u+\varepsilon$ are centred in the **infeasible** region, so the accepted
fraction $\to 0$. With $\lesssim$ a handful of survivors the MPPI average is
ill-defined and degenerates; the controller emits ≈0 motion → 0 % goal-reach at
near-zero clearance (it neither crashes nor progresses).

Key point: D1–D3 are **feasibility / conditioning** failures, not failures of
the convex-averaging *proof*. The proof (Props 1–2) is correct; it is simply
vacuous when the accepted set is empty or a sliver. The fix must restore a
*non-empty, well-conditioned* accepted set without weakening the guarantee.

---

## 2. Fix 1 — Affine **Higher-Order** DCBF (valid for relative degree 2, still a half-space)

Define the safety value as an affine functional of the **full state** $x=(p,v)$:
$$\boxed{\,h_{\mathrm{ho}}(x;\,p_o,n)\;=\;n^\top(p-p_o)\;+\;\eta\,n^\top v\,},\qquad \eta>0.$$

* **It is a valid relative-degree-aware barrier.** $n^\top v$ is the closing
  rate along the normal; $\eta$ is a look-ahead (braking) horizon. Writing
  $h_0=n^\top(p-p_o)$, this is the discrete **exponential/HO-CBF**
  $h_{\mathrm{ho}}=h_0+\eta\,\Delta_n h_0$ with $\Delta_n h_0 = n^\top v$ the
  per-normal first difference. Choosing $\eta$ places the two HO-CBF "poles" so
  the relative-degree-2 chain $p\!-\!v\!-\!a$ admits a feasible $u$; this is the
  textbook fix for R6's objection. The set $\{x:h_{\mathrm{ho}}\ge0\}$ is the
  *braking-feasible* sub-level set, not the naive position half-space.

* **It is still a half-space — in state space.** $h_{\mathrm{ho}}$ is affine in
  $x=(p,v)$, so $\mathcal C^{\mathrm{ho}}=\{x:h_{\mathrm{ho}}(x)\ge0\}$ is a
  convex half-space of $\mathbb R^{2d}$.

* **The averaging proof survives verbatim.** For the double integrator
  $x_{k+i}$ is affine in the control sequence, hence
  $x^\star_{k+i}=\sum_m\omega_m x^{(m)}_{k+i}$ is a convex combination; if every
  accepted $x^{(m)}_{k+i}\in\mathcal C^{\mathrm{ho}}_i$ (convex) then
  $x^\star_{k+i}\in\mathcal C^{\mathrm{ho}}_i$. **Propositions 1 and 2 hold
  unchanged, now in $(p,v)$-space.** The only change is *which* affine functional
  defines the half-space.

* **Feasibility restored (cures D1).** A robot moving toward the obstacle but
  *decelerating* satisfies $h_{\mathrm{ho}}\ge0$ even with small position margin,
  so samples that brake are accepted instead of universally rejected.

The discrete CBF condition is, as before, affine in $u_k$ (since $p_{k+1},v_{k+1}$
are affine in $u_k$):
$$h_{\mathrm{ho}}(x_{k+i+1})\ \ge\ (1-\gamma)\,h_{\mathrm{ho}}(x_{k+i}),\qquad
h_{\mathrm{ho}}(x_{k+i})\ \ge\ (1-\gamma)^i h_{\mathrm{ho}}(x_k).$$

---

## 3. Fix 2 — Relative-velocity (moving-obstacle) half-space

Predict the obstacle along the horizon, $p_o(k+i)=p_o(k)+i\Delta t\,v_o$, and use
the **relative** closing rate:
$$\boxed{\,h^{\mathrm{mov}}_{\mathrm{ho}}(x_{k+i})\;=\;n_i^\top\!\big(p_{k+i}-p_o(k\!+\!i)\big)\;+\;\eta\,n_i^\top\!\big(v_{k+i}-v_o\big)\,}.$$

* When the robot walks **alongside** a pedestrian, $v_{k+i}\approx v_o\Rightarrow
  n_i^\top(v_{k+i}-v_o)\approx0$, so the barrier reduces to the *position* margin
  and **does not demand braking** → nominal-aligned samples are accepted (cures
  D2 + the walking-with-crowd case directly).
* Still affine in the robot state per step ⇒ half-space ⇒ convex averaging
  preserved. Requires an obstacle-velocity estimate $v_o$, which the pedestrian
  datasets provide (finite-difference of tracks).

---

## 4. Fix 3 — Guidance instead of (only) rejection: predictive safety filter on the sampling **mean** (cures D3, the freeze)

Rejection alone collapses the surviving distribution. Add a one-shot projection
of the warm-start nominal onto the (convex) feasible polytope *before* sampling:
$$\bar u^{\mathrm{safe}}=\arg\min_{u}\ \tfrac12\|u-\bar u\|^2\quad
\text{s.t.}\quad h^{\mathrm{mov}}_{\mathrm{ho}}\big(x_{k+i}(u)\big)\ge(1-\gamma)^i\ \ \forall i.$$

Because every constraint is an affine half-space, the feasible set is a convex
polytope and this is a small convex QP (closed-form against the single active
half-space; a handful of dual iterations for several obstacles). **Centre the
Gaussian samples at $\bar u^{\mathrm{safe}}$.** Now the sampling cloud sits inside
the feasible cone, the accepted fraction is high, and MPPI is well-conditioned.

**The guarantee is untouched.** We still apply the hard half-space rejection as a
backstop, so the averaged plan is provably in the convex feasible set (Props 1–2).
Guidance only changes *where we sample*, never *what we certify*. Hence we keep
the rigorous safety statement **and** remove the freeze. This is a predictive
safety filter (PSF) acting on the sampling distribution, not on the output —
the cleanest way to reconcile "hard guarantee" with "don't reject everything."

---

## 5. Fix 4 — Covariance steering for multi-modality + diversity (ties to CC-/CS-MPPI)

Shape the sample covariance $\Sigma$ anisotropically: **wide along the half-space
boundary (tangent)**, **narrow along the normal**. Samples then spread into the
two homotopy classes (pass left / pass right) while staying feasible — this is
the multi-modality the dataset must contain and simultaneously answers the AE's
"insufficient sample diversity." The tangent/normal split is computed from $n_i$;
optionally use the optimal-covariance-steering update of CS-MPPI (arXiv 1905.13296)
/ CC-MPPI (arXiv 2109.12147) to drive a desired terminal covariance under the
half-space chance constraint. (See `RESEARCH.md` for the exact updates.)

---

## 6. Multiple obstacles, geodesic convexity / foliation

The intersection of the per-obstacle supporting half-spaces is a **convex
polytope**, so convex-combination safety is preserved for several *nearby*
obstacles simultaneously — the non-convexity R6 worried about (union of obstacle
*exteriors*) is avoided by **local convexification** into a corridor. Different
corridors = different leaves of a homotopy **foliation**; within a leaf the
problem is (geodesically) convex and averaging is safe. The CFM **context vector**
selects the leaf (which side to pass), so the learned policy is multi-modal
*across* leaves while each leaf is certified convex. This reframes the draft's
"multi-modal" claim as a foliation over convex corridors rather than a single
non-convex set — a defensible, novel framing.

---

## 7. Guided drifting model (learned policy with a runtime certificate)

The original limitation ("the generative model inherits safety only
statistically, no certificate") is removed by running the **same PSF projection
(§4) as a thin runtime filter on the one-step drifting generator's output**:
$\hat u_k \mapsto \Pi_{\mathcal C}(\hat u_k)$, a closed-form projection onto the
active half-space. One extra affine projection per step (µs-scale), preserving
the one-NFE real-time property, now yields a **hard** per-step guarantee on the
learned controller. This is the "guide the drifting model" idea: the generator
proposes a fast, multi-modal action; the affine filter certifies it.

---

## 8. What each reviewer concern maps to

| Concern | Addressed by |
|---|---|
| R6: (12) not a valid DCBF (ignores dynamics) | §2 affine HO-DCBF in $(p,v)$ |
| R6: multi-obstacle → non-convex; avg unsafe | §6 convex-polytope intersection / corridor foliation |
| R6: M=2 left/right avg unsafe | single *fixed* supporting half-space per step is convex; left/right are different leaves (§6), not both in one half-space |
| R1/R3/R4: no dynamic / real-ish setting, weak baselines | moving-pedestrian benchmark vs Mizuta CFM-MPPI (UCY/SDD) — `RESULTS.md` |
| R4: compare CFM-on-ours vs CFM-on-standard-MPPI | both trained + benchmarked |
| AE: insufficient sample diversity | §5 covariance steering preserves multi-modality |
| Limitation: learned policy lacks certificate | §7 guided drifting with runtime affine filter |

---

## 9. Falsifiable claims to verify experimentally (→ `RESULTS.md`)

1. **C1 (freeze cured).** Guided Safe MPPI accepted-sample fraction ≫ rejection
   MPPI on moving-pedestrian episodes; goal-reach rate rises from ~0 % to high.
2. **C2 (beats Mizuta on safety).** Lower collision rate and higher min-clearance
   than Mizuta CFM-MPPI at matched success, over ≥200 episodes, with confidence
   intervals and a paired significance test.
3. **C3 (beats Mizuta on success).** ≥ Mizuta success rate in moving crowds.
4. **C4 (tunable trade-off).** $\gamma$-schedule sweeps out a Pareto front
   (clearance ↑ as success/▽path-cost trades), which Mizuta (no knob) cannot.
5. **C5 (multi-modal).** Generated trajectories occupy ≥2 homotopy classes.
6. **C6 (guided drifting).** One-step guided drifting keeps the per-step
   certificate at ~1 NFE and matches/beats Mizuta latency.

---

## 10. Decoupling distribution learning from safety (the backbone)

**Algorithm (Guided Safe MPPI with a learned proposal).** Sample M control
sequences from a γ-conditioned proposal q_θ(·|o,γ) learned by flow matching on
accepted, low-cost rollouts; reject any sequence violating
h^aff_j(x_{k+i};x_k) ≥ (1−γ)^i for any active obstacle j; importance-weight-
average the survivors and project the executed control onto the active half-space
polytope — so q_θ is trained purely for feasibility and reward, never for safety.

**Powerful claim.** *Safety is a property of the rejection-and-projection
certificate, not of the sampler*; hence the proposal may be learned freely.

**Proposition 3 (proposal-agnostic average safety).** Let q be ANY proposal over
control sequences. Draw M samples, discard those violating
h^aff_j(x_{k+i};x_k) ≥ (1−γ)^i for any active j, and form
x*_{k+i}=Σ_{m∈A} ω_m x^{(m)}_{k+i}, ω_m≥0, Σω_m=1. For one-step control-affine
(and full-horizon LTI) dynamics, if A≠∅ then x*_{k+i} ∈ ⋂_j C^{aff}_{i,j},
independent of q.
*Proof.* Each accepted state lies in the half-space C^{aff}_{i,j} (rejection
test). x*_{k+i} is a convex combination (affine dynamics ⇒ state affine in
controls) of points in ⋂_j C^{aff}_{i,j}, an intersection of half-spaces (convex);
convex sets are closed under convex combination. q enters only the draw, not the
test or the convexity. ∎

**Proposition 4 (per-step certificate when A=∅).** Let Π be the projection onto
H(o,γ)={u : h^aff_j(x_{k+1}(u)) ≥ (1−γ)h^aff_j(x_k), ∀ active j}. Under pointwise
feasibility (Assumption 1, H≠∅), the executed control Π(u) gives
x_{k+1}∈⋂_j C^{aff}_{1,j} for any u and any proposal, even if no sample was accepted.
*Proof.* H is a nonempty closed convex polytope (intersection of half-spaces); the
Hilbert projection onto it exists, is unique, and satisfies every constraint by
definition ⇒ x_{k+1} safe. The averaging step is bypassed; the certificate is the
projection. ∎

**Consequence (separation of concerns).** Prop 3 + Prop 4 ⇒ the executed control
is certified safe whether or not the sampler produces feasible samples. So q_θ is
optimized ONLY for accept-rate, cost, and multimodality; safety is structural and
γ-tunable. This is strictly stronger than Mizuta's soft CBF guidance reward
r_safe = γ·min{0, ḣ+α(h)} (no hard guarantee): ours is a hard certificate
indifferent to how the distribution is learned.

**Honest caveats (state in paper).** (i) h^aff is a linear inner-approximation of
the obstacle exterior — conservative-safe (the tangent half-space excludes the
disk). (ii) Runtime Π uses a few Jacobi sweeps + a small margin buffer: exact for
a single active constraint, approximate for several (same flavor as Mizuta's
soft guidance). (iii) Guarantees are w.r.t. the model, including the
constant-velocity obstacle prediction; real pedestrian deviation is handled
empirically by the margin buffer and the relative-velocity (HOCBF ψ_1) term.
