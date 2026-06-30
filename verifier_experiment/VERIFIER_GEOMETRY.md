# Pillar-3 verifier — robot-centered level sets as a geometric feasibility/optimization problem

**Goal.** Given an H-step control sequence (→ trajectory `x_0,…,x_H`, `H` = the MPPI horizon, here 10), decide whether
there **exists at least one polytope** `P` (with a safety parameter `γ`) such that the trajectory satisfies the
recursive DTCBF inside `P` — i.e. the trajectory is *intrinsically safe for the next H steps*. The polytope's level
sets are **centered at the robot**, and the **slope of those level sets is the decision variable, set by `γ`**.
If no such `P` exists, **report infeasibility** (with the binding obstacle/step).

## 1. Robot-centered polytope and its barrier
Anchor at the robot's current position `c := x_0`. A polytope
`P = { x : aₖ·(x−c) ≤ bₖ , k=1..K }`, with unit normals `aₖ` and offsets `bₖ > 0` (so `c` is strictly interior).
Define the **robot-normalized barrier**
```
        H_P(x) = min_k  ( bₖ − aₖ·(x−c) ) / bₖ          (= 1 at x=c,  = 0 on ∂P,  < 0 outside).
```
Its super-level sets are **scaled copies of `P` about `c`**:
```
        { H_P ≥ ℓ } = { x : aₖ·(x−c) ≤ (1−ℓ) bₖ ∀k } = P shrunk by factor (1−ℓ) toward c.
```
So the nested sets `{H_P ≥ ℓ}` are concentric polytopes contracting to the robot as `ℓ→1`.

## 2. Recursive DTCBF = staying inside the contracting level sets
Require `H_P(x_{i+1}) ≥ (1−γ) H_P(x_i)` for `i=0..H−1`. Since `H_P(x_0)=H_P(c)=1`, this telescopes to
```
        H_P(x_i) ≥ (1−γ)^i ,   i = 0..H.                              (DTCBF)
```
**The decay rate `(1−γ)` is the *slope* of the nested level sets** — how fast the admissible set contracts per
step. Small `γ` (slow decay) ⇒ the trajectory must stay in *large* level sets ⇒ **conservative**; large `γ`
(fast decay) ⇒ it may approach `∂P` quickly ⇒ **aggressive**. `γ` is the single slope knob.

Per face, (DTCBF) is equivalent to the **linear** condition
```
        aₖ·(x_i − c) ≤ ( 1 − (1−γ)^i ) bₖ ,   ∀ k, i.                 (CONTAIN)
```

## 3. Safety = each obstacle excluded by some face
Obstacle `j` (center `oⱼ`, inflated radius `ρⱼ = rⱼ + r_robot`) must lie outside `P`: some face separates it,
```
        aₖ·(oⱼ − c) ≥ bₖ + ρⱼ   for some k.                          (SEP)
```

## 4. The existence problem and its closed form
**Verifier:** ∃ `(A, b>0)` and minimal `γ ≤ γ_max` satisfying (CONTAIN) ∧ (SEP)? With one face per obstacle along
the natural direction `aⱼ = (oⱼ−c)/‖oⱼ−c‖`, write the **clearance** `Cⱼ = ‖oⱼ−c‖ − ρⱼ` and the **trajectory
projection toward `j`** `pⱼ,ᵢ = aⱼ·(x_i − c)`. Then (SEP) is `bⱼ ≤ Cⱼ` and (CONTAIN) is
`bⱼ ≥ pⱼ,ᵢ /(1−(1−γ)^i)`, so a feasible offset exists iff
```
        max_i  pⱼ,ᵢ / (1 − (1−γ)^i)  ≤  Cⱼ           (per obstacle j;  only i with pⱼ,ᵢ > 0 bind).
```
Solving for the smallest admissible `γ` is **closed form**:
```
        req_γⱼ = max_{i: pⱼ,ᵢ>0}  [ 1 − (1 − pⱼ,ᵢ/Cⱼ)^{1/i} ] ,      req_γ = max_j req_γⱼ.
```
- **Certified** iff every `Cⱼ > 0` (robot starts clear), every `pⱼ,ᵢ < Cⱼ`, and `req_γ ≤ γ_max`. Return the
  certifying polytope (`aⱼ`, `bⱼ ∈ [pⱼ,·/(1−(1−req_γ)^·) , Cⱼ]`) and its level sets.
- **INFEASIBLE** (report the binding `(j,i)`) iff: `Cⱼ ≤ 0` (start in collision); or `pⱼ,ᵢ ≥ Cⱼ` (the trajectory
  projects *past* obstacle `j` — no obstacle-free `P` can contain it on the safe side); or `req_γ > γ_max`.

There is no LP solver needed in 2-D with per-obstacle normals — it is a per-(obstacle,step) closed form (and
generalizes to an LP/SOCP if normals are also optimized, or to 3-D). The construction is **sound**: `req_γ ≤ γ_max`
guarantees the recursive invariant, hence forward safety for the next H steps.

## 5. Why this is LESS conservative than the nominal `polytope_v2`
The nominal `polytope_v2` is the **fixed** sensing-disk K-gon (radius `R`) ∩ obstacle tangents — its faces and
offsets are *predetermined*, and it is **bounded by `R`**. A goal-seeking / gap-threading trajectory that travels
beyond `R`, or that needs a polytope *elongated along the gap*, leaves the nominal's level sets ⇒ the nominal
**rejects** it. The verifier instead **chooses** the polytope: it keeps only the faces that the trajectory could
violate, drops the `R`-bound, and orients/sizes `P` to the path — so it **certifies** the gap-thread (∃ `P`) while
remaining sound (still `req_γ ≤ γ_max`). Conservativeness is thus an artifact of *fixing* the polytope; the verifier
removes it by searching over polytopes.

## 6. Relation to `overnight_run_today/src/dtcbf.py`
`dtcbf.verify` implements the *obstacle-anchored* distance-DTCBF (per-step normal `(x_i−c_j)/‖·‖`, decay on the raw
clearance) — a sound sibling. This module uses the **robot-anchored** polytope barrier (fixed normals `aⱼ` at `x_0`,
level sets centered at the robot) per the project's framing; both are sound, and `req_γ` matches at first order.
Horizon: the verifier certifies exactly the **H-step** trajectory the MPPI produces (verifier horizon ≡ MPPI horizon).
