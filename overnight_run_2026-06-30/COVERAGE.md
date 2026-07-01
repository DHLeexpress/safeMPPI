# What is "coverage" for a generative policy when the input space is too large?

The FM policy's design is a whole control sequence `U ∈ R^{T×2}` (here `T≈80`, so ~160-D). We want to
say how much of the *safe, goal-reaching behavior* the policy generates ("coverage"), and show it rises
under safe flow expansion. This note argues why coverage must **not** be measured in the raw input space,
and defines the metrics we actually compute (`metrics.py`).

## Why not the raw control-sequence space
- **Curse of dimensionality.** A grid over `R^{160}` needs `~n^160` cells. Intractable to estimate or fill.
- **The valid set is measure-zero.** Collision-free, goal-reaching, DTCBF-certifiable control sequences
  form a thin, curved manifold in `R^{160}`; "volume covered" there is 0 and uninformative.
- **The control→behavior map is massively many-to-one.** Many different `U` roll out to essentially the
  *same path* (e.g. accelerate-then-coast vs constant-speed). Counting distinct `U` overcounts behaviors.

**Conclusion:** define coverage in a **low-dimensional behavior / outcome space** — a function of the
rolled-out path `ξ(U)` — not of `U`. And measure it only over the **verifier-reachable-safe set** `Ω*`
(you can only "cover" what is certifiable-safe-and-reaching).

## The metrics we compute (`metrics.py`)
Let `Ω* = { U : V_γ(U)=1 }` estimated by gating a broad "surrounding" proposal through the SAME compact
SOCP verifier (`build_omega_star_clutter`). All metrics are computed on the policy's **verifier-certified**
samples, normalized by `Ω*`.

1. **Spatial-occupancy coverage (headline).** Rasterize the free workspace into ~0.25 m cells. A trajectory
   "covers" the cells its tube passes through. `spatial_coverage = |cells(certified FM tubes) ∩ cells(Ω*)| /
   |cells(Ω*)|`. Directly the intuitive *"does the policy fill the reachable-safe free space?"* — robust,
   denominator-grounded, and the number that must go **up** during expansion.
2. **Homotopy / corridor mode coverage.** The signature of a path = which side (`±`) of each *on-corridor*
   obstacle it passes (a homotopy class). `mode_coverage = |signatures(certified FM) ∩ signatures(Ω*)| /
   |signatures(Ω*)|`. Captures "does it find *all the distinct routes*" (go-above vs go-below vs thread-gap).
3. **Descriptor-bin coverage.** Bin the lateral-offset-at-each-obstacle descriptor (reuses
   `overnight_run_today/src/descriptors.py`) and take the fraction of `Ω*` bins the policy populates. A finer
   secondary view of (2).
4. **Vendi diversity.** Effective number of distinct behaviors (denominator-free), from the RBF kernel
   spectrum of the descriptor — a sanity check that diversity, not just cell count, is rising.

## Why homotopy is a *secondary* headline in clutter
With `K` on-corridor obstacles there are up to `2^K` homotopy classes, most infeasible; the count is
combinatorially unstable and most classes are empty. So we lead with **spatial occupancy** (a stable,
bounded `[0,1]` scalar) and report homotopy/mode coverage as the interpretable "distinct routes" secondary.

## What "coverage ↑ while safe" means here
Every trajectory entering the numerator is **SOCP-certified** (`socp_gate.py`) — a specific polytope with a
`(1−γ)^t` level-set ruler is constructed and re-checked, sliding along the whole executed path. So the
safety claim is per-trajectory and exact; expansion raising spatial/mode coverage means the policy learns
to generate *more of the certifiable-safe behaviors* (both routes, tighter berths) without ever emitting an
uncertified one into the kept set. γ is the conservativeness knob: it enters the policy context **and** the
verifier ceiling, so lower γ certifies only wider-berth behaviors, higher γ admits tighter gap-threading.
