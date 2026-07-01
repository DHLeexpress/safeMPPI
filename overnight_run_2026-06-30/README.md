# overnight_run_2026-06-30 — Safe Flow Expansion smoke test (static cluttered scene)

A self-contained, qualitative smoke test of the Pillar-4/5 pipeline **before** the full pedestrian-data
run: pretrain a γ + scene-conditioned flow-matching (FM) policy on the *real* `best_area_mode4`
double-integrator SafeMPPI planner's rollouts in a static cluttered scene, then **expand** it under the
compact **SOCP polytope verifier**, proving coverage↑ while every kept trajectory stays certified-safe.
See `GOAL.md` (pillars) and `COVERAGE.md` (the coverage argument).

## Pipeline
```
env.py  →  build_dataset.py  →  pretrain.py  →  expand.py  →  video.py
run everything:  python run_smoke.py --stage all --smoke      # fast (CPU, ~minutes)
                 python run_smoke.py --stage all               # full run
```

| stage | file | what it does |
|---|---|---|
| 1 env | `env.py` | static cluttered scenes (`random_spheres_2d`, adapted to the planner's meter scale: ~5.6 m box, 8–10 obstacles, dt=0.1, u_max=2). |
| 2 data | `build_dataset.py` | deploy `best_area_mode4` SafeMPPI closed-loop over a **navigability-screened scene bank × γ × seeds**; record executed control sequences + conditions → `results/dataset.pt`. |
| 4 pretrain | `pretrain.py` | joint CFM training of the DeepSets scene encoder + FlowPolicy → `results/pretrained.pt` + `figures/pretrained_gamma_overlay.png` (γ-colored seed). |
| 3+5 expand | `expand.py` | rebind `safeflow.validity_label`→SOCP gate, `safeflow.evaluate`→clutter coverage; run `run_safeflow` (Alg 1) per γ, α=0, D₀ seeded with the certified Ω* → `results/expand_g*.pt`. |
| video | `video.py` | pretrained (faint) → certified trajectories fill the space over rounds → `figures/safeflow_expansion.gif` + `_compare.png`. |

## The core design answer — how the scene enters the FM conditioning (`scene_encoder.py`)
Mizuta/safeGPC conditioned on a **single nearest-obstacle vector**, which jumps as the nearest obstacle
changes and encodes "no obstacle" as an ambiguous `[0,0,0,0]`. Instead we use a **permutation-invariant
DeepSets set encoder** over *all* sensed obstacles: per-obstacle feature `[Δx, Δy, dist, r+r_robot,
presence]` (relative to start) → shared MLP → masked mean+max pool → scene token. Context =
`[start/S, goal/S, γ, scene_token, n_sensed/n_max]` (`ctx_dim = 6 + token_dim`).
- **Changing obstacle set** → handled by construction (permutation-invariant, count-agnostic; encodes the
  whole local set, not a jumpy nearest).
- **No obstacle** → learned `empty_token` + explicit `n_sensed=0` scalar, unambiguously distinct from
  "obstacle sitting at the robot" (fixing the old `[0,0,0,0]`).
Unit-tested in `scene_encoder.py` (`python scene_encoder.py`).

## The verifier (`socp_gate.py`) — `ieee_compact_polytope_verifier_package`
The compact **SOCP** `max Σ w_i m_i  s.t.  a_i·(q_t−c) ≤ β_t m_i,  r_i‖a_i‖ ≤ a_i·(o_i−c) − m_i,
‖a_i‖≤1, m_i≥m_min` (`β_t = 1−(1−γ)^t`), one variable tangent face per sensed obstacle + artificial
K-gon boundary anchors, each block solved **exactly** by its feasible angular interval; `check_certificate`
re-verifies `H_P(q_t) ≥ (1−γ)^t` (a sound certificate). The compact verifier certifies one H=10 local
rollout; a full ~80-step FM trajectory is gated by a **sliding window** (`socp_certify_trajectory`), re-centered
each step, certified iff every window certifies. γ is single-sourced (policy ctx **and** verifier ceiling).

## Reuse (not re-implemented)
- `overnight_run_today/src/{flow_policy, safeflow, dynamics, descriptors, uncertainty}.py` — `FlowPolicy`
  (unchanged; ctx just grows), `run_safeflow` (ACTFLOW Alg 1), descriptors/coverage. The SOCP gate + clutter
  coverage are dropped in by **rebinding** `safeflow.validity_label` / `safeflow.evaluate` (no loop fork).
  One minimal, backward-compatible edit: `run_safeflow(..., init_pos=None)` to seed D₀ (paradigm's
  "D₀ = verified-safe demos"); default None = unchanged behavior.
- `cfm_mppi/safegpc_adapter/safemppi.py` + `overnight_run_2026-06-28/best_area_mode4.json` — the frozen data engine.
- `random_spheres_2d` copied from the sibling `safeGPC/utils/`; SOCP core copied from the ieee package.

## Result (what to look for)
`figures/safeflow_expansion_compare.png` + `safeflow_expansion.gif`: the pretrained seed is a diffuse spray;
after expansion the SOCP-certified trajectories resolve into **distinct multi-modal routes** (go-above /
go-below) that all reach the goal, while every kept trajectory is verifier-certified. Full-run numbers
(8-scene bank, 55 demos, fixed scene 0, `expand_summary.json`):

| γ | spatial coverage | mode coverage | Vendi |
|---|---|---|---|
| 0.3 | 0.58 → **0.87** | 1.00 → 1.00 | 8.8 → 13.7 |
| 0.5 | 0.54 → **0.95** | 1.00 → 1.00 | 8.4 → 17.2 |
| 0.7 | 0.46 → **0.86** | 0.50 → 1.00 | 6.7 → 11.3 |

`spatial_coverage` (of the verifier-reachable-safe Ω*) rises ~0.5 → ~0.9 for every γ and `mode_coverage`
reaches 1.0. Raw-sample validity stays low (~0.04, dashed curve in the gif) — the verifier discards most
diffuse samples; the *kept* set covers the space (see caveats).

## Honest caveats (see also `NOTE.md`, `overnight_run_today/FINDINGS.md`)
- **Scope.** The encoder is trained across a scene **bank** so it genuinely uses obstacle geometry, but
  **expansion is on one fixed scene** (cross-scene generalization is out of scope for this smoke test).
- **σ-acquisition (Eq 9/10) is near-uniform** here; the broad "surrounding" proposal does the mode discovery
  (kept, `rho0>0`) — same finding as the toy. Its marginal value over uniform selection is unproven.
- **Raw-sample validity is modest** (weak few-demo seed): most FM samples are discarded by the verifier;
  the *kept* set covers the space. Full-run (more demos + steps) raises it. Safety is per-kept-trajectory exact.
- **DI position-space certificate.** For a static scene, verifying the full realized path (never certifies a
  collision; collision-free checked separately) is sound enough qualitatively; the relative-degree-2 braking
  gap only bites for moving obstacles / beyond-horizon guarantees (out of scope). This is the SI-safe framing
  `GOAL.md` endorses.
