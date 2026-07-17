# Forensic audit (2026-07-16d): expert V₂ < 1 and pretrained-model provenance

Two user concerns, both resolved with measurements. Everything here is reproducible from this
folder (`pretrain_audit/`: copied per-γ data, patched training script, fresh checkpoint, logs).

## Concern 1 — "expert V₂ should be ~1 by construction": RESOLVED, structural, not a bug

V₂ decomposition of the M=100 SafeMPPI expert rollouts (walled scene, goal (4.7,4.7), point robot):

| γ | taskspace fail | approach fail | SOCP fail |
|---|---|---|---|
| 0.1 | 0% | 12% | **0%** |
| 0.3 | 0% | 18% | 6% |
| 0.5 | 0% | 12% | **32%** |
| 1.0 | 0% | 16% | 15% |

- The approach criterion (net progress in every sliding window, ANDed over ~70 windows/trajectory)
  costs 12–18% on its own — that share of V₂<1 is by construction of V₂, as the user noted.
- The SOCP failures are structural to the certificate, NOT robot-body inflation:
  **r_robot = 0.0 (point robot) and PLAN_MARGIN = 0.0** — verified at runtime
  (`grid_scene.py:14,19`; `env.r_robot == 0.0`; every SOCP call passes `float(env.r_robot)`).
  Verifier folder (verified via `module.__file__`): `overnight_run_2026-07-01/verifier_polytope.py`.
- Mechanism: the verifier fits ONE tangent face per sensed obstacle per 10-step window
  (margin m ≤ distance-to-disk) and requires a·x_t ≤ β_t·m with β_t = 1−(1−γ)^t. A window that
  **wraps around** an obstacle cannot be linearly separated from the disk by a single face, and
  early fast approach at small clearance blows up (a·x_t)/β_t. Measured at γ0.5: failing expert
  windows have median true clearance 0.049 (passing: 0.159), median bearing sweep 1.85 rad, and
  **100% of failures sweep > 0.9 rad** around the nearest obstacle. γ-pattern follows the expert's
  own behavior: at γ0.1 it keeps 0.19 clearance → 0% SOCP fail; at γ0.5 it hugs at 0.06 → 32%.
- Reading: SafeMPPI satisfies ITS OWN stepwise CBF; our verifier is a strictly stronger sufficient
  certificate whose blind spot is tight wrap-throughs (its own docstring advertises fitted faces
  for tight *straight* threads). Consequences: (i) expert-vs-ours V₂ comparisons carry this
  asymmetry — caption accordingly; (ii) 6–32% of the pretraining windows are uncertifiable at
  their γ by our verifier; (iii) making expert V₂→1 by construction requires changing the
  certificate shape (window splitting at large bearing sweep / multiple faces per obstacle /
  per-step refits) — a verifier design decision, in the same geometry family as the near-goal
  NVP problem (short remaining distance + sweeping bearings at the plug corner).

## Concern 2 — "pretrained model too perfect at origin; maybe a contaminated variant": EXONERATED

Provenance chain, all verified:
- `results/hp_repr/pretrained_a32uni.pt`: keys {state_dict, config, best_val, data,
  per_gamma_cap}; `data='druni_'`, `per_gamma_cap=0`, best_val 1.0101; mtime Jul 9 00:04 —
  8 minutes after the druni dataset files (23:52–23:56). Creator: `overnight_run_07_06/
  pretrain_repr.py` (save signature matches the checkpoint keys exactly).
- **Dataset audit (`dataset/druni_windows_g*.pt`, copied per γ into `pretrain_audit/dataset/`)**:
  566 seeds/γ, start range x∈[0.16,4.84], min |y−x| = **1.02**, fraction |y−x| < 1 = **0.000**,
  fraction < 0.5 = 0.000. The |y−x|<0.5 variant and expert anchoring are ruled out at the data
  level; the creator script has no anchor/demo path.
- **True pretraining recipe (paper text must be corrected)**: AdamW lr 3e-4, cosine with 5-epoch
  warmup, weight decay 1e-4, **batch 256, 120 epochs**, ALL ~220k windows, 7 γ, val = 10% split,
  best-val checkpoint. The draft's "Adam 1e-4, batch 64, 4 inner steps" mixes in the old
  EXPANSION inner loop; none of it describes pretraining.
- **Decisive test — faithful from-scratch re-pretrain** (`pretrain_repr_audit.py`, identical
  recipe, copied data; val 1.0246 vs original 1.0101), then bare-policy M=100 random rollouts
  from the origin per γ:

| model | γ0.1 SR/CR | γ0.3 | γ0.5 | γ1.0 | V₂ pattern | raw up-frac |
|---|---|---|---|---|---|---|
| current a32uni | 0.91/0.09 | 0.88/0.12 | 0.89/0.11 | 0.95/0.05 | 0.00 / 0.67 / 0.72 / 0.79 | 0.145 |
| repro (fresh)  | 0.92/0.08 | 0.86/0.14 | 0.84/0.16 | 0.88/0.12 | 0.00 / 0.72 / 0.76 / 0.68 | 0.223 |

Differences are within seed-level variation at M=100 (Wilson ±~0.06); the repro reproduces origin
competence, the identical V₂ structure (γ0.1 exactly 0.00), the clearance/time regime, and the
same R-leaning prior. **Verdict: the checkpoint is exactly what it claims to be; origin competence
is genuine generalization** (position enters the model only through the relative-goal vector and
the local 32×32 polytope grid — both quasi-translation-invariant), not contamination.
**No re-expansion is required on provenance grounds**; re-expansion decisions can rest solely on
the NVP/verifier fixes.

## Files
`dataset/druni_windows_g{0.1..1.0}.pt` (copies), `pretrain_repr_audit.py` (path-patched copy of
the creator script), `out/pretrained_a32uni_repro.pt`, `../logs/repretrain_audit.log`,
`../results/true_eval/repro/paths_r0_g*.npz` (M=100 bare rollouts of the repro model).
