# Stage 5 — 20-iteration reduced-predicate sanity (v4 visualization grammar)

Status: **rerun, controlled ablation, visualization overwrite, and independent validation complete. No
deployment win is claimed.**

## Clean sanity semantics

| Arm | Whole-rollout acceptance | Sample classes |
|---|---|---|
| Ours | H10 ∧ task-space ∧ reach ∧ progress ∧ SOCP | easy + frontier |
| −SOCP | H10 ∧ task-space ∧ reach ∧ progress | easy + frontier |
| −Progress | H10 ∧ task-space ∧ reach ∧ SOCP | easy + frontier |
| −Curriculum | same as Ours | one class; no easy/frontier split |

The direct OOD expert pool is `data/canonical_seed_windows.pt`: `(0.05,0.05) → (5,5)`, all seven γ values,
and 819 exact executed H10 windows on the 8-plug scene. Demo distillation is `0.50 → 0.25`; batch 16 therefore
uses eight rows before the first accepted rollout and four rows afterward. Every arm latches at iteration 1.

The controlled `−Curriculum` arm consumes the full arm's exact positive and negative tensors at the same
iterations. Its accepted-window sequence is:

```text
106, 0, 73, 0, 95, 0, 0, 59, 85, 0, 77, 51, 0, 0, 0, 0, 0, 0, 0, 55
```

That is 601 accepted windows across eight nonempty updates. The independent validator also compares every
accepted `grid/low5/hist/U/gamma` tensor and every rejected `grid/low5/hist/U` tensor bit-for-bit. It verifies
3,367 bounded rejected rows across those eight updates. Thus data identity and volume are fixed; the class
split is the only curriculum difference.

## Why 20 iterations were necessary

Six iterations were not enough to decide whether gathering had stalled. The full arm found additional
batches at iterations 8, 9, 11, 12, and 20. The run used up to 12 gather attempts per iteration (240 nominal
attempt slots over the horizon), on GPU 2, before the bounded sanity stop.

All arms start from the same 25-step unfrozen OOD-expert checkpoint and use seed 5010, β=.2, α=5×10⁻⁴,
encoder LR 0.3× field LR, field LR 5×10⁻⁶, and batch 16.

| Arm | Accepted rollouts | Accepted windows | Nonempty updates | Faithful M=6×7 | Collisions |
|---|---:|---:|---:|---:|---:|
| Ours | 8 | 601 | 8 | 0/42 | 42/42 |
| −SOCP | 19 | 1,542 | 19 | 1/42 | 41/42 |
| −Progress | 11 | 1,416 | 11 | 0/42 | 42/42 |
| −Curriculum | 8 | 601 | 8 | 0/42 | 42/42 |

The only nonzero learned-arm result is `−SOCP` at γ=.1: one success, 41 collisions overall. This supports the
diagnosis that SOCP certification is the data bottleneck, but removing it is unsafe and is not a promising
deployment policy. Retaining four expert rows and waiting 20 iterations did not reveal a deployment-ready
“hope” configuration. The big dive remains unauthorized.

## Overwritten deliverables

- `viz/curriculum_it20.mp4` — slow H.264 curriculum, 2028×1014, 2 fps, 42 frames, 21 s.
- `viz/internals_v4.png/.pdf` — exact 2×3 v4 internals from `final_v7_ours`, iterations 1–20.
- `viz/rollouts_v4.png/.pdf` — exact 2×4 gallery, all three NO brothers, and both pretraining start/goal seeds.
- `viz/scatter_v4.png/.pdf` — marker=method and truncated-plasma=γ.
- `data/table_v4.md/.tex` — expert, ours, pretrained, low-guidance Kazuki, and all three ablations.

`python validate_v4_sanity.py` is **PASS**. It checks the 22/22 trainer suite, signed-negative objective,
predicate masks, 0.50→0.25 demo schedule, exact controlled replay, M=6 rows, baselines, artifact encoding, and
the unchanged active checkpoint SHA-256
`5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2`.

Decision: pause before any big dive. A future run needs a stronger way to bridge the certification/support
gap; simply relaxing SOCP produces unsafe trajectories.
