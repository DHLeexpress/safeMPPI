# Giant-obstacle bounded expansion report

The promoted run uses temperature 0.5, unfrozen encoder (0.3× LR), beta 0.2, window-native H=10 validity, and an actual OOD expert demo schedule of 50% through iteration 10 then 25%.

- Ours: SR 0.000, CR 1.000, coverage 0, mean boundary arc 0.343 rad.
- Learning health: PASS; route-mode audit: ALERT.
- Temperature sweep SR (0.1/0.5/1.0): 0.000/0.000/0.000.

## Requested artifacts

- `viz/rollouts_v4.png` — pretraining data, Expert, Pretrained, CFM-MPPI*, all three No brothers, Ours.
- `viz/internals_v4.png` — exact 2×3 training-internals grammar.
- `viz/scatter_v4.png` — gamma-colored reliability and successful-trajectory quality planes.
- `viz/curriculum_it20.mp4` — exact curriculum grammar, one second per iteration.

No long-run claim is made from this bounded Stage-5 sanity.
