# Coverage versus more iterations — evidence at t104

## Faithful M100 audit (seeds 0--99, reach .1, temp 1, NFE8)

| gamma | SR | CR | clearance mean | time mean (s) | coverage M25 | coverage M100 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 94% | 0% | .305 | 18.23 | 6 | 10 |
| 0.2 | 96% | 1% | .295 | 13.76 | 4 | 7 |
| 0.3 | 97% | 1% | .297 | 12.18 | 3 | 4 |
| 0.4 | 96% | 0% | .299 | 11.83 | 3 | 4 |
| 0.5 | 97% | 0% | .299 | 11.89 | 3 | 4 |
| 0.7 | 96% | 0% | .298 | 12.41 | 2 | 5 |
| 1.0 | 97% | 0% | .299 | 12.62 | 3 | 6 |

Artifacts: `results/p2/eval_corrected_mode2_it104_m100/` and
`analysis/origin_window_failure_probe_t104_m100.{md,json}`. Across 700 rollouts there are 673 successes,
7 origin-boundary failures (seed 12 at every gamma), 18 near-goal failures, and 2 other failures.

## What this proves

- M25 understated rare support: coverage rises from `2--6` to `4--10` at M100. The vector field has not
  irreversibly collapsed to one mode and has enough representational capacity to emit several rare modes.
- It still falls far short of the required >=14 at every gamma. Gamma .3--.5 each expose only 4 modes in
  100 faithful trials, so this is not merely an M25 counting artifact.
- More iterations of the **same** recipe are unlikely to solve coverage. At corrected iterations 101--105,
  coherent targeted proposals recorded zero exact target hits, even with 21--78 targeted attempts per
  iteration. A later cap-400 gather recorded one hit but could not satisfy the two-mode gamma-.1 readiness
  gate and therefore performed no update.
- The current readiness condition is only `min_modes_per_gamma=2` (`grid_expand_fixed.py:855-857`), not 14.
  Training batches are mode-balanced only over the modes that were actually gathered; the canonical
  `RURURURURU` mode still occupied 20--24 of 56 fresh rows at t103--t105.
- The M50 faithful coverage probe remained 4 across several hard-tail updates. By terminal hard-tail t118,
  M25 coverage was `2--5` while SR had regressed, demonstrating that additional repair iterations can move
  the field without expanding deployed mode support.

## Is the vector field overfit?

Not globally in the usual parameter-overfitting sense: t104 is only .97% from its fixed functional anchor,
the encoder is frozen, and M100 still reveals 4--10 modes. The more precise diagnosis is **probability-mass
concentration / mode-selection bias** in the fine-tuned vector field. Exact-valid expansion preferentially
repeats already-successful central staircases, while the soft target proposal almost never realizes its
requested uncovered staircase. Continuing that data distribution reinforces the same modes.

## Required change before a long coverage run

1. Do not launch more unchanged t104 iterations.
2. Make the coherent target proposal achieve and certify actual target hits before allowing a gradient
   update; `target_hits=0` must be an explicit failed coverage-readiness condition.
3. Use an absolute-iteration mode curriculum (for example 2 -> 4 -> 8 -> 12 -> 14 certified achieved modes
   per gamma), with higher rollout/sample budget and beta .2 only as the already-authorized exploration arm.
4. Preserve successful rare-mode latent fibers in replay so later updates cannot replace them with the
   canonical mode. Keep the normal exact Valid2/SOCP acceptance unchanged.
5. Measure coverage at M100 during coverage checkpoints. M25 remains useful for fast reliability screening
   but is too weak for a >=14 claim.

Reliability remains first: the new M100 audit also exposes CR 1% at gamma .2/.3 and a broader near-goal tail
than the 11 fixed M25 failures. A coverage curriculum should begin only from a checkpoint that passes the
independent M100 safety/reliability gate.

