# Stage 6 — expert and low-guidance Kazuki sanity baselines

SafeMPPI remains the target: **42/42 success, 0 collisions** across all seven γ values, with successful-path
clearance 0.220–0.265 m.

The earlier Kazuki setting (`w_safe=.3`, collision weight 20) looked like a dominating safety filter. The
replacement is deliberately softer:

`w_safe=.02`, `coll_w=2`, `goal_w=2`, `goal_coef=.1`, `N=100`, `elite=5`, `copy=50`.

It now shows the pretrained policy’s long diagonal/corridor behavior and reaches the goal neighborhood instead
of appearing stuck. On the fixed M=6×7 panel it gets **3/42 successes and 39/42 collisions**; the successes are
at γ=.4, .7, and 1.0, with 4.2–4.3 s time-to-goal and ≈0.215 m clearance. This is visually informative but still
an unsafe, weak baseline.

Canonical data: `results/kazuki_low_guidance_m6/`. The exact v4 rollout/scatter/table use this setting.

Decision: keep the low-guidance configuration for the sanity figures. Full 200/10/200 fidelity remains part of
the future big dive, not this bounded run.
