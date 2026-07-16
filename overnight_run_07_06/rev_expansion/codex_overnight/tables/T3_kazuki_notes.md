# T3 / Kazuki baseline — data provenance and reading notes

## What the method is (our faithful reimplementation)

`kazuki_baseline.py` reimplements the Mizuta/Kazuki guided CFM+MPPI pipeline on OUR task and OUR untouched
pretrained flow model (`../../results/hp_repr/pretrained_a32uni.pt` — never flow-expanded; this is the
frozen-benchmark requirement). Per replan step it samples N=200 flow candidates whose velocity field is
guided during integration:

    v  ←  v + goal_coef · g_goal + w_safe · g_cbf · markup(1.01^(T−t))

then MPPI-style elite selection scores candidates with a collision proximity cost (weight `coll_w`,
steepness `beta_mppi`) plus goal cost (`goal_w`), executes the first action, and replans. Their pipeline has
no γ input; the FM conditioning is supplied externally via `--gamma-ctx`.

## Row provenance (exact commands are in PROGRESS.md)

| Artifact | Config | M | How a–e were computed |
|---|---|---|---|
| `tables/T3_kazuki.{md,csv}` (headline) | TUNED: `--w-safe 0.3 --coll-w 20 --goal-w 2.0 --goal-coef 0.5 --beta-mppi 20 --gamma-ctx <γ>` | 200/γ (seeds 0–99 + 100–199 merged) | `eval_ae.py saved-worker` on the stored paths — the **identical metric code** used for P1 expert and every flow-expansion row (same reach=.1, same clearance/time/coverage definitions) |
| `results/kazuki_wsweep/kaz_w{.05,.3,.9,2,5}.json` (vulnerability sweep) | PUBLISHED-STYLE defaults: `coll_w=100, goal_w=0.1, goal_coef=0.1, beta_mppi=20, gamma_ctx=0.5`, single w_safe, all 200 samples on that coefficient | 25 each | SR/CR from episode outcomes; a–e undefined (0 successes — every episode times out at T=250) |
| Early proximity-wall attempts (PROGRESS 02:4x) | their published wall `100·exp(−20(d−r))` | 3–9 configs | froze at start; SR 0.00 |

## Reading (what the table rows mean)

1. **Headline T3 is a strong, honest baseline**: SR 100%, CR 0%, clearance .372–.375 m, time 8.96–10.47 s,
   coverage 5–8 at M=200. It is faster and wider-clearance than our current expansion because MPPI
   re-optimizes every step against the true obstacle field at deployment time (an online optimizer),
   while ours is a single certified generative policy.
2. **The tuned point is a needle.** Getting T3 to work required a manual 4-knob retune
   (`coll_w 100→20, goal_w 0.1→2.0, goal_coef 0.1→0.5`, w_safe fixed 0.3) after the published-style
   configuration failed completely. With published-style weights, EVERY single w_safe in {0.05, 0.3, 0.9,
   2.0, 5.0} gives SR 0% — 25/25 timeouts each — even when all 200 MPPI samples are devoted to that
   coefficient (no sample-count confound).
3. **No safety-level mechanism.** Even at the tuned point, a–e are FLAT across γ (clearance .372–.375 at
   every γ; time monotone-ish 9.0–10.5 s): w_safe is a global cost weight, not a conditioning input, so the
   operator cannot request a safety level and get a graded behavior. Our method carries γ as a conditioning
   variable with a per-window verifier certificate: clearance/time vary systematically with γ
   (e.g. t104: 18.2 s cautious at γ.1 → 11.7 s at γ.5).
4. Everything is saved for re-testing: per-seed paths (`results/kazuki_final*/paths_g*.npz`,
   `results/kazuki_wsweep/paths_*.npz`), per-row JSONs with the full parameter set embedded, and the exact
   launch commands in PROGRESS.md.
