# Overnight Run — 2026-06-23 → morning of 2026-06-24

**Objective (user).** By morning: objective statistics + proof that *Guided Safe
MPPI* (ours) beats Mizuta CFM-MPPI in **moving-pedestrian** settings, an
aesthetic safety-parameter-guidance video, and an exploration of guiding the
drifting model. Full autonomy granted — no check-ins.

**Deliverables in this folder**
- `THEORY.md` — the proof / theoretical contribution (DONE).
- `RESEARCH.md` — mechanisms from CC-MPPI / Shield-MPPI / CS-MPPI / HOCBF (from research agent).
- `CODE_MAP.md` — where to inject the new sampler (from code-map agent).
- `RESULTS.md` — final statistics, significance tests, tables (the headline deliverable).
- `STATUS.md` — live progress log, updated as jobs finish.
- Video(s) under `../results/benchmark_videos/`.

## Baseline to beat (measured tonight, episode 110, moving peds)
- Mizuta CFM-MPPI: success ✅, no collision, goal reached, min-clearance 0.295.
- Ours (rejection safemppi_gamma): **0 % success, freezes** (never reaches goal),
  clearance 0.015. Aggregate sfm/doubleintegrator: 0 % success, 100 % collision.
- So Mizuta currently wins. Target: overturn this.

## Pipeline
0. [done] Context, baseline numbers, theory.
1. [running] Research agent + code-map agent.
2. Implement Guided Safe MPPI (§2–§5 of THEORY) in `safegpc_adapter/` + a guided
   sampler hook. CPU smoke first.
3. Sanity: rerun episode-110 moving comparison; confirm freeze is cured
   (accepted-fraction ↑, goal reached) before scaling.
4. Regenerate moving-pedestrian dataset with Guided Safe MPPI across γ grid.
5. Train (GPUs 0,1): contextual CFM on guided data; guided drifting generator.
6. Evaluate ≥200 moving-pedestrian episodes: ours (CFM + guided MPPI + guided
   drifting) vs Mizuta CFM-MPPI vs old rejection MPPI. Metrics: success,
   collision, min-clearance, path/control cost, latency/NFE. Bootstrap CIs +
   paired test. Iterate hyperparameters until C2+C3 hold.
7. Aesthetic γ-guidance video.
8. (stretch) clone a racing MPPI benchmark for a second domain.
9. Write `RESULTS.md`.

## Compute
- GPUs 0,1 free (GPU 3 busy — leave it). Pin training to `CUDA_VISIBLE_DEVICES=0,1`.
- conda env `cfm_mppi`; scripts use `conda run -n cfm_mppi`.

## Stop / honesty rule
Report measured numbers regardless of outcome. If after iteration ours does not
beat Mizuta on a metric, say so explicitly and show where it does win (e.g.
clearance/tunability) and why.
