# Safe Flow Expansion handoff

This folder is the compact entry point for continuing the experiments without re-reading the full overnight
history. All paths below are relative to `codex_overnight/`.

## Current fixed-recipe implementation

- Trainer: `../grid_expand_fixed.py`
- Valid2 + true certificate-slack helper: `../grid_metrics2.py`
- Shared a--e evaluator: `../eval_ae.py`
- Curriculum renderer: `../video_curriculum_fixed.py`
- P2 internals renderer: `../plot_p2_diagnostics.py`
- Active controlled runs: `../results/p2/qmix75_socp_d125_b03/` and
  `../results/p2/qmix50_socp_d125_b03/`

The fixed recipe uses an absolute-iteration AND cell:

`frontier = (sigma >= q_sigma(N)) AND (certificate_slack <= q_margin(1-N)) AND (progress >= q_progress(N))`.

Its initial schedule is `{0: .50, 200: .60, 400: .70}`, beta is constant `.3`, Valid2 is unchanged,
and missing classes trigger more gathering up to the attempt cap rather than demo backfill. The selected final
checkpoint and complete command will be added to `latest_model.json` after the all-gamma goal audit.

## Historical recipes requested by the user

- `m_base_recipe.json`: locked quantile-OR recipe. It is useful as the clean historical baseline, but it
  reinforces high-uncertainty near-origin dither and therefore produces CR near zero with inadequate SR.
- `m_combo_recipe.json`: the stability winner before the fixed AND experiment. It blocks sigma >= .25 from
  the easy pool and adds demo backfill / first-window skipping; it is stable but the absolute sigma gate is
  an ad hoc paper mechanism.

Reference visualizations are retained in `../preliminary/`, especially `m_base_curriculum.mp4`,
`m_combo_curriculum.mp4`, and `micro100_internals_current_best.png`.

