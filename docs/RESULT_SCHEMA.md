# Result Schema

Unified benchmark JSONL files are written after every episode to:

`results/benchmark/<timestamp>/<dataset>/<dynamics>/<method>.jsonl`

Each line is a JSON object with at least:

- `success`: boolean
- `collision`: boolean
- `goal_reached`: boolean
- `min_clearance`: float
- `mean_clearance`: float
- `min_barrier_h`: float or null
- `num_barrier_violations`: integer
- `final_goal_distance`: float
- `path_length`: float
- `control_effort`: float
- `control_smoothness`: float
- `episode_return`: float
- `episode_cost`: float
- `planning_wall_time_mean`: seconds
- `planning_wall_time_median`: seconds
- `planning_wall_time_p95`: seconds
- `total_wall_time`: seconds
- `model_calls_per_step`: number
- `nfe`: number
- `gamma`: float or null
- `safe_coef`: float/list or null
- `safety_margin`: float
- `seed`: integer
- `dataset`: string
- `dynamics`: string
- `method`: string
- `checkpoint_path`: string or null
- `config_path`: string or null
- `safety_guarantee_scope`: `linear_system_theorem_relevant` or `empirical_only_unicycle`

At the end of a run, the evaluator writes:

- `summary.csv`
- `summary.json`
- `summary.md`

Summaries aggregate success rate, collision rate, clearance statistics, final distance, control effort/smoothness, planning time, NFE, and episode count.
