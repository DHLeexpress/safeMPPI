# Validation Comparison v3 Notes

Output: `results/benchmark_videos/smoke_moving_validation_comparison_v3.mp4`

## Command Settings

- dataset: `ucy`
- dynamics: `unicycle`
- pedestrian_source: `validation`
- episodes: `[0, 41, 110, 123]`
- steps: `80`
- gamma_grid: `[0.1, 0.5, 1.0]`
- no_safe_cfm: `True`
- safemppi_horizon: `40`
- safemppi_samples: `512`
- safemppi_running_goal_weight: `0.25`
- safemppi_terminal_goal_weight: `80.0`
- safemppi_control_weight: `0.03`
- safemppi_smooth_weight: `0.12`
- safemppi_soft_clearance_weight: `25.0`
- safemppi_progress_weight: `2.0`
- safemppi_use_sets_backup: `True`
- safemppi_sets_num_modes: `3`
- safemppi_sets_branch_scale: `0.85`
- safemppi_sets_include_cbf_backup: `True`
- safemppi_sets_cbf_push: `1.25`
- safemppi_sets_reverse_speed: `0.75`
- safemppi_sets_turn_rate: `1.4`
- debug_rollouts: `48`
- draw_hyperplanes: `True`
- hyperplane_horizon: `20`
- hyperplane_stride: `1`
- pedestrian_radius: `0.0`
- safety_margin: `0.5`
- show_backup_labels: `False`

## v3 Changes

- Hyperplanes are tangent to the effective safety disk, using `pedestrian_radius + r_safe`.
- The orange disk is the effective safety disk; the plotted affine CBF thresholds are parallel
  support planes between the robot-side plane and the tangent plane on that disk.
- The nearest safety disk for each SafeMPPI panel is highlighted in orange.
- SafeMPPI panels draw accepted rollouts in blue and rejected rollouts in red.
- This run uses a 4-scenario by 4-method grid: Mizuta, SafeMPPI gamma 0.1, 0.5, 1.0.
- Safe CFM is intentionally disabled here to isolate the SafeMPPI gamma tradeoff.
- SafeMPPI uses heading-aware unicycle nominal control, original-style input bounds `[-2, 2]`, unicycle noise `[0.3, 0.6]`, moving-pedestrian constant-velocity prediction, and tuned goal/smooth/safety costs.
- If enabled, SETS-style backup proposals are appended to the random MPPI samples. These
  branches linearize around the nominal trajectory, form `C C^T`, use `+-sqrt(lambda_i) q_i`
  terminal displacements, solve `C^+ delta_z`, clip normalized inputs to `[0,1]`, and add
  nearest-halfspace away/tangent/reverse backup controls.
- Backup modes are drawn with distinct colors; dashed backup branches were rejected by the
  affine barrier test and solid backup branches were accepted.

## Re-run Template

```bash
conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \
  --dataset ucy \
  --dynamics unicycle \
  --pedestrian-source validation \
  --episode-list 0 41 110 123 \
  --steps 80 \
  --gamma-grid 0.1 0.5 1.0 \
  --no-safe-cfm \
  --device cuda \
  --safemppi-horizon 40 \
  --safemppi-samples 512 \
  --safemppi-running-goal-weight 0.25 \
  --safemppi-terminal-goal-weight 80.0 \
  --safemppi-control-weight 0.03 \
  --safemppi-smooth-weight 0.12 \
  --safemppi-soft-clearance-weight 25.0 \
  --safemppi-progress-weight 2.0 \
  --safemppi-use-sets-backup \
  --safemppi-sets-num-modes 3 \
  --safemppi-sets-branch-scale 0.85 \
  --safemppi-sets-include-cbf-backup \
  --safemppi-sets-cbf-push 1.25 \
  --safemppi-sets-reverse-speed 0.75 \
  --safemppi-sets-turn-rate 1.4 \
  --debug-rollouts 48 \
  --hyperplane-horizon 20 \
  --hyperplane-stride 1 \
  --no-show-backup-labels \
  --figure-tag v3 \
  --output results/benchmark_videos/YOUR_NAME.mp4 \
  --gif-output results/benchmark_videos/YOUR_NAME.gif
```

## Episode Iteration Prompt

Pick four validation episodes that stress different tradeoffs, then only change `--episode-list`
and the output stem:

```bash
for stem in v3_episode_set_a v3_episode_set_b; do
  conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \
    --dataset ucy \
    --dynamics unicycle \
    --pedestrian-source validation \
    --episode-list EP0 EP1 EP2 EP3 \
    --steps 80 \
    --gamma-grid 0.1 0.5 1.0 \
    --no-safe-cfm \
    --device cuda \
    --safemppi-horizon 40 \
    --safemppi-samples 512 \
    --safemppi-use-sets-backup \
    --safemppi-sets-num-modes 3 \
    --safemppi-sets-branch-scale 0.85 \
    --debug-rollouts 48 \
    --hyperplane-horizon 20 \
    --hyperplane-stride 1 \
    --no-show-backup-labels \
    --figure-tag v3 \
    --output results/benchmark_videos/${stem}.mp4 \
    --gif-output results/benchmark_videos/${stem}.gif
done
```
