# Canonical Dataset Schema

Canonical split files are saved under:

- `dataset/canonical/train.pt`
- `dataset/canonical/val.pt`
- `dataset/canonical/test.pt`

Each file is a `torch.save` dictionary.

## Fields

- `states`: `float32 Tensor [N, T+1, state_dim]`
  - Double-integrator default state is `[px, py, vx, vy]`.
  - Unicycle-compatible data may use `[px, py, theta]`, padded by adapters when needed.
- `controls_dyn`: `float32 Tensor [N, T, control_dim]`
  - Control in the evaluation dynamics coordinates.
- `controls_si`: `float32 Tensor [N, T, 2]`
  - Single-integrator-style planar control when available. For safeGPC double-integrator records this is currently the same as `controls_dyn`.
- `start`: `float32 Tensor [N, 2]`
  - Start position.
- `goal`: `float32 Tensor [N, 2]`
  - Goal position.
- `ego_history`: `float32 Tensor [N, Hhist, ego_hist_dim]`
  - Default ego history is `[px, py, vx, vy]`.
- `action_history`: `float32 Tensor [N, Hhist, control_dim]`
  - Recent controls, zero-padded at the front.
- `nearest_obstacle_history`: `float32 Tensor [N, Hhist, obs_hist_dim]`
  - Default layout is `[relative_px, relative_py, relative_vx, relative_vy]`.
  - safeGPC artifacts provide nearest-surface relative position; relative velocity is zero-filled.
- `obstacles`: padded tensor when available, default `[N, max_obs, 3]` with `[cx, cy, radius]`, or NaN-filled when unknown.
- `gamma`: `float32 Tensor [N]`
  - safeGPC gamma when available; NaN for Mizuta source data.
- `dynamics_type`: `list[str]` length `N`
  - Examples: `doubleintegrator`, `unicycle`, `singleintegrator`.
- `safety_margin`: `float32 Tensor [N]`
  - Collision/safety margin used for evaluation.
- `source`: `list[str]` length `N`
  - Examples: `mizuta`, `safeGPC`.
- `metadata`: `dict`
  - Contains source path, raw shape, raw dtype, history length, split seed, and adapter notes.

## Adapters

- Mizuta adapter: `cfm_mppi.data.canonical_dataset.build_canonical_from_mizuta`
  - Input: `dataset/train80_ego.pt`, shape `[273989, 9, 80]`.
  - Uses channels `0:2` as positions and `2:4` as controls.
- safeGPC adapter: `cfm_mppi.data.canonical_dataset.build_canonical_from_safegpc`
  - Supports `.pt`, `.pkl`, `.pickle`, `.npz`, `.jsonl`, `.csv`, or a directory containing these.
  - Required per-step fields: `obs` or `obs_*`, `u` or `u_*`, and `gamma`.
  - Rows with `gamma=NaN` mark episode boundaries.

## Deterministic Splits

`save_canonical_splits(..., seed=0)` shuffles episode/sample indices with a Torch generator and writes 80/10/10 train/val/test splits by default.
