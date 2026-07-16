# Exact checkpoint/resume audit for `grid_expand_fixed.py`

Scope: read-only design audit. No trainer code was changed.

## Finding

The current files are policy snapshots, not resumable training checkpoints. For example,
`results/p2/finalunit_q50_k14_s15_from_it18/ckpt_100.pt` contains only
`config,state_dict,iter,srcr`, and its top-level iteration is 100 while `srcr.iter` is 90. The
periodic save happens before the iteration-100 measurement. A resume currently recreates Adam,
the GP query buffer, coverage, rolling metrics, selection/collapse counters, and any pile. More
importantly, with LwF enabled it deep-copies the resumed student as the new teacher, so the loss
itself changes at every process boundary.

## Minimal robust state

Keep the policy format accepted by `HP.load_hp`, but put all trainer data under one namespaced,
versioned top-level key, e.g. `trainer_state`:

```python
trainer_state = {
    "format": "grid_expand_fixed.train_state",
    "version": 1,
    "completed_iter": int(t),
    "optimizer": opt.state_dict(),          # Adam steps, moments, and actual group LRs
    "qbuf": qbuf,                           # CPU tensor dictionary, including tag if present
    "covered": {str(g): sorted(covered[g]) for g in gammas},
    "teacher_state_dict": None if teacher is None else teacher.state_dict(),
    "pile": pile_state_dict(pile),          # cap, T, margin/prog/widx/rid/it/use/labels
    "history": history,
    "rolling": {
        "reached": list(roll_reached), "collided": list(roll_coll),
    },
    "runtime": {
        "cooled": cooled, "best_sr": best_sr, "sr0": sr0,
        "best_safe_sr": best_safe_sr, "best_probe": best_probe,
        "best_probe_cov": best_probe_cov, "collapse_ct": collapse_ct,
        "last": last, "policy_training": policy.training,
    },
    "rng": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    },
    "signature": {
        # all state-affecting CurConfig values, excluding only iters/start_iter,
        # plus freeze_enc, enc_lr_mult, and ordered optimizer-group names
    },
}
```

The pile should use a plain dictionary, not pickle the `Pile` instance. Store labels as a boolean
`is_frontier` array rather than NumPy `object` strings. `GPUncertainty.X/_L` need not be saved:
the next `_gather_fresh` deterministically rebuilds them from the restored `qbuf` and current
policy, which is also what an uninterrupted next iteration does.

Also persist the rolling deques and best/collapse state: they do not change the next gradient, but
they do change online metrics, selected checkpoints, and early termination. Persist `last` because
a skipped update reuses it in logging/history.

## Restore order

1. Load the entire checkpoint on CPU (`HP.load_hp(path, device="cpu")`) and then move only the
   policy to the training device. `HP.load_hp(..., device="cuda")` maps *all* metadata tensors,
   including qbuf, pile, and RNG byte states, to CUDA; qbuf/pile are subsequently concatenated with
   CPU gathers and must remain on CPU.
2. Resolve `completed_iter` from `trainer_state` and require it to agree with top-level `iter` and
   any explicit `--start-iter`. A mismatch must fail unless the caller explicitly requests a cold
   fork.
3. Construct parameter groups exactly as before and validate the saved signature/group layout.
   Then call `opt.load_state_dict`. Do not apply the cooldown LR multiplier again: Adam's saved
   groups already contain the effective LR, and `runtime.cooled` is authoritative.
4. Restore qbuf, covered sets, pile, history, rolling deques, counters, and `last`.
5. Recreate the frozen teacher shell and load `teacher_state_dict`. If `lwf_eta > 0` and it is
   absent, exact resume is impossible; fail or explicitly label the run a cold fork.
6. Restore Python, NumPy, CPU Torch, and CUDA RNG **last**, after model/optimizer/device setup. Exact
   CUDA replay requires the same visible-device count and compatible GPU/kernel settings.
7. Only then run an RNG-isolated baseline measurement or enter the next gather. Checkpoint history
   is authoritative; do not silently replace it with `outdir/history.json`.

## Save/commit timing

Centralize all saves in one atomic helper (`path.tmp` followed by `os.replace`) and call it at a
completed-iteration boundary, after gather/update, pile relabel, probes/viz, scheduled measurement,
history, best trackers, and collapse counter. Delay `best.pt`, `safe_best.pt`, and probe-best writes
to that same boundary if they are intended to be resume-capable. The current periodic save occurs
before measurement and therefore carries stale metrics/state.

There are two additional segmentation hazards:

- `or local_t == cfg.iters` makes terminal measurement mutate history/best/collapse state only in a
  split run. Make terminal-only evaluation observational (separate metadata, no training-state
  mutations), or only claim exact resumes at scheduled measurement boundaries.
- `_save_viz_db` can consume global NumPy RNG while `_preserve_torch_rng` protects only Torch.
  Use a local `np.random.default_rng(fixed_seed_from_iter)` or a context that restores Python,
  NumPy, CPU Torch, and CUDA states around every diagnostic.

## `HP.save_hp` / `HP.load_hp` hazards

- `save_hp` flattens `extra` via `dict.update`; an accidental `extra["state_dict"]` or
  `extra["config"]` silently replaces the model. Keep one reserved `trainer_state` namespace and
  reject reserved-key collisions in the local save wrapper.
- `load_hp` returns `.eval()` and does not restore module training mode; restore `policy_training`
  (or deliberately force a documented mode before gathers).
- `map_location=device` applies to metadata as well as the model, hence the CPU-load recommendation.
- Model reconstruction only uses a subset of config fields and does no trainer-schema/config
  validation. Validate `config`, trainer-state version, optimizer layout, and training signature.
- Existing `safe_best.pt`, `best.pt`, `probe_best.pt`, and `final.pt` lack full state. They cannot be
  called exact continuations. Starting from one is a new cold fork (new Adam/query memory/teacher),
  even if a gather-only prime reduces the immediate qbuf failure.

## Deterministic split regression

After implementing state restore, run two independent two-iteration paths from the same legacy
input. Use `measure-every=1` so the one-iteration segment and uninterrupted run have identical
stateful measurement boundaries, disable viz/probes, and exercise Adam, qbuf, teacher, and pile:

```bash
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 CUBLAS_WORKSPACE_CONFIG=:4096:8
BASE=results/p2/finalunit_q50_k14_s15_from_it18/safe_best.pt
COMMON="--no-freeze --enc-lr-mult 0.3 --m-measure 1 --measure-every 1 \
 --rollouts-per-iter 28 --gather-attempt-cap 42 --batch 64 \
 --valid-prog-floor 0.15 --min-rollouts 1 --traj-prog-min 0 \
 --quantile-schedule 0:0.50 200:0.60 400:0.70 \
 --mix-start 0.75 0.25 --mix-end 0.50 0.50 --beta 0.3 \
 --early-until 100 --cooldown-from 400 --early-inner 1 --inner-steps 2 --cooldown-inner 1 \
 --demo-frac 0.125 --lwf-eta 0.05 --lr 1e-4 --fresh-frac 0.75 \
 --viz-db-every 0 --ckpt-every 1 --log-comp-every 0 --seed 123"

python grid_expand_fixed.py --ckpt "$BASE" --outdir analysis/resume_reg/a --iters 2 $COMMON
python grid_expand_fixed.py --ckpt "$BASE" --outdir analysis/resume_reg/b1 --iters 1 $COMMON
python grid_expand_fixed.py --ckpt analysis/resume_reg/b1/final.pt \
  --outdir analysis/resume_reg/b2 --iters 1 $COMMON --seed 999
```

The second process deliberately supplies a different seed; restored RNG must override it. Compare
`a/final.pt` with `b2/final.pt` recursively, not by file hash: every policy tensor, Adam tensor and
step, qbuf/pile tensor and array, covered set, rolling/runtime field, NumPy state, CPU Torch state,
and CUDA RNG byte tensor must match exactly. Also assert `completed_iter` is the same. For a strict
GPU bitwise test, enable `torch.use_deterministic_algorithms(True)`, disable TF32, and use the same
GPU model/visible-device topology. If bitwise CUDA equality is unavailable, the test must still
require exact non-model state and report model max-absolute error with a very tight documented
tolerance; it must not silently accept a policy-only comparison.
