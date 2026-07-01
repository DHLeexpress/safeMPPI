"""Tiny Weights & Biases helper for the smoke-test training stages.

Matches the repo convention (`cfm_mppi/training/train_safe_cfm.py`): **offline by default** (no API
key needed — `wandb sync <dir>` later to upload), graceful degradation if wandb is missing/broken,
and only numeric values are logged as metrics.  Import and use from `pretrain.py` / `expand.py`.

CLI flags to add to a stage:  --wandb-mode {offline,online,disabled}  --wandb-project  --wandb-name
"""
from __future__ import annotations

import os


def add_wandb_args(ap, default_project="cfm-mppi-safeflow"):
    ap.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    ap.add_argument("--wandb-project", default=default_project)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    return ap


def init_run(args, name, config, dir=None, group=None):
    """Return a wandb run (or None). mode=disabled or any failure -> None (stage still runs)."""
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    try:
        import wandb
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        os.environ.setdefault("WANDB_SILENT", "true")
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or name,
            group=group or getattr(args, "wandb_group", None),
            mode=args.wandb_mode,
            dir=dir,
            config=config,
            reinit=True,
        )
    except Exception as exc:  # wandb missing/broken -> jsonl/console only
        print(f"[wandb] disabled ({exc})", flush=True)
        return None


def log(run, data, step=None):
    if run is not None:
        run.log({k: v for k, v in data.items() if isinstance(v, (int, float, bool))}, step=step)


def log_image(run, key, path):
    if run is not None and path and os.path.exists(path):
        try:
            import wandb
            run.log({key: wandb.Image(path)})
        except Exception as exc:
            print(f"[wandb] image log failed for {key} ({exc})", flush=True)


def log_video(run, key, path):
    if run is not None and path and os.path.exists(path):
        try:
            import wandb
            run.log({key: wandb.Video(path)})
        except Exception as exc:
            print(f"[wandb] video log failed for {key} ({exc})", flush=True)


def finish(run, summary=None):
    if run is not None:
        try:
            for k, v in (summary or {}).items():
                run.summary[k] = v
            run.finish()
        except Exception as exc:
            print(f"[wandb] finish failed ({exc})", flush=True)
