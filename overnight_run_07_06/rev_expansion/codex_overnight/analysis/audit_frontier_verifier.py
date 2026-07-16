#!/usr/bin/env python3
"""Read-only audit of P2 ``viz_db`` snapshots.

This intentionally imports the same local modules as ``grid_expand_fixed.py``.
It checks the saved AND labels, strict-vs-legacy terminal radius, executed
Valid2, per-gamma sampling balance, and planned-window certificate validity.
No checkpoint or result is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REV = HERE.parent
WORK = REV.parent
sys.path[:0] = [str(HERE), str(REV), str(WORK)]

import grid_expand2 as GX2  # noqa: E402
import grid_metrics as GM  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_scene as GS  # noqa: E402


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _f(x):
    return float(x) if np.isfinite(x) else None


def audit_snapshot(path: Path, recompute: int = 64) -> dict:
    db = torch.load(path, map_location="cpu", weights_only=False)
    label = np.asarray(db["label"])
    sigma = np.asarray(db["sigma"], dtype=float)
    margin = np.asarray(db["margin"], dtype=float)
    prog = np.asarray(db["prog"], dtype=float)
    gamma = np.asarray(db["gamma"], dtype=float)
    rid = np.asarray(db["rid"], dtype=int)
    low5 = np.asarray(db["low5"])
    controls = np.asarray(db["U"])

    expected_front = (
        (sigma >= float(db["sigma_plane"]))
        & (margin <= float(db["margin_plane"]))
        & (prog >= float(db["prog_plane"]))
    )
    actual_front = label == "frontier"

    env = GS.make_grid()
    goal = np.asarray(env.goal, dtype=float)
    path_rows = []
    for i, p in enumerate(db.get("paths", [])):
        p = np.asarray(p, dtype=float)
        wi = np.flatnonzero(rid == i)
        g = float(np.median(gamma[wi])) if len(wi) else float("nan")
        path_rows.append(
            dict(
                gamma=g,
                final_distance=float(np.linalg.norm(p[-1] - goal)),
                strict_reach=bool(np.linalg.norm(p[-1] - goal) < 0.1),
                legacy_reach=bool(np.linalg.norm(p[-1] - goal) < GM.REACH),
                socp=bool(GM.socp_ok(p, env, g)),
                valid2=bool(GM2.traj_valid2(p, env, g)),
                taskspace=bool(GM.in_taskspace(p)),
            )
        )

    # Recompute raw (unclipped) verifier slack on a deterministic subset.
    n_check = min(max(recompute, 0), len(label))
    check_idx = np.linspace(0, len(label) - 1, n_check, dtype=int) if n_check else np.array([], int)
    raw_margin = []
    for i in check_idx:
        st = GX2.state_from_low5(low5[i])
        raw_margin.append(GM2.window_socp_margin(st, controls[i], env, gamma[i]))
    raw_margin = np.asarray(raw_margin, dtype=float)

    per_gamma = {}
    for g in GAMMAS:
        z = np.isclose(gamma, g)
        per_gamma[str(g)] = dict(
            windows=int(z.sum()),
            rollout_ids=int(len(np.unique(rid[z]))) if z.any() else 0,
            window_share=float(z.mean()),
            frontier_rate=float(actual_front[z].mean()) if z.any() else None,
            clipped_infeasible_rate=float((margin[z] <= -4.999).mean()) if z.any() else None,
            mean_sigma=float(sigma[z].mean()) if z.any() else None,
            mean_progress=float(prog[z].mean()) if z.any() else None,
        )

    def path_rate(key: str):
        return float(np.mean([x[key] for x in path_rows])) if path_rows else None

    return dict(
        snapshot=str(path),
        iteration=int(db["iter"]),
        n_windows=int(len(label)),
        n_paths=int(len(path_rows)),
        quantile=float(db["quantile"]),
        planes=dict(
            sigma=float(db["sigma_plane"]),
            margin=float(db["margin_plane"]),
            progress=float(db["prog_plane"]),
        ),
        label_mismatch=int(np.count_nonzero(expected_front != actual_front)),
        frontier_rate=float(actual_front.mean()),
        sigma=dict(min=float(sigma.min()), max=float(sigma.max()), std=float(sigma.std())),
        certificate=dict(
            clipped_infeasible_rate=float((margin <= -4.999).mean()),
            clipped_infeasible_frontier_rate=float((margin[actual_front] <= -4.999).mean())
            if actual_front.any()
            else None,
            zero_tie_rate=float((np.abs(margin) <= 1e-8).mean()),
            recomputed_n=int(n_check),
            recomputed_infeasible_rate=float((~np.isfinite(raw_margin)).mean()) if n_check else None,
            stored_recompute_max_abs_error=_f(
                np.max(np.abs(np.clip(raw_margin, -5.0, 5.0) - margin[check_idx]))
            )
            if n_check
            else None,
        ),
        paths=dict(
            strict_reach_rate=path_rate("strict_reach"),
            legacy_reach_rate=path_rate("legacy_reach"),
            executed_socp_rate=path_rate("socp"),
            executed_valid2_rate=path_rate("valid2"),
            taskspace_rate=path_rate("taskspace"),
            final_distance_min=min((x["final_distance"] for x in path_rows), default=None),
            final_distance_max=max((x["final_distance"] for x in path_rows), default=None),
        ),
        per_gamma=per_gamma,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", nargs="+", type=Path)
    ap.add_argument("--recompute", type=int, default=64)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = [audit_snapshot(p, args.recompute) for p in args.snapshot]
    text = json.dumps(rows, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
