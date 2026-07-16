#!/usr/bin/env python3
"""Read-only test of the near-origin-window explanation for SR<1 with CR=0.

The probe separates three claims that should not be conflated:
1. accepted training windows are concentrated near the origin;
2. near-origin target controls are numerically low-rank/ill-conditioned;
3. faithful deployment failures actually originate at the boundary.

It consumes saved viz_db snapshots and eval_ae path archives. It never trains or
modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
GOAL = np.array([5.0, 5.0])
R_GOAL = 5.0
TASK_EPS = 0.12


def named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = text.split("=", 1)
    return name, Path(path)


def scalar_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
    }


def summarize_mask(mask, label, sigma, progress, margin, condition, first_action):
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "easy_fraction": float(np.mean(label[mask] == "easy")),
        "frontier_fraction": float(np.mean(label[mask] == "frontier")),
        "sigma": scalar_summary(sigma[mask]),
        "progress": scalar_summary(progress[mask]),
        "margin": scalar_summary(margin[mask]),
        "control_centered_2d_condition": scalar_summary(condition[mask]),
        "first_action_mean": [float(v) for v in first_action[mask].mean(0)],
    }


def window_snapshot(path: Path, radius: float) -> dict:
    z = torch.load(path, map_location="cpu", weights_only=False)
    low5 = z["low5"].numpy()
    pos = GOAL[None] - low5[:, :2] * R_GOAL
    r = np.linalg.norm(pos, axis=1)
    controls = z["U"].numpy()
    centered = controls - controls.mean(axis=1, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    condition = s[:, 0] / np.maximum(s[:, 1], 1e-6)
    label = np.asarray(z["label"], dtype=str)
    sigma = np.asarray(z["sigma"], float)
    progress = np.asarray(z["prog"], float)
    margin = np.asarray(z["margin"], float)
    gamma = np.asarray(z["gamma"], float)
    near = r < radius
    rows = {}
    for g in GAMMAS:
        gm = np.isclose(gamma, g)
        rows[str(g)] = {
            "n": int(gm.sum()),
            "near_origin_fraction": float(np.mean(near[gm])) if gm.any() else None,
            "near_origin": summarize_mask(gm & near, label, sigma, progress, margin,
                                           condition, controls[:, 0]),
            "away_from_origin": summarize_mask(gm & ~near, label, sigma, progress, margin,
                                                 condition, controls[:, 0]),
        }
    return {
        "path": str(path),
        "iteration": int(z["iter"]),
        "n_windows": int(len(label)),
        "near_origin_radius": radius,
        "near_origin_fraction": float(np.mean(near)),
        "near_origin": summarize_mask(near, label, sigma, progress, margin,
                                       condition, controls[:, 0]),
        "away_from_origin": summarize_mask(~near, label, sigma, progress, margin,
                                            condition, controls[:, 0]),
        "by_gamma": rows,
    }


def path_kind(path: np.ndarray) -> str:
    xy = np.asarray(path, float)[:, :2]
    if np.linalg.norm(xy[-1] - GOAL) < 0.1:
        return "success"
    outside = bool((xy < -TASK_EPS).any() or (xy > 5.0 + TASK_EPS).any())
    radius = np.linalg.norm(xy, axis=1)
    if outside and len(xy) - 1 <= 25 and radius.max() < 1.0:
        return "origin_boundary_oob"
    if outside and np.linalg.norm(xy[-1] - GOAL) < 0.3:
        return "near_goal_oob"
    if outside:
        return "other_oob"
    return "timeout_or_nonreach"


def eval_archive(directory: Path) -> dict:
    counts = Counter()
    seed_kinds = defaultdict(list)
    rows = {}
    for g in GAMMAS:
        archive = directory / f"paths_g{g}.npz"
        if not archive.exists():
            continue
        with np.load(archive, allow_pickle=True) as z:
            paths = list(z["paths"])
            seeds = [int(v) for v in z["seeds"]]
        local = Counter()
        failures = []
        for seed, raw in zip(seeds, paths):
            path = np.asarray(raw, float)
            kind = path_kind(path)
            local[kind] += 1
            counts[kind] += 1
            seed_kinds[seed].append((g, kind))
            if kind != "success":
                failures.append({
                    "seed": seed,
                    "kind": kind,
                    "steps": int(len(path) - 1),
                    "endpoint": [float(v) for v in path[-1, :2]],
                    "final_goal_distance": float(np.linalg.norm(path[-1, :2] - GOAL)),
                    "near_origin_step_fraction": float(np.mean(np.linalg.norm(path[:, :2], axis=1) < 1.0)),
                })
        rows[str(g)] = {"counts": dict(local), "failures": failures}
    repeated = {}
    for seed, vals in seed_kinds.items():
        origin_gammas = [g for g, kind in vals if kind == "origin_boundary_oob"]
        if origin_gammas:
            repeated[str(seed)] = origin_gammas
    return {
        "directory": str(directory),
        "counts": dict(counts),
        "by_gamma": rows,
        "origin_boundary_oob_gammas_by_seed": repeated,
    }


def render_markdown(result: dict) -> str:
    out = [
        "# Origin-window / faithful-failure diagnostic",
        "",
        "This is a read-only causal triage. `control condition` is the ratio of the two singular values "
        "of each centered 10×2 target-control matrix; it is an explicit numerical proxy, not a synonym "
        "for poor closed-loop behavior.",
        "",
        "## Accepted training windows",
        "",
        "| Snapshot | Windows | near origin | easy near / away | σ near / away | progress near / away | condition median near / away |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["window_snapshots"].items():
        near, away = row["near_origin"], row["away_from_origin"]
        out.append(
            f"| {name} (it{row['iteration']}) | {row['n_windows']} | "
            f"{100 * row['near_origin_fraction']:.1f}% | "
            f"{100 * near['easy_fraction']:.1f}% / {100 * away['easy_fraction']:.1f}% | "
            f"{near['sigma']['mean']:.3f} / {away['sigma']['mean']:.3f} | "
            f"{near['progress']['mean']:.3f} / {away['progress']['mean']:.3f} | "
            f"{near['control_centered_2d_condition']['median']:.2f} / "
            f"{away['control_centered_2d_condition']['median']:.2f} |"
        )
    out += ["", "## Faithful evaluation failure taxonomy", "",
            "| Evaluation | success | origin-boundary OOB | near-goal OOB | other OOB/nonreach | repeated origin seed(s) |",
            "|---|---:|---:|---:|---:|---|"]
    for name, row in result["evaluations"].items():
        c = row["counts"]
        other = c.get("other_oob", 0) + c.get("timeout_or_nonreach", 0)
        repeated = ", ".join(f"{s}:γ={gs}" for s, gs in row["origin_boundary_oob_gammas_by_seed"].items())
        out.append(f"| {name} | {c.get('success', 0)} | {c.get('origin_boundary_oob', 0)} | "
                   f"{c.get('near_goal_oob', 0)} | {other} | {repeated or '—'} |")
    out += [
        "",
        "## Interpretation",
        "",
        "- Near-origin accepted windows are high-uncertainty and mostly land in the easy pool because the "
        "frontier is a three-way AND cell. The table quantifies their share rather than calling all of them bad.",
        "- The near-origin target-control condition number must be compared with the away group. Similar values "
        "argue against a numerical low-rank-window explanation.",
        "- Repeated origin-boundary OOB for the same latent seed across γ is direct evidence of a faithful-flow "
        "tail failure at the boundary. Near-goal OOB is a separate overshoot mechanism and needs separate treatment.",
        "- Do not loosen Valid2 or add an inference safety filter to make the metric pass. Diagnose and correct "
        "the generative tail while retaining faithful temp=1 evaluation.",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="append", type=named_path, default=[], metavar="NAME=PATH")
    ap.add_argument("--eval-dir", action="append", type=named_path, default=[], metavar="NAME=DIR")
    ap.add_argument("--origin-radius", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("analysis/origin_window_failure_probe.json"))
    ap.add_argument("--markdown", type=Path, default=Path("analysis/origin_window_failure_probe.md"))
    args = ap.parse_args()
    result = {
        "window_snapshots": {name: window_snapshot(path, args.origin_radius) for name, path in args.viz},
        "evaluations": {name: eval_archive(path) for name, path in args.eval_dir},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(render_markdown(result) + "\n")
    print(f"wrote {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
