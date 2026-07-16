#!/usr/bin/env python3
"""Assemble the WALLS-4 suite table, paper plots, and collision-location audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import eval_ae as EVAL
import grid_scene as GS


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
DEFAULT_INPUTS = (
    ("Full pipeline it100", ROOT / "results/p2/eval_walls4_base_it100_m100"),
    ("No curriculum it100", ROOT / "results/p2/eval_walls4_nocur_it100_m100"),
    ("No multi-step SOCP it100", ROOT / "results/p2/eval_walls4_nosocp_it100_m100"),
    ("No progress it100", ROOT / "results/p2/eval_walls4_noprog_it100_m100"),
    ("Walled expert", ROOT / "results/expert_gt_walls4"),
    ("Pretrained zero-shot (M25)", ROOT / "results/p2/eval_walls4_pretrained_m25"),
    ("s792 zero-shot (M25)", ROOT / "results/p2/eval_walls4_s792_m25"),
)


def gstr(g: float) -> str:
    return str(float(g))


def load_rows(inputs):
    rows = []
    for display, directory in inputs:
        for g in GAMMAS:
            path = directory / f"row_g{gstr(g)}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text())
            row["method"] = display
            rows.append(row)
    return rows


def write_table(rows, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(EVAL.FIELDS) + ["coverage_ids"]
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["coverage_ids"] = ";".join(out.get("coverage_ids", []))
            writer.writerow(out)

    lines = [
        "# WALLS-4 from-scratch suite (iteration 100)", "",
        "All policy rows use the faithful 4-plug scene, fixed gamma, seeds 0--99, temp=1, NFE=8, "
        "and no inference-time filter. Zero-shot context rows are explicitly marked M25.", "",
        "| Method | gamma | SR | CR | successful-episode clearance (m) | successful-episode time (s) | Coverage | n/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        c = "--" if not np.isfinite(row["clearance_mean"]) else f'{row["clearance_mean"]:.3f} +/- {row["clearance_std"]:.3f}'
        t = "--" if not np.isfinite(row["time_mean_s"]) else f'{row["time_mean_s"]:.2f} +/- {row["time_std_s"]:.2f}'
        lines.append(
            f'| {row["method"]} | {row["gamma"]:.1f} | {row["SR"]:.1%} | {row["CR"]:.1%} | '
            f'{c} | {t} | {row["coverage"]} | {row["n_success"]}/{row["M"]} |'
        )
    lines += [
        "", "Clearance is the episode mean over time of nearest-obstacle clearance, summarized only over successful episodes. "
        "Time and coverage likewise use successful episodes; failed/colliding early deaths cannot appear artificially fast.", "",
    ]
    prefix.with_suffix(".md").write_text("\n".join(lines))


def plot_clearance_time(rows, out: Path):
    primary = [name for name, _ in DEFAULT_INPUTS[:5]]
    colors = dict(zip(primary, ("#4477aa", "#ee7733", "#cc3311", "#aa4499", "#222222")))
    markers = dict(zip(primary, ("o", "s", "X", "^", "D")))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for name in primary:
        selected = sorted((r for r in rows if r["method"] == name), key=lambda r: r["gamma"])
        x = np.array([r["gamma"] for r in selected])
        axes[0].errorbar(x, [r["clearance_mean"] for r in selected],
                         yerr=[r["clearance_std"] for r in selected], label=name,
                         color=colors[name], marker=markers[name], lw=2, capsize=3)
        axes[1].errorbar(x, [r["time_mean_s"] for r in selected],
                         yerr=[r["time_std_s"] for r in selected], label=name,
                         color=colors[name], marker=markers[name], lw=2, capsize=3)
    axes[0].set_title("Safety distribution across gamma")
    axes[0].set_ylabel("Successful-episode clearance, mean +/- std (m)")
    axes[1].set_title("Performance distribution across gamma")
    axes[1].set_ylabel("Successful-episode completion time, mean +/- std (s)")
    for ax in axes:
        ax.set_xlabel("gamma")
        ax.set_xticks(GAMMAS)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("WALLS-4 faithful fixed-gamma evaluation (M100 per gamma)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collision_audit(inputs, out_prefix: Path):
    env = GS.make_grid()
    base_n = len(env.obstacles)
    EVAL._apply_wall_plugs_eval(env, 4)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    report = {}
    for display, directory in inputs[:4]:
        counts, total, max_depth = Counter(), 0, 0.0
        by_gamma = {}
        for g in GAMMAS:
            paths = EVAL.load_paths(directory / f"paths_g{gstr(g)}.npz")
            gc = Counter()
            for raw in paths:
                p = np.asarray(raw)[:, :2]
                d = np.linalg.norm(p[:, None] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
                if d.min() >= 0:
                    continue
                total += 1
                j = int(np.unravel_index(np.argmin(d), d.shape)[1])
                coord = f"({obs[j, 0]:.2f},{obs[j, 1]:.2f})"
                label = ("wall-plug " if j >= base_n else "interior ") + coord
                counts[label] += 1
                gc[label] += 1
                max_depth = max(max_depth, float(-d.min()))
            by_gamma[gstr(g)] = dict(gc)
        report[display] = dict(collision_episodes=total, locations=dict(counts),
                               max_penetration_m=max_depth, by_gamma=by_gamma)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# WALLS-4 collision-location audit", "",
             "Each colliding episode is assigned to the obstacle with the deepest penetration. "
             "The four added boundary plugs are labeled separately from interior obstacles.", "",
             "| Method | Collision episodes / 700 | Interior locations | Wall-plug locations | Max depth (m) |",
             "|---|---:|---|---|---:|"]
    for name, rec in report.items():
        interior = ", ".join(f"{k.removeprefix('interior ')}: {v}" for k, v in rec["locations"].items() if k.startswith("interior")) or "--"
        plugs = ", ".join(f"{k.removeprefix('wall-plug ')}: {v}" for k, v in rec["locations"].items() if k.startswith("wall-plug")) or "--"
        lines.append(f'| {name} | {rec["collision_episodes"]} | {interior} | {plugs} | {rec["max_penetration_m"]:.4f} |')
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-prefix", type=Path, default=ROOT / "tables/T_WALLS4_SUITE")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures/walls4_clearance_time_vs_expert.png")
    parser.add_argument("--collision-prefix", type=Path, default=ROOT / "analysis/walls4_collision_locations")
    args = parser.parse_args()
    rows = load_rows(DEFAULT_INPUTS)
    write_table(rows, args.table_prefix)
    plot_clearance_time(rows, args.figure)
    collision_audit(DEFAULT_INPUTS, args.collision_prefix)
    print(f"wrote {args.table_prefix}.md/.csv, {args.figure}, and {args.collision_prefix}.md/.json")


if __name__ == "__main__":
    main()
