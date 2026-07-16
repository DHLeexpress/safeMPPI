#!/usr/bin/env python3
"""Publishable table/figures/video for deployment-only adaptive gamma."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import eval_ae as EVAL
import grid_scene as GS


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def load_fixed(directory: Path, prefix: str):
    rows = []
    for gamma in GAMMAS:
        row = json.loads((directory / f"row_g{float(gamma)}.json").read_text())
        row.update(label=f"{prefix} gamma={gamma:.1f}", schedule=prefix, deployed_gamma=gamma)
        rows.append(row)
    return rows


def load_adaptive(directory: Path):
    rows = []
    for mode in ("heuristic", "verifier", "random"):
        row = json.loads((directory / f"row_{mode}.json").read_text())
        row.update(label=("random-gamma schedule" if mode == "random" else f"adaptive-{mode}"),
                   schedule=mode, deployed_gamma=None)
        rows.append(row)
    return rows


def pm(row, mean, std, digits):
    return f'{row[mean]:.{digits}f} +/- {row[std]:.{digits}f}'


def write_table(rows, path: Path, claim):
    fields = ("label", "schedule", "deployed_gamma", "SR", "CR", "clearance_mean", "clearance_std",
              "time_mean_s", "time_std_s", "coverage", "n_success", "M")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    lines = ["# Adaptive-gamma deployment on WALLS-4", "",
             "Adaptive rows are deployment-only and are not included in the faithful fixed-gamma suite table.", "",
             "| Method | SR | CR | Clearance (m) | Time (s) | Coverage | n/M |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["label"]} | {r["SR"]:.1%} | {r["CR"]:.1%} | '
                     f'{pm(r, "clearance_mean", "clearance_std", 3)} | '
                     f'{pm(r, "time_mean_s", "time_std_s", 2)} | {r["coverage"]} | '
                     f'{r["n_success"]}/{r["M"]} |')
    lines += ["", "## Pareto-front claim test", "",
              f'- Safest fixed-gamma mean clearance: {claim["safest_fixed_clearance"]:.3f} m.',
              f'- Fastest fixed-gamma mean completion: {claim["fastest_fixed_time"]:.2f} s.']
    for mode, result in claim["adaptive"].items():
        verdict = "PASSES" if result["above_front"] else "does not pass"
        lines.append(f'- **{mode} {verdict}** the strict mean-level claim '
                     f'(clearance {result["clearance"]:.3f} m, time {result["time"]:.2f} s).')
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def claim_test(fixed, adaptive):
    safest = max(r["clearance_mean"] for r in fixed)
    fastest = min(r["time_mean_s"] for r in fixed)
    result = {"safest_fixed_clearance": safest, "fastest_fixed_time": fastest, "adaptive": {}}
    for r in adaptive:
        if r["schedule"] == "random":
            continue
        result["adaptive"][r["schedule"]] = dict(
            clearance=r["clearance_mean"], time=r["time_mean_s"],
            safety_pass=r["clearance_mean"] >= safest,
            speed_pass=r["time_mean_s"] <= fastest,
            above_front=r["clearance_mean"] >= safest and r["time_mean_s"] <= fastest)
    return result


def pareto_plot(fixed, adaptive, experts, out: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(fixed):
        color = cmap(i / (len(fixed) - 1))
        ax.add_patch(Ellipse((r["time_mean_s"], r["clearance_mean"]),
                             2 * r["time_std_s"], 2 * r["clearance_std"],
                             facecolor=color, edgecolor=color, alpha=.15))
        ax.scatter(r["time_mean_s"], r["clearance_mean"], color=color, s=75, zorder=3)
        ax.annotate(f'g={r["deployed_gamma"]:.1f}\nSR {r["SR"]:.0%}/CR {r["CR"]:.0%}',
                    (r["time_mean_s"], r["clearance_mean"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    marks = {"heuristic": ("*", "#0077bb"), "verifier": ("P", "#ee3377"), "random": ("X", "#888888")}
    for r in adaptive:
        marker, color = marks[r["schedule"]]
        ax.add_patch(Ellipse((r["time_mean_s"], r["clearance_mean"]),
                             2 * r["time_std_s"], 2 * r["clearance_std"],
                             facecolor=color, edgecolor=color, alpha=.15))
        ax.scatter(r["time_mean_s"], r["clearance_mean"], marker=marker, color=color,
                   s=180, label=r["label"], zorder=4)
        ax.annotate(f'SR {r["SR"]:.0%}/CR {r["CR"]:.0%}',
                    (r["time_mean_s"], r["clearance_mean"]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8)
    ax.plot([r["time_mean_s"] for r in experts], [r["clearance_mean"] for r in experts],
            "--D", color="black", ms=5, lw=1.3, label="walled expert across gamma")
    ax.set_xlabel("Successful-episode completion time (s) — lower is better")
    ax.set_ylabel("Successful-episode clearance (m) — higher is better")
    ax.set_title("Adaptive gamma versus the fixed-gamma Pareto set (1-std ellipses)")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def load_trace(npz_path: Path):
    with np.load(npz_path, allow_pickle=True) as z:
        paths, traces, clearances = z["paths"], z["gamma_traces"], z["clearance_traces"]
    reached = [i for i, p in enumerate(paths) if np.linalg.norm(np.asarray(p)[-1] - np.array([5., 5.])) < .1]
    pool = reached or list(range(len(paths)))
    idx = min(pool, key=lambda i: abs(len(paths[i]) - np.median([len(paths[j]) for j in pool])))
    return np.asarray(paths[idx]), np.asarray(traces[idx]), np.asarray(clearances[idx]), idx


def trace_figure(adaptive_dir: Path, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for mode, color in (("heuristic", "#0077bb"), ("verifier", "#ee3377"), ("random", "#888888")):
        _, gamma, clearance, idx = load_trace(adaptive_dir / f"paths_{mode}.npz")
        axes[0].step(np.arange(len(gamma)) * .1, gamma, where="post", label=f"{mode} (seed index {idx})", color=color)
        axes[1].plot(np.arange(len(clearance)) * .1, clearance, label=mode, color=color)
    axes[0].set_ylabel("deployed gamma(t)"); axes[0].set_ylim(.05, 1.05); axes[0].legend()
    axes[1].set_ylabel("nearest-obstacle clearance (m)"); axes[1].set_xlabel("time (s)")
    axes[1].axhline(.3, ls=":", color="black", lw=1, label="heuristic d_lo")
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle("Adaptive gamma traces: lower near pinches, higher in open space")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def gamma_video(adaptive_dir: Path, out: Path, fps=12):
    path, gamma, _, idx = load_trace(adaptive_dir / "paths_verifier.npz")
    env = GS.make_grid(); EVAL._apply_wall_plugs_eval(env, 4)
    obs = env.obstacles.detach().cpu().numpy()
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(.1, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gamma_video_", dir=out.parent) as td:
        td = Path(td)
        for k in range(1, len(path)):
            fig, ax = plt.subplots(figsize=(7, 7))
            for o in obs:
                ax.add_patch(Circle(o[:2], o[2], color="#777777"))
            segments = np.stack([path[:-1], path[1:]], axis=1)[:k]
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
            lc.set_array(gamma[:k]); ax.add_collection(lc)
            ax.plot(0, 0, "ks"); ax.plot(5, 5, "*", color="gold", mec="black", ms=16)
            ax.set(xlim=(-.25, 5.25), ylim=(-.25, 5.25), aspect="equal",
                   title=f"Verifier-guided adaptive gamma — rollout {idx}, t={k * .1:.1f}s")
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="gamma(t)")
            fig.tight_layout(); fig.savefig(td / f"f{k-1:05d}.png", dpi=120); plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", str(td / "f%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", str(out)], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-dir", type=Path, required=True)
    ap.add_argument("--adaptive-dir", type=Path, required=True)
    ap.add_argument("--expert-dir", type=Path, default=ROOT / "results/expert_gt_walls4")
    ap.add_argument("--table", type=Path, default=ROOT / "tables/T_ADAPTIVE_GAMMA")
    ap.add_argument("--pareto", type=Path, default=ROOT / "figures/adaptive_gamma_pareto.png")
    ap.add_argument("--traces", type=Path, default=ROOT / "figures/adaptive_gamma_traces.png")
    ap.add_argument("--video", type=Path, default=ROOT / "video/adaptive_gamma_colored.mp4")
    args = ap.parse_args()
    fixed = load_fixed(args.fixed_dir, "fixed")
    adaptive = load_adaptive(args.adaptive_dir)
    experts = load_fixed(args.expert_dir, "walled expert")
    claim = claim_test(fixed, adaptive)
    rows = fixed + adaptive + experts
    write_table(rows, args.table, claim)
    args.table.with_suffix(".claim.json").write_text(json.dumps(claim, indent=2) + "\n")
    pareto_plot(fixed, adaptive, experts, args.pareto)
    trace_figure(args.adaptive_dir, args.traces)
    gamma_video(args.adaptive_dir, args.video)
    print(f"wrote {args.table}.md/.csv, {args.pareto}, {args.traces}, {args.video}")


if __name__ == "__main__":
    main()
