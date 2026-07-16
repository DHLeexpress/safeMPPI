"""Aesthetic overlays of saved DR training trajectories (2026-07-06).

This module does not re-run SafeMPPI. It decodes the saved training windows in
dataset/dr05_windows_g*.pt back into executed state paths and plots the same
randomly selected successful starts across all gammas.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import _paths  # noqa: F401
import grid_feats as GF
import grid_scene as GS
from di_grid_viz import di_step


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
OUTDIR = os.path.join(HERE, "figures", "dr_offdiag")

GAMMA_RE = re.compile(r"_g([0-9.]+)\.pt$")


@dataclass(frozen=True)
class Trajectory:
    start: np.ndarray
    path: np.ndarray


def configure_fonts() -> None:
    """Use LaTeX rendering when available, otherwise Computer Modern math fonts."""
    import shutil

    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def parse_gamma(path: str) -> float:
    match = GAMMA_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse gamma from {path}")
    return float(match.group(1))


def find_shards(data_dir: str, prefix: str) -> dict[float, str]:
    shards = {}
    for path in glob(os.path.join(data_dir, f"{prefix}*_g*.pt")):
        shards[parse_gamma(path)] = path
    if not shards:
        raise FileNotFoundError(f"No shards matching {prefix}*_g*.pt in {data_dir}")
    return dict(sorted(shards.items()))


def decode_saved_trajectories(shard_path: str, env) -> list[Trajectory]:
    """Decode sequential saved windows into per-trajectory paths.

    The dataset stores one record per time step: low5 contains the current
    state features and U[:, 0] contains the executed expert control. The final
    terminal state is not a low5 record, so we append it by integrating the last
    saved state and first target control once.
    """
    d = torch.load(shard_path, map_location="cpu")
    starts_t = d.get("starts")
    if starts_t is None:
        raise KeyError(f"{shard_path} has no 'starts' tensor")

    starts = starts_t.detach().cpu().numpy().astype(np.float32)
    low5 = d["low5"].detach().cpu().numpy().astype(np.float32)
    controls = d["U"][:, 0, :].detach().cpu().numpy().astype(np.float32)

    goal = env.goal.detach().cpu().numpy().astype(np.float32)
    pos = goal[None, :] - low5[:, :2] * GF.R_GOAL
    vel = low5[:, 2:4] * GF.V_SCALE
    states = np.concatenate([pos, vel], axis=1).astype(np.float32)

    segment_starts: list[int] = []
    cursor = 0
    for start in starts:
        delta = np.max(np.abs(pos[cursor:] - start[:2][None, :]), axis=1)
        hits = np.flatnonzero(delta < 1e-4)
        if len(hits) == 0:
            raise RuntimeError(
                f"Could not align start {start[:2].round(4).tolist()} in {os.path.basename(shard_path)}"
            )
        idx = cursor + int(hits[0])
        segment_starts.append(idx)
        cursor = idx + 1

    trajectories: list[Trajectory] = []
    for i, start_idx in enumerate(segment_starts):
        end_idx = segment_starts[i + 1] if i + 1 < len(segment_starts) else len(states)
        seg_states = states[start_idx:end_idx]
        seg_controls = controls[start_idx:end_idx]
        if len(seg_states) == 0:
            continue
        final_state = di_step(seg_states[-1], seg_controls[-1], dt=env.dt)
        path = np.vstack([seg_states[:, :2], final_state[:2][None, :]]).astype(np.float32)
        trajectories.append(Trajectory(start=starts[i].copy(), path=path))
    return trajectories


def draw_scene(ax, env, offdiag: float, *, label_scene: bool = False) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, color="#e6e1d8", lw=0.45, alpha=0.7)

    if offdiag > 0:
        xs = np.linspace(0.0, GS.GRID_M, 240)
        ax.fill_between(
            xs,
            xs - offdiag,
            xs + offdiag,
            color="#dedbd2",
            alpha=0.9,
            zorder=0,
            label="excluded band" if label_scene else None,
        )

    obstacles = env.obstacles.detach().cpu().numpy()
    for j, (ox, oy, rr) in enumerate(obstacles):
        is_wall = j >= GS.N_INTERIOR
        ax.add_patch(
            Circle(
                (float(ox), float(oy)),
                float(rr),
                facecolor="#6b6b68" if not is_wall else "#2f2f2d",
                edgecolor="white",
                lw=0.7 if not is_wall else 0.35,
                alpha=0.95 if not is_wall else 0.28,
                zorder=3 if not is_wall else 2,
            )
        )

    goal = env.goal.detach().cpu().numpy()
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        s=190,
        c="#d9272e",
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
        label="goal" if label_scene else None,
    )
    ax.set_xlim(0, GS.GRID_M)
    ax.set_ylim(0, GS.GRID_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, GS.GRID_M + 0.1, 1.0))
    ax.set_yticks(np.arange(0, GS.GRID_M + 0.1, 1.0))
    ax.tick_params(labelsize=8, colors="#4b4a45", length=2.5)
    for spine in ax.spines.values():
        spine.set_color("#3d3b36")
        spine.set_linewidth(0.8)


def colors_for_gammas(gammas: list[float]) -> dict[float, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("viridis")
    vals = np.linspace(0.08, 0.92, len(gammas))
    return {g: cmap(v) for g, v in zip(gammas, vals)}


def colors_for_starts(n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def plot_by_start(
    env,
    trajs_by_gamma: dict[float, list[Trajectory]],
    selected: np.ndarray,
    offdiag: float,
    out_path: str,
) -> None:
    gammas = sorted(trajs_by_gamma)
    gamma_colors = colors_for_gammas(gammas)

    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.7), constrained_layout=False)
    axes = axes.ravel()

    for panel_idx, traj_idx in enumerate(selected):
        ax = axes[panel_idx]
        draw_scene(ax, env, offdiag)
        for gamma in gammas:
            tr = trajs_by_gamma[gamma][int(traj_idx)]
            ax.plot(
                tr.path[:, 0],
                tr.path[:, 1],
                color=gamma_colors[gamma],
                lw=1.9,
                alpha=0.92,
                solid_capstyle="round",
                zorder=6,
            )
            ax.scatter(
                [tr.path[-1, 0]],
                [tr.path[-1, 1]],
                s=13,
                color=gamma_colors[gamma],
                edgecolor="none",
                zorder=7,
            )
        start = trajs_by_gamma[gammas[0]][int(traj_idx)].start
        ax.scatter(
            [start[0]],
            [start[1]],
            s=58,
            marker="o",
            facecolor="#101010",
            edgecolor="white",
            linewidth=1.0,
            zorder=9,
        )
        ax.set_title(
            f"start {int(traj_idx):02d}   ({start[0]:.2f}, {start[1]:.2f})",
            fontsize=10.5,
            color="#25231f",
            pad=7,
        )

    handles = [
        Line2D([0], [0], color=gamma_colors[g], lw=2.6, label=f"gamma {g:g}") for g in gammas
    ]
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(gammas),
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.suptitle(
        "Saved training trajectories: same 10 starts, gamma overlays",
        fontsize=16,
        fontweight="semibold",
        color="#181713",
        y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.992, top=0.91, bottom=0.125, wspace=0.08, hspace=0.2)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_gamma(
    env,
    trajs_by_gamma: dict[float, list[Trajectory]],
    selected: np.ndarray,
    offdiag: float,
    out_path: str,
) -> None:
    gammas = sorted(trajs_by_gamma)
    start_colors = colors_for_starts(len(selected))

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.05), constrained_layout=False)
    axes = axes.ravel()

    for ax_idx, gamma in enumerate(gammas):
        ax = axes[ax_idx]
        draw_scene(ax, env, offdiag)
        for local_idx, traj_idx in enumerate(selected):
            tr = trajs_by_gamma[gamma][int(traj_idx)]
            color = start_colors[local_idx]
            ax.plot(
                tr.path[:, 0],
                tr.path[:, 1],
                color=color,
                lw=1.45,
                alpha=0.84,
                solid_capstyle="round",
                zorder=6,
            )
            ax.scatter(
                [tr.start[0]],
                [tr.start[1]],
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=8,
            )
        ax.set_title(f"gamma {gamma:g}", fontsize=11.5, color="#25231f", pad=7)

    map_ax = axes[-1]
    draw_scene(map_ax, env, offdiag)
    for local_idx, traj_idx in enumerate(selected):
        start = trajs_by_gamma[gammas[0]][int(traj_idx)].start
        map_ax.scatter(
            [start[0]],
            [start[1]],
            s=74,
            color=start_colors[local_idx],
            edgecolor="white",
            linewidth=0.9,
            zorder=8,
        )
        map_ax.text(
            start[0],
            start[1],
            str(local_idx + 1),
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            fontweight="bold",
            zorder=9,
        )
    map_ax.set_title("selected starts", fontsize=11.5, color="#25231f", pad=7)

    handles = []
    for local_idx, traj_idx in enumerate(selected):
        start = trajs_by_gamma[gammas[0]][int(traj_idx)].start
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                lw=1.6,
                color=start_colors[local_idx],
                label=f"{local_idx + 1}: s{int(traj_idx):02d} ({start[0]:.1f},{start[1]:.1f})",
            )
        )
    fig.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.047),
    )
    fig.suptitle(
        "Saved training trajectories per gamma: 10 random off-diagonal starts",
        fontsize=16,
        fontweight="semibold",
        color="#181713",
        y=0.975,
    )
    fig.subplots_adjust(left=0.04, right=0.992, top=0.9, bottom=0.14, wspace=0.12, hspace=0.18)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_one_overlay(
    env,
    trajs_by_gamma: dict[float, list[Trajectory]],
    selected: np.ndarray,
    offdiag: float,
    out_path: str,
) -> None:
    gammas = sorted(trajs_by_gamma)
    base_cmap = plt.get_cmap("plasma")
    gamma_rgba = [base_cmap(v) for v in np.linspace(0.08, 0.92, len(gammas))]
    cmap = colors.ListedColormap(gamma_rgba, name="gamma_steps")
    edges = np.empty(len(gammas) + 1, dtype=float)
    edges[1:-1] = (np.array(gammas[:-1]) + np.array(gammas[1:])) / 2.0
    edges[0] = gammas[0] - (edges[1] - gammas[0])
    edges[-1] = gammas[-1] + (gammas[-1] - edges[-2])
    norm = colors.BoundaryNorm(edges, cmap.N)
    gamma_colors = {g: gamma_rgba[i] for i, g in enumerate(gammas)}

    fig, ax = plt.subplots(figsize=(13.0, 9.2))
    draw_scene(ax, env, offdiag)

    for gamma in gammas:
        for traj_idx in selected:
            tr = trajs_by_gamma[gamma][int(traj_idx)]
            ax.plot(
                tr.path[:, 0],
                tr.path[:, 1],
                color=gamma_colors[gamma],
                lw=1.75,
                alpha=0.34,
                solid_capstyle="round",
                zorder=6,
            )

    starts = np.stack([trajs_by_gamma[gammas[0]][int(i)].start for i in selected], axis=0)
    ax.scatter(
        starts[:, 0],
        starts[:, 1],
        s=58,
        marker="o",
        facecolor="#111111",
        edgecolor="white",
        linewidth=0.9,
        zorder=9,
        label="selected starts",
    )

    start_lines = []
    for rank, (traj_idx, start) in enumerate(zip(selected, starts), start=1):
        start_lines.append(
            rf"$\mathbf{{{rank}}}$: $s_{{{int(traj_idx):02d}}}$"
            rf"$\,({start[0]:.2f},\,{start[1]:.2f})$"
        )
    label_cols = 2 if len(start_lines) > 18 else 1
    rows_per_col = int(np.ceil(len(start_lines) / label_cols))
    ax.text(
        1.19,
        0.965,
        "Selected starts",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#171612",
    )
    for col in range(label_cols):
        lines = start_lines[col * rows_per_col : (col + 1) * rows_per_col]
        ax.text(
            1.19 + 0.38 * col,
            0.905,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5 if label_cols > 1 else 15,
            color="#171612",
            linespacing=1.28 if label_cols > 1 else 1.45,
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.13, 0.905, 0.39, 0.038])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=edges,
        ticks=gammas,
        spacing="proportional",
        drawedges=True,
    )
    cbar.ax.set_title(r"$\gamma$", fontsize=24, pad=8)
    cbar.ax.tick_params(labelsize=13, length=0, pad=4)
    cbar.outline.set_edgecolor("#2f2d28")
    cbar.outline.set_linewidth(0.8)
    if hasattr(cbar, "dividers"):
        cbar.dividers.set_color("white")
        cbar.dividers.set_linewidth(1.0)
    ax.set_xlabel(r"$x$ [m]", fontsize=18, labelpad=8)
    ax.set_ylabel(r"$y$ [m]", fontsize=18, labelpad=8)
    ax.tick_params(labelsize=16)
    fig.subplots_adjust(left=0.075, right=0.62, top=0.86, bottom=0.11)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def select_start_indices(starts: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(starts), size=min(n, len(starts)), replace=False)
    # Keep the random sample but lay it out spatially for easier comparison.
    selected = np.array(sorted(selected, key=lambda i: (-float(starts[i, 1]), float(starts[i, 0]))))
    return selected.astype(int)


def write_selection_summary(
    path: str,
    selected: np.ndarray,
    starts: np.ndarray,
    gammas: list[float],
    by_start_path: str | None,
    per_gamma_path: str | None,
    one_overlay_path: str,
) -> None:
    import json

    rows = [
        {
            "rank": int(k + 1),
            "trajectory_index": int(i),
            "start_xy": [round(float(starts[i, 0]), 5), round(float(starts[i, 1]), 5)],
        }
        for k, i in enumerate(selected)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gammas": gammas,
                "selected": rows,
                "figures": {
                    "one_overlay": one_overlay_path,
                    "by_start": by_start_path,
                    "per_gamma": per_gamma_path,
                },
            },
            f,
            indent=2,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--prefix", default="dr05_windows")
    ap.add_argument("--n-starts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=706)
    ap.add_argument("--offdiag", type=float, default=0.5)
    args = ap.parse_args()

    configure_fonts()
    os.makedirs(args.outdir, exist_ok=True)
    env = GS.make_grid()

    shards = find_shards(args.data_dir, args.prefix)
    trajs_by_gamma: dict[float, list[Trajectory]] = {}
    for gamma, path in shards.items():
        trajs_by_gamma[gamma] = decode_saved_trajectories(path, env)
        print(
            f"gamma {gamma:g}: decoded {len(trajs_by_gamma[gamma])} trajectories from {path}",
            flush=True,
        )

    gammas = sorted(trajs_by_gamma)
    n_traj = min(len(v) for v in trajs_by_gamma.values())
    starts = np.stack([trajs_by_gamma[gammas[0]][i].start for i in range(n_traj)], axis=0)
    selected = select_start_indices(starts, args.n_starts, args.seed)

    by_start = os.path.join(args.outdir, "training_data_gamma_overlay_by_start.png")
    per_gamma = os.path.join(args.outdir, "training_data_traj_per_gamma_10starts.png")
    one_overlay = os.path.join(args.outdir, "training_data_one_plot_gamma_overlay.png")
    plot_one_overlay(env, trajs_by_gamma, selected, args.offdiag, one_overlay)
    wrote_by_start = wrote_per_gamma = False
    if len(selected) == 10:
        plot_by_start(env, trajs_by_gamma, selected, args.offdiag, by_start)
        plot_per_gamma(env, trajs_by_gamma, selected, args.offdiag, per_gamma)
        wrote_by_start = wrote_per_gamma = True

    summary = os.path.join(args.outdir, "training_data_selected_starts.json")
    write_selection_summary(
        summary,
        selected,
        starts,
        gammas,
        by_start if wrote_by_start else None,
        per_gamma if wrote_per_gamma else None,
        one_overlay,
    )

    print("selected starts:", selected.tolist(), flush=True)
    print("saved", one_overlay, flush=True)
    if wrote_by_start:
        print("saved", by_start, flush=True)
    if wrote_per_gamma:
        print("saved", per_gamma, flush=True)
    print("saved", summary, flush=True)


if __name__ == "__main__":
    main()
