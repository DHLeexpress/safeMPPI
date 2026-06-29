"""Track 2b inference: SIMULATE grid coverage — draw the 7x7 map (3x3 block, start, goal), then overlay the
trajectories the trained conditional FM actually generates (pure-FM receding-horizon rollouts).

Run track2b first so the per-horizon checkpoints exist (track2b_H{4,9,14}.pt), then:
  python overnight_run_2026-06-28/track2b_inference_trajectories.py
"""
from __future__ import annotations
import math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track2b_grid_coverage import (FlowPolicy, cell_of, reached, traj_to_path,
                                    START, GOAL, N, BSIZE, UMAX, DT, NPATHS)
import lattice_paths as LP

FIG = os.path.join(HERE, "figures"); DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def rollout_positions(model, H, B, temp=1.0, nfe=10):
    """Pure-FM receding-horizon rollout; record the full continuous position trajectory per sample."""
    p = np.tile(START, (B, 1)).astype(float)
    paths = [[p[i].copy()] for i in range(B)]; done = np.zeros(B, bool)
    for _ in range(math.ceil(60 / H)):
        ctx = torch.tensor(p / N, dtype=torch.float32, device=DEV)
        U = model.sample(B, ctx, nfe=nfe, temp=temp).clamp(-UMAX, UMAX).cpu().numpy()
        for h in range(H):
            p = p + DT * U[:, h]
            for i in range(B):
                if not done[i]:
                    paths[i].append(p[i].copy())
        for i in range(B):
            if not done[i] and reached(p[i]):
                done[i] = True
        if done.all():
            break
    sigs = [traj_to_path([cell_of(q) for q in paths[i]]) for i in range(B)]
    return paths, sigs


def draw_map(ax):
    for (i, j) in LP.block_cells(N, BSIZE):
        ax.add_patch(Rectangle((i, j), 1, 1, facecolor="#444", edgecolor="#222", zorder=3))
    for k in range(N + 1):
        ax.axvline(k, color="0.85", lw=0.6, zorder=1); ax.axhline(k, color="0.85", lw=0.6, zorder=1)
    ax.add_patch(Circle(START, 0.22, facecolor="#1a9850", edgecolor="k", zorder=6))
    ax.scatter([GOAL[0]], [GOAL[1]], marker="*", s=240, c="#d62728", edgecolor="k", zorder=6)
    ax.text(START[0], START[1] - 0.45, "start", ha="center", fontsize=7)
    ax.text(GOAL[0], GOAL[1] + 0.3, "goal", ha="center", fontsize=7)
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def load(H):
    m = FlowPolicy(T=H, ctx_dim=2, width=128, depth=3, u_max=UMAX).to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE, f"track2b_H{H}.pt"), map_location=DEV)); m.eval()
    return m


def panel(ax, H, batches=10, B=400, temp=1.1):
    """Accumulate rollouts; keep ONE representative trajectory per DISTINCT safe path the FM generates,
    so the number of drawn green routes = the FM's coverage of the 74-path space."""
    model = load(H)
    rep = {}; n_roll = 0; n_bad = 0; bad_eg = None
    for _ in range(batches):
        paths, sigs = rollout_positions(model, H, B, temp=temp); n_roll += B
        for i, sig in enumerate(sigs):
            if sig is not None and sig not in rep:
                rep[sig] = np.array(paths[i])
            elif sig is None:
                n_bad += 1
                if bad_eg is None:
                    bad_eg = np.array(paths[i])
    draw_map(ax)
    if bad_eg is not None:                                   # one rejected example (non-monotone / hits block)
        ax.plot(bad_eg[:, 0], bad_eg[:, 1], "-", color="#d62728", lw=1.0, alpha=0.5, zorder=4, label="rejected")
    for arr in rep.values():
        ax.plot(arr[:, 0], arr[:, 1], "-", color="#1a9850", lw=1.0, alpha=0.30, zorder=5)
    ax.set_title(f"H={H}: FM covers {len(rep)}/{NPATHS} distinct safe paths ({100*len(rep)/NPATHS:.0f}%)\n"
                 f"{n_roll} inference rollouts", fontsize=9)


def main():
    horizons = [H for H in (4, 9, 14) if os.path.exists(os.path.join(HERE, f"track2b_H{H}.pt"))]
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.6 * len(horizons), 4.8))
    if len(horizons) == 1:
        axes = [axes]
    for ax, H in zip(axes, horizons):
        panel(ax, H)
    fig.suptitle("Track 2b — simulated grid coverage: inferred FM trajectories on the 7×7 map (3×3 block)\n"
                 "green = safe monotone path · red = rejected (non-monotone / hits block)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "track2b_inference_trajectories.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
