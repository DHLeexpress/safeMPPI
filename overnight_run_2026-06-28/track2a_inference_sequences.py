"""Track 2a inference: show the FM's GENERATED FULL SEQUENCES = the ODE sampling trajectories.

Each generated sample is produced by integrating the flow ODE from a Gaussian prior x_0 ~ N(0,I) to a
checkerboard point x_1 via Euler steps; the "full sequence" is that trajectory [x_0, x_{1/T}, ..., x_1].
We compare the SCARCE model (mis-specified pretraining = the left/round-0 figure) with the EXPANDED model
(after Safe Flow Expansion). Run track2a first so the checkpoints exist:

  python overnight_run_2026-06-28/track2a_chessboard_actflow.py --rounds 130 --inner 150 --temp 2.0
  python overnight_run_2026-06-28/track2a_inference_sequences.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track2a_chessboard_actflow import FM2D, valid, LO, HI, CELLW

FIG = os.path.join(HERE, "figures")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def sample_traj(model, n, nfe, temp=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (temp * torch.randn(n, 2, generator=g)).to(DEV)
    traj = [x.clone()]
    for i in range(nfe):
        t = torch.full((n,), i / nfe, device=DEV)
        x = x + (1.0 / nfe) * model.forward(x, t)
        traj.append(x.clone())
    x = torch.nan_to_num(x, nan=0.0).clamp(-4.2, 4.2); traj[-1] = x
    return torch.stack(traj, 1).cpu().numpy()          # [n, nfe+1, 2]


def load(name):
    m = FM2D().to(DEV); m.load_state_dict(torch.load(os.path.join(HERE, name), map_location=DEV)); m.eval()
    return m


def board(ax):
    for ii in range(3):
        for jj in range(3):
            if (ii + jj) % 2 == 0:
                ax.add_patch(plt.Rectangle((LO + ii * CELLW, LO + jj * CELLW), CELLW, CELLW,
                                           facecolor="0.86", edgecolor="0.6", zorder=0))
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def panel(ax, traj, title):
    board(ax)
    end = traj[:, -1, :]; v = valid(end)
    for k in range(traj.shape[0]):                     # the full generation sequences (ODE paths)
        c = "#1a9850" if v[k] else "#d62728"
        ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color=c, lw=0.4, alpha=0.18, zorder=2)
    ax.scatter(traj[:, 0, 0], traj[:, 0, 1], s=4, c="0.4", alpha=0.5, zorder=3, label="x₀ ~ N(0,I)")
    ax.scatter(end[v, 0], end[v, 1], s=6, c="#1a9850", zorder=4, label="x₁ valid")
    ax.scatter(end[~v, 0], end[~v, 1], s=6, c="#d62728", alpha=0.6, zorder=4, label="x₁ invalid")
    ax.set_title(f"{title}\nvalidity {100*v.mean():.0f}%", fontsize=10)
    ax.legend(fontsize=6, loc="lower left", framealpha=0.7)


def main():
    nfe = 40
    scarce = load("track2a_scarce.pt"); expanded = load("track2a_expanded.pt")
    ts = sample_traj(scarce, 220, nfe, temp=1.0); te = sample_traj(expanded, 220, nfe, temp=1.0)

    # static: full generated sequences, scarce vs expanded
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
    panel(a1, ts, "SCARCE model (mis-specified pretrain)\ngenerated sequences collapse to one cell")
    panel(a2, te, "EXPANDED model (after Safe Flow Expansion)\nsequences fan out to all 5 valid cells")
    fig.suptitle("Generated FULL sequences = flow ODE trajectories  x₀~N(0,I) → x₁ (checkerboard)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); p = os.path.join(FIG, "track2a_generated_sequences.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # GIF: animate the integration tau:0->1 (samples flowing from noise to the board), scarce vs expanded
    figg, (g1, g2) = plt.subplots(1, 2, figsize=(11, 5.3))

    def draw(f):
        for ax, traj, ttl in ((g1, ts, "SCARCE"), (g2, te, "EXPANDED")):
            ax.clear(); board(ax)
            pts = traj[:, f, :]; v = valid(traj[:, -1, :])
            ax.plot(traj[:, :f + 1, 0].T, traj[:, :f + 1, 1].T, "-", color="0.5", lw=0.3, alpha=0.12, zorder=2)
            ax.scatter(pts[v, 0], pts[v, 1], s=7, c="#1a9850", zorder=4)
            ax.scatter(pts[~v, 0], pts[~v, 1], s=7, c="#d62728", alpha=0.6, zorder=4)
            ax.set_title(f"{ttl}  τ={f/nfe:.2f}", fontsize=10)
        figg.suptitle("Flow ODE inference: x₀~N(0,I) → x₁  (generated full sequences)", fontsize=11)
        return []
    anim = FuncAnimation(figg, draw, frames=nfe + 1, interval=130)
    pg = os.path.join(FIG, "track2a_generated_sequences.gif")
    anim.save(pg, writer=PillowWriter(fps=8), dpi=95); plt.close(figg); print("saved", pg)
    print(f"scarce validity={100*valid(ts[:,-1,:]).mean():.0f}%  expanded validity={100*valid(te[:,-1,:]).mean():.0f}%")


if __name__ == "__main__":
    main()
