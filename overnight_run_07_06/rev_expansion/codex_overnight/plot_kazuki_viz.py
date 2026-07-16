#!/usr/bin/env python3
"""Kazuki generate/refine success/failure diagnostic with guidance arrows."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

import grid_scene as GS


def parse_record(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("record must be LABEL=PATH.pt")
    label, path = text.split("=", 1)
    return label, Path(path)


def scene(ax, env):
    obs = env.obstacles.detach().cpu().numpy()
    for o in obs:
        ax.add_patch(Circle(o[:2], o[2], color="#777777", alpha=.8, lw=0))
    ax.plot(0, 0, "s", color="black", ms=6, zorder=8)
    ax.plot(5, 5, "*", color="#F0C808", mec="black", ms=14, zorder=8)
    ax.set_xlim(-.25, 5.25); ax.set_ylim(-.25, 5.25); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.12)


def panel(ax, label, data, env):
    scene(ax, env)
    path = np.asarray(data["path"])
    recs = data["rec"]
    # A mid-episode snapshot shows local refinement away from the trivial origin/goal endpoints.
    idx = min(max(int(.45 * len(recs)), 0), len(recs) - 1)
    rec = recs[idx]
    cand = np.asarray(rec["cand"]); refined = np.asarray(rec["refined"]); best = np.asarray(rec["best"])
    for p in cand[::max(1, len(cand) // 40)]:
        ax.plot(p[:, 0], p[:, 1], color="#56B4E9", alpha=.10, lw=.7)
    for p in refined:
        ax.plot(p[:, 0], p[:, 1], color="#E69F00", alpha=.34, lw=1.0)
    st = np.asarray(rec["state"])
    ax.plot(np.r_[st[0], best[:, 0]], np.r_[st[1], best[:, 1]], color="#D55E00", lw=2.5,
            label="selected refined window")
    ax.plot(path[:, 0], path[:, 1], color="black", lw=2.1, label="executed trajectory")
    guide = np.asarray(rec.get("guidance", np.zeros((1, 2))))
    gm = guide.mean(axis=0)
    gn = np.linalg.norm(gm)
    if gn > 1e-9:
        vec = .45 * gm / gn
        ax.arrow(st[0], st[1], vec[0], vec[1], width=.015, head_width=.12, color="#CC79A7",
                 length_includes_head=True, zorder=10, label=r"mean reward guidance $\nabla_u R$")
    outcome = "success" if data.get("reached") else ("collision" if data.get("collided") else "timeout")
    cfg = f"w_safe={data.get('w_safe')}  coll_w={data.get('coll_w')}  goal_w={data.get('goal_w')}"
    ax.set_title(f"{label}: {outcome}, {len(path)-1} steps\n{cfg}\nsnapshot t={idx}")
    ax.legend(loc="lower right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="append", type=parse_record, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/kazuki_success_failure.png"))
    args = ap.parse_args()
    env = GS.make_grid()
    fig, axs = plt.subplots(1, len(args.record), figsize=(7 * len(args.record), 6.5), squeeze=False,
                            constrained_layout=True)
    for ax, (label, path) in zip(axs[0], args.record):
        data = torch.load(path, map_location="cpu", weights_only=False)
        panel(ax, label, data, env)
    fig.suptitle("Guidance baseline — generated candidates, MPPI refinement, and local reward direction",
                 fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
