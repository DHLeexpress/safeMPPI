"""AFE2 expansion video (user spec 2026-07-16b): for every round, ALL SEVEN gamma panels.

Colors (fixed by spec):
  gray        all K=64 generated plans (thinned over steps for legibility; count stated)
  orange      B SOCP queries whose verifier errored (socp_error: updates nothing)
  green       SOCP-positive queried plans
  red         rejected queried plans
  blue/thick  cost-selected plan (argmax progress) + the executed first-action path
  X           NO_VERIFIED_POSITIVE termination point
  text        positive count, min SOCP margin, raw untilted validity (audit), termination timestep

Usage: python video_afe2.py --run results/afe2/afe_s910 --out paper_results/afe2_afe.mp4
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _paths  # noqa: F401
import grid_scene as GS
import grid_expand_hardtail as HT

GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def draw_scene(ax, env, goal, x0):
    for o in env.obstacles.detach().cpu().numpy():
        ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="0.82", zorder=1))
    ax.plot([x0[0]], [x0[1]], "ks", ms=5, zorder=6)
    ax.plot([goal[0]], [goal[1]], "*", color="gold", mec="k", ms=13, zorder=6)
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])


def render_round(db, out_png, arm, step_thin=3):
    env = HT._apply_wall_plugs(GS.make_grid(), 8)
    goal = np.asarray(db["goal"], float)
    x0 = np.asarray(db["x0"], float)
    audit = db.get("audit") or {}
    vg = audit.get("V_gamma", {})
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6))
    by_g = {g: [] for g in GAMMAS}
    for v in db["viz"]:
        by_g[round(float(v["gamma"]), 2)].append(v)
    ep_by_g = {round(float(e["gamma"]), 2): e for e in db["eps"]}
    for ax, g in zip(axes.flat[:7], GAMMAS):
        draw_scene(ax, env, goal, x0)
        steps = by_g[g]
        ep = ep_by_g.get(g)
        npos_tot, min_marg = 0, np.inf
        for si, v in enumerate(steps):
            segs = np.asarray(v["segsK"], np.float32)
            if si % step_thin == 0:                    # gray: the K generated plans (thinned steps)
                for k in range(0, segs.shape[0], 2):
                    ax.plot(segs[k, :, 0], segs[k, :, 1], "-", color="0.55", lw=0.35,
                            alpha=0.10, zorder=2)
            for j, y in zip(v["drawn"], v["y"]):       # queried B: green / red / orange(err)
                c = {1: (0.10, 0.55, 0.15, 0.45), 0: (0.85, 0.12, 0.10, 0.40),
                     -1: (1.00, 0.55, 0.00, 0.60)}[int(y)]
                ax.plot(segs[j, :, 0], segs[j, :, 1], "-", color=c, lw=0.8, zorder=3)
            if v["sel"] >= 0 and si % step_thin == 0:  # blue: the cost-selected plan
                ax.plot(segs[v["sel"], :, 0], segs[v["sel"], :, 1], "-", color="#1155cc",
                        lw=1.1, alpha=0.8, zorder=4)
            npos_tot += int(np.sum(np.asarray(v["y"]) == 1))
            if np.isfinite(v.get("min_margin", np.nan)):
                min_marg = min(min_marg, float(v["min_margin"]))
        if ep is not None:                             # blue/thick: the executed path
            p = np.asarray(ep["path"], float)
            ax.plot(p[:, 0], p[:, 1], "-", color="#0b3d91", lw=2.4, zorder=5)
            if ep["status"] == "nvp":
                ax.plot([p[-1, 0]], [p[-1, 1]], "x", color="k", ms=13, mew=3.2, zorder=7)
            elif ep["status"] == "reached":
                ax.plot([p[-1, 0]], [p[-1, 1]], "*", color="#0b3d91", mec="k", ms=12, zorder=7)
        stat = ep["status"] if ep is not None else "-"
        tt = ep.get("term_t") if ep is not None else None
        ax.set_title(f"γ={g}", fontsize=13)
        ax.text(0.02, 0.98,
                f"pos {npos_tot}\nmin m {min_marg:.3f}\n"
                f"V̂ raw {float(vg.get(str(g), np.nan)):.2f}\n{stat}"
                + (f" t={tt}" if tt is not None else ""),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="0.6", alpha=0.85))
    axL = axes.flat[7]
    axL.axis("off")
    axL.legend(handles=[
        Line2D([], [], color="0.55", lw=1.5, alpha=0.6, label="K=64 generated plans"),
        Line2D([], [], color=(0.10, 0.55, 0.15), lw=2, label="queried: SOCP-positive"),
        Line2D([], [], color=(0.85, 0.12, 0.10), lw=2, label="queried: rejected"),
        Line2D([], [], color=(1.0, 0.55, 0.0), lw=2, label="queried: socp_error (updates nothing)"),
        Line2D([], [], color="#1155cc", lw=1.6, label="cost-selected plan (max progress)"),
        Line2D([], [], color="#0b3d91", lw=2.6, label="executed path"),
        Line2D([], [], color="k", marker="x", ls="", mew=3, ms=10,
               label="NO_VERIFIED_POSITIVE (terminate; no expert, no fallback)"),
    ], loc="center", fontsize=10.5, frameon=False)
    axL.text(0.5, 0.05, f"arm: {arm} — round {int(db['round'])}\n"
             f"evolving φ_s^(n); A rebuilt each round; σ used once (acquisition)",
             ha="center", fontsize=10, color="#333333", transform=axL.transAxes)
    fig.suptitle(f"AFE2 expert-free verified expansion — {arm}, round {int(db['round'])}",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=1)
    args = ap.parse_args()
    arm = os.path.basename(args.run.rstrip("/"))
    dbs = sorted(glob.glob(os.path.join(args.run, "viz_db", "round*.pt")),
                 key=lambda p: int(re.findall(r"round(\d+)\.pt", p)[0]))
    tmp = tempfile.mkdtemp(prefix="afe2_vid_")
    try:
        for k, p in enumerate(dbs):
            db = torch.load(p, map_location="cpu", weights_only=False)
            render_round(db, os.path.join(tmp, f"frame_{k:03d}.png"), arm)
            print(f"rendered round {db['round']}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", os.path.join(tmp, "frame_%03d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", args.out],
                       check=True)
        print("saved", args.out, f"({len(dbs)} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
