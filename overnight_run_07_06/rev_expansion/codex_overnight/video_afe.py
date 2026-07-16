"""Expansion-mechanism video for the AFE-minimal runs (user 2026-07-16: "add visualization (there is
no curriculum learning) ... how now flow expansion is happening, and visualize what samples are used
to train the model").

Per round (one frame each):
  A  the acquisition mechanism on the scene: every drawn planned window of the round, colored by its
     FULL-verifier label (green = certified -> D+, red = rejected -> only A_n); query-start dots
     colored by sigma at query time (viridis); executed episode paths (black), fallback steps
     (orange dots), reached (star) / dead (x).
  B  exactly which samples trained the model this round: context positions of the D+ rows drawn by
     the uniform replay (copper = round of origin -> shows the cumulative-replay memory), dot size
     = draw count.
  C  validity tracking up to this round: V_adverse pooled + gamma 0.1 (untilted audit), closed-loop
     SR, fallback rate.

Usage: python video_afe.py --run results/afe/A_s910 --out paper_results/afe_A_expansion.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
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


def draw_scene(ax, env, goal, x0):
    obs = env.obstacles.detach().cpu().numpy()
    for o in obs:
        ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="0.75", zorder=1))
    ax.plot([x0[0]], [x0[1]], "o", color="tab:blue", ms=7, zorder=6)
    ax.plot([goal[0]], [goal[1]], "*", color="gold", mec="black", ms=16, zorder=6)
    ax.set_xlim(-0.15, 5.15)
    ax.set_ylim(-0.15, 5.15)
    ax.set_aspect("equal")


def render_round(db, probe, out_png, max_segs=2200):
    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.28, wspace=0.18)
    n = int(db["round"])
    goal = np.asarray(db["goal"], float)
    x0 = np.asarray(db["x0"], float)
    env = HT._apply_wall_plugs(GS.make_grid(), 8)

    # ---- A: acquisition mechanism
    axA = fig.add_subplot(gs[:, 0])
    draw_scene(axA, env, goal, x0)
    seg = np.asarray(db["q_seg"], np.float32)
    y = np.asarray(db["q_y"], int)
    sig = np.asarray(db["q_sigma"], np.float32)
    stq = np.asarray(db["q_state"], np.float32)
    nq = len(y)
    keep = np.arange(nq)
    if nq > max_segs:
        keep = np.random.default_rng(0).choice(nq, max_segs, replace=False)
    for i in keep:
        c = (0.15, 0.55, 0.2, 0.10) if y[i] else (0.85, 0.15, 0.1, 0.10)
        axA.plot(seg[i, :, 0], seg[i, :, 1], "-", color=c, lw=0.7, zorder=2)
    sc = axA.scatter(stq[keep, 0], stq[keep, 1], c=sig[keep], cmap="viridis", vmin=0, vmax=1,
                     s=5, zorder=3, linewidths=0)
    plt.colorbar(sc, ax=axA, fraction=0.035, pad=0.01).set_label("σ at query (frozen φ⁰, A_n)")
    for p, fb, rch, dd in zip(db["ep_paths"], db["ep_fb"], db["ep_reached"], db["ep_dead"]):
        p = np.asarray(p, float)
        axA.plot(p[:, 0], p[:, 1], "-", color="black", lw=1.4, alpha=0.85, zorder=4)
        fb = np.asarray(fb, bool)
        if fb.any():
            axA.plot(p[1:][fb[:len(p) - 1], 0], p[1:][fb[:len(p) - 1], 1], ".",
                     color="tab:orange", ms=5, zorder=5)
        axA.plot([p[-1, 0]], [p[-1, 1]], "*" if rch else "x",
                 color="tab:green" if rch else "tab:red", ms=9, zorder=6)
    acc = y.mean() if nq else float("nan")
    axA.set_title(f"round {n} — verified-query acquisition (π∝e^{{(σ−σmax)/β}}, B/step, "
                  f"FULL SOCP before execution)\n{nq} queries, acc {acc:.2f} | "
                  f"D {int(db['n_D'])} D⁺ {int(db['n_Dpos'])} | shown {len(keep)}")
    axA.legend(handles=[Line2D([], [], color=(0.15, 0.55, 0.2, 0.6), lw=2, label="certified → D⁺ (trains)"),
                        Line2D([], [], color=(0.85, 0.15, 0.1, 0.6), lw=2, label="rejected → A_n only"),
                        Line2D([], [], color="black", lw=2, label="executed path (shielded)"),
                        Line2D([], [], color="tab:orange", marker=".", ls="", label="SafeMPPI fallback step")],
               fontsize=8, loc="upper left")

    # ---- B: trained-on samples
    axB = fig.add_subplot(gs[0, 1])
    draw_scene(axB, env, goal, x0)
    ts = np.asarray(db["train_state"], np.float32)
    tc = np.asarray(db["train_counts"], np.float64)
    tr = np.asarray(db["train_y_round"], np.float64)
    if len(ts):
        sc = axB.scatter(ts[:, 0], ts[:, 1], c=tr, cmap="copper", s=4 + 7 * tc,
                         vmin=1, vmax=max(n, 2), linewidths=0, zorder=3)
        plt.colorbar(sc, ax=axB, fraction=0.035, pad=0.01).set_label("round of origin")
    ndist = len(ts)
    axB.set_title(f"samples that TRAINED the model this round\nuniform replay over cumulative D⁺: "
                  f"{ndist} distinct rows (size = draw count)")

    # ---- C: validity tracking
    axC = fig.add_subplot(gs[1, 1])
    rounds, va, vg01, srs, fbs = [], [], [], [], []
    for r in probe:
        if r["round"] > n:
            break
        if r.get("V_adverse") is not None:
            rounds.append(r["round"])
            va.append(r["V_adverse"])
            vg01.append((r.get("V_gamma_adverse") or {}).get("0.1", np.nan))
        if r.get("SR") is not None:
            srs.append((r["round"], r["SR"]))
        if r.get("fb_rate") is not None:
            fbs.append((r["round"], r["fb_rate"]))
    if rounds:
        axC.plot(rounds, va, "-o", ms=3, color="tab:red", label="V̂ adverse (pooled)")
        axC.plot(rounds, vg01, "-o", ms=3, color="darkred", alpha=0.7, label="V̂ adverse γ0.1")
    if srs:
        axC.plot(*zip(*srs), "-s", ms=3, color="tab:green", label="closed-loop SR")
    if fbs:
        axC.plot(*zip(*fbs), "-", color="tab:orange", lw=1.2, label="fallback rate")
    axC.set_xlim(0, max(n + 1, 5))
    axC.set_ylim(-0.02, 1.02)
    axC.grid(alpha=0.3)
    axC.legend(fontsize=8, loc="lower right")
    axC.set_xlabel("round")
    axC.set_title("validity of the expansion (untilted audit ≠ query acceptance)")

    fig.text(0.005, 0.008,
             r"$c_t$=(H$_P$ grid, low5, hist)  $U_t\in\mathbb{R}^{10\times2}$ planned window | "
             r"$\sigma_n^2=z^\top A_n^{-1}z$, $z=\phi_s^0(U,c)/\|\cdot\|$ FROZEN, "
             r"$A_n=I+\lambda^{-1}\Sigma zz^\top$ over ALL verified queries | "
             r"draw B=8 w/o repl. $\propto\pi=e^{(\sigma-\sigma_{max})/\beta}$, FULL SOCP BEFORE exec | "
             r"exec $\propto\pi$ among verified-safe, else certified SafeMPPI | "
             r"update: $\min_\theta\,\ell_{CFM}(D_n^+)+\|\theta-\theta_n\|^2/2\eta$ (uniform replay, "
             r"no curriculum)", fontsize=8.2, color="#333333")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="render every k-th round")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()
    probe = [json.loads(l) for l in open(os.path.join(args.run, "probe.jsonl"))]
    dbs = sorted(glob.glob(os.path.join(args.run, "viz_db", "round*.pt")),
                 key=lambda p: int(re.findall(r"round(\d+)\.pt", p)[0]))
    dbs = [p for p in dbs if int(re.findall(r"round(\d+)\.pt", p)[0]) % args.every == 0 or
           p is dbs[0] or p is dbs[-1]]
    tmp = tempfile.mkdtemp(prefix="afe_vid_")
    try:
        for k, p in enumerate(dbs):
            db = torch.load(p, map_location="cpu", weights_only=False)
            render_round(db, probe, os.path.join(tmp, f"frame_{k:04d}.png"))
            if k % 10 == 0:
                print(f"rendered {k + 1}/{len(dbs)}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", os.path.join(tmp, "frame_%04d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # yuv420p needs even dims
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", args.out],
                       check=True)
        print("saved", args.out, f"({len(dbs)} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
