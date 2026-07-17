"""TRUE-evaluation figures (user 2026-07-16c). Curated gammas {0.1, 0.3, 0.5, 1.0} — the classic
MPC safety-vs-feasibility axis (stricter SOCP at low gamma <=> lower validity).

All metrics come from RAW stored paths with ONE uniform code path (no per-source row files):
  SR   = final point within reach of the goal
  CR   = min obstacle clearance < 0 anywhere OR left the task box (actually collided/dead)
  clr  = mean over SUCCESSES of the per-trajectory min obstacle clearance
  time = mean steps*dt over successes
  V2   = fraction of trajectories passing validity2 (taskspace AND goal-approach AND SOCP at gamma)

Figure A (gallery, rollouts_v5 style, 4 rows x 4 gammas):每 panel shows M=10 RATIO-MATCHED random
rollouts (round(10*SR) random successes + the rest random failures — no curation) out of M=100:
  row 1 Expert (SafeMPPI demos)        row 2 Purely pretrained (bare policy)
  row 3 After 10 AFE2 rounds (bare)    row 4 CFM-MPPI baseline (same pretrained model)
Figure B: SR / clearance / time / validity2 vs AFE2 round (0..10), one line per gamma, M=100 each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401
import grid_scene as GS
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_expand_hardtail as HT

GSEL = [0.1, 0.3, 0.5, 1.0]
PLA = plt.get_cmap("plasma")
GCOL = {0.1: PLA(0.08), 0.3: PLA(0.38), 0.5: PLA(0.58), 1.0: PLA(0.85)}
GOAL = np.array([4.7, 4.7])
REACH = 0.15

_ENV = None


def env():
    global _ENV
    if _ENV is None:
        _ENV = HT._apply_wall_plugs(GS.make_grid(), 8)
        _ENV.goal = __import__("torch").tensor([4.7, 4.7], dtype=_ENV.goal.dtype)
        GM2.GOAL_XY = GOAL.copy()
    return _ENV


def traj_metrics(p, g):
    """One trajectory -> success/collision/clearance/steps + validity2 AND its decomposition
    (v2_safe = taskspace AND SOCP@gamma only — the SAFETY part; approach separately)."""
    e = env()
    obs = e.obstacles.detach().cpu().numpy()
    rr = float(e.r_robot)
    p = np.asarray(p, float)
    d = np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
    clr = float(d.min())
    oob = bool((p < -GM.EPS_TASK).any() or (p > GM.GRID_M + GM.EPS_TASK).any())
    success = bool(np.linalg.norm(p[-1] - GOAL) < REACH)
    collided = bool(clr < 0.0 or oob)
    v2, st = GM2.traj_breakdown(p, e, float(g))
    return dict(success=success and not collided, collided=collided, clr=clr,
                steps=len(p) - 1, valid2=bool(v2),
                v2_safe=bool(st["taskspace"] and st["socp"]), appr=bool(st["approach"]))


def source_metrics(npz_path, g, cache):
    key = os.path.relpath(npz_path, _HERE) + f"|g{g}"
    if key in cache:
        return cache[key]
    z = np.load(npz_path, allow_pickle=True)
    ms = [traj_metrics(p, g) for p in z["paths"]]
    suc = [m for m in ms if m["success"]]
    out = dict(M=len(ms),
               SR=float(np.mean([m["success"] for m in ms])),
               CR=float(np.mean([m["collided"] for m in ms])),
               clr=float(np.mean([m["clr"] for m in suc])) if suc else float("nan"),
               time=float(np.mean([m["steps"] for m in suc])) * 0.1 if suc else float("nan"),
               V2=float(np.mean([m["valid2"] for m in ms])),
               V2S=float(np.mean([m["v2_safe"] for m in ms])),
               APPR=float(np.mean([m["appr"] for m in ms])),
               success_mask=[bool(m["success"]) for m in ms])
    cache[key] = out
    return out


def draw_panel(ax, npz_path, g, met, n_show=10, seed=0, title=None, row_label=None):
    e = env()
    for o in e.obstacles.detach().cpu().numpy():
        ax.add_patch(plt.Circle((o[0], o[1]), o[2], color="#cccccc", zorder=1))
    ax.plot(0.3, 0.3, "ks", ms=5, zorder=6)
    ax.plot(*GOAL, marker="*", c="gold", mec="k", ms=12, ls="", zorder=6)
    z = np.load(npz_path, allow_pickle=True)
    paths = list(z["paths"])
    mask = np.asarray(met["success_mask"], bool)
    rng = np.random.default_rng(seed)
    k = int(round(n_show * met["SR"]))                      # RATIO-MATCHED random selection
    si = np.where(mask)[0]
    fi = np.where(~mask)[0]
    pick = list(rng.choice(si, min(k, len(si)), replace=False)) + \
           list(rng.choice(fi, min(n_show - k, len(fi)), replace=False))
    for i in pick:
        p = np.asarray(paths[i], float)
        ok = mask[i]
        ax.plot(p[:, 0], p[:, 1], "-", color=GCOL[g], lw=1.2, alpha=0.85, zorder=3)
        ax.plot(p[::4, 0], p[::4, 1], ".", color="k", ms=1.3, alpha=0.5, zorder=4)
        if not ok:
            ax.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)
    ax.set_xlim(-0.35, 5.35); ax.set_ylim(-0.35, 5.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=14)
    if row_label:
        ax.set_ylabel(row_label, fontsize=13)
    tstr = "-" if not np.isfinite(met["time"]) else f"{met['time']:.1f}s"
    cstr = "-" if not np.isfinite(met["clr"]) else f"{met['clr']:.2f}"
    ax.text(0.02, 0.02, f"SR {met['SR']:.2f}  CR {met['CR']:.2f}\n"
                        f"clr {cstr}  t {tstr}\nV2 {met['V2']:.2f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(fc="white", ec="0.6", alpha=0.88))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afe-dir", default="results/true_eval/afe")
    ap.add_argument("--kaz-dir", default="results/true_eval/kazuki")
    ap.add_argument("--exp-dir", default="results/expert_g47")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--out-prefix", default="paper_results/true_eval")
    args = ap.parse_args()
    cache_f = os.path.join(_HERE, args.afe_dir, "metrics_cache.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}

    def P(rel):
        return os.path.join(_HERE, rel)

    rows = [
        ("Expert (SafeMPPI)", lambda g: P(f"{args.exp_dir}/paths_g{g}.npz")),
        ("Pretrained (bare)", lambda g: P(f"{args.afe_dir}/paths_r0_g{g}.npz")),
        (f"AFE2 round {args.rounds} (bare)", lambda g: P(f"{args.afe_dir}/paths_r{args.rounds}_g{g}.npz")),
        ("CFM-MPPI (same pretrained)", lambda g: P(f"{args.kaz_dir}/paths_g{g}.npz")),
    ]
    # -------- Figure A: gallery
    fig, axes = plt.subplots(4, 4, figsize=(16.4, 16.8))
    for ri, (rlab, pathf) in enumerate(rows):
        for ci, g in enumerate(GSEL):
            f = pathf(g)
            ax = axes[ri, ci]
            if not os.path.exists(f):
                ax.text(0.5, 0.5, "pending", ha="center", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            met = source_metrics(f, g, cache)
            draw_panel(ax, f, g, met, seed=17 * ri + ci,
                       title=(f"γ = {g}" if ri == 0 else None),
                       row_label=(rlab if ci == 0 else None))
    fig.suptitle("TRUE evaluation — M=100 random rollouts per cell; each panel shows 10 "
                 "ratio-matched random trajectories (no curation)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}_gallery.{ext}", dpi=130)
    plt.close(fig)

    # -------- Figure B: per-round metric curves (M=100 each point)
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.6))
    keys = [("SR", "success rate SR"), ("clr", "min clearance on successes [m]"),
            ("time", "time to success [s]"),
            ("V2", "validity: SAFETY (taskspace∧SOCP@γ, solid)\nvs full V₂ (∧approach, dashed)")]
    for g in GSEL:
        R, M = [], {k: [] for k in ("SR", "clr", "time", "V2", "V2S")}
        for n in range(0, args.rounds + 1):
            f = P(f"{args.afe_dir}/paths_r{n}_g{g}.npz")
            if not os.path.exists(f):
                continue
            met = source_metrics(f, g, cache)
            R.append(n)
            for k in M:
                M[k].append(met[k])
        for ax, (k, _) in zip(axs[:3], keys[:3]):
            ax.plot(R, M[k], "-o", color=GCOL[g], lw=1.8, ms=4, label=f"γ={g}")
        axs[3].plot(R, M["V2S"], "-o", color=GCOL[g], lw=1.9, ms=4, label=f"γ={g}")
        axs[3].plot(R, M["V2"], "--", color=GCOL[g], lw=1.1, alpha=0.7)
    for ax, (k, lab) in zip(axs, keys):
        ax.set_xlabel("AFE2 round"); ax.set_title(lab); ax.grid(alpha=.3)
        if k in ("SR", "V2"):
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
    fig.suptitle("Bare-policy metrics per AFE2 round — M=100 random rollouts per (round, γ)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}_curves.{ext}", dpi=140)
    with open(cache_f, "w") as fh:
        json.dump(cache, fh)
    print(f"wrote {args.out_prefix}_gallery.png/.pdf and _curves.png/.pdf "
          f"({len(cache)} cached cells)")


if __name__ == "__main__":
    main()
