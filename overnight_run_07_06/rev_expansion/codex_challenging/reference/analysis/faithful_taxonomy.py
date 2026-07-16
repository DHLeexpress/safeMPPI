"""FAITHFUL failure taxonomy (user 2026-07-13): SR<100 with low CR is NOT 'mostly safe' — the missing
mass is ILL-CONDITIONED episodes (out-of-bounds grazing / timeouts) that the CR metric hides. This tool
classifies every deployed episode RAW, no seed selection, no filtering:
    reach      : ended within `reach` of the goal, no obstacle collision, in-bounds
    collision  : hit an obstacle at any step (RAW CR)
    oob        : left the [0,GRID_M]^2 workspace at any step (ill-conditioned)
    timeout    : none of the above (wandered / stalled within T)
Deploys a ckpt fresh (faithful temp=1, NFE 8) OR reads a saved eval dir's paths.

  python analysis/faithful_taxonomy.py --ckpt <pretrained.pt> --tag pretrained --M 50 --out-dir grand_final_reports_rev
  python analysis/faithful_taxonomy.py --eval-dir results/p2/eval_greedy_it45_m100 --tag it45 --out-dir grand_final_reports_rev
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import grid_scene as GS      # noqa: E402
import grid_metrics as GM    # noqa: E402

GAMMAS = [0.1, 0.5, 1.0]
CLR = {"reach": "#009988", "collision": "#cc3311", "oob": "#ee7733", "timeout": "#888888"}


def classify(p, obs, goal, rr, reach):
    p = np.asarray(p, float)[:, :2]
    d = np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - rr
    coll = bool((d.min(axis=1) < 0.0).any())
    oob = bool((p < -GM.EPS_TASK).any() or (p > GM.GRID_M + GM.EPS_TASK).any())
    reached = bool(np.linalg.norm(p[-1] - goal) < reach and not coll)
    if coll:
        return "collision"
    if reached:
        return "reach"
    if oob:
        return "oob"
    return "timeout"


def get_paths(args, g):
    if args.eval_dir:
        f = os.path.join(P2, args.eval_dir, f"paths_g{g}.npz")
        if not os.path.exists(f):
            f = os.path.join(P2, args.eval_dir, f"paths_g{float(g)}.npz")
        z = np.load(f, allow_pickle=True)
        return [np.asarray(p, float) for p in z["paths"]]
    import grid_hp_expt as HP
    import sr_cr_eval as SR
    pol, _ = HP.load_hp(args.ckpt, device=args.device)
    env = GS.make_grid()
    if getattr(args, "wall_plugs", 0):
        import grid_expand_hardtail as HT
        HT._apply_wall_plugs(env, args.wall_plugs)
    if getattr(args, "start_eps", 0.0) > 0.0:
        import torch as _t
        env.x0 = _t.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    _, _, pbg = SR.eval_policy(pol, env, gammas=[float(g)], M=args.M, T_max=args.T, reach=args.reach,
                               temp=1.0, device=args.device, seed0=args.seed0, keep_paths=args.M,
                               log=lambda *a, **k: None)
    return pbg[float(g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--eval-dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--gammas", nargs="+", type=float, default=GAMMAS)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--reach", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wall-plugs", type=int, default=0, choices=[0, 2, 4, 8],
                    help="evaluate on the walled scene (collision detection includes the plugs)")
    ap.add_argument("--start-eps", type=float, default=0.0,
                    help="deploy from (eps,eps) — must match the trainer's --start-eps for the walled scene")
    ap.add_argument("--out-dir", default=os.path.join(P2, "grand_final_reports_rev"))
    args = ap.parse_args()
    assert args.ckpt or args.eval_dir, "need --ckpt or --eval-dir"

    env = GS.make_grid()
    if args.wall_plugs:
        import grid_expand_hardtail as HT
        HT._apply_wall_plugs(env, args.wall_plugs)
    if args.start_eps > 0.0:
        import torch as _t
        env.x0 = _t.tensor([args.start_eps, args.start_eps, 0.0, 0.0], dtype=env.x0.dtype)
    obs, goal, rr = env.obstacles.numpy(), env.goal.numpy(), float(env.r_robot)
    tax = {}
    ill = {}
    for g in args.gammas:
        paths = get_paths(args, g)
        cats = [classify(p, obs, goal, rr, args.reach) for p in paths]
        tax[g] = {k: cats.count(k) for k in CLR}
        tax[g]["M"] = len(paths)
        ill[g] = [(np.asarray(p, float)[:, :2], c) for p, c in zip(paths, cats)
                  if c in ("oob", "timeout")]

    os.makedirs(args.out_dir, exist_ok=True)
    # ---- figure: raw stacked bars + ill-conditioned trajectory panels ----
    ng = len(args.gammas)
    fig = plt.figure(figsize=(4.2 * (ng + 1), 4.6))
    axb = fig.add_subplot(1, ng + 1, 1)
    xs = np.arange(ng)
    bot = np.zeros(ng)
    for cat in ("reach", "collision", "oob", "timeout"):
        vals = np.array([100.0 * tax[g][cat] / tax[g]["M"] for g in args.gammas])
        axb.bar(xs, vals, bottom=bot, color=CLR[cat], label=cat)
        bot += vals
    axb.set_xticks(xs); axb.set_xticklabels([f"$\\gamma$={g}" for g in args.gammas])
    axb.set_ylabel(f"% of episodes (RAW, M={tax[args.gammas[0]]['M']})")
    axb.set_title(f"{args.tag}: faithful outcome split")
    axb.legend(fontsize=9, loc="lower center", ncol=2)
    for j, g in enumerate(args.gammas):
        ax = fig.add_subplot(1, ng + 1, j + 2)
        for o in obs:
            ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
        ax.plot(0, 0, "ks", ms=5); ax.plot(5, 5, "*", c="gold", mec="k", ms=12)
        for p, c in ill[g][:25]:
            ax.plot(p[:, 0], p[:, 1], color=CLR[c], lw=1.0, alpha=0.7, zorder=3)
            ax.plot(p[-1, 0], p[-1, 1], "x", color=CLR[c], ms=6, mew=1.6, zorder=5)
        ax.axhline(0, color="k", lw=0.5, ls=":"); ax.axhline(5, color="k", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5, ls=":"); ax.axvline(5, color="k", lw=0.5, ls=":")
        ax.set_xlim(-1.2, 6.2); ax.set_ylim(-1.2, 6.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        no, nt = tax[g]["oob"], tax[g]["timeout"]
        ax.set_title(f"$\\gamma$={g}: {no} OOB (orange) + {nt} timeout (grey)")
    fig.suptitle(f"FAITHFUL taxonomy — {args.tag}: low CR hides out-of-bounds ill-conditioning "
                 "(dotted = workspace [0,5])", fontsize=13)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"taxonomy_{args.tag}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")

    print(f"\n=== {args.tag} FAITHFUL taxonomy (RAW, no filtering) ===")
    for g in args.gammas:
        t = tax[g]; M = t["M"]
        print(f"  g{g}: reach {100*t['reach']/M:.0f}% | collision(RAW CR) {100*t['collision']/M:.0f}% "
              f"| OOB {100*t['oob']/M:.0f}% | timeout {100*t['timeout']/M:.0f}%  (M={M})")
    json.dump(tax, open(os.path.join(args.out_dir, f"taxonomy_{args.tag}.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
