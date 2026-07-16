"""DR off-diagonal smoke + starts viz (user 2026-07-06).

Off-diagonal (|y-x| >= offdiag) domain-randomized starts, obstacle clearance = obs_margin, fixed goal,
per-gamma SafeMPPI expert, usual H=10 spec. This is the PRE-FLIGHT for the full gen_dr_data.py run:
  1. samples N starts (zero velocity) and scatters them over the grid  -> <outdir>/starts_scatter.png
  2. rolls out n_rollout trajectories per gamma, reports success rate   -> stdout table + smoke_summary.json
  3. plots one successful trajectory per gamma                          -> <outdir>/traj_per_gamma.png
Reuses gen_dr_data.sample_start / rollout_dr VERBATIM so the smoke exercises the exact code path the full
generation uses.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch  # noqa: F401  (import parity with the generator / CUDA init)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths  # noqa: F401
import grid_scene as GS
from gen_dr_data import sample_start, rollout_dr

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def draw_scene(ax, env, offdiag):
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    if offdiag > 0:  # shade the excluded diagonal band |y-x| < offdiag
        xs = np.linspace(0, 5, 200)
        ax.fill_between(xs, xs - offdiag, xs + offdiag, color="0.85", zorder=0,
                        label=f"|y-x|<{offdiag} (excluded)")
    for o in obs:
        orad = o[2] if len(o) > 2 else GS.OBS_R
        ax.add_patch(Circle((o[0], o[1]), float(orad), color="0.45", zorder=2))
    ax.plot(goal[0], goal[1], marker="*", color="tab:red", ms=16, zorder=5, label="goal")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    ap.add_argument("--n-rollout", type=int, default=10)
    ap.add_argument("--n-starts", type=int, default=1400)
    ap.add_argument("--offdiag", type=float, default=0.5)
    ap.add_argument("--obs-margin", type=float, default=0.05)
    ap.add_argument("--outdir", default="figures/dr_offdiag")
    args = ap.parse_args()
    outdir = os.path.join(HERE, args.outdir); os.makedirs(outdir, exist_ok=True)

    env = GS.make_grid(); cfg = GS.mode1_config()
    print(f"[smoke] env T={env.T} dt={env.dt} r_robot={float(env.r_robot):.3f} "
          f"obstacles={len(env.obstacles)} offdiag={args.offdiag} obs_margin={args.obs_margin}", flush=True)

    # --- 1. starts scatter (cheap: sample_start only; v=0 by construction) ---
    rng = np.random.default_rng(2026)
    starts = np.array([sample_start(env, rng, obs_margin=args.obs_margin, offdiag=args.offdiag)
                       for _ in range(args.n_starts)], dtype=np.float32)
    fb = env.x0.detach().cpu().numpy()[:2]
    n_fb = int((np.linalg.norm(starts[:, :2] - fb[None], axis=1) < 1e-6).sum())
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    draw_scene(ax, env, args.offdiag)
    ax.scatter(starts[:, 0], starts[:, 1], s=6, c="tab:blue", alpha=0.5, zorder=3, label="starts (v=0)")
    ax.set_title(f"DR off-diagonal starts  N={len(starts)}  fallback={n_fb}\n"
                 f"|y-x|>={args.offdiag},  clearance {args.obs_margin} m")
    ax.legend(loc="upper left", fontsize=7)
    f1 = os.path.join(outdir, "starts_scatter.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[starts] {len(starts)} sampled, {n_fb} fell back to origin -> {f1}", flush=True)

    # --- 2. smoke rollouts: success rate + one successful trajectory per gamma ---
    rows = []; trajs = {}
    for g in args.gammas:
        n_ok = 0; kept = None; t0 = time.time()
        for s in range(args.n_rollout):
            try:
                states, controls, start = rollout_dr(env, g, cfg, s, offdiag=args.offdiag,
                                                     obs_margin=args.obs_margin)
            except Exception as e:  # surface but keep going so one bad seed doesn't sink the table
                print(f"  ! γ{g} seed{s} rollout error: {e}", flush=True); continue
            ok, _ = GS.is_success(states[:, :2], env)
            if ok and len(controls) >= 2:
                n_ok += 1
                if kept is None:
                    kept = (states, start)
        sec = (time.time() - t0) / max(args.n_rollout, 1)
        rows.append((g, n_ok, args.n_rollout, sec)); trajs[g] = kept
        print(f"[roll] g{g}: {n_ok}/{args.n_rollout} success  {sec:.2f}s/seed", flush=True)

    # traj-per-gamma panel
    ncol = 4; nrow = int(np.ceil(len(args.gammas) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    ok_by_g = {r[0]: r[1] for r in rows}
    for i, g in enumerate(args.gammas):
        ax = axes[i // ncol][i % ncol]; draw_scene(ax, env, args.offdiag)
        kept = trajs[g]
        if kept is not None:
            states, start = kept
            ax.plot(states[:, 0], states[:, 1], "-", color="tab:green", lw=2, zorder=4)
            ax.scatter([start[0]], [start[1]], c="tab:green", s=70, marker="o",
                       edgecolor="k", zorder=6, label="start")
        ax.set_title(f"g={g}  ({ok_by_g[g]}/{args.n_rollout} ok)")
    for j in range(len(args.gammas), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    f2 = os.path.join(outdir, "traj_per_gamma.png")
    fig.savefig(f2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[traj] one trajectory per gamma -> {f2}", flush=True)

    # summary
    summ = dict(offdiag=args.offdiag, obs_margin=args.obs_margin, n_rollout=args.n_rollout,
                n_starts=int(len(starts)), n_fallback=n_fb,
                success={str(r[0]): dict(ok=r[1], n=r[2], sec_per_seed=round(r[3], 3)) for r in rows})
    with open(os.path.join(outdir, "smoke_summary.json"), "w") as fjs:
        json.dump(summ, fjs, indent=2)
    tot_ok = sum(r[1] for r in rows); tot = sum(r[2] for r in rows)
    print(f"[SMOKE DONE] overall success {tot_ok}/{tot} = {100 * tot_ok / max(tot,1):.1f}%  "
          f"starts_fallback {n_fb}/{len(starts)}", flush=True)


if __name__ == "__main__":
    main()
