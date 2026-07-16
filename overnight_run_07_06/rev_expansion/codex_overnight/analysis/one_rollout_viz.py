"""(a) Off-diagonal WALLED SafeMPPI expert rollouts for rollouts_v4 panel 1 (analysis/runs/
offdiag_expert_walls8.npz), and (b) ONE clean rollout of a checkpoint per gamma on the walled scene
(user 2026-07-14: show the greedy it100 single rollout before extending).
  python analysis/one_rollout_viz.py --mode single --ckpt results/p2/gs3_final/final.pt --tag greedy_it100
  python analysis/one_rollout_viz.py --mode offdiag
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2)), HERE]
import grid_scene as GS  # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402
GSEL = [0.1, 0.5, 1.0]


def walled_env(start=(0.05, 0.05)):
    import torch
    e = GS.make_grid(); HT._apply_wall_plugs(e, 8)
    e.x0 = torch.tensor([start[0], start[1], 0.0, 0.0], dtype=e.x0.dtype)
    return e


def offdiag():
    """3 off-diagonal-start walled SafeMPPI expert rollouts per gamma -> npz for panel 1."""
    import grid_scene as GS_, grid_rollout  # noqa
    from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
    from di_grid_viz import di_step
    cfg = GS.mode1_config(); ad = SafeMPPIAdapter(**cfg)
    env = walled_env(); goal = env.goal.numpy(); goal_t = env.goal.detach().cpu().float()
    obs_plan = GS.planner_obstacles(env)
    starts = [(0.3, 2.6), (2.6, 0.3), (0.4, 3.6)]     # clearly off-diagonal, in free space
    paths, sts, gs = [], [], []
    for g in GSEL:
        for (sx, sy) in starts:
            st = np.array([sx, sy, 0.0, 0.0], np.float32); states = [st.copy()]
            for t in range(env.T):
                import torch
                a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_plan, gamma=g, seed=t)
                st = di_step(st, a.detach().cpu().numpy().astype(np.float32), dt=env.dt)
                states.append(st.copy())
                if np.linalg.norm(st[:2] - goal) < 0.15:
                    break
            paths.append(np.array(states, np.float32)[:, :2]); sts.append([sx, sy]); gs.append(g)
    out = os.path.join(P2, "analysis/runs/offdiag_expert_walls8.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pa = np.empty(len(paths), object)
    for i, p in enumerate(paths):
        pa[i] = p
    np.savez_compressed(out, paths=pa, starts=np.array(sts), gammas=np.array(gs))
    print("wrote", out, "(", len(paths), "rollouts )")


def single(ckpt, tag):
    """ONE deploy per gamma from the canonical start, on the walled scene."""
    import torch, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import grid_hp_expt as HP, grid_rollout as GR
    PLA = plt.get_cmap("plasma"); GC = {0.1: PLA(0.08), 0.5: PLA(0.55), 1.0: PLA(0.85)}
    pol, _ = HP.load_hp(ckpt, device="cuda"); pol.eval()
    env = walled_env()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, g in zip(axes, GSEL):
        for o in env.obstacles.numpy():
            ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
        torch.manual_seed(0)
        out = GR.fm_deploy(pol, env, float(g), T=250, temp=1.0, nfe=8, device="cuda", reach=0.15)
        p = np.asarray(out["path"], float)
        ax.plot(p[:, 0], p[:, 1], "-", color=GC[g], lw=2.4, zorder=4)
        ax.plot(p[::4, 0], p[::4, 1], ".", color="k", ms=2.4, alpha=.6, zorder=5)
        ok = np.linalg.norm(p[-1] - [5, 5]) < 0.15
        ax.plot(0.05, 0.05, "ks", ms=7, zorder=6); ax.plot(5, 5, "*", c="gold", mec="k", ms=16, zorder=6)
        if not ok:
            ax.plot(p[-1, 0], p[-1, 1], "x", c="#cc3311", ms=12, mew=3, zorder=7)
        ax.set_xlim(-0.45, 5.45); ax.set_ylim(-0.45, 5.45); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"γ={g}  {'REACHED' if ok else 'FAIL'}  ({len(p)-1} steps)", fontsize=13)
    fig.suptitle(f"Single rollout — {tag} (walled, canonical start)", fontsize=15)
    fig.tight_layout()
    out = os.path.join(P2, "grand_final_reports_rev", f"one_rollout_{tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offdiag", "single"], required=True)
    ap.add_argument("--ckpt"); ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    if args.mode == "offdiag":
        offdiag()
    else:
        single(args.ckpt, args.tag)


if __name__ == "__main__":
    main()
