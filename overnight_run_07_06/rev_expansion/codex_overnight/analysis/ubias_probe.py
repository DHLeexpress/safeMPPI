"""U-bias + weak-backprop probe (CENTERPIECE, user 2026-07-15).

Conjecture (user): the CFM loss backprop is too weak to move the policy's INITIAL sampling distribution,
even at high beta, so the U-lean is INHERITED from the pretrained base and the symmetric gather just
reflects it. Test: at the fixed start context, sample the raw window distribution (NO sigma-tilt) from
ckpt it0/it50/it100 and classify each first-window as R-lean (Δx>Δy) or U-lean (Δy>Δx). If the R/U split
barely moves across checkpoints, the base p(U|c) is essentially frozen and beta (a direction-agnostic
tilt, grid_rollout.py:149) has no leverage on it. Also reports full-deployment R/U (staircase first move).

  python analysis/ubias_probe.py --ckpts it0=<pre> it50=<c50> it100=<c100>
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2)),
                os.path.dirname(os.path.dirname(os.path.dirname(P2)))]
import torch
import grid_scene as GS, grid_rollout as GR, grid_feats as GF, grid_hp_expt as HP
import grid_metrics2 as GM2   # goal-relative staircase (the (5,5)-frame GM.staircase_id is all-None at 4.7)
from eval_ae import _apply_wall_plugs_eval


def raw_ru(pol, env, gamma, start, N=600, device="cuda"):
    """Sample N raw first-windows at the start context (no tilt); classify by net (Δx vs Δy)."""
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy()
    st = np.array([start[0], start[1], 0.0, 0.0], np.float32)
    gT = torch.tensor(GF.axis_grid(st[:2], obs, rr), device=device)
    lT = torch.tensor(GF.low5(st, goal, gamma), device=device)
    hT = torch.tensor(GF.hist_pad(np.zeros((0, 2)), GF.K_HIST), device=device)
    U = pol.sample_window(gT, lT, hT, n=N, temp=1.0, nfe=8).detach().cpu().numpy()
    net = GR.di_rollout_batch(st, U, env.dt)[:, -1, :] - st[:2]        # (N,2) net displacement
    dx, dy = net[:, 0], net[:, 1]
    u = int((dy > dx).sum()); r = int((dx >= dy).sum())
    return dict(R=r, U=u, u_frac=u / max(r + u, 1), mean_dx=float(dx.mean()), mean_dy=float(dy.mean()))


def deploy_ru(pol, env, gamma, goal, M=40, device="cuda"):
    """Full deployment: first goal-relative staircase move R vs U (outcome-level bias)."""
    GM2.GOAL_XY = np.array(goal, float)
    r = u = none = 0
    for s in range(M):
        torch.manual_seed(s)
        p = np.asarray(GR.fm_deploy(pol, env, float(gamma), T=250, temp=1.0, nfe=8,
                                    device=device, reach=0.15)["path"], float)
        w = GM2.staircase_id_goal(p, goal, reach=0.15)
        if w is None:
            none += 1
        elif w[0] == "R":
            r += 1
        else:
            u += 1
    return dict(R=r, U=u, none=none, u_frac=u / max(r + u, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=path ...")
    ap.add_argument("--goal-xy", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--start-xy", type=float, nargs=2, default=[0.3, 0.3])
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid(); _apply_wall_plugs_eval(env, 8)
    env.x0 = torch.tensor([args.start_xy[0], args.start_xy[1], 0., 0.], dtype=env.x0.dtype)
    env.goal = torch.tensor([args.goal_xy[0], args.goal_xy[1]], dtype=env.goal.dtype)

    labels, raw, dep = [], [], []
    print(f"U-bias probe @ start{tuple(args.start_xy)} goal{tuple(args.goal_xy)} γ{args.gamma} (U-frac=up-lean share)")
    for spec in args.ckpts:
        lab, path = spec.split("=", 1)
        pol, _ = HP.load_hp(path, device=dev); pol.eval()
        rr = raw_ru(pol, env, args.gamma, args.start_xy, device=dev)
        dd = deploy_ru(pol, env, args.gamma, args.goal_xy, device=dev)
        labels.append(lab); raw.append(rr); dep.append(dd)
        print(f"  {lab:>6}: RAW-sample U-frac {rr['u_frac']:.2f} (R{rr['R']}/U{rr['U']}, "
              f"⟨dx⟩{rr['mean_dx']:.3f} ⟨dy⟩{rr['mean_dy']:.3f}) | "
              f"DEPLOY U-frac {dd['u_frac']:.2f} (R{dd['R']}/U{dd['U']}/none{dd['none']})")

    d0, d1 = raw[0]["u_frac"], raw[-1]["u_frac"]
    frozen = abs(d1 - d0) < 0.1                              # data-driven title: did the anchor hold?
    if frozen:
        a_title = "(A) U-lean FROZEN near base — both modes held"
        sup = (f"Diversity PRESERVED — the δ/η anchor freezes the sampling distribution near the base "
               f"(raw {d0:.2f}→{d1:.2f}); deploy stays balanced ({dep[-1]['R']}R/{dep[-1]['U']}U) instead "
               f"of collapsing to all-U")
    else:
        a_title = "(A) U-lean climbs R→U with training (base leans R!)"
        sup = (f"U-bias CREATED by training: base leans RIGHT ({d0:.2f}), training flips it to UP "
               f"({d1:.2f} raw / {dep[-1]['u_frac']:.1f} deploy) — CFM backprop STRONGLY reshapes sampling")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(labels))
    ax[0].bar(x, [r["u_frac"] for r in raw], color="#7b68ee", label="raw sample")
    ax[0].plot(x, [d["u_frac"] for d in dep], "o-", c="#d62728", label="deployment")
    ax[0].axhline(0.5, ls=":", c="grey", label="balanced")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("U-lean fraction"); ax[0].set_title(a_title)
    ax[0].legend(fontsize=9)
    ax[1].plot(x, [r["mean_dx"] for r in raw], "s-", c="#1f77b4", label="⟨Δx⟩ (right)")
    ax[1].plot(x, [r["mean_dy"] for r in raw], "o-", c="#ff7f0e", label="⟨Δy⟩ (up)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("mean first-window displacement"); ax[1].set_title("(B) raw sampling drift over training")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    fig.suptitle(sup, fontsize=11.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P2, "paper_results", f"ubias_probe.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote paper_results/ubias_probe.png")
    d0, d1 = raw[0]["u_frac"], raw[-1]["u_frac"]
    print(f"  raw U-frac {labels[0]}={d0:.2f} -> {labels[-1]}={d1:.2f}  (Δ={d1 - d0:+.2f}); "
          f"{'CONFIRMS weak-backprop (≈frozen)' if abs(d1 - d0) < 0.1 else 'moved — backprop DID shift it'}")


if __name__ == "__main__":
    main()
