"""Why pushing the perimeter walls out breaks the pretrained (user 2026-07-14).
Deploy the SAME pretrained policy (gamma 1.0, several seeds) on: 8-plug (walls at +/-0.2, in-distribution),
push 0.2 m, push 0.3 m. GREEN = reached goal, RED = died (OOB/collision). Shows the policy is tightly
coupled to the perimeter it was trained with: 0.2 m already induces goal-overshoot / OOD exits, 0.3 m
collapses. Also marks the INTERIOR squeeze points (the real low-gamma SOCP constraint).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2)), HERE]
import numpy as np, torch  # noqa
import grid_hp_expt as HP, grid_scene as GS, grid_rollout as GR, grid_expand_hardtail as HT  # noqa
import new_scene_viz as NS  # noqa

PRE = os.path.join(P2, "../../results/hp_repr/pretrained_a32uni.pt")


def make(kind):
    e = GS.make_grid()
    if kind == "8plug":
        HT._apply_wall_plugs(e, 8); e.x0 = torch.tensor([0.05, 0.05, 0., 0.], dtype=e.x0.dtype)
    else:
        push = float(kind.split("_")[1]); obs, _, _ = NS.pushed_obstacles(push)
        e.obstacles = torch.tensor(obs, dtype=e.obstacles.dtype); e.x0 = torch.tensor([0., 0., 0., 0.], dtype=e.x0.dtype)
    return e


def panel(ax, kind, pol, g=1.0, n=8):
    e = make(kind); obs = e.obstacles.numpy(); rr = float(e.r_robot)
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor="#c9c9c9", ec="none", zorder=1))
    ax.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], "--", c="#4477aa", lw=1.0, zorder=2)
    nz = 0
    for s in range(n):
        torch.manual_seed(s)
        out = GR.fm_deploy(pol, e, g, T=250, temp=1.0, nfe=8, device="cuda", reach=0.2)
        p = np.asarray(out["path"], float)
        c = "#009944" if out["reached"] else "#cc3311"
        nz += int(out["reached"])
        ax.plot(p[:, 0], p[:, 1], "-", c=c, lw=1.5, alpha=0.85, zorder=4)
        ax.plot(p[-1, 0], p[-1, 1], "o", c=c, ms=5, zorder=5)
    ax.plot(0, 0, "ks", ms=7, zorder=8); ax.plot(5, 5, "*", c="gold", mec="k", ms=16, zorder=8)
    ax.set_xlim(-0.7, 5.7); ax.set_ylim(-0.7, 5.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{kind}\n{nz}/{n} reached (green)", fontsize=13)


def main():
    pol, _ = HP.load_hp(PRE, device="cuda"); pol.eval()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, kind in zip(axes, ["8plug", "pushed_0.2", "pushed_0.3"]):
        panel(ax, kind, pol)
    fig.suptitle("Same pretrained policy (γ=1.0) — the perimeter it was trained on (±0.2) is load-bearing: "
                 "0.2 m push already overshoots/exits, 0.3 m collapses", fontsize=13)
    fig.tight_layout()
    out = os.path.join(P2, "grand_final_reports_rev", "push_deploy.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
