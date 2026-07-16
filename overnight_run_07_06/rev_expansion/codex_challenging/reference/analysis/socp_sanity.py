"""SOCP sanity check (user 2026-07-13): is the goal 'attached to the obstacle' the reason the walled-scene
SOCP starves? Deploy a policy, solve the per-window SOCP, and VISUALIZE the verifier polytope (GREEN)
along the trajectory — on the WALLED scene (goal on the plug boundary) vs a NO-OUTER-WALL scene (goal in
free space). Compare the valid2 rate and where the SOCP fails.

  python analysis/socp_sanity.py --ckpt results/p2/w8_nosocp/final.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import grid_scene as GS          # noqa: E402
import grid_metrics2 as GM2      # noqa: E402
import grid_rollout as GR        # noqa: E402
import grid_hp_expt as HP        # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402
import verifier_polytope as VP   # noqa: E402

GREEN = "#009944"


def poly_from_faces(faces, c, R=2.0):
    """2D polygon of {a_i.(y) <= m_i} (y = x - c) from feasible faces, as world-frame vertices."""
    A, b = [], []
    for f in faces:
        if not getattr(f, "feasible", True) or f.m <= 1e-9:
            continue
        a = np.asarray(f.a, float)[:2]
        n = np.linalg.norm(a)
        if n < 1e-9:
            continue
        A.append(a / n); b.append(f.m / n)
    if len(A) < 3:
        return None
    A = np.array(A); b = np.array(b)
    try:
        from scipy.spatial import HalfspaceIntersection, ConvexHull
        hs = HalfspaceIntersection(np.hstack([A, -b[:, None]]), np.zeros(2))
        V = hs.intersections
        if len(V) < 3:
            return None
        V = V[ConvexHull(V).vertices]
        return V + c[None, :2]
    except Exception:
        return None


def window_faces(seg, obs, r_robot, gamma, n_theta=180):
    c = np.asarray(seg[0], float)[:2]
    path = np.asarray(seg, float)[:, :2]
    ok, faces, _raw, reff = VP.certify_window(path, obs, float(r_robot), float(gamma),
                                              R=2.0, n_theta=n_theta)
    return ok, faces, c


def deploy(pol, env, g, seed):
    return np.asarray(GR.fm_deploy(pol, env, float(g), T=250, temp=1.0, nfe=8, device="cuda",
                                   reach=0.2)["path"], float)


def valid_rate(pol, env, g, M):
    ok = 0
    for s in range(M):
        import torch; torch.manual_seed(s)
        if GM2.traj_valid2(deploy(pol, env, g, s), env, float(g)):
            ok += 1
    return 100.0 * ok / M


def panel(ax, env, pol, g, title, H=10):
    obs = env.obstacles.numpy(); rr = float(env.r_robot); goal = env.goal.numpy()
    for o in obs:
        ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    ax.plot(env.x0.numpy()[0], env.x0.numpy()[1], "ks", ms=6, zorder=6)
    ax.plot(goal[0], goal[1], "*", c="gold", mec="k", ms=15, zorder=7)
    import torch; torch.manual_seed(0)
    p = deploy(pol, env, g, 0)
    ax.plot(p[:, 0], p[:, 1], color="#333333", lw=1.6, zorder=4)
    # SOCP polytope (green) at a few windows incl. the one nearest the goal
    T = len(p) - 1
    ks = sorted(set([2, T // 2, max(0, T - H - 1), max(0, T - 2)]))
    n_fail = 0
    for k in ks:
        seg = p[k:k + H + 1]
        if len(seg) < 3:
            continue
        ok, faces, c = window_faces(seg, obs, rr, g)
        poly = poly_from_faces(faces, c)
        col = GREEN if ok else "#cc3311"
        if poly is not None:
            ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=0.14, zorder=2)
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=col, lw=1.6, zorder=5)
        ax.plot(c[0], c[1], "o", color=col, ms=6, zorder=6)
        if not ok:
            n_fail += 1
            ax.plot(c[0], c[1], "x", color="#cc3311", ms=11, mew=2.5, zorder=8)
    ax.set_xlim(-0.7, 5.7); ax.set_ylim(-0.7, 5.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{title}\n{n_fail}/{len(ks)} sampled windows SOCP-FAIL (red)", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/p2/w8_nosocp/final.pt")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--M", type=int, default=20)
    args = ap.parse_args()
    pol, _ = HP.load_hp(args.ckpt, device="cuda"); pol.eval()

    walled = GS.make_grid(); HT._apply_wall_plugs(walled, 8)
    walled.x0 = __import__("torch").tensor([0.05, 0.05, 0.0, 0.0], dtype=walled.x0.dtype)
    nowall = GS.make_grid(walls=False)

    vr_w = valid_rate(pol, walled, args.gamma, args.M)
    vr_n = valid_rate(pol, nowall, args.gamma, args.M)
    print(f"SOCP valid2 rate @g{args.gamma} M{args.M}: WALLED {vr_w:.0f}% | NO-OUTER-WALL {vr_n:.0f}%")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.5))
    panel(a1, walled, pol, args.gamma, f"WALLED (8-plug, goal ON plug): valid2 {vr_w:.0f}%")
    panel(a2, nowall, pol, args.gamma, f"NO OUTER WALL (goal in free space): valid2 {vr_n:.0f}%")
    fig.suptitle("SOCP verifier polytope (GREEN=certified, RED=infeasible) along one rollout — "
                 "is the goal 'attached to the obstacle' the problem?", fontsize=12)
    fig.tight_layout()
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = "trained" if tag == "final" else tag
    out = os.path.join(P2, "grand_final_reports_rev", f"socp_sanity_{tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
