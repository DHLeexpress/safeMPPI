"""New eval/train scene (user 2026-07-14): DON'T treat gamma 0.1/0.2 differently (ad-hoc). Instead fix
the geometry so all 7 gammas are uniform: PUSH every perimeter wall 0.3 m OUT from the [0,5] boundary
(so the goal/origin/boundary gain 0.3 m clearance -> low-gamma SOCP becomes satisfiable) and ADD 4 corner
plugs to seal the diagonal gaps the push opens. Renders CURRENT (8-plug, goal on wall) vs NEW side by side.

  python analysis/new_scene_viz.py --push 0.3
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import grid_scene as GS            # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402

GRID = 5.0


def current_obstacles():
    e = GS.make_grid(); HT._apply_wall_plugs(e, 8)
    return e.obstacles.numpy().copy(), e.goal.numpy()[:2], float(e.r_robot)


def pushed_obstacles(push=0.3):
    """Interior obstacles unchanged; every perimeter/plug obstacle moved `push` further OUT of [0,GRID]
    along whichever axis it sits outside; then 4 corner plugs seal the new diagonal gaps."""
    e = GS.make_grid(); HT._apply_wall_plugs(e, 8)
    o = e.obstacles.numpy().copy()
    for i in range(len(o)):
        x, y = o[i, 0], o[i, 1]
        if x < -1e-3:
            o[i, 0] = x - push
        elif x > GRID + 1e-3:
            o[i, 0] = x + push
        if y < -1e-3:
            o[i, 1] = y - push
        elif y > GRID + 1e-3:
            o[i, 1] = y + push
    r = 0.2
    c = GRID + 0.2 + push        # pushed corner coordinate (matches the pushed edge rows)
    corners = np.array([[-0.2 - push, -0.2 - push, r], [c, -0.2 - push, r],
                        [-0.2 - push, c, r], [c, c, r]], dtype=o.dtype)
    o = np.concatenate([o, corners], axis=0)
    return o, e.goal.numpy()[:2], float(e.r_robot)


def min_clear(pt, obs):
    d = np.linalg.norm(obs[:, :2] - np.asarray(pt)[None, :], axis=1) - obs[:, 2]
    return float(d.min())


def draw(ax, obs, goal, title, push_note=""):
    ax.set_facecolor("#f7f6f4")
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor="#8a8a8a", ec="none", zorder=2))
    ax.plot([0, GRID, GRID, 0, 0], [0, 0, GRID, GRID, 0], "--", c="#4477aa", lw=1.3, zorder=3)  # task box
    ax.plot(0, 0, "s", c="k", ms=9, zorder=8)
    ax.plot(GRID, GRID, "*", c="gold", mec="k", ms=20, zorder=8)
    cg = min_clear(goal, obs); co = min_clear([0, 0], obs)
    ax.annotate(f"goal clear\n{cg:.2f} m", (GRID, GRID), (GRID - 1.35, GRID - 0.95),
                fontsize=12, ha="center", color="#aa3311",
                arrowprops=dict(arrowstyle="->", color="#aa3311", lw=1.6))
    ax.annotate(f"origin clear\n{co:.2f} m", (0, 0), (0.95, 0.75),
                fontsize=12, ha="center", color="#aa3311",
                arrowprops=dict(arrowstyle="->", color="#aa3311", lw=1.6))
    ax.set_xlim(-1.0, 6.0); ax.set_ylim(-1.0, 6.0); ax.set_aspect("equal")
    ax.set_xticks(range(0, 6)); ax.set_yticks(range(0, 6))
    ax.set_title(f"{title}\n{len(obs)} obstacles{push_note}", fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(P2, "grand_final_reports_rev", "new_scene.png"))
    args = ap.parse_args()

    oc, goal, rr = current_obstacles()
    on, _, _ = pushed_obstacles(args.push)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.6))
    draw(a1, oc, goal, "CURRENT (8-plug): goal ON the wall", "  (r_robot=0)")
    draw(a2, on, goal, f"NEW: walls pushed {args.push:.1f} m out + 4 corner plugs",
         f"  (+{len(on)-len(oc)} vs current)")
    fig.suptitle("Eval/train scene fix — uniform across all 7 gammas (no per-gamma special-casing)",
                 fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"goal clearance: current {min_clear(goal, oc):.3f} m -> new {min_clear(goal, on):.3f} m")
    print(f"origin clearance: current {min_clear([0,0], oc):.3f} m -> new {min_clear([0,0], on):.3f} m")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
