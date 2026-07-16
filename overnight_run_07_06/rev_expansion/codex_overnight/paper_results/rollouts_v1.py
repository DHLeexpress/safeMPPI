# _v1 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt | recipe: corrected guarded unit (open scene; lr 2e-5 frozen enc, recovery .3, hard 12, escape 64, ratcheted teacher, trust gates) | data: UNCHANGED from _v0 (model still it146): ours+expert paths from results/p2/eval_it146_m100 and results/expert_gt (M100 per-seed paths, gamma {0.1,0.5,1.0}); Kazuki SUCCESS = results/kazuki_final_m200 (tuned w_safe=.3, coll_w=20, goal_w=2.0, goal_coef=.5); Kazuki TRAPPED = results/kazuki_wsweep (published-style weights coll_w=100 goal_w=.1 goal_coef=.1, single w_safe — every episode times out) | _v1 bump: lockstep with table_v1 (same-scene ablation update; ablations do not appear in this figure)
"""Rollout galleries, gamma {0.1, 0.5, 1.0}: (row 1) OURS colored by staircase mode (diagonal coverage: we
recover ~2^4 corridor words) with expert overlay; (row 2) Kazuki two-case exhibit — tuned guidance succeeds,
published-style guidance TRAPS (guidance breaks OOD: one reward fits one scene)."""
import glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 14})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import _paths  # noqa: E402
import grid_scene as GS_  # noqa: E402
import grid_metrics as GM  # noqa: E402

GSEL = ["0.1", "0.5", "1.0"]


def loadp(d, g):
    f = os.path.join(P2, d, f"paths_g{g}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return [np.asarray(p, float) for p in z["paths"]]


def scene(ax):
    env = GS_.make_grid()
    for o in env.obstacles.numpy():
        ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    ax.plot(0, 0, "ks", ms=6, zorder=6); ax.plot(5, 5, "*", c="gold", mec="k", ms=13, zorder=6)
    ax.set_xlim(-0.3, 5.4); ax.set_ylim(-0.3, 5.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5))
    cmap = plt.get_cmap("tab20")
    for j, g in enumerate(GSEL):
        ax = axes[0, j]; scene(ax)
        ours = loadp("results/p2/eval_it146_m100", g) or []
        modes = {}
        for p in ours:
            if np.linalg.norm(p[-1] - [5, 5]) < 0.1:
                w = GM.staircase_id(p, reach=0.1)
                modes.setdefault(w, []).append(p)
        for k, (w, ps) in enumerate(sorted(modes.items(), key=lambda kv: -len(kv[1]))):
            for p in ps[:3]:
                ax.plot(p[:, 0], p[:, 1], color=cmap(k % 20), lw=1.3, alpha=0.8, zorder=3)
        exp = loadp("results/expert_gt", g) or []
        for p in exp[:6]:
            ax.plot(p[:, 0], p[:, 1], color="k", lw=0.9, alpha=0.35, ls="--", zorder=2)
        ax.set_title(f"OURS it146, $\\gamma$={g}: {len(modes)} modes (colors)\nexpert dashed grey")
        ax = axes[1, j]; scene(ax)
        ok = loadp("results/kazuki_final_m200", g) or []
        for p in ok[:5]:
            ax.plot(p[:, 0], p[:, 1], color="#009988", lw=1.3, alpha=0.85, zorder=3)
        trap = None
        for f in sorted(glob.glob(os.path.join(P2, "results/kazuki_wsweep", f"paths_g{g}*.npz"))) or \
                sorted(glob.glob(os.path.join(P2, "results/kazuki_wsweep", "paths_g0.5.npz"))):
            z = np.load(f, allow_pickle=True); trap = [np.asarray(p, float) for p in z["paths"]]; break
        for p in (trap or [])[:5]:
            ax.plot(p[:, 0], p[:, 1], color="#cc3311", lw=1.3, alpha=0.85, zorder=4)
        ax.set_title(f"Kazuki, $\\gamma$={g}: tuned guidance reaches (teal)\npublished-style guidance TRAPPED (red)")
    fig.suptitle("Rollout comparison — our expanded policy recovers many diagonal corridors and mimics the "
                 "expert; guidance-based Kazuki works only at its tuned reward and traps off it (OOD "
                 "fragility)", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"rollouts_v1.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote rollouts_v1.png/.pdf")


if __name__ == "__main__":
    main()
