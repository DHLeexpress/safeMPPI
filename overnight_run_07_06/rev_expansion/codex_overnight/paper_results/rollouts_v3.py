# _v3 | model: GRAND FINAL = results/p2/openscratch_base_s870/final.pt (pretrained_a32uni -> OPEN scene FROM SCRATCH, walls4_phased recipe ADAPTED to the open scene: FROZEN encoder (open scene = pretraining distribution; the unfrozen variant collapses by it40 — 4/4 runs, see PROGRESS 07-12 evening), t104-proven update magnitudes lr 2e-5 + single inner step + min-modes 2 (lr 1e-4 x2 collapses on the open scene regardless of encoder freezing — walls geometry funnels data, open scene does not), cap 600, trust gates off, phased curriculum .85/2, perp-brake targeting, recovery bands + hard-quota, 100 iters) | data: demo = ../../dataset/dr05_windows_g{0.1,0.5,1.0}.pt (ALL attempted starts + window support cloud), ours OOD = analysis/runs/ood_starts_grandfinal.npz (6 held-out starts x 3 gamma), ours M100 = results/p2/eval_grandfinal_m100, pretrained = results/p2/eval_pretrained_m25, Kazuki detuned = results/kazuki_sweep_smoke/w09 (kazuki_baseline.py on untouched pretrained, single knob w_safe .3->.9, M=10), ablations 5.1-5.3 = FROM-SCRATCH same-recipe arms results/p2/eval_openscratch_{nosocp,noprog,nocur}_m100 (one flag off each, FROZEN encoder, matching the full recipe)
"""Rollout gallery _v3 (2x4): (1) actual demo data + ours from held-out starts, (2) demo expert,
(3) pretrained + failure inset, (4) detuned Kazuki + trap inset; (5-7) from-scratch ablations,
(8) GRAND FINAL with BALANCED modes (U-first and R-first corridor words shown together)."""
import glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import _paths  # noqa: E402
import grid_scene as GS_  # noqa: E402
import grid_metrics as GM  # noqa: E402

GSEL = [0.1, 0.5, 1.0]
PLA = plt.get_cmap("plasma")
GCOL = {0.1: PLA(0.08), 0.5: PLA(0.55), 1.0: PLA(0.85)}

OURS_DIR = "results/p2/eval_grandfinal_m100"
OOD_NPZ = "analysis/runs/ood_starts_grandfinal.npz"
KAZ_DIR = "results/kazuki_sweep_smoke/w09"
ABL = [("results/p2/eval_openscratch_nosocp_m100", "NO safety validity check"),
       ("results/p2/eval_openscratch_noprog_m100", "NO progress check"),
       ("results/p2/eval_openscratch_nocur_m100", "NO curriculum")]
DEMO_TPL = os.path.join(P2, "..", "..", "dataset", "dr05_windows_g{}.pt")


def loadp(d, g):
    f = os.path.join(P2, d, f"paths_g{g}.npz")
    if not os.path.exists(f):
        return []
    z = np.load(f, allow_pickle=True)
    return [np.asarray(p, float) for p in z["paths"]]


def scene(ax, title, bold=False):
    env = GS_.make_grid()
    for o in env.obstacles.numpy():
        ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    ax.plot(0, 0, "ks", ms=5, zorder=6)
    ax.plot(5, 5, "*", c="gold", mec="k", ms=12, zorder=6)
    ax.set_xlim(-0.3, 5.4); ax.set_ylim(-0.3, 5.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=6, fontsize=18, fontweight=("bold" if bold else "normal"))


def draw(ax, p, g, dashed=False, lw=1.4, dots=True):
    ok = np.linalg.norm(p[-1] - [5, 5]) < 0.1
    ax.plot(p[:, 0], p[:, 1], color=GCOL[g], lw=lw, ls="--" if dashed else "-", alpha=0.9, zorder=3)
    if dots:
        ax.plot(p[::3, 0], p[::3, 1], ".", color="k", ms=1.6, alpha=0.55, zorder=4)
    if not ok:
        ax.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)
    return ok


def pick(paths, n_fail=1, n_total=3):
    ok = [p for p in paths if np.linalg.norm(p[-1] - [5, 5]) < 0.1]
    bad = [p for p in paths if np.linalg.norm(p[-1] - [5, 5]) >= 0.1 and len(p) > 5]
    out = bad[:n_fail] + ok[: max(0, n_total - min(len(bad), n_fail))]
    return out[:n_total] if out else paths[:n_total]


def word_of(p):
    try:
        return GM.staircase_id(p, reach=0.1)
    except Exception:
        return None


def add_zoom(ax, box, paths_gs, w=0.42):
    """Inset zoom on a failure region. box = (x0, x1, y0, y1)."""
    axi = inset_axes(ax, width=f"{int(w*100)}%", height=f"{int(w*100)}%", loc="lower right",
                     borderpad=0.4)
    env = GS_.make_grid()
    for o in env.obstacles.numpy():
        axi.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    for p, g in paths_gs:
        axi.plot(p[:, 0], p[:, 1], color=GCOL[g], lw=1.6, alpha=0.95, zorder=3)
        axi.plot(p[::2, 0], p[::2, 1], ".", color="k", ms=2.0, alpha=0.6, zorder=4)
        if np.linalg.norm(p[-1] - [5, 5]) >= 0.1:
            axi.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=9, mew=2.4, zorder=6)
    axi.set_xlim(box[0], box[1]); axi.set_ylim(box[2], box[3]); axi.set_aspect("equal")
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color("#cc3311"); s.set_linewidth(1.6)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                           fill=False, ec="#cc3311", lw=1.3, zorder=7))


def main():
    import torch
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.0))

    # (1) PRE-TRAINED DATA: the UNIFORM-GRID start recipe (gen_uniform_data.uniform_starts: 32x32 grid,
    #     |y-x|>=1, obstacle-free, +-0.02 jitter -> ~566 grey seeds) + 8 OFF-DIAGONAL SafeMPPI expert
    #     rollouts per gamma (starts far from the diagonal; analysis/runs/offdiag_expert.npz)
    ax = axes[0, 0]; scene(ax, "Pre-trained data")
    import gen_uniform_data as GUD
    us = GUD.uniform_starts()
    ax.plot(us[:, 0], us[:, 1], ".", color="#999999", ms=3.5, alpha=0.55, zorder=2)   # uniform IC seeds
    od = os.path.join(P2, "analysis/runs/offdiag_expert.npz")
    if os.path.exists(od):
        z = np.load(od, allow_pickle=True)
        for p, st, g in zip(z["paths"], z["starts"], z["gammas"]):
            p = np.asarray(p, float)
            ax.plot(p[:, 0], p[:, 1], color=GCOL[round(float(g), 2)], lw=1.3, alpha=0.9, zorder=3)
            ax.plot(st[0], st[1], "o", color="k", ms=5, zorder=6)   # off-diagonal start

    # (2) demo expert
    ax = axes[0, 1]; scene(ax, "Expert")
    for g in GSEL:
        for p in loadp("results/expert_gt", g)[:3]:
            draw(ax, p, g)

    # (3) pretrained + failure inset (dies at the origin)
    ax = axes[0, 2]; scene(ax, "Pretrained")
    fail_pg = []
    for g in GSEL:
        for p in pick(loadp("results/p2/eval_pretrained_m25", g), n_fail=2):
            draw(ax, p, g)
            if np.linalg.norm(p[-1] - [5, 5]) >= 0.1:
                fail_pg.append((p, g))
    add_zoom(ax, (-0.25, 0.9, -0.25, 0.9), fail_pg[:4])

    # (4) Kazuki detuned + trap inset
    ax = axes[0, 3]; scene(ax, r"CFM-MPPI$^{*}$")
    kfail = []
    for g in GSEL:
        for p in pick(loadp(KAZ_DIR, g), n_fail=2):
            draw(ax, p, g)
            if np.linalg.norm(p[-1] - [5, 5]) >= 0.1:
                kfail.append((p, g))
    if kfail:
        tail = kfail[0][0][-60:]
        cx, cy = tail[:, 0].mean(), tail[:, 1].mean()
        add_zoom(ax, (cx - 0.8, cx + 0.8, cy - 0.8, cy + 0.8), kfail[:2])

    # (5-7) FROM-SCRATCH ablations
    for ax, (d, t) in zip(axes[1, :3], ABL):
        scene(ax, t)
        got = False
        for g in GSEL:
            ps = loadp(d, g)
            for p in pick(ps, n_fail=2):
                draw(ax, p, g); got = True
        if not got:
            ax.text(2.5, 2.6, "from-scratch arm\ntraining", ha="center", fontsize=12, color="#888888")

    # (8) GRAND FINAL — BALANCED modes (U-first and R-first words together)
    ax = axes[1, 3]; scene(ax, "Ours", bold=True)
    got = False
    for g in GSEL:
        groups = {}
        for p in loadp(OURS_DIR, g):
            if np.linalg.norm(p[-1] - [5, 5]) < 0.1:
                w = word_of(p)
                if w:
                    groups.setdefault(w, []).append(p)
        uw = sorted([w for w in groups if str(w).startswith("U")], key=lambda k: -len(groups[k]))
        rw = sorted([w for w in groups if str(w).startswith("R")], key=lambda k: -len(groups[k]))
        seq = [w for pair in zip(uw, rw) for w in pair] + uw[len(rw):] + rw[len(uw):]
        for w in seq[:6]:
            draw(ax, groups[w][0], g, lw=1.5); got = True
    if not got:
        ax.text(2.5, 2.6, "grand-final M100 pending", ha="center", fontsize=12, color="#888888")

    cmap3 = ListedColormap([GCOL[g] for g in GSEL])
    sm = plt.cm.ScalarMappable(cmap=cmap3, norm=BoundaryNorm([0, 1, 2, 3], 3))
    cb = fig.colorbar(sm, ax=axes, location="right", fraction=0.022, pad=0.02, ticks=[0.5, 1.5, 2.5])
    cb.ax.set_yticklabels(["0.1", "0.5", "1.0"]); cb.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"rollouts_v3.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote rollouts_v3.png/.pdf")


if __name__ == "__main__":
    main()
