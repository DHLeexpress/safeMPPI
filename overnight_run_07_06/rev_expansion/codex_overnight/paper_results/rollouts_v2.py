# _v2 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt (champion; gen3/gen4 perp-brake continuations FAILED their it146-baseline gates and were not promoted) | recipe: corrected guarded unit (open scene; lr 2e-5 frozen enc, recovery .3, hard 12, escape 64, ratcheted teacher, trust gates) | data: OOD-start rollouts analysis/runs/ood_starts_it146.npz (4 off-diagonal starts x gamma {.1,.5,1}, faithful deploy, 12/12 reached); expert results/expert_gt; pretrained results/p2/eval_pretrained_m25 (M25); Kazuki detuned = KAZ_DIR below from the M=10 smoke sweep (kazuki_baseline.py = guided CFM+MPPI on our untouched pretrained_a32uni.pt, gamma via context only); ablation rollouts = SHORT-WINDOW arms results/p2/eval_openabl_{nosocp,noprog,nocur}_m100 (12-update it134->146 ablations, PROVISIONAL: faithful from-scratch arms openscratch_s870-873 are training and will replace these in the next _vN); full-model modes from results/p2/eval_it146_m100
"""Rollout gallery 2x4: (top) OOD-start recovery / demo expert / pretrained / detuned Kazuki;
(bottom) w/o SOCP / w/o progress / w/o curriculum / full model with distinct corridor modes.
gamma encoded by color (shared colorbar outside); small black dots = robot states; x = failed episode."""
import glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 12.5})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import _paths  # noqa: E402
import grid_scene as GS_  # noqa: E402
import grid_metrics as GM  # noqa: E402

GSEL = [0.1, 0.5, 1.0]
# plasma (truncated): distinct from the viridis used for sigma/uncertainty in the curriculum videos
PLA = plt.get_cmap("plasma")
GCOL = {0.1: PLA(0.08), 0.5: PLA(0.55), 1.0: PLA(0.85)}

# Kazuki detuned exhibit — sweep sweet spot: single knob w_safe .3 -> .9 (SR 10/60/90 at g.1/.5/1, all traps)
KAZ_DIR = "results/kazuki_sweep_smoke/w09"
KAZ_LABEL = r"$w_{safe}$ .3$\to$.9: traps"


def loadp(d, g):
    f = os.path.join(P2, d, f"paths_g{g}.npz")
    if not os.path.exists(f):
        return []
    z = np.load(f, allow_pickle=True)
    return [np.asarray(p, float) for p in z["paths"]]


def scene(ax, title):
    env = GS_.make_grid()
    for o in env.obstacles.numpy():
        ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    ax.plot(0, 0, "ks", ms=5, zorder=6)
    ax.plot(5, 5, "*", c="gold", mec="k", ms=12, zorder=6)
    ax.set_xlim(-0.3, 5.4); ax.set_ylim(-0.3, 5.4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, pad=4)


def pick(paths, n_fail=1, n_total=3):
    """Failure-aware exemplar selection: up to n_fail failed paths (visible length), rest successes."""
    ok = [p for p in paths if np.linalg.norm(p[-1] - [5, 5]) < 0.1]
    bad = [p for p in paths if np.linalg.norm(p[-1] - [5, 5]) >= 0.1 and len(p) > 5]
    out = bad[:n_fail] + ok[: max(0, n_total - min(len(bad), n_fail))]
    return out[:n_total] if out else paths[:n_total]


def draw(ax, p, g, dashed=False, lw=1.4):
    ok = np.linalg.norm(p[-1] - [5, 5]) < 0.1
    ax.plot(p[:, 0], p[:, 1], color=GCOL[g], lw=lw, ls="--" if dashed else "-",
            alpha=0.9, zorder=3)
    ax.plot(p[::3, 0], p[::3, 1], ".", color="k", ms=1.6, alpha=0.55, zorder=4)
    if not ok:
        ax.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)
    return ok


def main():
    fig, axes = plt.subplots(2, 4, figsize=(19.5, 10.0))

    # (1) ours from OOD starts (dashed, colored by gamma)
    ax = axes[0, 0]; scene(ax, "Ours (it146) from OOD starts — all reach")
    z = np.load(os.path.join(P2, "analysis/runs/ood_starts_it146.npz"), allow_pickle=True)
    for p, st, g in zip(z["paths"], z["starts"], z["gammas"]):
        draw(ax, np.asarray(p, float), round(float(g), 2), dashed=True)
        ax.plot(st[0], st[1], "s", color="k", ms=5.5, zorder=6)

    # (2) demo expert
    ax = axes[0, 1]; scene(ax, "Demo expert — diagonal corridors only")
    for g in GSEL:
        for p in loadp("results/expert_gt", g)[:3]:
            draw(ax, p, g)

    # (3) pretrained (failure-heavy exemplars: it is SR 24-48%)
    ax = axes[0, 2]; scene(ax, "Pretrained (before expansion)")
    for g in GSEL:
        for p in pick(loadp("results/p2/eval_pretrained_m25", g), n_fail=2):
            draw(ax, p, g)

    # (4) Kazuki detuned (trap exemplars + a success)
    ax = axes[0, 3]; scene(ax, r"Kazuki detuned ($w_s{=}.9$) — traps")
    if KAZ_DIR:
        for g in GSEL:
            for p in pick(loadp(KAZ_DIR, g), n_fail=2):
                draw(ax, p, g)
    else:
        ax.text(2.5, 2.6, "M=10 sweep\nrunning", ha="center", fontsize=13, color="#888888")

    # (5-7) ablations (short-window arms, provisional)
    for ax, d, t in [(axes[1, 0], "results/p2/eval_openabl_nosocp_m100", "w/o SOCP (multi-step safety)$^{\\dagger}$"),
                     (axes[1, 1], "results/p2/eval_openabl_noprog_m100", "w/o progress condition$^{\\dagger}$"),
                     (axes[1, 2], "results/p2/eval_openabl_nocur_m100", "w/o curriculum$^{\\dagger}$")]:
        scene(ax, t)
        for g in GSEL:
            for p in loadp(d, g)[:3]:
                draw(ax, p, g)

    # (8) full model — distinct corridor modes, 6 per gamma
    ax = axes[1, 3]; scene(ax, "Ours full (it146) — distinct modes")
    for g in GSEL:
        modes = {}
        for p in loadp("results/p2/eval_it146_m100", g):
            if np.linalg.norm(p[-1] - [5, 5]) < 0.1:
                w = GM.staircase_id(p, reach=0.1)
                modes.setdefault(w, []).append(p)
        for w in sorted(modes, key=lambda k: -len(modes[k]))[:6]:
            draw(ax, modes[w][0], g, lw=1.5)

    # shared gamma colorbar OUTSIDE + line-style legend OUTSIDE
    cmap3 = ListedColormap([GCOL[g] for g in GSEL])
    sm = plt.cm.ScalarMappable(cmap=cmap3, norm=BoundaryNorm([0, 1, 2, 3], 3))
    cb = fig.colorbar(sm, ax=axes, location="right", fraction=0.022, pad=0.02, ticks=[0.5, 1.5, 2.5])
    cb.ax.set_yticklabels(["0.1", "0.5", "1.0"]); cb.set_label(r"safety level $\gamma$", fontsize=13)
    fig.legend(handles=[
        Line2D([], [], color="#555555", ls="--", label="OOD-start rollout (dashed)"),
        Line2D([], [], color="#555555", ls="-", label="origin-start rollout"),
        Line2D([], [], color="k", ls="", marker=".", label="robot states"),
        Line2D([], [], color="#cc3311", ls="", marker="x", mew=2, label="failed episode"),
        Line2D([], [], color="k", ls="", marker="s", label="start"),
    ], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.001), fontsize=11)
    fig.text(0.01, 0.005, r"$\dagger$ short-window ablation arms (it134$\to$146); faithful from-scratch "
                          "ablation retrains (openscratch_s870-873) are training and replace these next _vN",
             fontsize=9, color="#666666")
    fig.suptitle("Rollout gallery — expansion recovers OOD starts and many corridor modes; "
                 "guidance-based baseline is reward-fragile", y=0.965, fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"rollouts_v2.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote rollouts_v2.png/.pdf")


if __name__ == "__main__":
    main()
