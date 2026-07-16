# _v6 | PARADIGM CHANGE (2026-07-16): pure AFE-minimal Safe Flow Expansion — NO curriculum, NO
# easy/frontier, NO demo/LwF/anchor/encoder-freeze. One object identity: the planned window U_t is
# sampled from p_theta(.|c_t), sigma-scored on the FROZEN phi0 (cumulative 32x32 A_n over ALL verified
# queries), Gibbs-drawn (pi ~ e^{(sigma-max)/beta}, B=8/step), FULL-SOCP verified BEFORE execution,
# stored in D_n (pos+neg), and replayed (uniform over cumulative D+) under the single proximal
# objective l_CFM + ||theta-theta_n||^2/(2 eta). Execution samples ~pi among verified-safe plans
# (the diversity lever: covS 52 vs argmax 26-30); certified SafeMPPI backup when none verifies.
# | ours = results/p2/eval_afe_pure_pi_s910 (final.pt of results/afe/pure_pi_s910, 100 rounds, seed
# 910, lam 10, eta .01, M40 eval T350) | brothers = matched pi-rule runs with ONE gate removed each:
# eval_afe_bro_noverif (no pre-execution verifier: every drawn plan enters D+), eval_afe_bro_nofallback
# (no certified backup: unshielded execution when nothing certifies), eval_afe_bro_noprox (eta 1e18 +
# no fstep bound: unbounded round updates) | Kazuki = results/kazuki_g47_trap (safety-CALIBRATED
# CFM-MPPI, w_safe .72: safe but TRAPS, SR .35, circles to timeout — the safety-reliability trap)
# | expert = results/expert_g47 M100; pretrained = results/p2/eval_pretrained_g47 M40 with goal-corner
# overshoot inset | scene: 8-plug walls, start (0.3,0.3), goal (4.7,4.7), reach 0.15 | gamma colors =
# plasma {0.1,0.5,1.0} (viridis reserved for sigma)
"""Rollout gallery _v6 (2x4, pure-AFE paradigm): (1) pre-trained data, (2) expert, (3) pretrained +
goal-corner failure inset, (4) safety-calibrated CFM-MPPI + trap inset; (5-7) the method's OWN gate
ablations (one load-bearing piece removed each, matched recipe), (8) Ours — verified-plan flow with
pi-execution, balanced route words."""
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
import grid_metrics2 as GM2  # noqa: E402  (goal-relative staircase words)
from eval_ae import _apply_wall_plugs_eval  # noqa: E402

GSEL = [0.1, 0.5, 1.0]
PLA = plt.get_cmap("plasma")
GCOL = {0.1: PLA(0.08), 0.5: PLA(0.55), 1.0: PLA(0.85)}
REACH = 0.15

OURS_DIR = "results/p2/eval_afe_pure_pi_s910"     # pure AFE, exec ~ pi among verified-safe
PRE_DIR = "results/p2/eval_pretrained_g47"
EXP_DIR = "results/expert_g47"
KAZ_DIR = "results/kazuki_g47_trap"               # safety-CALIBRATED CFM-MPPI (w_safe .72): traps
ABL = [("results/p2/eval_afe_bro_noverif", "NO pre-execution verifier",
        "8.5% of gather episodes DIE\nuncertified plans train the flow"),
       ("results/p2/eval_afe_bro_nofallback", "NO certified fallback",
        "4.9% of gather episodes DIE\nat contexts the flow can't certify"),
       ("results/p2/eval_afe_bro_noprox", "NO proximal bound",
        "routes collapse: covΣ 18 (Ours 34)\naudit validity erodes (−4.5 pts)")]
START_XY = (0.3, 0.3)
DEMO_TPL = os.path.join(P2, "..", "..", "dataset", "w8d_windows_g{}.pt")


def loadp(d, g):
    f = os.path.join(P2, d, f"paths_g{g}.npz")
    if not os.path.exists(f):
        return []
    z = np.load(f, allow_pickle=True)
    return [np.asarray(p, float) for p in z["paths"]]


def scene(ax, title, bold=False):
    env = GS_.make_grid()
    _apply_wall_plugs_eval(env, 8)
    for o in env.obstacles.numpy():
        ax.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    ax.plot(START_XY[0], START_XY[1], "ks", ms=6, zorder=6)
    ax.plot(4.7, 4.7, "*", c="gold", mec="k", ms=13, zorder=6)
    ax.set_xlim(-0.45, 5.45); ax.set_ylim(-0.45, 5.45); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, pad=6, fontsize=18, fontweight=("bold" if bold else "normal"))


def draw(ax, p, g, dashed=False, lw=1.4, dots=True):
    ok = np.linalg.norm(p[-1] - [4.7, 4.7]) < REACH
    ax.plot(p[:, 0], p[:, 1], color=GCOL[g], lw=lw, ls="--" if dashed else "-", alpha=0.9, zorder=3)
    if dots:
        ax.plot(p[::3, 0], p[::3, 1], ".", color="k", ms=1.6, alpha=0.55, zorder=4)
    if not ok:
        ax.plot(p[-1, 0], p[-1, 1], "x", color="#cc3311", ms=8, mew=2.2, zorder=6)
    return ok


def pick(paths, n_fail=1, n_total=3):
    ok = [p for p in paths if np.linalg.norm(p[-1] - [4.7, 4.7]) < REACH]
    bad = [p for p in paths if np.linalg.norm(p[-1] - [4.7, 4.7]) >= REACH and len(p) > 5]
    out = bad[:n_fail] + ok[: max(0, n_total - min(len(bad), n_fail))]
    return out[:n_total] if out else paths[:n_total]


def word_of(p):
    try:
        return GM2.staircase_id_goal(p, [4.7, 4.7], reach=REACH)
    except Exception:
        return None


def add_zoom(ax, box, paths_gs, w=0.42, loc="lower right"):
    axi = inset_axes(ax, width=f"{int(w*100)}%", height=f"{int(w*100)}%", loc=loc, borderpad=0.4)
    env = GS_.make_grid()
    _apply_wall_plugs_eval(env, 8)
    for o in env.obstacles.numpy():
        axi.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    axi.plot(4.7, 4.7, "*", c="gold", mec="k", ms=11, zorder=6)
    for p, g in paths_gs:
        axi.plot(p[:, 0], p[:, 1], color=GCOL[g], lw=1.6, alpha=0.95, zorder=3)
        axi.plot(p[::2, 0], p[::2, 1], ".", color="k", ms=2.0, alpha=0.6, zorder=4)
        if np.linalg.norm(p[-1] - [4.7, 4.7]) >= REACH:
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

    # (1) PRE-TRAINED DATA: uniform-grid IC seeds + off-diagonal walled-SafeMPPI expert rollouts
    ax = axes[0, 0]; scene(ax, "Pre-trained data")
    try:
        import gen_uniform_data as GUD
        us = GUD.uniform_starts()
        ax.plot(us[:, 0], us[:, 1], ".", color="#999999", ms=3.5, alpha=0.55, zorder=2)
    except Exception:
        for g in ["0.1", "0.5", "1.0"]:
            f = DEMO_TPL.format(g)
            if os.path.exists(f):
                st = torch.load(f, map_location="cpu", weights_only=False).get("starts")
                if st is not None and len(st):
                    ax.plot(st.numpy()[:, 0], st.numpy()[:, 1], ".", color="#999999", ms=3.5, alpha=0.55, zorder=2)
    od = os.path.join(P2, "analysis/runs/offdiag_expert_walls8.npz")
    if os.path.exists(od):
        z = np.load(od, allow_pickle=True)
        for p, st, g in zip(z["paths"], z["starts"], z["gammas"]):
            p = np.asarray(p, float)
            ax.plot(p[:, 0], p[:, 1], color=GCOL[round(float(g), 2)], lw=1.3, alpha=0.9, zorder=3)
            ax.plot(st[0], st[1], "o", color="k", ms=5, zorder=6)
    else:
        for g in GSEL:
            for p in loadp(EXP_DIR, g)[:3]:
                draw(ax, p, g, dots=False, lw=1.2)

    # (2) expert
    ax = axes[0, 1]; scene(ax, "Expert")
    for g in GSEL:
        for p in loadp(EXP_DIR, g)[:3]:
            draw(ax, p, g)

    # (3) pretrained + GOAL-CORNER failure inset (the overshoot mass the verifier keeps rejecting —
    #     the same corner where the certified fallback carries Ours' terminal approach)
    ax = axes[0, 2]; scene(ax, "Pretrained")
    fail_pg = []
    for g in GSEL:
        for p in pick(loadp(PRE_DIR, g), n_fail=2):
            draw(ax, p, g)
            if np.linalg.norm(p[-1] - [4.7, 4.7]) >= REACH:
                fail_pg.append((p, g))
    near_goal = [(p, g) for p, g in fail_pg if np.linalg.norm(p[-1] - [4.7, 4.7]) < 1.2]
    add_zoom(ax, (3.9, 5.45, 3.9, 5.45), (near_goal or fail_pg)[:4], loc="lower right")

    # (4) safety-CALIBRATED CFM-MPPI + trap inset: safe clearance but circles to timeout (SR .35)
    ax = axes[0, 3]; scene(ax, r"CFM-MPPI$^{*}$ (safety-calibrated)")
    kfail = []
    for g in GSEL:
        for p in pick(loadp(KAZ_DIR, g), n_fail=3, n_total=3):
            draw(ax, p, g)
            if np.linalg.norm(p[-1] - [4.7, 4.7]) >= REACH:
                kfail.append((p, g))
    if kfail:
        tail = kfail[0][0][-60:]
        cx, cy = tail[:, 0].mean(), tail[:, 1].mean()
        add_zoom(ax, (cx - 0.8, cx + 0.8, cy - 0.8, cy + 0.8), kfail[:2], loc="upper left")

    # (5-7) the method's OWN gate ablations (matched pi-rule recipe, one piece removed each);
    # the annotation states each removed gate's MEASURED cost (gather-time deaths / collapse)
    for ax, (d, t, note) in zip(axes[1, :3], ABL):
        scene(ax, t)
        got = False
        for g in GSEL:
            ps = loadp(d, g)
            for p in pick(ps, n_fail=2):
                draw(ax, p, g); got = True
        if not got:
            ax.text(2.5, 2.6, "ablation arm\ntraining", ha="center", fontsize=12, color="#888888")
        else:
            ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=10,
                    color="#aa2211", bbox=dict(fc="white", ec="#cc3311", alpha=0.85, lw=1.0))

    # (8) OURS — pure AFE with pi-execution: balanced route words (U-first and R-first together)
    ax = axes[1, 3]; scene(ax, "Ours", bold=True)
    got = False
    for g in GSEL:
        groups = {}
        for p in loadp(OURS_DIR, g):
            if np.linalg.norm(p[-1] - [4.7, 4.7]) < REACH:
                w = word_of(p)
                if w:
                    groups.setdefault(w, []).append(p)
        uw = sorted([w for w in groups if str(w).startswith("U")], key=lambda k: -len(groups[k]))
        rw = sorted([w for w in groups if str(w).startswith("R")], key=lambda k: -len(groups[k]))
        seq = [w for pair in zip(uw, rw) for w in pair] + uw[len(rw):] + rw[len(uw):]
        for w in seq[:6]:
            draw(ax, groups[w][0], g, lw=1.5); got = True
    if not got:
        ax.text(2.5, 2.6, "eval pending", ha="center", fontsize=12, color="#888888")

    cmap3 = ListedColormap([GCOL[g] for g in GSEL])
    sm = plt.cm.ScalarMappable(cmap=cmap3, norm=BoundaryNorm([0, 1, 2, 3], 3))
    cb = fig.colorbar(sm, ax=axes, location="right", fraction=0.022, pad=0.02, ticks=[0.5, 1.5, 2.5])
    cb.ax.set_yticklabels(["0.1", "0.5", "1.0"]); cb.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"rollouts_v6.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote rollouts_v6.png/.pdf")


if __name__ == "__main__":
    main()
