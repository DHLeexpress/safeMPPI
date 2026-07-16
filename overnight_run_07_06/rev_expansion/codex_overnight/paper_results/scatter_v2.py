# _v2 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt (champion; gen3/gen4 continuations gate-failed, not promoted) | recipe: corrected guarded unit (lr 2e-5, frozen enc, recovery .3, hard-quota 12, escape 64, ratcheted teacher, trust gates, OPEN scene) | data: expert=results/expert_gt M100 (full 7-gamma sweep), ours=results/p2/eval_it146_m100 M100, kazuki TUNED=results/kazuki_final_m200 M200 (kazuki_baseline.py guided CFM+MPPI on our untouched pretrained_a32uni.pt, gamma via context only; w_safe=.3 coll_w=20 goal_w=2.0 goal_coef=.5 beta_mppi=20), kazuki DETUNED=KAZ_DET_DIR below (M=10 smoke sweep variant), pretrained=results/p2/eval_pretrained_m25 M25 | layout: 1x2 phase planes (SR-CR and clearance-time), marker=method, color=gamma (viridis colorbar outside), legend outside
"""'Proof we won' phase planes: (left) reliability SR vs CR — ideal corner (100, 0);
(right) quality clearance vs time — up-left is safer AND faster. Marker = method, color = gamma."""
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
# truncated plasma: distinct from the viridis used for sigma/uncertainty in the curriculum videos
VIR = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(matplotlib.colors.Normalize()(range(256)))[10:225])
NORM = matplotlib.colors.Normalize(vmin=0.1, vmax=1.0)

KAZ_DET_DIR = "results/kazuki_sweep_smoke/w09"       # sweet spot: w_safe .3 -> .9, M=10
KAZ_DET_LABEL = r"CFM-MPPI$^{*}$"

SERIES = [
    ("Expert", "results/expert_gt", "o", 95, dict()),
    ("Our approach", "results/p2/eval_it146_m100", "*", 240, dict(edgecolors="k", linewidths=0.9)),
    ("Pretrained", "results/p2/eval_pretrained_m25", "s", 70, dict(alpha=0.55)),
]


def rows(d):
    out = {}
    for f in glob.glob(os.path.join(P2, d, "row_g*.json")):
        r = json.load(open(f)); out[round(float(r["gamma"]), 2)] = r
    return out


def main():
    series = list(SERIES)
    if KAZ_DET_DIR:
        series.insert(3, (KAZ_DET_LABEL, KAZ_DET_DIR, "v", 95, dict()))
    fig, (aR, aQ) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for name, d, mark, size, kw in series:
        R = rows(d)
        for g in GAMMAS:
            if g not in R:
                continue
            r = R[g]; c = [VIR(NORM(g))]
            aR.scatter(r["SR"] * 100, r["CR"] * 100, c=c, marker=mark, s=size, zorder=3, **kw)
            aQ.scatter(r["time_mean_s"], r["clearance_mean"], c=c, marker=mark, s=size, zorder=3, **kw)
    aR.set_xlabel("success rate SR [%]"); aR.set_ylabel("collision rate CR [%]")
    aR.set_ylim(-0.6, None); aR.grid(alpha=0.3)
    aQ.set_xlabel("time to goal [s]"); aQ.set_ylabel("min clearance (successes) [m]")
    aQ.grid(alpha=0.3)

    def _lbl(n):
        return r"$\mathbf{Our\ approach}$" if n == "Our approach" else n
    handles = [Line2D([], [], color="#666666", marker=m, ls="", ms=11 if m == "*" else 8, label=_lbl(n))
               for n, _, m, _, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(series), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=VIR, norm=NORM); sm.set_array([])
    cb = fig.colorbar(sm, ax=[aR, aQ], location="right", fraction=0.03, pad=0.015, ticks=GAMMAS)
    cb.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"scatter_v2.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote scatter_v2.png/.pdf")


if __name__ == "__main__":
    main()
