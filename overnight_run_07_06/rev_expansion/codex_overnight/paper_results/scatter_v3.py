# _v3 | model: GRAND FINAL = results/p2/openscratch_base_s870/final.pt (pretrained_a32uni -> OPEN scene FROM SCRATCH, walls4_phased recipe ADAPTED to the open scene: FROZEN encoder (open scene = pretraining distribution; the unfrozen variant collapses by it40 — 4/4 runs, see PROGRESS 07-12 evening), t104-proven update magnitudes lr 2e-5 + single inner step + min-modes 2 (lr 1e-4 x2 collapses on the open scene regardless of encoder freezing — walls geometry funnels data, open scene does not), cap 600, trust gates off, phased curriculum .85/2, perp-brake targeting, recovery bands + hard-quota, 100 iters) | data: expert=results/expert_gt M100, ours=results/p2/eval_grandfinal_m100 M100, kazuki DETUNED=results/kazuki_sweep_smoke/w09 (M=10, w_safe .3->.9 on untouched pretrained; tuned variant removed per user), pretrained=results/p2/eval_pretrained_m25 M25, ablations=FROM-SCRATCH same-recipe arms results/p2/eval_openscratch_{nosocp,noprog,nocur}_m100
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
KAZ_DET_LABEL = r"Kazuki detuned $w_s{=}.9$ (M=10)"

SERIES = [
    ("Demo expert (target)", "results/expert_gt", "o", 95, dict()),
    ("Ours GRAND FINAL", "results/p2/eval_grandfinal_m100", "*", 240, dict(edgecolors="k", linewidths=0.9)),
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
    aR.set_title("reliability (a, b) — ideal: bottom-right")
    aR.annotate("ideal", xy=(100, 0), xytext=(90.5, 1.6), fontsize=11, color="#009988",
                arrowprops=dict(arrowstyle="->", color="#009988"))
    aR.set_ylim(-0.6, None); aR.grid(alpha=0.3)
    aQ.set_xlabel("time to goal [s]"); aQ.set_ylabel("min clearance (successes) [m]")
    aQ.set_title("quality (c, d) — ideal: top-left (safer, faster)")
    aQ.grid(alpha=0.3)
    handles = [Line2D([], [], color="#666666", marker=m, ls="", ms=11 if m == "*" else 8, label=n)
               for n, _, m, _, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=len(series), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=VIR, norm=NORM); sm.set_array([])
    cb = fig.colorbar(sm, ax=[aR, aQ], location="right", fraction=0.03, pad=0.015, ticks=GAMMAS)
    cb.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"scatter_v3.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote scatter_v3.png/.pdf")


if __name__ == "__main__":
    main()
