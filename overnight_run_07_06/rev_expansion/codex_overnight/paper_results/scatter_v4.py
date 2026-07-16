# _v4 | model: it200 = results/p2/final_b02/final.pt (GRAND FINAL walled: pretrained_a32uni -> 8-plug WALLED scene, EMERGENT-gamma curriculum (--emergent-gamma: all 7 gammas gathered uniformly, zero-certified gammas don't block the update; gamma0.2 joined 0->16% valid2), faithful no-rec recipe: frozen enc, lr 2e-5, beta .2, mix .4/.6, q .30, rollouts 28, gp=qbuf 500, demo .125 + LwF .05 on OPEN dr05 demos, start-eps .05, reach .2, 200 iters) | data: expert=results/expert_gt_walls8 M100 (SAME walled scene, start-eps .05, reach .15 — like-for-like re-baseline 2026-07-14; walls squeeze the expert too: clr .232-.270 vs .281-.333 open, g0.1 SR .91), ours=results/p2/eval_final_b02_it200 M40, kazuki DETUNED=results/kazuki_walls8_w09 (w_safe .9 on untouched pretrained, WALLED: SR 0.00 at all 3 gammas — guidance stuck, all timeouts; clearance undefined), pretrained=results/p2/eval_pretrained_it0 M40 walled | layout: 1x2 phase planes (SR-CR and clearance-time), marker=method, color=gamma (plasma_trunc colorbar outside — NEVER viridis, that's sigma's), legend outside; KEY RESULT: ours is SAFER (higher clearance) than the walled expert at ALL 7 gammas
"""'Proof' phase planes on the WALLED scene: (left) reliability SR vs CR — ideal corner (100, 0);
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

KAZ_DET_DIR = "results/kazuki_g47"           # TUNED walled Kazuki @ start-eps 0.3 (SR 0.3-0.55)
KAZ_DET_LABEL = r"CFM-MPPI$^{*}$"

# start-eps 0.3 (cleared-start) big dive; all baselines re-run on the SAME start-eps-0.3 scene
SERIES = [
    ("Expert", "results/expert_g47", "o", 95, dict()),
    ("Our approach", "results/p2/eval_faithful_it100", "*", 240, dict(edgecolors="k", linewidths=0.9)),
    ("Pretrained", "results/p2/eval_pretrained_g47", "s", 70, dict(alpha=0.55)),
]
# ablation brothers (add if their eval dirs exist)
BROTHERS = [(r"$-$SOCP", "results/p2/eval_faithbro_nosocp", "P", 80, dict(alpha=.8)),
            (r"$-$Progress", "results/p2/eval_faithbro_noprog", "X", 80, dict(alpha=.8)),
            (r"$-$Curriculum", "results/p2/eval_faithbro_nocur", "D", 60, dict(alpha=.8))]


def rows(d):
    out = {}
    for f in glob.glob(os.path.join(P2, d, "row_g*.json")):
        r = json.load(open(f)); out[round(float(r["gamma"]), 2)] = r
    return out


def main():
    import numpy as np
    series = list(SERIES)
    if KAZ_DET_DIR:
        series.insert(3, (KAZ_DET_LABEL, KAZ_DET_DIR, "v", 95, dict()))
    # safety-reliability tradeoff: the SAFE-tuned Kazuki (ws0.72) reaches ours' clearance but TRAPS (SR↓)
    if os.path.isdir(os.path.join(P2, "results/kazuki_g47_trap")):
        series.insert(4, (r"CFM-MPPI$^{*}$ (safe-tuned)", "results/kazuki_g47_trap", "^", 110,
                          dict(edgecolors="k", linewidths=0.6)))
    for br in BROTHERS:                                   # include brothers only when evaluated
        if os.path.isdir(os.path.join(P2, br[1])):
            series.append(br)
    fig, (aR, aQ) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    for name, d, mark, size, kw in series:
        R = rows(d)
        for g in GAMMAS:
            if g not in R:
                continue
            r = R[g]; c = [VIR(NORM(g))]
            aR.scatter(r["SR"] * 100, r["CR"] * 100, c=c, marker=mark, s=size, zorder=3, **kw)
            if r.get("clearance_mean") is not None and np.isfinite(r["clearance_mean"]):
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
        fig.savefig(os.path.join(HERE, f"scatter_v4.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote scatter_v4.png/.pdf")


if __name__ == "__main__":
    main()
