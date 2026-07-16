# _v6 | PARADIGM CHANGE (2026-07-16): pure AFE-minimal Safe Flow Expansion (see rollouts_v6.py header
# for the full method line). ours = eval_afe_pure_pi_s910 (pi-execution among verified-safe: the
# diversity lever, covS 52); brothers = eval_afe_bro_{noverif,nofallback,noprox} (matched pi-rule, one
# gate off each); CFM-MPPI* = kazuki_g47 (goal-tuned) and kazuki_g47_trap (safety-calibrated w_safe
# .72 — reaches our clearance but traps: the safety-reliability trap); curriculum recipe (prev. Ours,
# faithful_div_it100) kept as a grey reference; expert = expert_g47 M100; pretrained M40.
# | layout: 1x2 phase planes (SR-CR, clearance-time), marker=method, color=gamma (plasma_trunc,
# NEVER viridis — that is sigma's colormap in the expansion video/internals).
"""'Proof' phase planes, pure-AFE paradigm: (left) reliability SR vs CR — ideal corner (100, 0);
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
VIR = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(matplotlib.colors.Normalize()(range(256)))[10:225])
NORM = matplotlib.colors.Normalize(vmin=0.1, vmax=1.0)

SERIES = [
    ("Expert", "results/expert_g47", "o", 95, dict()),
    ("Our approach", "results/p2/eval_afe_pure_pi_s910", "*", 240, dict(edgecolors="k", linewidths=0.9)),
    ("Pretrained", "results/p2/eval_pretrained_g47", "s", 70, dict(alpha=0.55)),
    (r"CFM-MPPI$^{*}$", "results/kazuki_g47", "v", 95, dict()),
    (r"CFM-MPPI$^{*}$ (safety-calib.)", "results/kazuki_g47_trap", "^", 110,
     dict(edgecolors="k", linewidths=0.6)),
    ("Curriculum recipe (prev.)", "results/p2/eval_faithful_div_it100", "h", 70,
     dict(alpha=0.45, edgecolors="none")),
]
BROTHERS = [(r"$-$Verifier", "results/p2/eval_afe_bro_noverif", "P", 80, dict(alpha=.8)),
            (r"$-$Fallback", "results/p2/eval_afe_bro_nofallback", "X", 80, dict(alpha=.8)),
            (r"$-$Prox", "results/p2/eval_afe_bro_noprox", "D", 60, dict(alpha=.8))]


def rows(d):
    out = {}
    for f in glob.glob(os.path.join(P2, d, "row_g*.json")):
        r = json.load(open(f)); out[round(float(r["gamma"]), 2)] = r
    return out


def main():
    import numpy as np
    series = list(SERIES)
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
    fig.legend(handles=handles, loc="upper center", ncol=min(5, len(series)), frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=VIR, norm=NORM); sm.set_array([])
    cb = fig.colorbar(sm, ax=[aR, aQ], location="right", fraction=0.03, pad=0.015, ticks=GAMMAS)
    cb.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"scatter_v6.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote scatter_v6.png/.pdf")


if __name__ == "__main__":
    main()
