# _v0 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt | recipe: corrected guarded unit (lr 2e-5, frozen enc, recovery .3, hard-quota 12, escape 64, ratcheted teacher, trust gates, OPEN scene) | data: expert=results/expert_gt M100 (full gamma sweep), ours=results/p2/eval_it146_m100 M100, kazuki=results/kazuki_final_m200 M200 (tuned single w_safe=.3 with coll_w=20 goal_w=2.0 goal_coef=.5 beta_mppi=20 on OUR pretrained model, gamma-conditioned per point), pretrained=results/p2/eval_pretrained_m25 M25 (context series)
"""'Proof we won' scatter: 4 subplots (SR, CR, clearance, time). One color per method; gamma encoded by
marker for expert & ours & kazuki (all gamma-conditioned; Kazuki uses our pretrained model + guidance)."""
import json, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
MARKS = {0.1: "o", 0.2: "v", 0.3: "^", 0.4: "s", 0.5: "D", 0.7: "P", 1.0: "*"}
SERIES = [("Demo expert (target)", "results/expert_gt", "#000000"),
          ("Ours it146", "results/p2/eval_it146_m100", "#0077bb"),
          ("Kazuki guidance", "results/kazuki_final_m200", "#cc3311"),
          ("Pretrained (M25)", "results/p2/eval_pretrained_m25", "#bbbbbb")]


def rows(d):
    out = {}
    for f in glob.glob(os.path.join(P2, d, "row_g*.json")):
        r = json.load(open(f)); out[round(float(r["gamma"]), 2)] = r
    return out


def main():
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    panels = [("SR (%)", lambda r: r["SR"] * 100), ("CR (%)", lambda r: r["CR"] * 100),
              ("clearance (m)", lambda r: r["clearance_mean"]), ("time (s)", lambda r: r["time_mean_s"])]
    for ax, (title, fn) in zip(axes, panels):
        for name, d, c in SERIES:
            R = rows(d)
            for g in GAMMAS:
                if g not in R:
                    continue
                ax.scatter(g, fn(R[g]), color=c, marker=MARKS[g], s=110 if name.startswith("Ours") else 70,
                           edgecolors="k" if name.startswith("Ours") else "none", linewidths=0.8,
                           alpha=0.95 if name != "Pretrained (M25)" else 0.55, zorder=3)
        ax.set_title(title); ax.set_xlabel(r"$\gamma$"); ax.grid(alpha=0.3)
        ax.set_xticks(GAMMAS); ax.set_xticklabels([str(g) for g in GAMMAS], fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker="s", ls="", ms=9, label=n) for n, _, c in SERIES]
    axes[0].legend(handles=handles, fontsize=10, loc="lower right")
    axes[1].set_ylim(-0.5, max(axes[1].get_ylim()[1], 5))
    fig.suptitle("Per-$\\gamma$ deployment metrics — expert vs Kazuki (our pretrained + guidance, tuned "
                 "$w_{safe}{=}.3$, $c_w{=}20$, $g_w{=}2$, $g_c{=}.5$) vs OURS (it146)", fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"scatter_v0.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote scatter_v0.png/.pdf")


if __name__ == "__main__":
    main()
