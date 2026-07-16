"""Paper scatter (2026-07-14): 'proof we won' phase planes at gamma {0.1,0.5,1.0}.
x = time (faster = LEFT), y = clearance (safer = UP). Expert (black) is the target; OURS (blue) wins by
sitting UP-and-LEFT of it (safer AND faster). Kazuki (orange) for reference. SR/CR annotated per point.
Reads row_g*.json from --ours-dir (walled eval) + expert_gt + kazuki (kept as-is).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)


def row(d, g):
    p = os.path.join(d, f"row_g{float(g)}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--expert-dir", default=os.path.join(P2, "results/expert_gt"))
    ap.add_argument("--kazuki-dir", default=os.path.join(P2, "results/kazuki_sweep_smoke/w09"))
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--out", default=os.path.join(P2, "paper_results/scatter_v4.png"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    series = [("Demo expert", args.expert_dir, "k", "o"),
              (r"CFM-MPPI$^{*}$", args.kazuki_dir, "#e69f00", "s"),
              ("Ours", args.ours_dir, "#0072B2", "^")]

    fig, axes = plt.subplots(1, len(args.gammas), figsize=(5.0 * len(args.gammas), 5.0))
    if len(args.gammas) == 1:
        axes = [axes]
    for ax, g in zip(axes, args.gammas):
        e = row(args.expert_dir, g)
        pts = []
        for name, d, c, m in series:
            r = row(d, g)
            if r is None or not np.isfinite(r.get("clearance_mean", np.nan)):
                continue
            pts.append((r["time_mean_s"], r["clearance_mean"]))
            ax.scatter([r["time_mean_s"]], [r["clearance_mean"]], c=c, marker=m, s=170,
                       edgecolors="k", linewidths=1.1, zorder=5, label=name)
            ax.annotate(f"SR {r['SR']*100:.0f} / CR {r['CR']*100:.0f}",
                        (r["time_mean_s"], r["clearance_mean"]), (6, -12), textcoords="offset points",
                        fontsize=8.5, color=c)
        # zoom to the data (clearances cluster 0.24-0.37); keep the axes tight so the gap is legible
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        xlo, xhi = min(xs) - 0.5, max(xs) + 0.9; ylo, yhi = min(ys) - 0.02, max(ys) + 0.03
        if e is not None:  # win region: safer (above expert clearance) AND faster (left of expert time)
            ax.axhline(e["clearance_mean"], color="k", ls=":", lw=1, alpha=.5)
            ax.axvline(e["time_mean_s"], color="k", ls=":", lw=1, alpha=.5)
            ax.fill_betweenx([e["clearance_mean"], yhi], xlo, e["time_mean_s"],
                             color="#009944", alpha=0.10, zorder=0)
            ax.text(0.03, 0.97, "beats expert\n(safer + faster)", transform=ax.transAxes, va="top",
                    fontsize=9, color="#007733")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("time to goal (s)  — faster →", fontsize=11)
        ax.set_ylabel("clearance (m)  — safer ↑", fontsize=11)
        ax.set_title(f"γ = {g:.1f}", fontsize=13)
        ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Proof of dominance: Ours vs demo expert and CFM-MPPI$^{*}$" + (f" — {args.note}" if args.note else ""),
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
