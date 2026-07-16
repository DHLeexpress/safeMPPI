"""GREEDY internals: what the hill-climb actually optimized, per step, from greedy_log.jsonl.

Row 1: the PAIRED a-d the selection scored each step (SR / CR / clearance / time / coverage), baseline
(grey) vs promoted winner (blue), with the eval-M annotated (M8 until it45, M20 after — the switch is
drawn as a vertical line). Row 2: the chosen knobs each step (beta, frontier%) + net gamma-cells improved.
This is the honest record: it shows exactly how much each metric moved and which config was picked.

  python analysis/greedy_internals.py --log results/p2/greedy_gf_s870/greedy_log.jsonl \
      --out grand_final_reports/greedy_internals.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 11})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(P2, "results/p2/greedy_gf_s870/greedy_log.jsonl"))
    ap.add_argument("--out", default=os.path.join(P2, "grand_final_reports/greedy_internals.png"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.log) if l.strip()]
    if not recs:
        print("no greedy steps yet"); return
    it = [r["iter"] for r in recs]
    ms = [r.get("eval_M", 8) for r in recs]
    switch = next((r["iter"] for r in recs if r.get("eval_M", 8) >= 20), None)

    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    (aSR, aCR, aClr, aTime), (aCov, aBeta, aFront, aCells) = axes

    def panel(ax, key, title, better):
        base = [r["baseline"][key] for r in recs]
        win = [r["winner"][key] for r in recs]
        ax.plot(it, base, "-o", color="#999999", ms=4, label="baseline (prev ckpt)")
        ax.plot(it, win, "-o", color="#0077bb", ms=5, label="promoted")
        ax.set_title(title); ax.grid(alpha=0.3); ax.set_xlabel("iteration")
        if switch:
            ax.axvline(switch - 0.5, color="#cc3311", ls="--", lw=1.2)

    panel(aSR, "SR", "SR (pooled 3$\\gamma$) — higher better", True)
    aSR.legend(loc="best")
    panel(aCR, "CR", "CR — lower better", False)
    panel(aClr, "clr", "clearance [m] — higher better", True)
    panel(aTime, "time", "time [s] — lower better", False)
    panel(aCov, "cov", "coverage (modes) — context", True)

    beta = [r["chosen_beta"] for r in recs]
    front = [r["chosen_frontier"] for r in recs]
    strict = [r.get("strict", False) for r in recs]
    aBeta.step(it, beta, where="mid", color="#ee7733", lw=2)
    aBeta.scatter(it, beta, c=["#009988" if s else "#cc3311" for s in strict], s=45, zorder=5)
    aBeta.set_title("chosen $\\beta$ (green=STRICT, red=best-effort)"); aBeta.grid(alpha=0.3)
    aBeta.set_xlabel("iteration"); aBeta.set_ylabel(r"$\beta$")
    aFront.step(it, front, where="mid", color="#33bbee", lw=2)
    aFront.scatter(it, front, c="#33bbee", s=35, zorder=5)
    aFront.set_title("chosen frontier fraction"); aFront.grid(alpha=0.3)
    aFront.set_xlabel("iteration"); aFront.set_ylabel("frontier %")
    net = [r.get("gamma_cells", {}).get("net", 0) for r in recs]
    aCells.bar(it, net, color=["#009988" if n > 0 else "#cc3311" for n in net])
    aCells.axhline(0, color="k", lw=0.8)
    aCells.set_title("net $\\gamma$-cells improved (of 12: 3$\\gamma\\times$4 metrics)")
    aCells.grid(alpha=0.3); aCells.set_xlabel("iteration")

    for ax in (aBeta, aFront):
        if switch:
            ax.axvline(switch - 0.5, color="#cc3311", ls="--", lw=1.2)
    mtxt = f"eval M: {ms[0]}" + (f" -> 20 at it{switch}" if switch else "")
    fig.suptitle(f"Greedy hill-climb internals — paired 3$\\gamma$ (0.1/0.5/1.0) a-d, {mtxt}; "
                 "red dashed = M8->M20 switch", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=125, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
