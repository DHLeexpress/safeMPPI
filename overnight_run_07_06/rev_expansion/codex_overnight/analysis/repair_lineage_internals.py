"""Internals of the repair SEARCH itself: every gated one-step/short arm in chronological order.

Reads every analysis/fixed_seed_gate_*.json (each = one bounded experiment judged by the frozen t104 M25
per-seed baseline) and renders the checkpoint-indexed trajectory of aggregate M25 SR, fixed-failure flips
(0-11), and regressions. This is the proof-plot that the promotion rule (flips WITH zero regressions) was
enforced across the whole search, and how s671/s766 emerged.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)

MILESTONES = {"origin_tightgate_production_s671": "s671\norigin fixed",
              "goal_brake_gammaaug_s766": "s766\n11/11 flips",
              "goal_gamma1_brake_focus_s790": "s790\nreject",
              "hardtail108": "ht108\nfirst flips"}


def main(out=os.path.join(P2, "figures", "internals_repair_lineage.png")):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "fixed_seed_gate_*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("per_gamma_SR"):
            continue
        name = re.sub(r"^fixed_seed_gate_|\.json$", "", os.path.basename(f))
        srs = [v["SR"] for v in d["per_gamma_SR"].values()]
        flips = int(str(d.get("fixed_flipped", "0/11")).split("/")[0])
        rows.append(dict(name=name, t=os.path.getmtime(f), agg=float(np.mean(srs)),
                         mn=float(np.min(srs)), flips=flips, regs=int(d.get("n_regressions", -1)),
                         gate=bool(d.get("gate_pass", False))))
    if not rows:
        raise SystemExit("no gate jsons")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows))
    agg = [r["agg"] for r in rows]; mn = [r["mn"] for r in rows]
    flips = [r["flips"] for r in rows]; regs = [r["regs"] for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(20, 9), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    a1.axhline(0.937, color="#888", ls="--", lw=1)
    a1.text(0.2, 0.939, "t104 baseline .937", fontsize=9, color="#666")
    a1.axhline(1.0, color="green", ls=":", lw=1)
    a1.plot(x, agg, "-o", ms=3.5, lw=1.2, color="#4477aa", label="aggregate M25 SR")
    a1.plot(x, mn, "-s", ms=2.5, lw=0.8, color="#cc6677", alpha=0.7, label="worst-γ SR")
    for i, r in enumerate(rows):
        for key, lab in MILESTONES.items():
            if key in r["name"]:
                a1.annotate(lab, xy=(i, r["agg"]), xytext=(i, min(r["agg"] + 0.06, 1.06)),
                            fontsize=9, ha="center", color="darkgreen" if "s766" in key or "s671" in key else "black",
                            arrowprops=dict(arrowstyle="->", lw=0.8))
    a1.set_ylim(0.3, 1.09); a1.set_ylabel("M25 SR"); a1.legend(loc="lower right", fontsize=10)
    a1.set_title(f"Repair-search internals — {len(rows)} gated bounded experiments (chronological), "
                 "judged per-seed vs the frozen t104 M25 archive", fontsize=13)
    a2.bar(x - 0.2, flips, 0.4, color="#009988", label="fixed-failure flips (of 11)")
    a2.bar(x + 0.2, regs, 0.4, color="#cc3311", label="new regressions")
    a2.axhline(11, color="#009988", ls=":", lw=1)
    a2.set_ylabel("count"); a2.set_ylim(0, 12.5); a2.legend(loc="upper left", fontsize=10)
    a2.set_xlabel("gated experiment (chronological)")
    step = max(1, len(rows) // 28)
    a2.set_xticks(x[::step])
    a2.set_xticklabels([rows[i]["name"][:22] for i in range(0, len(rows), step)],
                       rotation=60, ha="right", fontsize=7)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"{len(rows)} experiments -> {out}")
    best = max(rows, key=lambda r: (r["flips"] - 100 * max(r["regs"], 0), r["agg"]))
    print("best by (flips, no-regs):", best["name"], best)


if __name__ == "__main__":
    main()
