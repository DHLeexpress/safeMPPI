"""GRAND FINAL internals — pretrained -> OPEN scene, walls4_phased recipe from it0
(FROZEN encoder + t104-proven lr 2e-5 x1 step (both unfrozen and lr 1e-4 collapse on the open scene), cap 600, phased curriculum .85/2, perp-brake; 100 iters).
Grey = the SAME recipe on the walled scene (walls4_phased_s830) as reference.
8 panels: loss | field grad | enc grad | trust telemetry | batch composition | SR | CR | coverage.
Usage: python analysis/grand_final_internals.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
                            "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)

RUNS = [
    ("GRAND FINAL: open scene from scratch (phased+perp)", "#0077bb",
     ["results/p2/openscratch_base_s870"]),
    ("walls-4 reference (same recipe, walled scene)", "#bbbbbb",
     ["results/p2/walls4_phased_s830"]),
]


def load(dirs):
    recs, hist = [], []
    for d in dirs:
        pj = os.path.join(P2, d, "probe.jsonl")
        if os.path.exists(pj):
            recs += [json.loads(l) for l in open(pj)]
        hj = os.path.join(P2, d, "history.json")
        if os.path.exists(hj):
            hist += json.load(open(hj))
    seen = {r["iter"]: r for r in recs}
    recs = [seen[k] for k in sorted(seen)]
    hseen = {h["iter"]: h for h in hist}
    hist = [hseen[k] for k in sorted(hseen)]
    return recs, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P2, "grand_final_reports", "internals_latest.png"))
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    (aL, aF, aE, aT), (aB, aS, aC, aP) = axes
    for name, color, dirs in RUNS:
        recs, hist = load(dirs)
        if not recs:
            continue
        x = [r["iter"] for r in recs]
        kw = dict(color=color)
        lw = 2.2 if color == "#0077bb" else 1.4
        al = 1.0 if color == "#0077bb" else 0.65
        aL.plot(x, [r.get("loss") for r in recs], "-o", ms=3.5, lw=lw, alpha=al, label=name, **kw)
        aF.plot(x, [r.get("fld") for r in recs], "-o", ms=3.5, lw=lw, alpha=al, **kw)
        aE.plot(x, [r.get("enc") for r in recs], "-o", ms=3.5, lw=lw, alpha=al, **kw)
        aT.plot(x, [100 * (r.get("functional_step") or 0) for r in recs], "-o", ms=3, lw=lw, alpha=.8 * al, **kw)
        aT.plot(x, [100 * (r.get("anchor_drift") or 0) for r in recs], "--s", ms=3, lw=lw, alpha=al, **kw)
        be = np.array([r.get("batch_e") or 0 for r in recs], float)
        bf = np.array([r.get("batch_f") or 0 for r in recs], float)
        bh = np.array([r.get("batch_hard") or 0 for r in recs], float)
        bd = np.array([r.get("batch_d") or 0 for r in recs], float)
        tot = np.maximum(be + bf + bh + bd, 1)
        aB.plot(x, bf / tot, "-o", ms=3, lw=lw, alpha=al, **kw)
        aB.plot(x, bh / tot, ":^", ms=3, lw=lw, alpha=.8 * al, **kw)
        sx = [r["iter"] for r in recs if r.get("sr50") is not None]
        aS.plot(sx, [r["sr50"] for r in recs if r.get("sr50") is not None], ":^", ms=5, lw=1.4, alpha=.9 * al, **kw)
        aC.plot(sx, [r["cr50"] for r in recs if r.get("sr50") is not None], ":^", ms=5, lw=1.4, alpha=.9 * al, **kw)
        aP.plot(sx, [r.get("cov50") for r in recs if r.get("sr50") is not None], "-o", ms=4, lw=lw, alpha=al, **kw)
        if hist:
            hx = [h["iter"] for h in hist]
            aS.plot(hx, [h["SR"] for h in hist], "-o", ms=5, lw=lw, alpha=al, **kw)
            aC.plot(hx, [h["CR"] for h in hist], "-o", ms=5, lw=lw, alpha=al, **kw)
    aL.set_title("cfm + aux loss (fresh data each iter)"); aL.legend(loc="upper right")
    aF.set_title("field grad-RMS (aggressiveness)")
    aE.set_title("encoder grad-RMS (FROZEN on open scene; unfrozen collapsed)")
    aT.set_title("functional step % (solid) / teacher drift % (dashed)\ntrust gates OFF (from-scratch recipe)")
    aB.set_title("batch portions: frontier (solid), hard-strip (dotted)\nphase switch = frontier share leaves 0")
    aS.axhline(1.0, color="green", ls=":", lw=1); aS.set_ylim(-0.02, 1.03)
    aS.set_title("SR from origin — M5 gate (solid), SR50 probe (dotted)")
    aC.set_ylim(-0.02, 1.03); aC.set_title("CR from origin (M5 solid, CR50 dotted)")
    aP.set_title("coverage: cov50 distinct staircase modes")
    for ax in axes.flat:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    fig.suptitle("GRAND FINAL internals — pretrained → OPEN scene, from-scratch phased+perp-brake recipe "
                 "(100 iters); grey = same recipe on walled scene (reference)", fontsize=18)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
