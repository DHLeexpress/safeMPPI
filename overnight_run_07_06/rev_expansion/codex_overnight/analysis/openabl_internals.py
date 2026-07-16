"""Report internals (Image-1 style, large fonts) for the OPEN-SCENE ablation suite at it146's recipe.

The three ablation arms branch from the SAME s792 base at abs it134 and run 12 iterations to it146 with
it146's exact guarded-unit recipe (lr 2e-5, frozen enc, trust gates 2.5%/1.6% WITH rollback, recovery .3,
hard-quota 12, escape 64) — differing ONLY by their single ablation flag. Black = the full-recipe lineage
(gen1 s801 + gen2 s802) that produced it146 over the same window; blue = the live gen3 continuation
(+perp-brake unification).
8 panels: cfm+aux loss | field grad-RMS | encoder grad-RMS | trust telemetry | batch composition |
SR from origin (M5 gate + SR50 probe) | CR from origin | coverage + goal-strip probe.
"""
from __future__ import annotations

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
    ("FULL recipe (it146 lineage, gen1+gen2)", "#000000",
     ["results/p2/unit_s792_esc64_s801", "results/p2/unit_ratchet_gen2_s802"]),
    ("(3.1) no multi-step SOCP", "#d62728", ["results/p2/openabl_nosocp_s861"]),
    ("(3.2) no progress condition", "#9467bd", ["results/p2/openabl_noprog_s862"]),
    ("(3.3) no curriculum", "#ff7f0e", ["results/p2/openabl_nocur_s863"]),
    ("gen3 continuation (live, +perp-brake)", "#0077bb", ["results/p2/unit146_gen3_s853"]),
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
    seen = {}
    for r in recs:
        seen[r["iter"]] = r
    recs = [seen[k] for k in sorted(seen)]
    hseen = {}
    for h in hist:
        hseen[h["iter"]] = h
    hist = [hseen[k] for k in sorted(hseen)]
    return recs, hist


def main(out=os.path.join(P2, "figures", "internals_openabl_it146.png")):
    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    (aL, aF, aE, aT), (aB, aS, aC, aP) = axes
    for name, color, dirs in RUNS:
        recs, hist = load(dirs)
        if not recs:
            continue
        x = [r["iter"] for r in recs]
        kw = dict(color=color) if color else {}
        lw = 2.4 if color == "#000000" else 1.8
        aL.plot(x, [r.get("loss") for r in recs], "-o", ms=3.5, lw=lw, label=name, **kw)
        aF.plot(x, [r.get("fld") for r in recs], "-o", ms=3.5, lw=lw, **kw)
        aE.plot(x, [r.get("enc") for r in recs], "-o", ms=3.5, lw=lw, **kw)
        aT.plot(x, [100 * (r.get("functional_step") or 0) for r in recs], "-o", ms=3, lw=lw, alpha=.85, **kw)
        aT.plot(x, [100 * (r.get("anchor_drift") or 0) for r in recs], "--s", ms=3, lw=lw, **kw)
        be = np.array([r.get("batch_e") or 0 for r in recs], float)
        bf = np.array([r.get("batch_f") or 0 for r in recs], float)
        bh = np.array([r.get("batch_hard") or 0 for r in recs], float)
        bd = np.array([r.get("batch_d") or 0 for r in recs], float)
        tot = np.maximum(be + bf + bh + bd, 1)
        aB.plot(x, bf / tot, "-o", ms=3, lw=lw, **kw)
        aB.plot(x, bh / tot, ":^", ms=3, lw=lw, alpha=.8, **kw)
        sx = [r["iter"] for r in recs if r.get("sr50") is not None]
        aS.plot(sx, [r["sr50"] for r in recs if r.get("sr50") is not None], ":^", ms=5, lw=1.4, alpha=.9, **kw)
        aC.plot(sx, [r["cr50"] for r in recs if r.get("sr50") is not None], ":^", ms=5, lw=1.4, alpha=.9, **kw)
        aP.plot(sx, [r.get("cov50") for r in recs if r.get("sr50") is not None], "-o", ms=4, lw=lw, **kw)
        spx = [r["iter"] for r in recs if r.get("strip_probe_goal") is not None]
        aP.plot(spx, [10 * r["strip_probe_goal"] for r in recs if r.get("strip_probe_goal") is not None],
                "--", lw=1.2, alpha=.7, **kw)
        if hist:
            hx = [h["iter"] for h in hist]
            aS.plot(hx, [h["SR"] for h in hist], "-o", ms=5, lw=lw, **kw)
            aC.plot(hx, [h["CR"] for h in hist], "-o", ms=5, lw=lw, **kw)
    aL.set_title("cfm + aux loss (fresh data each iter)"); aL.legend(loc="upper right")
    aF.set_title("field grad-RMS (aggressiveness)")
    aE.set_title("encoder grad-RMS (frozen: 0 by design)")
    aT.set_title("functional step % (solid) vs cumulative anchor drift % (dashed)\ntrust gates 2.5%/1.6% WITH rollback (guarded unit)")
    aB.set_title("batch portions: frontier (solid), hard-strip (dotted)")
    aS.axhline(1.0, color="green", ls=":", lw=1); aS.set_ylim(-0.02, 1.03)
    aS.set_title("SR from origin — M5 gate (solid), SR50 probe (dotted)")
    aC.set_ylim(-0.02, 1.03); aC.set_title("CR from origin (M5 solid, CR50 dotted)")
    aP.set_title("coverage: cov50 (solid) + 10×goal-strip win-OOB (dashed)")
    for ax in axes.flat:
        ax.set_xlabel("absolute iteration")
        ax.grid(alpha=0.25)
    fig.suptitle("OPEN-SCENE ablation internals — branch at it134 from the s792 base, 12 iters in it146's "
                 "EXACT recipe, one ablation flag each; black = full-recipe lineage over the same window",
                 fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
