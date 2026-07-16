"""Internals proof-plot: per-iteration unit telemetry (probe.jsonl) + per-step losses of the bounded arms.

Row 1: guarded-unit generations stitched — CFM loss, per-step functional drift, cumulative anchor drift
       (with the 1.6% bound and rollbacks marked), M5 gate SR and SR50 probe.
Row 2: the bounded one-step arms' loss_steps curves (s791..s799) + their gate outcomes.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)

UNIT_DIRS = sorted(glob.glob(os.path.join(P2, "results/p2/unit_s792_esc64_s801"))) + \
    sorted(glob.glob(os.path.join(P2, "results/p2/unit_ratchet_gen*")))
ARMS = [("s791 γ1-focus", "goal_gamma1_preserve_t104_s791"),
        ("s792 all-γ+preserve", "goal_allg_preserve_lowdose_s792"),
        ("s794 band", "goal_topband_preserve_s794"),
        ("s795 band 10×", "goal_topband_dose2_s795"),
        ("s799 oracle-brake", "goal_oraclebrake_s799")]


def main(out=os.path.join(P2, "figures", "internals_unit_and_arms.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    (a1, a2, a3), (b1, b2, b3) = axes

    xoff = 0; bounds = []
    L, F, A, M5, S50, XI = [], [], [], [], [], []
    m5x, s50x = [], []
    for d in UNIT_DIRS:
        pj = os.path.join(d, "probe.jsonl")
        if not os.path.exists(pj):
            continue
        recs = [json.loads(l) for l in open(pj)]
        for r in recs:
            i = xoff + len(L)
            L.append(r.get("loss")); F.append(r.get("functional_step"))
            A.append(r.get("anchor_drift")); XI.append(r.get("iter"))
            if r.get("sr50") is not None:
                S50.append(r["sr50"]); s50x.append(i)
        bounds.append(xoff + len(L))
    x = np.arange(len(L))
    Lp = [v if (v is not None and np.isfinite(v)) else np.nan for v in L]
    a1.plot(x, Lp, "-o", ms=3, color="#4477aa")
    a1.set_title("unit CFM+aux loss per iteration (gens stitched)"); a1.set_ylabel("loss")
    a2.plot(x, [v if v else np.nan for v in F], "-o", ms=3, color="#009988", label="per-step fstep")
    a2.plot(x, [v if v else np.nan for v in A], "-s", ms=3, color="#cc3311", label="cumulative anchor")
    a2.axhline(0.016, color="#cc3311", ls="--", lw=1); a2.axhline(0.025, color="#009988", ls=":", lw=1)
    for bx in bounds[:-1]:
        a2.axvline(bx - 0.5, color="k", lw=0.8, ls=":")
    a2.legend(fontsize=9); a2.set_title("trust telemetry (dashed = bounds; dotted verticals = ratchet)")
    a3.plot(s50x, S50, "-o", ms=4, color="#4477aa", label="SR50 probe (γ.5, M50)")
    a3.set_ylim(0, 1.05); a3.legend(fontsize=9); a3.set_title("stability probes")
    for ax in (a1, a2, a3):
        ax.set_xlabel("unit iteration (stitched)")

    for name, tag in ARMS:
        f = os.path.join(P2, f"results/p2/{tag}.json")
        if not os.path.exists(f):
            continue
        st = json.load(open(f))["stats"]
        b1.plot(st.get("loss_steps", []), "-o", ms=3, label=name)
        b2.plot(np.array(st.get("anchor_steps", [])) * 100, "-o", ms=3, label=name)
    b1.set_title("bounded arms — loss per optimizer step"); b1.legend(fontsize=8); b1.set_xlabel("step")
    b2.axhline(1.6, color="k", ls="--", lw=1)
    b2.set_title("bounded arms — cumulative anchor drift (%)"); b2.legend(fontsize=8); b2.set_xlabel("step")

    gates = []
    for f in sorted(glob.glob(os.path.join(HERE, "fixed_seed_gate_*.json")), key=os.path.getmtime):
        d = json.load(open(f))
        if not d.get("per_gamma_SR"):
            continue
        nm = os.path.basename(f)[16:-5]
        if any(t in nm for t in ("s791", "s792", "s794", "s795", "s799", "unit_g1")):
            srs = [v["SR"] for v in d["per_gamma_SR"].values()]
            gates.append((nm[:24], float(np.mean(srs)),
                          int(str(d["fixed_flipped"]).split("/")[0]), d["n_regressions"]))
    if gates:
        xx = np.arange(len(gates))
        b3.bar(xx - 0.2, [g[2] for g in gates], 0.4, color="#009988", label="flips (of 11)")
        b3.bar(xx + 0.2, [g[3] for g in gates], 0.4, color="#cc3311", label="regressions")
        for i, g in enumerate(gates):
            b3.text(i, 11.4, f"{g[1]*100:.0f}%", ha="center", fontsize=8)
        b3.set_xticks(xx); b3.set_xticklabels([g[0] for g in gates], rotation=45, ha="right", fontsize=7)
        b3.set_ylim(0, 13); b3.legend(fontsize=8, loc="center left")
        b3.set_title("gate outcomes (top: aggregate M25 SR)")
    fig.suptitle("Internals — every update is measured: losses, trust bounds, rollbacks, gates", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
