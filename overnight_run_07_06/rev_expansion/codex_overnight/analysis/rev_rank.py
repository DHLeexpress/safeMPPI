"""Rank the REV sweep configs by FAITHFUL reach (min OOB+collision), keep the best, build the comparison
figure. Runs faithful_taxonomy on each config's final.pt (M30, 3 gamma), reads the RAW split, ranks by
pooled reach%. Writes grand_final_reports_rev/rev_sweep_compare.png + rev_sweep_log.json + copies the
winner ckpt to results/p2/rev_best.pt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
ENV = dict(os.environ, LD_LIBRARY_PATH="/home/dohyun/miniforge3/lib", OMP_NUM_THREADS="6",
           CUDA_VISIBLE_DEVICES="3")
GAMMAS = [0.1, 0.5, 1.0]
CONFIGS = ["rev_b3_f50_r3", "rev_b3_f75_r3", "rev_b2_f75_r3",
           "rev_b2_f50_r3", "rev_b3_f75_r5", "rev_b3_f25_r3"]
REVDIR = os.path.join(P2, "grand_final_reports_rev")
SWEEPDIR = os.path.join(REVDIR, "sweep")
CLR = {"reach": "#009988", "collision": "#cc3311", "oob": "#ee7733", "timeout": "#888888"}


def eval_tax(tag):
    ck = os.path.join(P2, "results/p2", tag, "final.pt")
    if not os.path.exists(ck):
        return None
    subprocess.run(["python", os.path.join(HERE, "faithful_taxonomy.py"), "--ckpt", ck,
                    "--tag", tag, "--M", "30", "--out-dir", SWEEPDIR], env=ENV,
                   capture_output=True, text=True, timeout=1200)
    jf = os.path.join(SWEEPDIR, f"taxonomy_{tag}.json")
    if not os.path.exists(jf):
        return None
    return json.load(open(jf))


def pooled(tax):
    r = {k: 0 for k in CLR}; M = 0
    for g in GAMMAS:
        t = tax.get(str(g)) or tax.get(str(float(g)))
        if not t:
            continue
        for k in CLR:
            r[k] += t[k]
        M += t["M"]
    return {k: 100.0 * r[k] / max(M, 1) for k in CLR}, M


def main():
    os.makedirs(SWEEPDIR, exist_ok=True)
    results = {}
    for tag in CONFIGS:
        tax = eval_tax(tag)
        if tax is None:
            print(f"{tag}: no final.pt / eval failed")
            continue
        pool, M = pooled(tax)
        results[tag] = dict(pool=pool, M=M, per_gamma=tax)
        print(f"{tag}: reach {pool['reach']:.0f}% | CR {pool['collision']:.0f}% | "
              f"OOB {pool['oob']:.0f}% | timeout {pool['timeout']:.0f}%")

    if not results:
        print("no results"); return
    # rank: max reach, then min (CR+OOB)
    order = sorted(results, key=lambda t: (results[t]["pool"]["reach"],
                                           -(results[t]["pool"]["collision"] + results[t]["pool"]["oob"])),
                   reverse=True)
    best = order[0]
    print(f"\nBEST: {best} (reach {results[best]['pool']['reach']:.0f}%)")
    import shutil
    shutil.copy(os.path.join(P2, "results/p2", best, "final.pt"), os.path.join(P2, "results/p2/rev_best.pt"))

    # comparison figure: stacked bars per config
    fig, ax = plt.subplots(figsize=(1.7 * len(order) + 2, 5.2))
    xs = np.arange(len(order)); bot = np.zeros(len(order))
    for cat in ("reach", "collision", "oob", "timeout"):
        vals = np.array([results[t]["pool"][cat] for t in order])
        ax.bar(xs, vals, bottom=bot, color=CLR[cat], label=cat)
        bot += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("rev_", "") for t in order], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% episodes (faithful, RAW, pooled 3$\\gamma$, M30)")
    ax.set_title("REV sweep (pretrained -> it10): frontier-early / low-easy / $\\beta$ + recovery — "
                 f"ranked by reach; BEST = {best.replace('rev_','')}")
    ax.legend(ncol=4, loc="lower center", fontsize=9)
    ax.axhline(results[best]["pool"]["reach"], color="#009988", ls="--", lw=1, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(REVDIR, "rev_sweep_compare.png"), dpi=130, bbox_inches="tight")
    json.dump(dict(order=order, best=best, results=results),
              open(os.path.join(REVDIR, "rev_sweep_log.json"), "w"), indent=1)
    print("wrote", os.path.join(REVDIR, "rev_sweep_compare.png"))


if __name__ == "__main__":
    main()
