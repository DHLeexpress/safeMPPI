"""AFE2 diagnostics report (user spec 2026-07-16b): two arms (prox control vs afe deep update) on
identical acquisition/representation/seeds. Panels:
  A pooled expert-free controller SR / NVP / CR      B per-gamma controller SR
  C untilted raw validity per gamma (audit)          D update: CFM loss, per-module grad norms,
                                                       relative parameter change, fixed-probe
                                                       representation cosine drift (NO "encoder loss")
  E uncertainty: all-K vs selected-B sigma medians, A effective rank
  F acquisition: ESS/K, normalized entropy, selected-vs-pool uplift
  G data: |D|, |D+|, distinct trained, dither share   H text: final per-gamma table with Wilson CIs
Usage: python analysis/afe2_report.py --arms results/afe2/prox_s910 results/afe2/afe_s910
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
PLA = plt.get_cmap("plasma")
GC = {g: PLA(0.08 + 0.77 * i / 6) for i, g in enumerate(GAMMAS)}
LS = {0: "-", 1: "--"}


def wilson(p, n, z=1.0):
    if n <= 0:
        return 0.0, 0.0
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, p - (ctr - hw)), max(0.0, (ctr + hw) - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", default="paper_results/afe2_report_v1.png")
    args = ap.parse_args()
    arms = []
    for a in args.arms:
        recs = [json.loads(l) for l in open(os.path.join(a, "probe.jsonl"))]
        arms.append((os.path.basename(a.rstrip("/")).replace("_s910", ""), recs))

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.6))

    ax = axes[0, 0]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for key, c in (("SR", "#009944"), ("NVP", "#cc7711"), ("CR", "#cc3311")):
            ys = [r["ctrl_pooled"][key] for r in recs]
            ax.plot(R, ys, LS[ai], color=c, lw=2.0, marker="o", ms=3,
                    label=(key if ai == 0 else None))
    ax.set_title("(A) expert-free verified controller (pooled)\nSR / NO_VERIFIED_POSITIVE / CR")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.5, " / ".join(f"{LS[ai]} = {lab}" for ai, (lab, _) in enumerate(arms)),
            transform=ax.transAxes, fontsize=9, color="dimgray")

    ax = axes[0, 1]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for g in GAMMAS:
            ys = [r["ctrl"][g]["SR"] for r in recs]
            ax.plot(R, ys, LS[ai], color=GC[g], lw=1.3, label=(f"γ{g}" if ai == 0 else None))
    ax.set_title("(B) controller SR per γ"); ax.set_xlabel("round")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 2]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for g in GAMMAS:
            ys = [r["V_gamma"][g] for r in recs]
            ax.plot(R, ys, LS[ai], color=GC[g], lw=1.3, label=(f"γ{g}" if ai == 0 else None))
        ax.plot(R, [r.get("V_adverse") for r in recs], LS[ai], color="k", lw=1.8,
                label=("adverse (pooled)" if ai == 0 else None))
    ax.set_title("(C) untilted raw validity per γ (fixed ρ_eval audit)")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 3]
    ax2 = ax.twinx()
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("cfm") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["cfm"] for r in rr], LS[ai], color="#D55E00", lw=1.8,
                label=("CFM loss" if ai == 0 else None))
        for kg, c in (("E_g", "#4477aa"), ("trunk", "#009988"), ("head", "#aa3377")):
            ax2.plot(R, [r["grad_norm"].get(kg, np.nan) for r in rr], LS[ai], color=c, lw=1.0,
                     alpha=0.8, label=(f"|grad| {kg}" if ai == 0 else None))
    ax2.set_yscale("log"); ax2.set_ylabel("per-module grad norm (log)")
    ax.set_title("(D) update: total CFM loss + encoder/trunk/head grad norms")
    ax.set_xlabel("round"); ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

    ax = axes[1, 0]
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("rep_cos") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["rep_cos"] for r in rr], LS[ai], color="#0b3d91", lw=1.8,
                label=("rep cosine vs φ⁰ (fixed probe)" if ai == 0 else None))
        rr2 = [r for r in recs if r.get("rel_param_change")]
        for kg, c in (("E_g", "#4477aa"), ("trunk", "#009988"), ("head", "#aa3377")):
            ax.plot([r["round"] for r in rr2],
                    [10 * r["rel_param_change"].get(kg, np.nan) for r in rr2], LS[ai], color=c,
                    lw=1.0, alpha=0.8, label=(f"10×Δθ/θ {kg}" if ai == 0 else None))
    ax.set_title("(E) representation drift + relative parameter change")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("ess_med") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["ess_med"] for r in rr], LS[ai], color="#117733", lw=1.8,
                label=("median ESS/K" if ai == 0 else None))
        ax.plot(R, [r["ent_med"] for r in rr], LS[ai], color="#88ccee", lw=1.4,
                label=("median entropy/log K" if ai == 0 else None))
        ax.plot(R, [r["uplift_med"] for r in rr], LS[ai], color="#999933", lw=1.4,
                label=("σ uplift (sel−pool)" if ai == 0 else None))
    ax.axhspan(0.25, 0.5, color="#117733", alpha=0.08)
    ax.set_title("(F) acquisition: ESS/K (band = calibration target), entropy, uplift")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax2 = ax.twinx()
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("sig_all_med") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["sig_all_med"] for r in rr], LS[ai], color="#440154", lw=1.6,
                label=("median σ all-K" if ai == 0 else None))
        ax.plot(R, [r["sig_sel_med"] for r in rr], LS[ai], color="#35b779", lw=1.6,
                label=("median σ selected-B" if ai == 0 else None))
        ax2.plot(R, [r["A_eff_rank"] for r in rr], LS[ai], color="#888888", lw=1.2)
    ax2.set_ylabel("effective rank of A (grey)", color="#666666")
    ax.set_title("(G) uncertainty distribution (evolving rep, A rebuilt/round)")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1, 3]
    ax.axis("off")
    lines = []
    for lab, recs in arms:
        last = recs[-1]
        M = 8
        lines.append(f"[{lab}] final round {last['round']}  D {last.get('n_D')}  D+ {last.get('n_Dpos')}")
        lines.append("  γ    SR(±)      NVP(±)     V̂raw   q/pos")
        for g in GAMMAS:
            c = last["ctrl"][g]
            lo, hi = wilson(c["SR"], M)
            lo2, hi2 = wilson(c["NVP"], M)
            pg = (last.get("per_gamma") or {}).get(g, {})
            lines.append(f"  {g}  {c['SR']:.2f}(−{lo:.2f}+{hi:.2f}) "
                         f"{c['NVP']:.2f}(−{lo2:.2f}+{hi2:.2f})  "
                         f"{last['V_gamma'][g]:.2f}   {pg.get('n_q', '-')}/{pg.get('n_pos', '-')}")
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=8, va="top")
    ax.set_title("(H) final per-γ table (M=8 fixed-index rollouts, Wilson ±1σ)")

    fig.suptitle("AFE2 corrected two-arm study — evolving φ_s^(n), rebuilt A, expert-free "
                 "(NO fallback; NO_VERIFIED_POSITIVE terminates), β fixed by ESS calibration",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=135)
    print("saved", args.out)


if __name__ == "__main__":
    main()
