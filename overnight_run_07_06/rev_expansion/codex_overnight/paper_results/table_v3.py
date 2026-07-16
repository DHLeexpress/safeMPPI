# _v3 | model: GRAND FINAL = results/p2/openscratch_base_s870/final.pt (pretrained_a32uni -> OPEN scene FROM SCRATCH, walls4_phased recipe ADAPTED to the open scene: FROZEN encoder (open scene = pretraining distribution; the unfrozen variant collapses by it40 — 4/4 runs, see PROGRESS 07-12 evening), t104-proven update magnitudes lr 2e-5 + single inner step + min-modes 2 (lr 1e-4 x2 collapses on the open scene regardless of encoder freezing — walls geometry funnels data, open scene does not), cap 600, trust gates off, phased curriculum .85/2, perp-brake targeting, recovery bands + hard-quota, 100 iters) | data: expert=results/expert_gt M100, ours=results/p2/eval_grandfinal_m100 M100, kazuki DETUNED=results/kazuki_sweep_smoke/w09 (M=10, w_safe .3->.9 on untouched pretrained; tuned variant removed per user), pretrained=results/p2/eval_pretrained_m25 M25, ablations=FROM-SCRATCH same-recipe arms results/p2/eval_openscratch_{nosocp,noprog,nocur}_m100
"""IEEE double-column .tex table _v2: gamma {0.1,0.5,1.0}; adds the detuned-Kazuki fragility row."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GS = ["0.1", "0.5", "1.0"]

ROWS = [
    ("Demo expert (SafeMPPI)", "results/expert_gt", ""),
    (r"Kazuki guidance ($w_{s}{=}.9$)$^{\ddagger}$", "results/kazuki_sweep_smoke/w09", ""),
    ("Ours GRAND FINAL (from scratch)", "results/p2/eval_grandfinal_m100", ""),
    (r"\;-- w/o SOCP$^{\dagger}$", "results/p2/eval_openscratch_nosocp_m100", ""),
    (r"\;-- w/o progress$^{\dagger}$", "results/p2/eval_openscratch_noprog_m100", ""),
    (r"\;-- w/o curriculum$^{\dagger}$", "results/p2/eval_openscratch_nocur_m100", ""),
]


def cell(d, g):
    f = os.path.join(P2, d, f"row_g{g}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def main():
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Safe flow expansion vs.\ baselines and ablations "
         r"(faithful deployment, reach $0.1$\,m). Clearance is the successful-episode mean "
         r"nearest-obstacle distance.}", r"\label{tab:main}",
         r"\setlength{\tabcolsep}{3.2pt}\footnotesize",
         r"\begin{tabular}{l c ccc ccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{SR\,\% / CR\,\%} & \multicolumn{3}{c}{clearance [m] / time [s]} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Method & Cov. & $\gamma{=}.1$ & $\gamma{=}.5$ & $\gamma{=}1$ & $\gamma{=}.1$ & $\gamma{=}.5$ & $\gamma{=}1$ \\",
         r"\midrule"]
    for name, d, _tag in ROWS:
        cs = [cell(d, g) for g in GS]
        if not any(cs):
            continue
        covs = [c["coverage"] for c in cs if c]
        cov = f"{min(covs)}--{max(covs)}" if covs else "--"
        sr = " & ".join(f"{c['SR']*100:.0f}/{c['CR']*100:.0f}" if c else "--" for c in cs)
        ct = " & ".join(f"{c['clearance_mean']:.2f}/{c['time_mean_s']:.1f}" if c and c.get('clearance_mean') is not None else "--" for c in cs)
        L.append(f"{name} & {cov} & {sr} & {ct} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\raggedright\scriptsize $^{\ddagger}$Single guidance weight moved from its tuned "
          r"value ($M{=}10$ probe): the safety guidance over-dominates and the planner stalls in local "
          r"minima --- reward-fragility of guidance-based deployment. $^{\dagger}$Short-window ablations "
          r"(final 12 updates, identical recipe, one component removed); full from-scratch ablation "
          r"retrains in progress.\par}",
          r"\end{table}"]
    out = os.path.join(HERE, "table_v3.tex")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
