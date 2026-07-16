# _v1 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt | recipe: corrected guarded unit (lr 2e-5, frozen encoder, 1 step/iter, fresh certified gathers cap 300, recovery-starts 0.3, hard-quota 12 + oob-x0 pairing, escape-replay 64 = certified t104-preservation, teacher/anchor ratcheted at generation boundaries, trust gates 2.5%/1.6%, beta .3, q .5 absolute, OPEN scene) | data: expert=results/expert_gt (M100), kazuki=results/kazuki_final_m200 (M200, tuned w_safe=.3 coll_w=20 goal_w=2.0 goal_coef=.5, gamma-conditioned), ours=results/p2/eval_it146_m100 (M100), ablations = SAME-SCENE retrains results/p2/openabl_{nosocp_s861,noprog_s862,nocur_s863} (branch the s792 base at abs it134, 12 updates to it146 in the IDENTICAL recipe with exactly one component removed; internals figures/internals_openabl_it146.png) evaluated M100 open scene at results/p2/eval_openabl_{nosocp,noprog,nocur}_m100 — replaces _v0's provisional WALLED rows
"""IEEE double-column .tex table: gamma {0.1, 0.5, 1.0}; rows = expert / Kazuki / ours(it146) / -SOCP / -progress / -curriculum (ablations now open-scene, exact-recipe)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GS = ["0.1", "0.5", "1.0"]

ROWS = [
    ("Demo expert (SafeMPPI)", "results/expert_gt", ""),
    ("Kazuki guidance", "results/kazuki_final_m200", ""),
    ("Ours (full, it146)", "results/p2/eval_it146_m100", ""),
    (r"\;-- w/o SOCP", "results/p2/eval_openabl_nosocp_m100", ""),
    (r"\;-- w/o progress", "results/p2/eval_openabl_noprog_m100", ""),
    (r"\;-- w/o curriculum", "results/p2/eval_openabl_nocur_m100", ""),
]


def cell(d, g):
    f = os.path.join(P2, d, f"row_g{g}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def main():
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Safe flow expansion vs.\ baselines and ablations "
         r"(faithful deployment, reach $0.1$\,m; $M{\ge}100$ episodes per $\gamma$). Clearance is the "
         r"successful-episode mean nearest-obstacle distance.}", r"\label{tab:main}",
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
        ct = " & ".join(f"{c['clearance_mean']:.2f}/{c['time_mean_s']:.1f}" if c else "--" for c in cs)
        L.append(f"{name} & {cov} & {sr} & {ct} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", "",
          r"\vspace{2pt}{\raggedright\scriptsize Ablations: the final 12 updates of the identical "
          r"guarded recipe (abs.\ it.\ $134{\to}146$, same base and trust gates) with exactly one "
          r"component removed; same open scene, same faithful evaluation.\par}",
          r"\end{table}"]
    out = os.path.join(HERE, "table_v1.tex")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
