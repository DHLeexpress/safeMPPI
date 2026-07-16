# _v2 | model: it146 = results/p2/unit_ratchet_gen2_s802/ckpt_146.pt (champion; gen3/gen4 perp-brake continuations gate-failed vs the it146 M25 baseline, not promoted) | recipe: corrected guarded unit (lr 2e-5, frozen encoder, 1 step/iter, cap 300, recovery .3, hard-quota 12 + oob-x0, escape 64, ratcheted teacher, trust 2.5%/1.6%, beta .3, q .5, OPEN scene) | data: expert=results/expert_gt (M100); Kazuki = kazuki_baseline.py (faithful Mizuta/Kazuki guided CFM+MPPI reimpl on our UNTOUCHED pretrained_a32uni.pt, gamma only via FM context): TUNED=results/kazuki_final_m200 (M200, w_safe=.3 coll_w=20 goal_w=2 goal_coef=.5 beta=20), DETUNED=results/kazuki_sweep_smoke/w09 (M=10 smoke, w_safe=.9 only knob moved -> guidance over-conservative, traps); ours=results/p2/eval_it146_m100 (M100); ablations=SHORT-WINDOW arms eval_openabl_* (12 updates it134->146, one flag off; PROVISIONAL — faithful from-scratch arms openscratch_s870-873 training, will replace)
"""IEEE double-column .tex table _v2: gamma {0.1,0.5,1.0}; adds the detuned-Kazuki fragility row."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GS = ["0.1", "0.5", "1.0"]

ROWS = [
    ("Demo expert (SafeMPPI)", "results/expert_gt", ""),
    (r"Kazuki guidance ($w_{s}{=}.9$)$^{\ddagger}$", "results/kazuki_sweep_smoke/w09", ""),
    ("Ours (full, it146)", "results/p2/eval_it146_m100", ""),
    (r"\;-- w/o SOCP$^{\dagger}$", "results/p2/eval_openabl_nosocp_m100", ""),
    (r"\;-- w/o progress$^{\dagger}$", "results/p2/eval_openabl_noprog_m100", ""),
    (r"\;-- w/o curriculum$^{\dagger}$", "results/p2/eval_openabl_nocur_m100", ""),
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
    out = os.path.join(HERE, "table_v2.tex")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
