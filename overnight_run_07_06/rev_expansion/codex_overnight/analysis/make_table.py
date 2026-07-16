"""Paper table (2026-07-14): demo expert vs Kazuki(detuned) vs OURS at gamma {0.1,0.5,1.0}.
Expert (results/expert_gt) and Kazuki (results/kazuki_sweep_smoke/w09) are KEPT AS-IS (open scene, user);
OURS is the walled-scene eval (report_at wrote row_g*.json into --ours-dir). Writes .md + IEEE .tex.
  python analysis/make_table.py --ours-dir results/p2/eval_final_b03_it200 --out-prefix paper_results/table_v4
"""
from __future__ import annotations
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)


def row(d, g):
    p = os.path.join(d, f"row_g{float(g)}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def cell(r, safer_than=None, faster_than=None):
    if r is None:
        return dict(sr="—", cr="—", clr="—", tim="—", cov="—")
    return dict(sr=f"{r['SR']*100:.0f}", cr=f"{r['CR']*100:.0f}",
                clr=f"{r['clearance_mean']:.3f}", tim=f"{r['time_mean_s']:.2f}", cov=str(r['coverage']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--expert-dir", default=os.path.join(P2, "results/expert_gt"))
    ap.add_argument("--kazuki-dir", default=os.path.join(P2, "results/kazuki_sweep_smoke/w09"))
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--out-prefix", default=os.path.join(P2, "paper_results/table_v4"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    methods = [("Demo expert (SafeMPPI)", args.expert_dir),
               ("CFM-MPPI$^{*}$ (Kazuki)", args.kazuki_dir),
               (r"\textbf{Ours (Safe Flow Expansion)}", args.ours_dir)]

    # markdown
    L = [f"# Paper table {args.note}", "",
         "| γ | Method | SR% | CR% | Clearance (m) | Time (s) | Coverage |",
         "|---:|---|---:|---:|---:|---:|---:|"]
    for g in args.gammas:
        for name, d in methods:
            c = cell(row(d, g))
            L.append(f"| {g:.1f} | {name.replace('$^{*}$','*').replace(chr(92)+'textbf{','').replace('}','')} "
                     f"| {c['sr']} | {c['cr']} | {c['clr']} | {c['tim']} | {c['cov']} |")
    open(args.out_prefix + ".md", "w").write("\n".join(L) + "\n")

    # IEEE tex
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Safe Flow Expansion vs.\ the demo expert and CFM-MPPI$^{*}$ at three safety levels "
         r"$\gamma$. SR/CR success/collision rate; clearance and time on successful episodes "
         r"(higher clearance = safer, lower time = faster); coverage = distinct modes." + (" " + args.note if args.note else "") + "}",
         r"\label{tab:main}", r"\begin{tabular}{clrrrrr}", r"\toprule",
         r"$\gamma$ & Method & SR & CR & Clear. & Time & Cov. \\", r"\midrule"]
    for i, g in enumerate(args.gammas):
        for j, (name, d) in enumerate(methods):
            c = cell(row(d, g))
            gcol = f"\\multirow{{3}}{{*}}{{{g:.1f}}}" if j == 0 else ""
            T.append(f"{gcol} & {name} & {c['sr']}\\% & {c['cr']}\\% & {c['clr']} & {c['tim']} & {c['cov']} \\\\")
        T.append(r"\midrule" if i < len(args.gammas) - 1 else r"\bottomrule")
    T += [r"\end{tabular}", r"\end{table}"]
    open(args.out_prefix + ".tex", "w").write("\n".join(T) + "\n")
    print("wrote", args.out_prefix + ".md/.tex")


if __name__ == "__main__":
    main()
