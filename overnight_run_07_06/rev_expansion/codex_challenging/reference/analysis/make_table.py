#!/usr/bin/env python3
"""Exact v4 Markdown + IEEE table for the challenging sanity comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

METHODS = [
    ("Demo expert (SafeMPPI)", "Demo expert (SafeMPPI)", "stage_results/06_baselines/results/expert_m6"),
    ("Our approach", r"\textbf{Ours (Safe Flow Expansion)}", "stage_results/05_sanity/data/eval_final_v7_ours"),
    ("Pretrained", "Pretrained", "stage_results/04_canonical/data/pretrained_m6"),
    ("CFM-MPPI* (low guidance)", r"CFM-MPPI$^{*}$ (low guidance)",
     "stage_results/06_baselines/results/kazuki_low_guidance_m6"),
    ("NO safety validity check", r"$-$SOCP", "stage_results/05_sanity/data/eval_final_v7_no_socp"),
    ("NO progress check", r"$-$Progress", "stage_results/05_sanity/data/eval_final_v7_no_progress"),
    ("NO curriculum", r"$-$Curriculum", "stage_results/05_sanity/data/eval_final_v7_no_curriculum"),
]


def load_row(directory: Path, gamma: float):
    path = directory / f"row_g{float(gamma)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def cells(row):
    if row is None:
        return ["—"] * 5
    return [
        f"{row['SR'] * 100:.0f}",
        f"{row['CR'] * 100:.0f}",
        f"{row['clearance_mean']:.3f}" if finite(row.get("clearance_mean")) else "—",
        f"{row['time_mean_s']:.2f}" if finite(row.get("time_mean_s")) else "—",
        str(row.get("coverage", "—")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--out-prefix", type=Path,
                        default=ROOT / "stage_results/05_sanity/data/table_v4")
    args = parser.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    markdown = [
        "# Challenging sanity comparison (v4 style)", "",
        "| γ | Method | SR% | CR% | Clearance (m) | Time (s) | Coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for gamma in args.gammas:
        for md_name, _, relative in METHODS:
            values = cells(load_row(ROOT / relative, gamma))
            markdown.append(f"| {gamma:.1f} | {md_name} | " + " | ".join(values) + " |")
    markdown += ["", "Sanity sample size is M=6 per gamma; this table is diagnostic, not a final claim.", ""]
    args.out_prefix.with_suffix(".md").write_text("\n".join(markdown))

    tex = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Challenging-scene sanity comparison at three safety levels $\gamma$. "
        r"SR/CR denote success/collision rate; clearance and time are computed on successful episodes. "
        r"All learned arms use $M{=}6$ and are diagnostic rather than final.}",
        r"\label{tab:challenging_sanity}", r"\begin{tabular}{clrrrrr}", r"\toprule",
        r"$\gamma$ & Method & SR & CR & Clear. & Time & Cov. \\", r"\midrule",
    ]
    for gamma_index, gamma in enumerate(args.gammas):
        for method_index, (_, tex_name, relative) in enumerate(METHODS):
            sr, cr, clearance, time_s, coverage = cells(load_row(ROOT / relative, gamma))
            gamma_cell = rf"\multirow{{{len(METHODS)}}}{{*}}{{{gamma:.1f}}}" if method_index == 0 else ""
            sr = sr + (r"\%" if sr != "—" else "")
            cr = cr + (r"\%" if cr != "—" else "")
            tex.append(f"{gamma_cell} & {tex_name} & {sr} & {cr} & {clearance} & {time_s} & {coverage} " + r"\\")
        tex.append(r"\midrule" if gamma_index < len(args.gammas) - 1 else r"\bottomrule")
    tex += [r"\end{tabular}", r"\end{table*}"]
    args.out_prefix.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {args.out_prefix}.md/.tex")


if __name__ == "__main__":
    main()
