"""T4 — quantitative beyond-teacher proof: which staircase modes does the expanded policy DEPLOY that the
SafeMPPI expert (the teacher that generated all demonstrations) does not?

Compares per-gamma M100 deployed mode sets (staircase words of successful faithful rollouts) between:
  - P1 expert (`results/expert_gt`, M100/γ — the teacher's own deployed support),
  - any number of policy eval dirs (t104, candidate, final),
and reports: shared modes, NEW modes (never deployed by the expert at that γ), LOST modes, plus the a–e
beat table rows. Every counted success is collision-free by the shared metric code (eval_ae), i.e. every
new mode was reached under the same faithful protocol — the verifier gated its training data, and
deployment confirms it.

Output: tables/T4_mode_discovery.md + analysis/mode_discovery.json (+ optional per-mode gallery figure).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]


def mode_sets(evdir):
    out = {}
    for g in GAMMAS:
        f = os.path.join(evdir, f"row_g{g}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        out[g] = dict(ids=set(r.get("coverage_ids", [])), row=r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-dir", default=os.path.join(P2, "results/expert_gt"))
    ap.add_argument("--policy", action="append", nargs=2, metavar=("NAME", "DIR"),
                    default=None, help="repeatable: NAME EVAL_DIR")
    ap.add_argument("--out-md", default=os.path.join(P2, "tables/T4_mode_discovery.md"))
    ap.add_argument("--out-json", default=os.path.join(HERE, "mode_discovery.json"))
    args = ap.parse_args()
    policies = args.policy or [["t104_M100", os.path.join(P2, "results/p2/eval_corrected_mode2_it104_m100")]]

    exp = mode_sets(args.expert_dir)
    res = {"expert_dir": args.expert_dir, "policies": {}}
    expert_label = os.path.relpath(args.expert_dir, P2)
    lines = ["# T4 — mode discovery beyond the teacher (per-γ M100 deployed staircase modes)", "",
             f"Expert = SafeMPPI (`{expert_label}`, M=100/γ): the SAME controller that produced every",
             "demonstration the policy was ever pretrained on. A NEW mode = a staircase word the policy",
             "deploys successfully (faithful temp=1/NFE8/reach=.1, collision-free) that the expert never",
             "deployed at that γ in its own 100 trials.", ""]
    for name, d in policies:
        pol = mode_sets(d)
        pres = {}
        lines += [f"## {name}  (`{d}`)", "",
                  "| γ | expert modes | policy modes | shared | **NEW (beyond teacher)** | lost | new words |",
                  "|---|---|---|---|---|---|---|"]
        tot_new = 0
        for g in GAMMAS:
            if g not in pol or g not in exp:
                continue
            E, Pm = exp[g]["ids"], pol[g]["ids"]
            new = sorted(Pm - E); lost = sorted(E - Pm)
            tot_new += len(new)
            pres[g] = dict(expert=len(E), policy=len(Pm), shared=len(E & Pm),
                           new=new, lost=lost)
            lines.append(f"| {g} | {len(E)} | {len(Pm)} | {len(E & Pm)} | **{len(new)}** | {len(lost)} | "
                         f"{', '.join(new) if new else '—'} |")
        lines += ["", f"Total NEW modes across γ: **{tot_new}**", ""]
        res["policies"][name] = pres
        # a–e beat rows
        lines += ["| γ | SR | CR | clearance vs expert | time vs expert | coverage vs expert |", "|---|---|---|---|---|---|"]
        for g in GAMMAS:
            if g not in pol or g not in exp:
                continue
            r, e = pol[g]["row"], exp[g]["row"]
            lines.append(f"| {g} | {r['SR']*100:.0f}% | {r['CR']*100:.0f}% | "
                         f"{r['clearance_mean']:.3f} vs {e['clearance_mean']:.3f} | "
                         f"{r['time_mean_s']:.2f} vs {e['time_mean_s']:.2f} | "
                         f"{r['coverage']} vs {e['coverage']} |")
        lines.append("")
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, default=list)
    print("\n".join(lines[:24]))
    print("wrote", args.out_md, "and", args.out_json)


if __name__ == "__main__":
    main()
