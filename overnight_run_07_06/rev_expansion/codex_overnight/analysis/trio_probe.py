"""Fast pre-screen of three (or more) marginal gamma/seed fibers.

Full M25 gates cost ~6 GPU-minutes; these three rollouts decide most failures. Sweep any number of
checkpoints; print PASS only if all three reach. Full fixed-seed gate still required before promotion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import seed12_tail_trace as ST  # noqa: E402

TRIO = [(0.1, 22), (1.0, 5), (1.0, 14)]


def probe(ckpt, device="cuda", cases=TRIO, wall_plugs=0):
    pol, _ = ST.HP.load_hp(str(ckpt), device=device)
    pol.eval()
    env = ST.GS.make_grid()
    if wall_plugs:
        import grid_expand_hardtail as HT
        HT._apply_wall_plugs(env, wall_plugs)
    out = []
    for g, s in cases:
        tr = ST.trace_deploy(pol, env, g, s, device=device)
        out.append("R" if tr["reached"] else ("D" if tr["dead"] else "T"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wall-plugs", type=int, choices=(0, 2, 4), default=0)
    ap.add_argument("--cases", nargs="*", default=[], metavar="GAMMA:SEED")
    args = ap.parse_args()
    cases = TRIO if not args.cases else [(float(x.split(":", 1)[0]), int(x.split(":", 1)[1]))
                                         for x in args.cases]
    for c in args.ckpts:
        try:
            r = probe(c, args.device, cases=cases, wall_plugs=args.wall_plugs)
        except Exception as e:  # noqa: BLE001
            print(f"{c}: ERROR {e}")
            continue
        verdict = "TRIO-PASS" if all(v == "R" for v in r) else "fail"
        labels = " ".join(f"g{g:g}s{s}={v}" for (g, s), v in zip(cases, r))
        print(f"{c}: {labels}  {verdict}", flush=True)


if __name__ == "__main__":
    main()
