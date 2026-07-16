"""Generate OOD-START rollouts for the paper rollout gallery (panel 1).

Demo data and deployment evals always start at the origin (0,0) — the demo support hugs the
diagonal corridors. Here we deploy the expanded policy FAITHFULLY (temp 1, NFE 8, no filtering)
from off-diagonal cell centers the demonstrations never visited, per gamma {0.1, 0.5, 1.0}:
evidence that expansion recovered the state space, not just the origin fiber.
Output: analysis/runs/ood_starts_<tag>.npz  (paths object array + starts + gammas + reached).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import seed12_tail_trace as ST  # noqa: E402

STARTS = [(3.5, 0.5), (4.5, 1.5), (0.5, 3.5), (1.5, 4.5), (4.5, 0.5), (0.5, 4.5)]
GAMMAS = [0.1, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="it146")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed0", type=int, default=500)
    args = ap.parse_args()

    pol, _ = ST.HP.load_hp(args.ckpt, device=args.device)
    pol.eval()
    env = ST.GS.make_grid()
    paths, starts, gammas, reached = [], [], [], []
    k = 0
    for sx, sy in STARTS:
        for g in GAMMAS:
            env.x0 = torch.tensor([sx, sy, 0.0, 0.0], dtype=torch.float32)
            tr = ST.trace_deploy(pol, env, g, args.seed0 + k, device=args.device)
            paths.append(np.asarray(tr["path"], np.float32))
            starts.append((sx, sy)); gammas.append(g); reached.append(bool(tr["reached"]))
            print(f"start ({sx},{sy}) g{g}: reached={tr['reached']} dead={tr['dead']} len={len(tr['path'])}",
                  flush=True)
            k += 1
    pa = np.empty(len(paths), dtype=object)
    for i, p in enumerate(paths):
        pa[i] = p
    out = HERE / "runs" / f"ood_starts_{args.tag}.npz"
    os.makedirs(out.parent, exist_ok=True)
    np.savez_compressed(out, paths=pa, starts=np.array(starts, np.float32),
                        gammas=np.array(gammas, np.float32), reached=np.array(reached))
    print("wrote", out, f"({int(np.sum(reached))}/{len(reached)} reached)")


if __name__ == "__main__":
    main()
