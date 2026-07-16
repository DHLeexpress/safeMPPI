"""Distill a fixed teacher on states induced by a repaired candidate (independent seeds)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import seed12_tail_trace as ST


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--region", type=float, nargs=4, default=(0.0, 2.2, 0.0, 1.8),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cand, _ = ST.HP.load_hp(args.candidate, device=args.device)
    teacher, _ = ST.HP.load_hp(args.teacher, device=args.device)
    cand.eval(); teacher.eval(); env = ST.GS.make_grid()
    rows = []
    for seed in range(args.seed0, args.seed0 + args.M):
        tr = ST.trace_deploy(cand, env, args.gamma, seed, nfe=8, reach=.1, device=args.device)
        kept = 0
        for t, step in enumerate(tr["steps"]):
            st = np.asarray(step["state"])
            xmin, xmax, ymin, ymax = args.region
            if not (xmin <= st[0] <= xmax and ymin <= st[1] <= ymax):
                continue
            gT = torch.tensor(step["grid"], device=args.device)
            lT = torch.tensor(step["low5"], device=args.device)
            hT = torch.tensor(step["hist"], device=args.device)
            ctx = ST._ctx_of(teacher, gT, lT, hT)
            x0 = torch.tensor(step["x0"], device=args.device)[None]
            _, xs = ST.integrate(teacher, ctx, x0, nfe=8, keep_states=True)
            rows.append((step, xs[-1][0].cpu().numpy(), seed, t)); kept += 1
        print(f"g{args.gamma} seed{seed}: candidate_reached={tr['reached']} dead={tr['dead']} rows={kept}", flush=True)
    if not rows:
        raise RuntimeError("no on-policy rows")
    out = {
        "grid": torch.tensor(np.stack([r[0]["grid"] for r in rows])),
        "low5": torch.tensor(np.stack([r[0]["low5"] for r in rows])),
        "hist": torch.tensor(np.stack([r[0]["hist"] for r in rows])),
        "x0": torch.tensor(np.stack([r[0]["x0"] for r in rows])),
        "target_x": torch.tensor(np.stack([r[1] for r in rows])),
        "gamma": torch.full((len(rows),), args.gamma),
        "seed": torch.tensor([r[2] for r in rows]),
        "step": torch.tensor([r[3] for r in rows]),
        "metadata": {"candidate": args.candidate, "teacher": args.teacher,
                     "seed0": args.seed0, "M": args.M, "gamma": args.gamma,
                     "region": list(args.region), "nfe": 8},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
