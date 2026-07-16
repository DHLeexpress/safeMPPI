"""Build immutable successful/rare-mode replay from independent faithful seeds.

Each row stores a selected context, the exact deployment base latent, and the
teacher's raw NFE8 endpoint. Only complete successful trajectories are kept;
optional staircase filters retain certified rare-mode fibers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import seed12_tail_trace as ST


def worker(args) -> None:
    policy, _ = ST.HP.load_hp(args.ckpt, device=args.device)
    policy.eval(); env = ST.GS.make_grid()
    if args.wall_plugs:
        import grid_expand_hardtail as HT
        HT._apply_wall_plugs(env, args.wall_plugs)
    rows = []
    for seed in range(args.seed0, args.seed0 + args.M):
        tr = ST.trace_deploy(policy, env, args.gamma, seed, nfe=8, reach=0.1, device=args.device)
        if not tr["reached"] or tr["dead"]:
            continue
        sid = ST.GM.staircase_id(tr["path"], reach=0.1)
        if args.modes and sid not in set(args.modes):
            continue
        kept = 0
        for t, step in enumerate(tr["steps"]):
            st = np.asarray(step["state"])
            xmin, xmax, ymin, ymax = args.region
            if xmin <= st[0] <= xmax and ymin <= st[1] <= ymax:
                rows.append((step, seed, t, sid)); kept += 1
        print(f"g{args.gamma} seed{seed}: success mode={sid}, replay_rows={kept}", flush=True)
    if not rows:
        raise RuntimeError("no successful escape rows")
    out = {
        "grid": torch.tensor(np.stack([r[0]["grid"] for r in rows])),
        "low5": torch.tensor(np.stack([r[0]["low5"] for r in rows])),
        "hist": torch.tensor(np.stack([r[0]["hist"] for r in rows])),
        "x0": torch.tensor(np.stack([r[0]["x0"] for r in rows])),
        "target_x": torch.tensor(np.stack([r[0]["euler"][-1] for r in rows])),
        "gamma": torch.full((len(rows),), float(args.gamma)),
        "seed": torch.tensor([r[1] for r in rows], dtype=torch.long),
        "step": torch.tensor([r[2] for r in rows], dtype=torch.long),
        "mode": [r[3] for r in rows],
        "metadata": {"ckpt": args.ckpt, "seed0": args.seed0, "M": args.M,
                     "gamma": args.gamma, "faithful_temp": 1.0, "nfe": 8,
                     "reach": 0.1, "success_only": True, "region": list(args.region),
                     "wall_plugs": args.wall_plugs, "mode_filter": list(args.modes)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


def merge(args) -> None:
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.inputs]
    wall_plugs = {int(p.get("metadata", {}).get("wall_plugs", 0)) for p in parts}
    if len(wall_plugs) != 1:
        raise ValueError(f"cannot merge replay parts from different scenes: wall_plugs={sorted(wall_plugs)}")
    keys = ("grid", "low5", "hist", "x0", "target_x", "gamma", "seed", "step")
    out = {k: torch.cat([p[k] for p in parts]) for k in keys}
    out["mode"] = sum((list(p.get("mode", [None] * len(p["x0"]))) for p in parts), [])
    out["metadata"] = {"inputs": list(args.inputs), "rows": len(out["x0"]),
                       "seed_disjoint_from_m25": True, "wall_plugs": wall_plugs.pop(),
                       "mode_filtered": any(p.get("metadata", {}).get("mode_filter") for p in parts)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"merged {len(parts)} parts / {len(out['x0'])} rows -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--region", type=float, nargs=4, default=(0.0, 1.25, 0.0, 0.65),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--inputs", nargs="+")
    ap.add_argument("--modes", nargs="*", default=[], help="optional successful staircase IDs to retain")
    ap.add_argument("--wall-plugs", type=int, choices=(0, 2, 4), default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.inputs:
        merge(args)
    else:
        if args.ckpt is None or args.gamma is None:
            ap.error("worker mode requires --ckpt and --gamma")
        worker(args)


if __name__ == "__main__":
    main()
