"""Certify-and-filter a teacher replay at each row's DESTINATION gamma (START_HERE requirement).

The replay's training target is the teacher's raw NFE8 endpoint x; as a control window that is
U = clamp(x * u_max). The verifier certificate is gamma-dependent (grid_metrics2.py), so every row must
pass `window_socp_stats(state, U_target, env, row_gamma)` before it may supervise training.
Rows that fail are DROPPED (never retargeted); metadata and seed provenance are audited and preserved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import seed12_tail_trace as ST  # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402
import grid_expand2 as GX2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-min", type=int, default=100)
    ap.add_argument("--seed-max", type=int, default=199)
    ap.add_argument("--u-max", type=float, default=1.0)
    ap.add_argument("--wall-plugs", type=int, choices=(0, 2, 4), default=0)
    args = ap.parse_args()

    rp = torch.load(args.replay, map_location="cpu", weights_only=False)
    n = len(rp["x0"])
    seeds = rp["seed"].numpy()
    assert seeds.min() >= args.seed_min and seeds.max() <= args.seed_max, \
        f"replay uses seeds outside [{args.seed_min},{args.seed_max}]: {seeds.min()}..{seeds.max()}"
    assert "metadata" in rp, "replay missing metadata"
    replay_wall_plugs = rp["metadata"].get("wall_plugs")
    if replay_wall_plugs is not None and int(replay_wall_plugs) != args.wall_plugs:
        raise ValueError(f"replay wall_plugs={replay_wall_plugs} but audit requested {args.wall_plugs}")
    env = ST.GS.make_grid()
    HT._apply_wall_plugs(env, args.wall_plugs)
    T = rp["target_x"].shape[-1] // 2 if rp["target_x"].dim() == 2 else rp["target_x"].shape[1]
    keep, margins = [], []
    fails_by_g = {}
    for i in range(n):
        st4 = np.asarray(GX2.state_from_low5(rp["low5"][i].numpy()), dtype=np.float32)
        x = rp["target_x"][i]
        U = (x.reshape(-1, 2) * args.u_max).clamp(-args.u_max, args.u_max).numpy()
        g = float(rp["gamma"][i])
        ok, margin, _res = HT.GM2.window_socp_stats(st4, U, env, g)
        if ok:
            keep.append(i); margins.append(margin)
        else:
            fails_by_g[g] = fails_by_g.get(g, 0) + 1
    keep_t = torch.as_tensor(keep, dtype=torch.long)
    out = {}
    for k, value in rp.items():
        if torch.is_tensor(value):
            out[k] = value[keep_t]
        elif k == "mode" and isinstance(value, (list, tuple)) and len(value) == n:
            out[k] = [value[i] for i in keep]
        else:
            out[k] = value
    out["metadata"] = dict(rp.get("metadata", {}),
                           destination_gamma_certified=True,
                           certified_rows=len(keep), input_rows=n,
                           dropped_by_gamma={str(k): v for k, v in sorted(fails_by_g.items())},
                           certified_margin_min=(float(np.min(margins)) if margins else None),
                           certified_margin_median=(float(np.median(margins)) if margins else None),
                           wall_plugs=args.wall_plugs,
                           seed_range=[int(seeds.min()), int(seeds.max())])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    per_g = {str(float(g)): int((rp["gamma"][keep_t].numpy() == g).sum())
             for g in sorted(set(rp["gamma"].numpy().tolist()))}
    print(f"certified {len(keep)}/{n} rows (dropped {n-len(keep)}: {fails_by_g}); per-gamma kept {per_g}")
    print(f"margin min/median: {out['metadata']['certified_margin_min']}/{out['metadata']['certified_margin_median']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
