#!/usr/bin/env python3
"""Bounded strict-validity probe of rollout-coherent staircase targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REV = os.path.dirname(ROOT)
WORK = os.path.dirname(REV)
sys.path[:0] = [WORK, REV, ROOT]

import grid_expand_fixed as FIX  # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
import grid_metrics as GM  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_rollout as GR  # noqa: E402
import grid_scene as GS  # noqa: E402
import sr_cr_eval as SR  # noqa: E402
from uncertainty import GPUncertainty  # noqa: E402


def summarize(rows):
    modes = Counter(r["sid"] for r in rows if r["sid"] is not None)
    n = sum(modes.values())
    entropy = -sum((c / n) * math.log(c / n) for c in modes.values()) if n else 0.0
    return dict(n=len(rows), reach=float(np.mean([r["reached"] for r in rows])),
                collision=float(np.mean([r["collision"] for r in rows])),
                valid2=float(np.mean([r["valid2"] for r in rows])),
                coherent_cert=float(np.mean([r["coherent_cert"] for r in rows])),
                coherent_progress=float(np.mean([r["coherent_progress"] for r in rows])),
                target_hit=float(np.mean([r["target_hit"] for r in rows if r["target"] is not None]))
                if any(r["target"] is not None for r in rows) else None,
                distinct_modes=len(modes), mode_entropy=entropy, modes=dict(sorted(modes.items())),
                mean_steps=float(np.mean([r["steps"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wall-plugs", type=int, choices=(0, 2, 4), default=0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--n-target", type=int, default=40)
    ap.add_argument("--align-temp", type=float, default=0.45)
    ap.add_argument("--perp-brake", action="store_true")
    args = ap.parse_args()
    policy, _ = HP.load_hp(args.checkpoint, device=args.device)
    policy.eval(); env = GS.make_grid(); HT._apply_wall_plugs(env, args.wall_plugs)
    canonical = "RURURURURU"
    targets = [canonical] + sorted(GM.neighbors(canonical))
    rows = []
    for arm in ("ordinary", "targeted"):
        for rep in range(args.repeats):
            for ti, word in enumerate(targets):
                torch.manual_seed(73000 + rep * 100 + ti)
                unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
                target = word if arm == "targeted" else None
                with HT._target_proposal_override(bool(args.perp_brake and target is not None)):
                    out = GR.fm_deploy(
                        policy, env, args.gamma, T=250, target=target,
                        tilt=dict(unc=unc, beta=args.beta, N=64, s=0.9, broad=0, feature="phi_s",
                                  temp=1.0, churn=0.05, safe_filter=True,
                                  n_target=args.n_target, align_temp=args.align_temp),
                        nfe=8, record=True, verify_fn=GM2.window_label_cheap,
                        reach=0.1, device=args.device)
                coherent = FIX._executed_horizon_tensors(out["recs"])
                cert, prog = [], []
                if coherent is not None:
                    _G, L, _H, U = coherent
                    ix = np.linspace(0, len(U) - 1, min(48, len(U))).astype(int)
                    for i in ix:
                        p, pts, d = FIX._window_progress(L[i].numpy(), U[i].numpy(), env)
                        ok, margin, _resid = GM2.window_socp_stats(
                            FIX.GX2.state_from_low5(L[i].numpy()), U[i].numpy(), env, args.gamma)
                        cert.append(bool(ok and np.isfinite(margin) and margin > 0))
                        prog.append(bool(GM.in_taskspace(pts) and GM2.approach_ok(d) and
                                         p >= min(0.15, 0.5 * d[0])))
                sid = GM.staircase_id(out["path"], reach=0.1) if out["reached"] else None
                rows.append(dict(arm=arm, repeat=rep, target=target, sid=sid,
                                 target_hit=bool(target is not None and sid == target),
                                 reached=bool(out["reached"]), collision=bool(SR.path_collides(out["path"], env)),
                                 valid2=bool(GM2.traj_valid2(out["path"], env, args.gamma)),
                                 coherent_cert=float(np.mean(cert)) if cert else 0.0,
                                 coherent_progress=float(np.mean(prog)) if prog else 0.0,
                                 steps=int(out["steps"])))
    result = dict(checkpoint=args.checkpoint, gamma=args.gamma, targets=targets,
                  wall_plugs=args.wall_plugs, beta=args.beta, n_target=args.n_target,
                  align_temp=args.align_temp, perpendicular_brake=args.perp_brake,
                  ordinary=summarize([r for r in rows if r["arm"] == "ordinary"]),
                  targeted=summarize([r for r in rows if r["arm"] == "targeted"]), rows=rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("ordinary", "targeted")}, indent=2))


if __name__ == "__main__":
    main()
