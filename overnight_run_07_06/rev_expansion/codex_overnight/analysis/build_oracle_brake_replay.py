"""Certified oracle-brake replay: executed windows from band-brake-guided exploration rollouts.

Exploration override (a legitimate exploration mechanism, like tilt/targeted/recovery starts): deploy the
candidate faithfully EXCEPT that inside the rising interception band (y in [band]) with vy>vy_min the
executed action's y-component is set to -u_max (full brake). Trajectories must REACH (strict .1) and pass
the UNCHANGED traj_valid2; every extracted H=10 executed window must pass taskspace + approach_ok +
progress floor + the exact destination-gamma certificate — identical acceptance to the trainer's gather.
Only windows whose CONTEXT lies in the band are kept (the medicine rows).

Output format = escape-replay: each kept window is written K times with distinct deterministic base latents
x0 and target_x = (U/u_max).flatten(), so the endpoint loss distills the certified brake window onto K
latent fibers at that context. Seeds are independent (>=100); M25 evaluation seeds 0-24 are never used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import seed12_tail_trace as ST  # noqa: E402
import grid_expand_hardtail as HT  # noqa: E402
import grid_expand2 as GX2  # noqa: E402


@torch.no_grad()
def oracle_deploy(policy, env, g, seed, band, vy_min, device, T=250, reach=0.1):
    torch.manual_seed(seed)
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    hist, path, steps = [], [st[:2].copy()], []
    reached = dead = False
    for t in range(T):
        gT, lT, hT = ST._ctx_tensors(st, goal, g, hist, env, device)
        ctx = ST._ctx_of(policy, gT, lT, hT)
        x0 = torch.randn(1, policy.d, device=device)
        U, _ = ST.integrate(policy, ctx, x0, nfe=8)
        a = U[0, 0].detach().cpu().numpy().copy()
        oracled = False
        if band[0] <= st[1] <= band[1] and st[3] > vy_min:
            a[1] = -float(policy.u_max); oracled = True
        steps.append(dict(state=st.copy(), grid=gT.detach().cpu().numpy(),
                          low5=lT.detach().cpu().numpy(), hist=hT.detach().cpu().numpy(),
                          a=a.copy(), oracled=oracled))
        st = ST.GR.di_step(st, np.asarray(a, np.float32), dt=env.dt)
        hist.append(np.asarray(a, np.float32)); path.append(st[:2].copy())
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True; break
        if (st[:2] < -ST.GM.EPS_TASK).any() or (st[:2] > ST.GM.GRID_M + ST.GM.EPS_TASK).any():
            dead = True; break
        if len(obs) and (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min() < 0.0:
            dead = True; break
    return dict(path=np.array(path, np.float32), reached=reached, dead=dead, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--band", type=float, nargs=2, default=(4.85, 5.12))
    ap.add_argument("--vy-min", type=float, default=0.05)
    ap.add_argument("--ctx-band", type=float, nargs=2, default=(4.80, 5.12),
                    help="keep windows whose CONTEXT y is in this range with vy>0")
    ap.add_argument("--seeds", type=int, nargs=2, default=(100, 199))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--k-latents", type=int, default=8)
    ap.add_argument("--h", type=int, default=10)
    args = ap.parse_args()
    dev = args.device
    pol, _ = ST.HP.load_hp(args.ckpt, device=dev); pol.eval()
    env = ST.GS.make_grid()
    cfg = HT.CurConfig()
    rows, n_traj, n_kept_traj = [], 0, 0
    for g in args.gammas:
        for seed in range(args.seeds[0], args.seeds[1] + 1):
            n_traj += 1
            tr = oracle_deploy(pol, env, float(g), seed, args.band, args.vy_min, dev)
            if not tr["reached"] or tr["dead"]:
                continue
            if not HT.GM2.traj_valid2(tr["path"], env, float(g)):
                continue
            n_kept_traj += 1
            S = tr["steps"]; H = args.h
            for t in range(len(S) - H):
                stt = S[t]["state"]
                if not (args.ctx_band[0] <= stt[1] <= args.ctx_band[1] and stt[3] > 0.0):
                    continue
                U = np.stack([S[t + j]["a"] for j in range(H)])
                low5 = S[t]["low5"]
                p_i, pts, dists = HT._window_progress(low5, U, env)
                if not HT.GM.in_taskspace(pts):
                    continue
                if not HT.GM2.approach_ok(dists):
                    continue
                if p_i < min(cfg.valid_prog_floor, 0.5 * dists[0]):
                    continue
                ok, margin, _res = HT.GM2.window_socp_stats(
                    GX2.state_from_low5(low5), U, env, float(g))
                if not ok:
                    continue
                rows.append((S[t], U, float(g), seed, t, margin,
                             any(S[t + j]["oracled"] for j in range(H))))
        print(f"g{g}: cumulative rows={len(rows)} kept_traj={n_kept_traj}/{n_traj}", flush=True)
    if not rows:
        raise RuntimeError("no certified band windows")
    gen = torch.Generator().manual_seed(20260713)
    K, d = args.k_latents, pol.d
    G = torch.tensor(np.stack([r[0]["grid"] for r in rows])).repeat_interleave(K, 0)
    L = torch.tensor(np.stack([r[0]["low5"] for r in rows])).repeat_interleave(K, 0)
    Hh = torch.tensor(np.stack([r[0]["hist"] for r in rows])).repeat_interleave(K, 0)
    X0 = torch.randn(len(rows) * K, d, generator=gen)
    TX = torch.tensor(np.stack([(r[1] / pol.u_max).reshape(-1) for r in rows]),
                      dtype=torch.float32).repeat_interleave(K, 0)
    out = {
        "grid": G, "low5": L, "hist": Hh, "x0": X0, "target_x": TX,
        "gamma": torch.tensor([r[2] for r in rows], dtype=torch.float32).repeat_interleave(K, 0),
        "seed": torch.tensor([r[3] for r in rows], dtype=torch.long).repeat_interleave(K, 0),
        "step": torch.tensor([r[4] for r in rows], dtype=torch.long).repeat_interleave(K, 0),
        "metadata": {"ckpt": args.ckpt, "kind": "oracle_brake_certified_windows",
                     "band": list(args.band), "ctx_band": list(args.ctx_band),
                     "vy_min": args.vy_min, "k_latents": K, "seeds": list(args.seeds),
                     "unique_windows": len(rows), "kept_traj": n_kept_traj,
                     "margin_min": float(min(r[5] for r in rows)),
                     "oracled_window_frac": float(np.mean([r[6] for r in rows])),
                     "destination_gamma_certified": True},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"{len(rows)} unique certified band windows x {K} latents -> {args.out}")
    print("metadata:", out["metadata"])


if __name__ == "__main__":
    main()
