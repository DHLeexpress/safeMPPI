#!/usr/bin/env python3
"""Deployment-only adaptive-gamma evaluation on the 4-plug scene.

The verifier selector maps the *same* base latent through every gamma-conditioned
flow at each replan.  No training, rejection sampling, or fixed-gamma table data
is touched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import grid_expand_hardtail as HT  # path bootstrap + canonical local verifier
import grid_scene as GS
import eval_ae as EVAL


GAMMAS = np.array((0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0), dtype=np.float32)


def _object_array(items):
    out = np.empty(len(items), dtype=object)
    for i, value in enumerate(items):
        out[i] = np.asarray(value, dtype=np.float32)
    return out


@torch.no_grad()
def windows_from_same_latent(policy, grid_np, low5_np, hist_np, gammas, latent, nfe=8):
    """Map one x0 through all gamma contexts; output [K,H,2]."""
    device = policy.head.weight.device
    gs = np.asarray(gammas, dtype=np.float32)
    k = len(gs)
    grid = torch.as_tensor(grid_np, device=device).unsqueeze(0).repeat(k, 1, 1, 1)
    low5 = torch.as_tensor(low5_np, device=device).unsqueeze(0).repeat(k, 1)
    low5[:, 4] = torch.as_tensor(gs, device=device)
    hist = torch.as_tensor(hist_np, device=device).unsqueeze(0).repeat(k, 1, 1)
    ctx = policy.ctx_from(grid, low5, hist)
    x = latent.to(device).reshape(1, -1).repeat(k, 1)
    for i in range(nfe):
        tau = torch.full((k,), i / nfe, device=device)
        x = x + policy.forward(x, tau, ctx) / nfe
    return (x.reshape(k, policy.T, 2) * policy.u_max).clamp(-policy.u_max, policy.u_max)


def proximity_gamma(state, env, d_lo=.3, d_hi=1.0, gamma_min=.1, gamma_max=1.0):
    obs = env.obstacles.detach().cpu().numpy()
    p = np.asarray(state, dtype=float)[:2]
    clearance = np.linalg.norm(p[None] - obs[:, :2], axis=1) - obs[:, 2] - float(env.r_robot)
    d_min = float(clearance.min())
    u = np.clip((d_min - d_lo) / (d_hi - d_lo), 0.0, 1.0)
    return float(gamma_min + (gamma_max - gamma_min) * u), d_min


def verifier_scores(state, windows, gammas, env):
    """Certificate-first score with literal face margin and window progress."""
    goal = env.goal.detach().cpu().numpy()
    d0 = float(np.linalg.norm(np.asarray(state)[:2] - goal))
    records = []
    for U, gamma in zip(np.asarray(windows), np.asarray(gammas)):
        seg = HT.GR.window_positions(state, U, env.dt)
        progress = d0 - float(np.linalg.norm(seg[-1] - goal))
        task = bool(HT.GM.in_taskspace(seg))
        approach = bool(HT.GM2.approach_ok(np.linalg.norm(
            np.vstack([np.asarray(state)[:2], seg]) - goal[None], axis=1)))
        ok, face_margin, residual = HT.GM2.window_socp_stats(state, U, env, float(gamma))
        if not ok:
            face_margin = HT.GM2.window_min_clearance(state, U, env)
        valid = bool(ok and task and approach)
        # 1000 makes exact-validity lexicographically dominant.  The remaining
        # terms are the requested literal face margin + goal progress.
        score = 1000.0 * valid + 10.0 * bool(ok) + float(face_margin) + progress
        records.append(dict(gamma=float(gamma), score=score, valid=valid, certificate=bool(ok),
                            face_margin=float(face_margin), progress=progress,
                            cert_residual=float(residual)))
    return records


@torch.no_grad()
def deploy(policy, env, mode, seed, T=250, reach=.1, nfe=8, d_lo=.3, d_hi=1.0):
    if mode not in {"heuristic", "verifier", "random"}:
        raise ValueError(mode)
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 1000003)
    random.seed(seed)
    obs = env.obstacles.detach().cpu().numpy()
    rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    hist, path, gamma_trace, clearance_trace, selector = [], [st[:2].copy()], [], [], []
    reached = dead = False
    for _ in range(T):
        grid = HT.GF.axis_grid(st[:2], obs, rr)
        hist_pad = HT.GF.hist_pad(np.asarray(hist).reshape(-1, 2) if hist else np.zeros((0, 2)))
        latent = torch.randn(policy.d, device=policy.head.weight.device)
        if mode == "heuristic":
            gamma, d_min = proximity_gamma(st, env, d_lo=d_lo, d_hi=d_hi)
            candidates = np.array([gamma], dtype=np.float32)
        elif mode == "random":
            gamma = float(np_rng.choice(GAMMAS))
            d_min = proximity_gamma(st, env, d_lo=d_lo, d_hi=d_hi)[1]
            candidates = np.array([gamma], dtype=np.float32)
        else:
            d_min = proximity_gamma(st, env, d_lo=d_lo, d_hi=d_hi)[1]
            candidates = GAMMAS
        low5 = HT.GF.low5(st, goal, float(candidates[0]))
        windows = windows_from_same_latent(policy, grid, low5, hist_pad, candidates, latent, nfe=nfe)
        if mode == "verifier":
            scored = verifier_scores(st, windows.detach().cpu().numpy(), candidates, env)
            choice = int(np.argmax([r["score"] for r in scored]))
            gamma = float(candidates[choice])
            selector.append(scored[choice])
        else:
            choice = 0
        U = windows[choice].detach().cpu().numpy()
        action = U[0]
        st = HT.GR.di_step(st, action.astype(np.float32), dt=env.dt)
        hist.append(action.astype(np.float32))
        path.append(st[:2].copy())
        gamma_trace.append(gamma)
        clearance_trace.append(d_min)
        if np.linalg.norm(st[:2] - goal) < reach:
            reached = True
            break
        if (st[:2] < -HT.GM.EPS_TASK).any() or (st[:2] > HT.GM.GRID_M + HT.GM.EPS_TASK).any():
            dead = True
            break
        if (np.linalg.norm(st[:2][None] - obs[:, :2], axis=1) - obs[:, 2] - rr).min() < 0:
            dead = True
            break
    return dict(path=np.asarray(path, np.float32), gamma=np.asarray(gamma_trace, np.float32),
                clearance=np.asarray(clearance_trace, np.float32), selector=selector,
                reached=reached, dead=dead, steps=len(path) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("heuristic", "verifier", "random"), required=True)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--d-lo", type=float, default=.3)
    ap.add_argument("--d-hi", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    policy, _ = HT.HP.load_hp(args.ckpt, device=args.device)
    env = GS.make_grid()
    HT._apply_wall_plugs(env, 4)
    rollouts = []
    for i in range(args.M):
        out = deploy(policy, env, args.mode, args.seed0 + i, T=args.T, nfe=args.nfe,
                     d_lo=args.d_lo, d_hi=args.d_hi)
        rollouts.append(out)
        if (i + 1) % 10 == 0 or i + 1 == args.M:
            print(f"[{args.mode}] {i + 1}/{args.M}", flush=True)
    paths = [r["path"] for r in rollouts]
    method = f"adaptive-{args.mode}"
    row = EVAL.summarize_paths(paths, env, float("nan"), method)
    row.update(schedule=args.mode, d_lo=args.d_lo, d_hi=args.d_hi, checkpoint=str(Path(args.ckpt).resolve()))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    EVAL.save_paths(outdir / f"paths_{args.mode}.npz", paths,
                    gamma_traces=_object_array([r["gamma"] for r in rollouts]),
                    clearance_traces=_object_array([r["clearance"] for r in rollouts]),
                    seeds=np.arange(args.seed0, args.seed0 + args.M))
    (outdir / f"row_{args.mode}.json").write_text(json.dumps(row, indent=2, allow_nan=True) + "\n")
    print(json.dumps(row, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
