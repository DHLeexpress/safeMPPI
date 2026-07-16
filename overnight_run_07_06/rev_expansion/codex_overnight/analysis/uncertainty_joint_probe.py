"""Reconstruct a representative GP buffer and test whether sigma tilt enriches valid progress.

The trainer does not checkpoint qbuf, so this is explicitly an approximation: it draws the same
384-feature GP budget from the ten latest accepted-window snapshots before the selected checkpoint.
It is still useful for testing the ranking mechanism, because all candidates at a state are scored
against the same reconstructed buffer and the exact production phi_s/RBF/beta implementation is used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REV = os.path.dirname(ROOT)
WORK = os.path.dirname(REV)
sys.path[:0] = [ROOT, REV, WORK]

import grid_feats as GF  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_rollout as GR  # noqa: E402
import grid_scene as GS  # noqa: E402
from uncertainty import GPUncertainty  # noqa: E402


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/p2/finalunit_q50_k14_s15_from_it18/ckpt_100.pt")
    ap.add_argument("--viz-dir", default="results/p2/finalunit_q50_k14_s15_from_it18/viz_db")
    ap.add_argument("--snapshots", type=int, default=10)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(HERE, "uncertainty_joint_probe.json"))
    args = ap.parse_args()
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
    viz_dir = args.viz_dir if os.path.isabs(args.viz_dir) else os.path.join(ROOT, args.viz_dir)
    policy, _ = HP.load_hp(ckpt, "cpu")
    env = GS.make_grid()

    files = sorted(glob.glob(os.path.join(viz_dir, "it*.pt")),
                   key=lambda p: int(os.path.basename(p)[2:-3]))[-args.snapshots:]
    Gs, Ls, Us = [], [], []
    for path in files:
        db = torch.load(path, map_location="cpu", weights_only=False)
        Gs.append(db["grid"])
        Ls.append(db["low5"])
        Us.append(db["U"])
    G, L, U = torch.cat(Gs), torch.cat(Ls), torch.cat(Us)
    idx = torch.randperm(len(U), generator=torch.Generator().manual_seed(5))[:384]
    hist = torch.zeros(len(idx), GF.K_HIST, 2)
    with torch.no_grad():
        phi_buffer = policy.phi_s(U[idx], policy.ctx_from(G[idx], L[idx], hist), s=0.9)
    unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
    unc.set_buffer(phi_buffer)

    obs = env.obstacles.detach().cpu().numpy()
    state = env.x0.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    grid = torch.tensor(GF.axis_grid(state[:2], obs, float(env.r_robot)))
    hist1 = torch.zeros(GF.K_HIST, 2)
    result = {"checkpoint": ckpt, "buffer_snapshot_files": files, "buffer_n": int(len(idx)), "rows": {}}
    for gamma in GAMMAS:
        low = torch.tensor(GF.low5(state, goal, gamma))
        torch.manual_seed(2026)
        candidates = policy.sample_window(grid, low, hist1, n=args.N, temp=1.0, nfe=6)
        safe = GR.safe_mask(state, candidates.numpy(), obs, float(env.r_robot), env.dt)
        candidates = candidates[torch.as_tensor(np.where(safe)[0])]
        with torch.no_grad():
            sigma = unc.sigma(policy.phi_s_at(candidates, grid, low, hist1, s=0.9)).numpy()
        weights = np.exp((sigma - sigma.max()) / args.beta)
        weights /= weights.sum()
        progress, certified = [], []
        for controls in candidates.numpy():
            seg = GR.window_positions(state, controls, env.dt)
            d = np.linalg.norm(np.vstack([state[:2], seg]) - goal, axis=1)
            progress.append(float(d[0] - d[-1]))
            margin = GM2.window_socp_margin(state, controls, env, gamma)
            certified.append(bool(margin >= -1e-8))
        progress = np.asarray(progress)
        certified = np.asarray(certified)
        joint = certified & (progress >= 0.15)
        result["rows"][str(gamma)] = {
            "candidates_after_cheap_safe_filter": int(len(candidates)),
            "sigma_min_mean_std_max": [float(x) for x in (sigma.min(), sigma.mean(), sigma.std(), sigma.max())],
            "weight_ess": float(1.0 / np.square(weights).sum()),
            "weight_ess_fraction": float(1.0 / np.square(weights).sum() / len(weights)),
            "joint_certified_progress_uniform_probability": float(joint.mean()),
            "joint_certified_progress_sigma_weighted_probability": float(weights[joint].sum()),
            "sigma_progress_pearson": corr(sigma, progress),
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
