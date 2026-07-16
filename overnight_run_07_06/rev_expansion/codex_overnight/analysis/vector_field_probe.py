"""Read-only diagnostics for the Safe Flow Expansion vector field.

This script never trains or saves a policy.  It evaluates checkpoints on the same
fixed latent draws and balanced demo contexts so that changes are attributable to
the learned field rather than Monte-Carlo noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REV = os.path.dirname(ROOT)
WORK = os.path.dirname(REV)
sys.path[:0] = [WORK, REV, ROOT]

import grid_feats as GF  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
import grid_scene as GS  # noqa: E402
import pretrain_repr as PR  # noqa: E402


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _balanced_demo(n_per_gamma: int):
    """Deterministic, equal-count demo slice (pretraining itself used all windows)."""
    Gs, Ls, Hs, Us = [], [], [], []
    for gi, gamma in enumerate(GAMMAS):
        g, l, h, u = PR.load_data("dr05_", [str(gamma)], n_per_gamma, seed=100 + gi)
        Gs.append(g)
        Ls.append(l)
        Hs.append(h)
        Us.append(u)
    return tuple(torch.cat(xs) for xs in (Gs, Ls, Hs, Us))


def _fixed_cfm(policy, batch, seed: int = 7123):
    """CFM MSE/cosine at fixed (x0,tau), reported overall and per gamma."""
    G, L, H, U = batch
    with torch.no_grad():
        ctx = policy.ctx_from(G, L, H)
        B = U.shape[0]
        gen = torch.Generator().manual_seed(seed)
        x1 = (U / policy.u_max).reshape(B, policy.d)
        x0 = torch.randn(x1.shape, generator=gen)
        tau = torch.rand(B, generator=gen).clamp(1e-4, 1.0)
        xt = (1 - tau)[:, None] * x0 + tau[:, None] * x1
        target = x1 - x0
        pred = policy(xt, tau, ctx)
        mse = ((pred - target) ** 2).mean(1)
        cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    out = {"all": {"mse": float(mse.mean()), "cos": float(cos.mean())}}
    for gamma in GAMMAS:
        m = torch.isclose(L[:, 4], torch.tensor(gamma, dtype=L.dtype))
        out[str(gamma)] = {"mse": float(mse[m].mean()), "cos": float(cos[m].mean())}
    # A fixed tau sweep distinguishes endpoint-field drift from a generic loss change.
    tau_rows = {}
    with torch.no_grad():
        for tau_value in (0.0, 0.25, 0.5, 0.75, 0.9999):
            tau_fixed = torch.full((B,), tau_value)
            xt_fixed = (1 - tau_fixed)[:, None] * x0 + tau_fixed[:, None] * x1
            pred_fixed = policy(xt_fixed, tau_fixed, ctx)
            mse_fixed = ((pred_fixed - target) ** 2).mean(1)
            cos_fixed = torch.nn.functional.cosine_similarity(pred_fixed, target, dim=1)
            tau_rows[str(tau_value)] = {
                "mse": float(mse_fixed.mean()), "cos": float(cos_fixed.mean()),
                "field_norm": float(pred_fixed.norm(dim=1).mean()),
            }
    out["tau_sweep"] = tau_rows
    return out


def _origin_inputs(env, n: int, seed: int = 9901):
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    state = env.x0.detach().cpu().numpy()
    grid = torch.tensor(GF.axis_grid(state[:2], obs, float(env.r_robot)))
    hist = torch.zeros(GF.K_HIST, 2)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n, GF.H_PRED * 2), generator=gen)
    direction = torch.randn((n, GF.H_PRED * 2), generator=gen)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-9)
    contexts = {}
    for gamma in GAMMAS:
        low = torch.tensor(GF.low5(state, goal, gamma))
        contexts[gamma] = (grid, low, hist)
    return x, direction, contexts


def _origin_field(policy, env, n: int = 256):
    """Field norm, local x-Lipschitz ratio, and fixed-seed output statistics at origin."""
    x, direction, contexts = _origin_inputs(env, n)
    eps = 1e-2
    tau = torch.full((n,), 0.5)
    field_by_gamma = {}
    out = {}
    with torch.no_grad():
        for gamma, (grid, low, hist) in contexts.items():
            ctx = policy.ctx_from(grid, low, hist)
            ctx = policy._expand_ctx(ctx[0], n)
            v = policy(x, tau, ctx)
            vp = policy(x + eps * direction, tau, ctx)
            lip = (vp - v).norm(dim=1) / eps
            field_by_gamma[gamma] = v

            # Identical latent draws across checkpoints and gammas.
            torch.manual_seed(314159)
            U = policy.sample_window(grid, low, hist, n=n, temp=1.0, nfe=6)
            flat = U.reshape(n, -1)
            centered = flat - flat.mean(0, keepdim=True)
            svals = torch.linalg.svdvals(centered / np.sqrt(max(n - 1, 1)))
            energy = svals.square()
            effective_rank = float(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
            out[str(gamma)] = {
                "field_norm": float(v.norm(dim=1).mean()),
                "local_lipschitz_mean": float(lip.mean()),
                "local_lipschitz_p95": float(torch.quantile(lip, 0.95)),
                "sample_std": float(flat.std(0).mean()),
                "sample_effective_rank": effective_rank,
                "sample_clip_fraction": float((flat.abs() >= policy.u_max - 1e-7).float().mean()),
                "first_action_mean": [float(z) for z in U[:, 0].mean(0)],
                "first_action_std": [float(z) for z in U[:, 0].std(0)],
            }

        # Same x and physical context, only raw gamma differs.
        stack = torch.stack([field_by_gamma[g] for g in GAMMAS])
        consecutive = (stack[1:] - stack[:-1]).norm(dim=2).mean()
        total = (stack[-1] - stack[0]).norm(dim=1).mean()
        noise_scale = stack.norm(dim=2).std(dim=1).mean().clamp_min(1e-9)
        out["gamma_sensitivity"] = {
            "mean_consecutive_field_l2": float(consecutive),
            "g1_minus_g01_field_l2": float(total),
            "field_l2_over_within_gamma_norm_std": float(total / noise_scale),
        }
    return out, {str(g): field_by_gamma[g] for g in GAMMAS}


def _parameter_stats(policy, base):
    sd, bd = policy.state_dict(), base.state_dict()
    out = {}
    for group, prefix in (("trunk", "trunk."), ("head", "head."), ("encoder", "enc_grid.")):
        num = den = 0.0
        for name, value in sd.items():
            if name.startswith(prefix):
                num += float(((value - bd[name]) ** 2).sum())
                den += float((bd[name] ** 2).sum())
        out[group] = float(np.sqrt(num / max(den, 1e-30)))
    W = policy.trunk[0].weight.detach()
    # Input order is x20, raw low5, grid token32, Fourier time32.
    out["first_layer_input_fro"] = {
        "x20": float(W[:, :20].norm()),
        "relgoal_velocity4": float(W[:, 20:24].norm()),
        "gamma1": float(W[:, 24:25].norm()),
        "grid_token32": float(W[:, 25:57].norm()),
        "time32": float(W[:, 57:89].norm()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "vector_field_probe.json"))
    ap.add_argument("--demo-per-gamma", type=int, default=96)
    ap.add_argument("--origin-n", type=int, default=256)
    ap.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"))
    args = ap.parse_args()
    default = [
        ("base", "../../results/hp_repr/pretrained_a32uni.pt"),
        ("it18", "results/p2/balanced_k14_s7_from_it15/probe_best.pt"),
        ("s15_it30", "results/p2/finalunit_q50_k14_s15_from_it18/ckpt_30.pt"),
        ("s15_it100", "results/p2/finalunit_q50_k14_s15_from_it18/ckpt_100.pt"),
        ("s16_it100", "results/p2/finalunit_q50_k14_s16_from_it18/ckpt_100.pt"),
    ]
    entries = args.checkpoint or default
    entries = [(name, path if os.path.isabs(path) else os.path.join(ROOT, path)) for name, path in entries]
    policies = OrderedDict((name, HP.load_hp(path, "cpu")[0].eval()) for name, path in entries)
    base = policies[next(iter(policies))]
    demo = _balanced_demo(args.demo_per_gamma)
    env = GS.make_grid()
    result = {"checkpoints": {}, "pairwise_origin_field_relative_l2": {}}
    fields = {}
    for name, policy in policies.items():
        origin, fields[name] = _origin_field(policy, env, args.origin_n)
        result["checkpoints"][name] = {
            "path": dict(entries)[name],
            "parameters": _parameter_stats(policy, base),
            "balanced_demo_fixed_cfm": _fixed_cfm(policy, demo),
            "origin": origin,
        }
    names = list(policies)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            vals = []
            for gamma in GAMMAS:
                a, b = fields[left][str(gamma)], fields[right][str(gamma)]
                vals.append(float((a - b).norm(dim=1).mean() / a.norm(dim=1).mean().clamp_min(1e-9)))
            result["pairwise_origin_field_relative_l2"][f"{left}__{right}"] = {
                "mean": float(np.mean(vals)), "per_gamma": dict(zip(map(str, GAMMAS), vals))
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
