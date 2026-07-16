#!/usr/bin/env python3
"""Same-latent support probe at the WALLS-4 interior `(1,1)` pinch.

Reuses `seed12_tail_trace`'s faithful trace and explicit NFE8 integration machinery.
For each fixed failing fiber, the final pre-collision context from a source checkpoint
is held constant while identical latents are mapped through every candidate checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT.parent), str(ROOT.parent.parent)]

import seed12_tail_trace as ST
import grid_expand_hardtail as HT


DEFAULT_CASES = ((0.4, 16), (0.4, 28), (0.4, 48))


@torch.no_grad()
def score(policy, step, env, latents, fail_x0, device):
    g = torch.as_tensor(step["grid"], device=device)
    l = torch.as_tensor(step["low5"], device=device)
    h = torch.as_tensor(step["hist"], device=device)
    ctx = ST._ctx_of(policy, g, l, h)
    X = latents.to(device).clone(); X[0] = torch.as_tensor(fail_x0, device=device)
    U, _ = ST.integrate(policy, ctx, X, nfe=8)
    U = U.detach().cpu().numpy()
    state = np.asarray(step["state"], np.float32)
    positions = ST.GR.di_rollout_batch(state, U, env.dt)
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    d = np.linalg.norm(positions[:, :, None] - obs[None, None, :, :2], axis=3) - obs[None, None, :, 2] - rr
    min_d = d.min(axis=(1, 2)); first_d = d[:, 0].min(axis=1)
    return dict(one_step_collision=float((first_d < 0).mean()),
                window_collision=float((min_d < 0).mean()),
                clearance_p01=float(np.percentile(min_d, 1)),
                clearance_p10=float(np.percentile(min_d, 10)),
                clearance_median=float(np.median(min_d)),
                fail_one_step_collision=bool(first_d[0] < 0),
                fail_window_collision=bool(min_d[0] < 0),
                fail_min_clearance=float(min_d[0]),
                min_clearance=min_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="checkpoint whose failing contexts/latents are traced")
    ap.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--cases", nargs="*", default=[], metavar="GAMMA:SEED")
    ap.add_argument("--n-latents", type=int, default=4096)
    ap.add_argument("--context-offset", type=int, default=0,
                    help="number of replans before the final pre-collision context")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    args = ap.parse_args()
    cases = DEFAULT_CASES if not args.cases else tuple(
        (float(x.split(":", 1)[0]), int(x.split(":", 1)[1])) for x in args.cases)
    env = ST.GS.make_grid(); HT._apply_wall_plugs(env, 4)
    source, _ = ST.HP.load_hp(args.source, device=args.device); source.eval()
    policies = {}
    for name, path in args.checkpoint:
        pol, _ = ST.HP.load_hp(path, device=args.device); pol.eval(); policies[name] = pol
    gen = torch.Generator().manual_seed(20260711)
    latents = torch.randn(args.n_latents, source.d, generator=gen)
    result = dict(source=args.source, checkpoints={k: p for k, p in args.checkpoint},
                  cases=[list(x) for x in cases], n_latents=args.n_latents, wall_plugs=4, results={})
    hist_data = {name: [] for name in policies}
    for gamma, seed in cases:
        trace = ST.trace_deploy(source, env, gamma, seed, nfe=8, reach=.1, device=args.device)
        verified, faithful = ST.verify_trace(source, env, gamma, seed, trace, device=args.device)
        if not verified:
            raise RuntimeError(f"source trace mismatch for gamma={gamma}, seed={seed}")
        if trace["reached"]:
            raise RuntimeError(f"source case unexpectedly succeeds: gamma={gamma}, seed={seed}")
        step_index = max(0, len(trace["steps"]) - 1 - args.context_offset)
        step = trace["steps"][step_index]
        case_key = f"g{gamma}_s{seed}"
        result["results"][case_key] = dict(source_steps=len(trace["steps"]), context_step=step_index,
                                            context_offset=args.context_offset,
                                            source_dead=bool(trace["dead"]), faithful_match=verified,
                                            checkpoints={})
        for name, policy in policies.items():
            rec = score(policy, step, env, latents, step["x0"], args.device)
            hist_data[name].append(rec.pop("min_clearance"))
            result["results"][case_key]["checkpoints"][name] = rec
            print(case_key, name, rec, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 5), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ci, (gamma, seed) in enumerate(cases):
        ax = axes[0, ci]
        lo = min(float(x[ci].min()) for x in hist_data.values())
        hi = max(float(np.percentile(x[ci], 99.5)) for x in hist_data.values())
        bins = np.linspace(lo, hi, 65)
        for j, (name, values) in enumerate(hist_data.items()):
            ax.hist(values[ci], bins=bins, histtype="step", density=True, lw=2,
                    color=colors(j), label=name)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(f"gamma={gamma}, seed={seed}\n{args.context_offset} replans before final pre-collision context")
        ax.set_xlabel("planned-window minimum clearance (m)"); ax.set_ylabel("density")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("WALLS-4 `(1,1)` pinch latent support — identical contexts and base latents")
    fig.tight_layout(); args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", args.out, "and", args.fig)


if __name__ == "__main__":
    main()
