"""Pretrain the gamma + scene-conditioned FM policy (Pillar 4) on the SafeMPPI dataset.

Jointly trains the DeepSets scene encoder and the FlowPolicy trunk/head with the conditional
flow-matching loss (CondOT path, velocity MSE — the mid-flow-noise mechanism is left intact).
Produces the headline "pretrained policies" figure: sampled trajectories per gamma, colored by
gamma, overlaid on one fixed cluttered scene (the conservative, γ-graded seed that expansion opens up).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths
import env as E
from dynamics import rollout, clip_controls
from scene_encoder import SceneConditionedFlowPolicy
import wandb_utils as W


def rebuild_bank(data, device="cpu"):
    box = data["bank_params"]["scene"]["box"]
    return [E.env_from_obstacles(s["obstacles"], s["start"], s["goal"], T=data["T"], dt=data["dt"],
                                 u_max=data["u_max"], r_robot=data["r_robot"], box=box, device=device)
            for s in data["scenes"]]


def make_policy(data, token_dim=32, width=256, depth=3, device="cpu"):
    box = data["bank_params"]["scene"]["box"]
    pol = SceneConditionedFlowPolicy(
        T=data["T"], token_dim=token_dim, width=width, depth=depth,
        S=2.0 * box, R_enc=2.5 * box, r_robot=data["r_robot"], u_max=data["u_max"],
    ).to(device)
    return pol


def train(policy, data, steps=2500, lr=5e-4, batch=128, device="cpu", log=print, run=None):
    U = data["U"].to(device)
    scene_id = data["scene_id"].to(device)
    gamma = data["gamma"].to(device)
    M = U.shape[0]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    last = float("nan")
    for i in range(steps):
        bi = torch.randint(0, M, (min(batch, M),), device=device)
        ctx = policy.ctx_for(scene_id[bi], gamma[bi])
        loss = policy.cfm_loss(U[bi], ctx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
        W.log(run, {"pretrain/cfm_loss": last, "pretrain/lr": lr}, step=i)
        if i % max(1, steps // 6) == 0 or i == steps - 1:
            log(f"  pretrain {i}/{steps} loss={last:.4f}")
    policy.eval()
    return last


@torch.no_grad()
def plot_gamma_overlay(policy, env, gammas, path, n=80, device="cpu", title=None):
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    box = float(env.ylim[1] - 0.1)
    gx = np.linspace(*env.xlim, 200)
    gy = np.linspace(*env.ylim, 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = E.clearance_field(env, GX, GY)
    ax.contourf(GX, GY, (Z < 0).astype(float), levels=[0.5, 1.5], colors=["#d9d9d9"], alpha=0.5, zorder=0)
    ax.contour(GX, GY, Z, levels=[0.0], colors="#525252", linewidths=0.8, zorder=1)
    for (ox, oy, rr) in env.obstacles.cpu().numpy():
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.8, alpha=0.75, zorder=4))
    cmap = plt.get_cmap("viridis")
    gmin, gmax = min(gammas), max(gammas)
    for g in gammas:
        color = cmap((g - gmin) / max(gmax - gmin, 1e-6))
        ctx = policy.ctx_for_env(env, g).detach()
        U = clip_controls(policy.sample(n, ctx), env)
        paths = rollout(U, env)[:, :, :2].cpu().numpy()
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=0.7, alpha=0.35, zorder=5)
        ax.plot([], [], "-", color=color, lw=2.0, label=f"γ={g}")
    ax.scatter([env.x0[0]], [env.x0[1]], s=70, c="#00a000", edgecolor="k", zorder=10, label="start")
    ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=200, c="gold", edgecolor="k", zorder=10, label="goal")
    ax.set_xlim(*env.xlim)
    ax.set_ylim(*env.ylim)
    ax.set_aspect("equal")
    ax.set_title(title or "Pretrained FM policy — γ-colored trajectories (conservative seed)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_paths.HERE, "results", "dataset.pt"))
    ap.add_argument("--out", default=os.path.join(_paths.HERE, "results", "pretrained.pt"))
    ap.add_argument("--fig", default=os.path.join(_paths.HERE, "figures", "pretrained_gamma_overlay.png"))
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--token-dim", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--viz-scene", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    if args.smoke:
        args.steps = min(args.steps, 500)

    data = torch.load(args.data, weights_only=False)
    device = args.device
    bank = rebuild_bank(data, device=device)
    policy = make_policy(data, token_dim=args.token_dim, width=args.width, depth=args.depth, device=device)
    policy.attach_bank(bank)
    n_demos = int(data["U"].shape[0])
    print(f"=== pretrain: {n_demos} demos, {len(bank)} scenes, T={data['T']}, "
          f"ctx_dim={6 + args.token_dim}, steps={args.steps} ===", flush=True)

    run = W.init_run(args, name=f"pretrain-{n_demos}demos", dir=os.path.join(_paths.HERE, "results"),
                     config={"stage": "pretrain", "steps": args.steps, "batch": 128, "lr": 5e-4,
                             "token_dim": args.token_dim, "width": args.width, "depth": args.depth,
                             "T": data["T"], "ctx_dim": 6 + args.token_dim, "n_demos": n_demos,
                             "n_scenes": len(bank), "gammas": data["gammas"], "smoke": args.smoke})
    final_loss = train(policy, data, steps=args.steps, device=device, run=run)

    ckpt = {
        "state_dict": policy.state_dict(),
        "T": data["T"], "token_dim": args.token_dim, "width": args.width, "depth": args.depth,
        "S": policy.S, "R_enc": policy.R_enc, "r_robot": data["r_robot"], "n_max": policy.n_max,
        "u_max": data["u_max"], "dt": data["dt"], "box": data["bank_params"]["scene"]["box"],
        "scenes": data["scenes"], "gammas": data["gammas"],
    }
    torch.save(ckpt, args.out)
    print("saved", args.out, flush=True)

    viz_env = bank[args.viz_scene % len(bank)]
    plot_gamma_overlay(policy, viz_env, data["gammas"], args.fig, device=device)
    print("saved", args.fig, flush=True)
    W.log_image(run, "pretrained_gamma_overlay", args.fig)
    W.finish(run, summary={"final_cfm_loss": final_loss, "n_demos": n_demos})


if __name__ == "__main__":
    main()
