"""Progressive safe-flow-expansion video (Pillar 5 headline visual).

Base layer = the pretrained FM policy's trajectories (faint gray).  Then, over expansion rounds,
overlay the SOCP-VERIFIER-CERTIFIED FM trajectories (colored by gamma) as they fill the free
space, alongside the rising spatial-coverage / validity curves.  Produces a per-gamma-combined
gif/mp4 and a static pretrained-vs-expanded comparison.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

import _paths
import env as E
from dynamics import rollout, clip_controls
from scene_encoder import SceneConditionedFlowPolicy
from socp_gate import make_socp_validity_label


def rebuild(expand):
    pc = expand["policy_cfg"]
    sc = expand["scene"]
    env = E.env_from_obstacles(sc["obstacles"], sc["start"], sc["goal"], T=pc["T"], dt=pc["dt"],
                               u_max=pc["u_max"], r_robot=pc["r_robot"], box=pc["box"])

    def make_pol():
        return SceneConditionedFlowPolicy(T=pc["T"], token_dim=pc["token_dim"], width=pc["width"],
                                          depth=pc["depth"], n_max=pc["n_max"], S=pc["S"],
                                          R_enc=pc["R_enc"], r_robot=pc["r_robot"], u_max=pc["u_max"])
    return env, make_pol


@torch.no_grad()
def sample_paths(state, make_pol, env, ctx, n):
    pol = make_pol()
    pol.load_state_dict(state)
    pol.eval()
    U = clip_controls(pol.sample(n, ctx), env)
    return rollout(U, env)[:, :, :2].cpu().numpy(), U


@torch.no_grad()
def certified_paths(state, make_pol, env, ctx, gamma, n, gate, target=40, max_batches=6):
    """Oversample until ~target verifier-certified trajectories are collected (dense video)."""
    pol = make_pol()
    pol.load_state_dict(state)
    pol.eval()
    out, got = [], 0
    for _ in range(max_batches):
        U = clip_controls(pol.sample(n, ctx), env)
        valid, safe, states, _ = gate(U, env, gamma, None)
        pv = states[valid][:, :, :2].cpu().numpy()
        if len(pv):
            out.append(pv)
            got += len(pv)
        if got >= target:
            break
    return np.concatenate(out, 0) if out else np.zeros((0, env.T + 1, 2))


def draw_scene(ax, env):
    gx = np.linspace(*env.xlim, 200)
    gy = np.linspace(*env.ylim, 200)
    GX, GY = np.meshgrid(gx, gy)
    Z = E.clearance_field(env, GX, GY)
    ax.contourf(GX, GY, (Z < 0).astype(float), levels=[0.5, 1.5], colors=["#e0e0e0"], alpha=0.6, zorder=0)
    ax.contour(GX, GY, Z, levels=[0.0], colors="#737373", linewidths=0.7, zorder=1)
    for (ox, oy, rr) in env.obstacles.cpu().numpy():
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
    ax.scatter([env.x0[0]], [env.x0[1]], s=70, c="#00a000", edgecolor="k", zorder=10)
    ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=200, c="gold", edgecolor="k", zorder=10)
    ax.set_xlim(*env.xlim)
    ax.set_ylim(*env.ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def make_video(expands, out_path, n_paths=120, device="cpu", log=print):
    """expands: dict gamma -> loaded expand file (all share the scene)."""
    gammas = sorted(expands.keys())
    env, make_pol = rebuild(expands[gammas[0]])
    gate = make_socp_validity_label(env, R_ver=2.0, H_win=10, stride=2, reach_radius=0.6)
    cmap = plt.get_cmap("viridis")
    gmin, gmax = min(gammas), max(gammas)
    gcolor = {g: cmap((g - gmin) / max(gmax - gmin, 1e-6)) for g in gammas}

    # faint pretrained base (per gamma) + rounds list (shared)
    base = {}
    for g in gammas:
        ctx = expands[g]["ctx"].to(device)
        base[g], _ = sample_paths(expands[g]["pretrained_state"], make_pol, env, ctx, n_paths)
    rounds = sorted(expands[gammas[0]]["snapshots"].keys())

    # precompute certified paths per (round, gamma)
    log(f"rendering {len(rounds)} rounds x {len(gammas)} gamma ...")
    certs = {}
    for r in rounds:
        for g in gammas:
            ctx = expands[g]["ctx"].to(device)
            certs[(r, g)] = certified_paths(expands[g]["snapshots"][r], make_pol, env, ctx, g, n_paths, gate)
    hist = {g: expands[g]["history"] for g in gammas}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle("Pretrained γ-colored policy (faint) → verifier-certified trajectories fill the free space",
                 fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.02, right=0.97, wspace=0.16)

    def frame(i):
        r = rounds[i]
        axL.clear()
        draw_scene(axL, env)
        for g in gammas:                                    # faint pretrained base
            for p in base[g]:
                axL.plot(p[:, 0], p[:, 1], "-", color="0.6", lw=0.4, alpha=0.12, zorder=4)
        n_cert = 0
        for g in gammas:                                    # certified, colored by gamma
            for p in certs[(r, g)]:
                axL.plot(p[:, 0], p[:, 1], "-", color=gcolor[g], lw=0.8, alpha=0.5, zorder=6)
            n_cert += len(certs[(r, g)])
        handles = [plt.Line2D([], [], color=gcolor[g], lw=2, label=f"γ={g}") for g in gammas]
        handles.append(plt.Line2D([], [], color="0.6", lw=2, alpha=0.5, label="pretrained"))
        axL.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
        axL.set_title(f"Safe Flow Expansion — round {r}  ·  {n_cert} SOCP-certified trajectories", fontsize=11)

        axR.clear()
        for g in gammas:
            hr = [h["round"] for h in hist[g]]
            sc = [h["spatial_coverage"] for h in hist[g]]
            va = [h["validity"] for h in hist[g]]
            axR.plot(hr, sc, "-o", color=gcolor[g], ms=3, lw=1.4, label=f"cov γ={g}")
            axR.plot(hr, va, "--", color=gcolor[g], lw=1.0, alpha=0.7)
        axR.axvline(r, color="0.5", lw=0.8, ls=":")
        axR.set_xlabel("expansion round")
        axR.set_ylabel("spatial coverage (—) / validity (- -)")
        axR.set_ylim(-0.02, 1.02)
        axR.legend(fontsize=8, loc="upper left")
        axR.set_title("coverage ↑  (of verifier-reachable-safe Ω*)", fontsize=11)
        return []

    anim = FuncAnimation(fig, frame, frames=len(rounds), interval=700)
    anim.save(out_path, writer=PillowWriter(fps=2), dpi=90)
    plt.close(fig)
    log(f"saved {out_path}")

    # static pretrained-vs-expanded comparison (last round)
    fig2, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    for ax, (which, title) in zip(axes, [("pre", "Pretrained seed"), ("post", "After Safe Flow Expansion")]):
        draw_scene(ax, env)
        for g in gammas:
            paths = base[g] if which == "pre" else certs[(rounds[-1], g)]
            for p in paths:
                axL_c = "0.6" if which == "pre" else gcolor[g]
                ax.plot(p[:, 0], p[:, 1], "-", color=axL_c, lw=0.7, alpha=0.35, zorder=5)
        cov = "" if which == "pre" else \
            "\ncov " + ", ".join(f"γ{g}={hist[g][-1]['spatial_coverage']:.2f}" for g in gammas)
        ax.set_title(title + cov, fontsize=10)
    fig2.suptitle("Coverage of the verified-safe set expands (each trajectory SOCP-certified)", fontsize=12, y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    still = out_path.replace(".gif", "_compare.png")
    fig2.savefig(still, dpi=130)
    plt.close(fig2)
    log(f"saved {still}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_paths.HERE, "results"))
    ap.add_argument("--out", default=os.path.join(_paths.HERE, "figures", "safeflow_expansion.gif"))
    ap.add_argument("--n-paths", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "expand_g*.pt")))
    if not files:
        raise SystemExit("no expand_g*.pt files — run expand.py first")
    expands = {}
    for f in files:
        d = torch.load(f, weights_only=False)
        expands[float(d["gamma"])] = d
    print(f"=== video from {len(expands)} gamma runs: {sorted(expands.keys())} ===", flush=True)
    make_video(expands, args.out, n_paths=args.n_paths, device=args.device)


if __name__ == "__main__":
    main()
