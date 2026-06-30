"""Step 2 visualization: grid of synchronized GIFs (episodes x gamma) of the SINGLE-INTEGRATOR SafeMPPI
rollout on REAL UCY/SDD pedestrian crowds, with polytope_v2 + nested {H>=(1-gamma)^i} level sets.

Rows = episodes, columns = gamma. gamma is ISOLATED (use_guidance=False, use_aniso_cov=False, safety_margin=0)
so we can read its effect cleanly. Horizon H=10 (= MPPI horizon). Mizuta-matched core params.

  LD_PRELOAD=.../libstdc++.so.6 python overnight_run_2026-06-28/step2_safemppi_grid.py --dataset ucy
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.evaluation.render_validation_comparison import get_parser as _vparser, _make_scene

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DT = 0.1; GAMMAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def norm_barrier(poly, grid):
    mr = (poly.b - poly.A @ poly.ref).clamp_min(1e-3)
    val = (poly.b.unsqueeze(0) - grid @ poly.A.T) / mr.unsqueeze(0)
    return val.min(dim=1).values


def _si_step(state, a, dt):
    s = state.copy(); s[0] += dt * a[0]; s[1] += dt * a[1]; s[2] = a[0]; s[3] = a[1]; return s


def load_scene(dataset, episode, steps):
    args = _vparser().parse_args([])
    args.dataset = dataset; args.pedestrian_source = "validation"; args.dynamics = "doubleintegrator"
    args.steps = steps; args.pedestrian_radius = 0.5; args.episode = episode; args.seed = episode
    state0, goal, obstacles_seq, velocities_seq, label = _make_scene(args)
    s0 = np.asarray(state0, float).reshape(-1)
    s0 = np.array([s0[0], s0[1], 0.0, 0.0]) if s0.shape[0] >= 2 else np.zeros(4)
    return s0, np.asarray(goal, float).reshape(2), np.asarray(obstacles_seq, float), np.asarray(velocities_seq, float), label


def clutter_score(obstacles_seq, s0, goal):
    # pedestrians near the straight start->goal corridor, averaged over time
    a = s0[:2]; b = goal; ab = b - a; L = np.linalg.norm(ab) + 1e-9; u = ab / L
    sc = 0.0; T = obstacles_seq.shape[0]
    for t in range(0, T, 5):
        ob = obstacles_seq[min(t, T - 1)]
        if ob.shape[0] == 0:
            continue
        ok = ~np.isnan(ob[:, :2]).any(1)
        p = ob[ok, :2] - a
        proj = np.clip(p @ u, 0, L); foot = a + np.outer(proj, u)
        dist = np.linalg.norm(ob[ok, :2] - foot, axis=1)
        sc += np.sum(dist < 1.5)
    return sc


def pick_episodes(dataset, steps, k, n_scan=80):
    scored = []
    for i in range(n_scan):
        try:
            s0, goal, obs, vel, _ = load_scene(dataset, i, steps)
            scored.append((clutter_score(obs, s0, goal), i))
        except Exception:
            continue
    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def make_adapter(gamma, horizon, samples, umax, sensing):
    return SafeMPPIAdapter(
        horizon=horizon, dt=DT, num_samples=samples, gamma=gamma,
        noise_sigma=(0.4, 0.4), u_min=(-umax, -umax), u_max=(umax, umax),
        safety_margin=0.0, dynamics_type="singleintegrator",
        use_ho_barrier=False, eta=0.0, use_guidance=False, use_aniso_cov=False,
        barrier_topk=0, barrier_activation_radius=sensing,
    )


def rollout(adapter, s0, goal, obstacles_seq, velocities_seq, steps, gamma, dev, ep):
    state = s0.astype(np.float32).copy(); traj = [state[:2].copy()]; reached = False
    T = obstacles_seq.shape[0]
    for t in range(steps):
        if not reached:
            ob = obstacles_seq[min(t, T - 1)]; vel = velocities_seq[min(t, velocities_seq.shape[0] - 1)]
            ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vel = vel[ok] if vel.shape[0] == ok.shape[0] else vel
            a, _ = adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=dev),
                torch.tensor(goal, dtype=torch.float32, device=dev),
                torch.tensor(ob, dtype=torch.float32, device=dev),
                gamma=gamma,
                obstacle_velocities=torch.tensor(vel, dtype=torch.float32, device=dev),
                seed=ep * 100000 + t,
            )
            state = _si_step(state, a.detach().cpu().numpy(), DT)
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True
        traj.append(state[:2].copy())
    return np.array(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy", choices=["ucy", "sdd"])
    ap.add_argument("--episodes", nargs="+", type=int, default=None)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--umax", type=float, default=2.0)
    ap.add_argument("--sensing", type=float, default=3.5)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    eps = args.episodes or pick_episodes(args.dataset, args.steps, 3)
    print("episodes:", eps)

    # precompute scenes + rollouts
    scenes = {}; trajs = {}
    for ep in eps:
        s0, goal, obs, vel, label = load_scene(args.dataset, ep, args.steps)
        scenes[ep] = (s0, goal, obs, vel)
        for g in GAMMAS:
            adapter = make_adapter(g, args.horizon, args.samples, args.umax, args.sensing)
            trajs[(ep, g)] = rollout(adapter, s0, goal, obs, vel, args.steps, g, dev, ep)
        print(f"  episode {ep} ({label}): {obs.shape[1]} peds, rolled {len(GAMMAS)} gammas", flush=True)

    # plot window per episode
    lims = {}
    for ep in eps:
        s0, goal, obs, _ = scenes[ep]
        allxy = obs[..., :2].reshape(-1, 2); allxy = allxy[~np.isnan(allxy).any(1)]
        pts = np.vstack([allxy, s0[:2][None], goal[None]])
        pad = args.sensing * 0.6
        lims[ep] = ((pts[:, 0].min() - pad, pts[:, 0].max() + pad), (pts[:, 1].min() - pad, pts[:, 1].max() + pad))

    R, C = len(eps), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(2.7 * C, 2.7 * R), squeeze=False)
    nframes = args.steps + 1

    def draw(f):
        for r, ep in enumerate(eps):
            s0, goal, obs, _ = scenes[ep]; (xl, yl) = lims[ep]
            T = obs.shape[0]; ob = obs[min(f, T - 1)]
            ok = ~np.isnan(ob[:, :2]).any(1); peds = ob[ok]
            for c, g in enumerate(GAMMAS):
                ax = axes[r][c]; ax.clear()
                tr = trajs[(ep, g)]; cpos = tr[min(f, len(tr) - 1)]
                poly, info = build_polytope_v2(cpos, peds, sensing_range=args.sensing, n_base=args.K, margin=0.0)
                gx = np.linspace(*xl, 90); gy = np.linspace(*yl, 90); GX, GY = np.meshgrid(gx, gy)
                grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32)
                Hh = norm_barrier(poly, grid).numpy().reshape(GX.shape)
                lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
                ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.55, zorder=1)
                ax.contour(GX, GY, Hh, levels=[l for l in lv if l > 0], colors="#2166ac", linewidths=0.4, alpha=0.6, zorder=2)
                ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.3, zorder=3)
                for (px, py, rr) in peds:
                    ax.add_patch(Circle((px, py), rr, facecolor="#7b3294", alpha=0.45, edgecolor="#4d004b", lw=0.6, zorder=4))
                ax.plot(tr[:min(f, len(tr) - 1) + 1, 0], tr[:min(f, len(tr) - 1) + 1, 1], "-", color="#1a9850", lw=1.2, alpha=0.7, zorder=5)
                ax.scatter([cpos[0]], [cpos[1]], s=40, c="#1a9850", edgecolor="k", zorder=9)
                ax.scatter([goal[0]], [goal[1]], marker="*", s=90, c="#d62728", edgecolor="k", zorder=9)
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"γ={g}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"ep {ep}", fontsize=9)
        fig.suptitle(f"Step 2 — single-integrator SafeMPPI on real {args.dataset.upper()} crowds  ·  H={args.horizon}  ·  "
                     f"γ isolated (no guidance, margin 0)  ·  t={f}/{args.steps}", fontsize=11)
        return []

    anim = FuncAnimation(fig, draw, frames=nframes, interval=1000 // args.fps)
    p = os.path.join(FIG, "step2_safemppi_grid.gif")
    anim.save(p, writer=PillowWriter(fps=args.fps), dpi=85); print("saved", p)
    draw(nframes - 1); fig.savefig(os.path.join(FIG, "step2_safemppi_grid.png"), dpi=130)
    print("saved", os.path.join(FIG, "step2_safemppi_grid.png"))


if __name__ == "__main__":
    main()
