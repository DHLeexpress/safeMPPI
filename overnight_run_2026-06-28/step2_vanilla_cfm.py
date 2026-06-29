"""Step 2: VANILLA obstacle-conditioned flow-matching policy (lightened Mizuta backbone).

Trains `ContextualTransformerModel` (small) with the repo's `safe_cfm_loss` on a fixed scene whose
demo set covers all homotopy modes, then samples MANY control sequences with a plain Euler ODE (NO
MPPI, NO rejection) -> shows multimodal behavior. This is the "Mizuta vanilla generative policy".

  python overnight_run_2026-06-28/step2_vanilla_cfm.py --scenes single gap --device cuda
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "src"))

from cfm_mppi.models.contextual_transformer import ContextualTransformerModel
from cfm_mppi.training.train_loop_safe_cfm import safe_cfm_loss
from cfm_mppi.models.context_encoder import context_kwargs_from_batch
import scenes as S

FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
HIST = 4


def mode_of(states_xy, sc):
    """lateral y at the obstacle longitude -> mode. single:0 left(+),1 right(-); gap:0 left,1 gap,2 right."""
    ox = sc.obstacles[:, 0].mean()
    k = np.argmin(np.abs(states_xy[:, 0] - ox))
    y = states_xy[k, 1]
    if sc.name == "single":
        return 0 if y >= 0 else 1
    if y >= 0.5:
        return 0
    if y <= -0.5:
        return 2
    return 1


def build_dataset(sc, per_mode=240, noise=0.25):
    """Safe + goal-reaching demos covering all modes -> controls (accelerations) + states."""
    if sc.name == "single":
        ranges = [(1.1, 1.9), (-1.9, -1.1)]                # left, right
    else:
        ranges = [(1.7, 2.3), (-0.35, 0.35), (-2.3, -1.7)]  # left, gap, right
    Cs, States = [], []
    rng = np.random.default_rng(0)
    for lo, hi in ranges:
        got = 0; tries = 0
        while got < per_mode and tries < per_mode * 30:
            tries += 1
            lat = rng.uniform(lo, hi)
            st = S.make_trajectory(sc, lat, sigma=noise, seed=int(rng.integers(1 << 30)))
            xy = st[:, :2]
            if S.clearance(xy, sc).min() < 0.0:
                continue
            if np.linalg.norm(xy[-1] - sc.goal) > 0.9:
                continue
            v = st[:, 2:4]
            u = (v[1:] - v[:-1]) / sc.dt                    # realized accelerations [T,2]
            Cs.append(u.astype(np.float32)); States.append(st.astype(np.float32)); got += 1
    controls = torch.tensor(np.stack(Cs))                  # [N,T,2]
    states = torch.tensor(np.stack(States))                # [N,T+1,4]
    return controls, states


def scene_context(sc, n, device):
    start = torch.tensor(sc.x0[:2], dtype=torch.float32).repeat(n, 1)
    goal = torch.tensor(sc.goal, dtype=torch.float32).repeat(n, 1)
    o = sc.obstacles
    nearest = o[np.argmin(np.linalg.norm(o[:, :2] - sc.x0[:2], axis=1))]
    rel = np.array([nearest[0] - sc.x0[0], nearest[1] - sc.x0[1], 0.0, 0.0], dtype=np.float32)
    ego0 = torch.tensor(sc.x0, dtype=torch.float32)
    return {
        "start": start.to(device), "goal": goal.to(device),
        "ego_current": ego0.repeat(n, 1).to(device),
        "ego_history": ego0.repeat(n, HIST, 1).to(device),
        "action_history": torch.zeros(n, HIST, 2, device=device),
        "nearest_obstacle_history": torch.tensor(rel).repeat(n, HIST, 1).to(device),
        "gamma": torch.full((n,), 0.5, device=device),
        "safety_margin": torch.full((n,), 0.5, device=device),
    }


@torch.no_grad()
def sample_controls(model, sc, n, device, nfe=10):
    ctx = scene_context(sc, n, device)
    x = torch.randn(n, 2, sc.T, device=device)
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    for i in range(nfe):
        t = torch.full((n,), float(ts[i]), device=device)
        x = x + (ts[i + 1] - ts[i]) * model(x, t, **ctx)
    return x.transpose(1, 2)                                # [n,T,2] controls


def rollout(controls, sc):
    n = controls.shape[0]
    x = torch.tensor(sc.x0, dtype=torch.float32, device=controls.device).repeat(n, 1)
    out = [x]
    for t in range(sc.T):
        u = controls[:, t]; p, v = x[:, :2], x[:, 2:4]
        x = torch.cat([p + sc.dt * v + 0.5 * sc.dt ** 2 * u, v + sc.dt * u], 1)
        out.append(x)
    return torch.stack(out, 1)                              # [n,T+1,4]


def train_and_plot(sc, device, steps=1500, bs=256, nsamp=300):
    controls, states = build_dataset(sc)
    N = controls.shape[0]
    print(f"[{sc.name}] dataset N={N} T={sc.T}")
    model = ContextualTransformerModel.from_mizuta_defaults(
        d_model=128, nhead=4, num_layers=3, dim_feedforward=256, history_len=HIST).to(device)
    ctx_full = scene_context(sc, N, device)
    batch_base = {"controls_si": controls.to(device), "states": states.to(device), **ctx_full}
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for it in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=device)
        batch = {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == N else v) for k, v in batch_base.items()}
        loss, _ = safe_cfm_loss(model, batch, device)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 5) == 0:
            print(f"  it {it} loss {float(loss):.4f}")
    model.eval()

    U = sample_controls(model, sc, nsamp, device).clamp(-sc.u_max, sc.u_max)
    st = rollout(U, sc).cpu().numpy()
    safe = S.clearance(st[:, :, :2], sc).min(1) >= 0.0
    modes = np.array([mode_of(st[i, :, :2], sc) for i in range(nsamp)])
    cols = {0: "#1a9850", 1: ("#2166ac" if sc.name == "single" else "#f46d43"), 2: "#2166ac"}
    names = (["LEFT", "RIGHT"] if sc.name == "single" else ["LEFT", "GAP", "RIGHT"])

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for (cx, cy, r) in sc.obstacles:
        ax.add_patch(Circle((cx, cy), r, facecolor="#7b3294", alpha=0.32, edgecolor="#4d004b", lw=1.1, zorder=4))
        ax.add_patch(Circle((cx, cy), r + sc.r_robot, facecolor="none", edgecolor="#7b3294", ls="--", lw=0.8, alpha=0.5, zorder=4))
    for i in range(nsamp):
        c = cols[int(modes[i])] if safe[i] else "0.6"
        ax.plot(st[i, :, 0], st[i, :, 1], color=c, alpha=0.18 if safe[i] else 0.06, lw=0.7, zorder=(6 if safe[i] else 5))
    ax.scatter(*sc.x0[:2], s=80, c="#1a9850", edgecolor="k", zorder=9)
    ax.scatter(*sc.goal, s=150, c="gold", edgecolor="k", marker="*", zorder=9)
    ax.set_xlim(*sc.xlim); ax.set_ylim(*sc.ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cnt = [int(((modes == m) & safe).sum()) for m in range(len(names))]
    ax.set_title(f"ENV {sc.name} — Stage 2: vanilla CFM (lightened Mizuta backbone)\n"
                 f"{nsamp} samples · safe {int(safe.sum())} · " + "  ".join(f"{names[m]}:{cnt[m]}" for m in range(len(names))),
                 fontsize=9)
    fig.tight_layout(); p = os.path.join(FIGDIR, f"{sc.name}_stage2_vanilla_cfm.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved {p}  modes={cnt} safe={int(safe.sum())}/{nsamp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["single", "gap"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    for name in args.scenes:
        train_and_plot(S.make_scene(name), args.device, steps=args.steps)


if __name__ == "__main__":
    main()
