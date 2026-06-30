"""ep15 diagnostic — fix ONE UCY episode (15) to succeed collision-free, di_gamma-style viz.

Columns = gamma {0.1, 0.3, 0.5}. Rows = cumulative fixes (each builds on the previous):
  0 baseline   : pure SafeMPPI (SI, default, temperature on) — expected to fail (all-rejected -> goal-seek -> collide)
  1 small kappa: + small velocity-predictive gain (polytope must NOT vanish)
  2 +ctrl cost : + higher control-effort quadratic (stop straight-lining to goal)
  3 +sensing   : + reduced sensing range (pull level sets in; fewer DTCBF inactive; feasible samples exist)
  4 +escape    : + escape-bias (Suggestion 2); also shows the escape gradient + mean/cov

Viz (the new DEFAULT, like results/benchmark_videos/di_gamma.mp4): accepted (green) / rejected (red, X at endpoint)
ROLLOUT TRAJECTORIES + nested {H>=(1-gamma)^i} level sets + executed path + accept/reject counts.

  python overnight_run_2026-06-28/ep15_diagnostic.py            # full rows x gamma grid
  python overnight_run_2026-06-28/ep15_diagnostic.py --row 3    # tune a single row (prints collision per gamma)
"""
from __future__ import annotations
import argparse, os, sys
from copy import deepcopy
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2
from cfm_mppi.mppi.sweep import _load, velocity_inflate, _si_step, DT

FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
GAMMAS = [0.1, 0.3, 0.5]
R_ROBOT = 0.2
EP = 15

# base SI config. check_first_control_only=True is the ALL-REJECTION FIX: the full-horizon DCBF check against the
# single-nearest (jumpy) barrier rejects every rollout in a dense crowd; checking only the executed first control
# (+ replanning each step) restores feasible samples. noise_sigma raised so the sample spread is visibly Gaussian.
BASE = dict(horizon=10, dt=DT, num_samples=128, noise_sigma=(0.5, 0.5), temperature=1.0,
            u_min=(-2.0, -2.0), u_max=(2.0, 2.0), safety_margin=0.5, dynamics_type="singleintegrator",
            use_ho_barrier=False, eta=0.0, use_guidance=False, use_aniso_cov=False,
            umax_react_dist=0.0, umax_min_frac=0.0,
            barrier_topk=0, barrier_activation_radius=3.5, control_weight=0.03, escape_gain=0.0)

# cumulative fix rows. Label = the ACTUAL parameter change (old -> new). FULL recursive check stays on.
# Zeroth fix = proximity-adaptive DIRECTIONAL speed cap + variance blow-open (umax_react_dist); carried by all rows.
ADP = dict(umax_react_dist=2.0, umax_min_frac=0.0, sigma_expand_gain=1.0)
ROWS = [
    ("baseline\nu_max=2 fixed",         dict(),                                                                       0.0),
    ("0: u_max->adaptive\n+variance (react=2)", dict(ADP),                                                            0.0),
    ("1: +predict_gain 0->0.4",         dict(ADP),                                                                    0.4),
    ("2: +control_weight 0.03->0.15",   dict(ADP, control_weight=0.15),                                               0.4),
    ("3: +sensing 3.5->2.0",            dict(ADP, control_weight=0.15, barrier_activation_radius=2.0),                0.4),
    ("4: +escape_gain 0->0.10",         dict(ADP, control_weight=0.15, barrier_activation_radius=2.0, escape_gain=0.10), 0.4),
]


def escape_dir(p, peds):
    if peds.shape[0] == 0:
        return np.zeros(2), 0.0
    rel = peds[:, :2] - p[None]; d = np.linalg.norm(rel, axis=1)
    clr = np.clip(d - peds[:, 2] - R_ROBOT, 1e-3, None)
    m = rel / np.clip(d, 1e-9, None)[:, None]
    g = (m / clr[:, None]).sum(0); gn = float(np.linalg.norm(g))
    return (-g / gn if gn > 1e-6 else np.zeros(2)), gn


def record(cfg, predict_gain, gamma, steps=80, dev="cuda"):
    s0, goal, obs, vel = _load("ucy", EP, steps)
    tau = cfg["horizon"] * DT
    ad = SafeMPPIAdapter(**cfg)
    state = s0.astype(np.float32).copy(); T = obs.shape[0]; rec = []; reached = False; min_clear = np.inf
    gt = torch.tensor(goal, dtype=torch.float32, device=dev); path = [state[:2].copy()]
    for t in range(steps):
        ob = obs[min(t, T - 1)]; vl = vel[min(t, vel.shape[0] - 1)]
        ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vl = vl[ok] if vl.shape[0] == ok.shape[0] else vl
        if not reached:
            obp = velocity_inflate(ob, vl, state[:2], predict_gain, tau)
            a, info = ad.plan(torch.tensor(state, dtype=torch.float32, device=dev), gt,
                              torch.tensor(obp, dtype=torch.float32, device=dev), gamma=gamma,
                              obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev),
                              seed=t, return_rollouts=True)
            dr = info["debug_rollouts"]
            ed, gn = escape_dir(state[:2], ob)
            rate = float(info["infeasibility_rate"]); n_rej = int(info["num_barrier_violations"])
            n_tot = int(round(n_rej / rate)) if rate > 1e-9 else int(cfg["num_samples"])
            rec.append(dict(p=state[:2].copy(), crowd=ob.copy(), crowd_infl=obp.copy(), traj=dr["states"],
                            feas=dr["feasible"], n_acc=max(0, n_tot - n_rej), n_rej=n_rej,
                            mean=info["mean_control"], sigma=info["sigma"], esc=ed, gn=gn, v=state[2:4].copy()))
            state = _si_step(state, a.detach().cpu().numpy(), DT)
            if ob.shape[0]:
                min_clear = min(min_clear, float(np.min(np.linalg.norm(ob[:, :2] - state[:2], axis=1) - ob[:, 2] - R_ROBOT)))
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True
        else:
            rec.append(rec[-1] | {"p": state[:2].copy()})
        path.append(state[:2].copy())
    collided = bool(min_clear < 0.0)
    return rec, np.array(path), goal, dict(reached=reached, collided=collided, min_clear=float(min_clear),
                                           success=bool(reached and not collided))


def draw_cell(ax, st, path_upto, goal, gamma, sensing, xl, yl, show_escape=False):
    # polytope from the velocity-INFLATED obstacles (Suggestion 1): faces retreat for peds closing on the robot.
    infl = st.get("crowd_infl", st["crowd"])
    poly, _ = build_polytope_v2(st["p"], infl, sensing_range=sensing, n_base=16, margin=0.0)
    gx = np.linspace(*xl, 80); gy = np.linspace(*yl, 80); GX, GY = np.meshgrid(gx, gy)
    mr = (poly.b - poly.A @ poly.ref).clamp_min(1e-3)
    Hh = ((poly.b[None] - torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32) @ poly.A.T) / mr[None]).min(1).values.numpy().reshape(GX.shape)
    lv = sorted({round((1 - gamma) ** i, 4) for i in range(8)} | {0.0})
    ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.45, zorder=1)
    ax.contour(GX, GY, Hh, levels=[l for l in lv if l > 0], colors="#2166ac", linewidths=0.3, alpha=0.6, zorder=2)
    ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.2, zorder=3)
    for i, (ox, oy, rr) in enumerate(st["crowd"]):
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.55, edgecolor="#7b3294", lw=0.4, zorder=2))
        ir = float(infl[i, 2]) if i < len(infl) else rr                          # inflated (predicted) radius
        if ir > rr + 1e-3:
            ax.add_patch(Circle((ox, oy), ir, facecolor="none", edgecolor="#d62728", ls="--", lw=0.5, alpha=0.6, zorder=2))
    # accepted/rejected ROLLOUT trajectories (di_gamma style)
    traj = st["traj"]; feas = np.asarray(st["feas"], bool)
    for k in range(traj.shape[0]):
        xy = traj[k, :, :2]; col = "#1a9850" if feas[k] else "#d62728"
        ax.plot(xy[:, 0], xy[:, 1], "-", color=col, lw=0.4, alpha=0.25 if feas[k] else 0.35, zorder=4)
        if not feas[k]:
            ax.plot(xy[-1, 0], xy[-1, 1], "x", color="#d62728", ms=3, mew=0.7, zorder=5)
    ax.plot(path_upto[:, 0], path_upto[:, 1], "-", color="#e6191b", lw=1.6, zorder=6)        # executed path
    ax.scatter([st["p"][0]], [st["p"][1]], s=45, c="#1a9850", edgecolor="k", zorder=9)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=130, c="gold", edgecolor="k", zorder=9)
    if show_escape and st["gn"] > 1e-6:
        ax.annotate("", xy=(st["p"][0] + 1.2 * st["esc"][0], st["p"][1] + 1.2 * st["esc"][1]),
                    xytext=(st["p"][0], st["p"][1]), arrowprops=dict(arrowstyle="-|>", color="#9400d3", lw=1.8), zorder=8)
        ax.add_patch(Ellipse((st["p"][0] + 0.6 * st["mean"][0], st["p"][1] + 0.6 * st["mean"][1]),
                             1.2 * st["sigma"][0], 1.2 * st["sigma"][1], facecolor="none", edgecolor="#ff7f00", lw=1.2, zorder=8))
    ax.text(0.02, 0.98, f"accept {st['n_acc']}\nreject {st['n_rej']}", transform=ax.transAxes,
            va="top", ha="left", fontsize=6.5, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=None, help="tune a single row index (prints collision per gamma)")
    ap.add_argument("--fps", type=int, default=5); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = [ROWS[args.row]] if args.row is not None else ROWS

    data = {}; verdict = {}
    for ri, (name, ov, kappa) in enumerate(rows):
        for g in GAMMAS:
            cfg = deepcopy(BASE); cfg.update(ov)
            rec, path, goal, v = record(cfg, kappa, g, dev=args.device)
            data[(name, g)] = (rec, path); verdict[(name, g)] = v
            print(f"  [{name}] γ={g}: success={v['success']} collided={v['collided']} reached={v['reached']} min_clear={v['min_clear']:.2f}", flush=True)
    if args.row is not None:
        return

    s0, goal, obs, _ = _load("ucy", EP, 80)
    vR = 3.2   # robot-centered view half-width so the accept/reject trajectory fans are visible (di_gamma style)
    sensings = {name: dict(BASE, **ov).get("barrier_activation_radius", 3.5) for name, ov, _ in ROWS}

    R, C = len(ROWS), len(GAMMAS)
    fig, axes = plt.subplots(R, C, figsize=(3.4 * C, 3.0 * R), squeeze=False)
    nF = 81

    def draw(f):
        for ri, (name, ov, kappa) in enumerate(ROWS):
            for ci, g in enumerate(GAMMAS):
                ax = axes[ri][ci]; ax.clear()
                rec, path = data[(name, g)]; st = rec[min(f, len(rec) - 1)]
                xl = (st["p"][0] - vR, st["p"][0] + vR); yl = (st["p"][1] - vR, st["p"][1] + vR)  # robot-centered
                draw_cell(ax, st, path[:min(f, len(path) - 1) + 1], goal, g, sensings[name], xl, yl,
                          show_escape=(ri == len(ROWS) - 1))
                if ri == 0:
                    ax.set_title(f"γ={g}", fontsize=11)
                if ci == 0:
                    ax.set_ylabel(name, fontsize=8)
        fig.suptitle(f"ep15 v1 adaptive · accept=green/reject=red ✗ trajectories · red path · dashed=velocity-inflated "
                     f"obstacle (predict retreat) · zeroth row = directional speed cap + variance blow-open · t={f}", fontsize=10)
        return []

    anim = FuncAnimation(fig, draw, frames=nF, interval=1000 // args.fps)
    p = os.path.join(FIG, "ep15_diagnostic_v1_adaptive.gif")
    anim.save(p, writer=PillowWriter(fps=args.fps), dpi=80); print("saved", p)
    draw(40); fig.savefig(os.path.join(FIG, "ep15_diagnostic_v1_adaptive.png"), dpi=110); print("saved png")
    # verdict table
    print("\n=== ep15 verdict table (success / collided) ===")
    for name, _, _ in ROWS:
        cells = "  ".join(f"γ{g}:{'OK' if verdict[(name,g)]['success'] else ('COLLIDE' if verdict[(name,g)]['collided'] else 'no-reach')}" for g in GAMMAS)
        print(f"  {name:18s} {cells}")


if __name__ == "__main__":
    main()
