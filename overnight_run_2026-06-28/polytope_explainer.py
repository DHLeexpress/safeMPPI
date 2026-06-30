"""Polytope SafeMPPI EXPLAINER — small, zoomed, annotated panels that show CLEARLY how each part works on ONE
episode, before any full grid GIF:
  top row  (one panel per selected step): scene zoomed on the robot -> nominal polytope + {H_P>=(1-g)^i} level sets,
           accepted (green) / rejected (red x) rollout trajectories, the free-space centroid arrow (mean steering),
           and the accept/reject counts.
  bottom row (same steps): the CONTROL-space proposal -> mean control vector (arrow) + covariance ellipse; the
           covariance grows when the polytope is small (trapped). Polytope size annotated.

  python overnight_run_2026-06-28/polytope_explainer.py --gamma 0.5 --episode auto
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.mppi.sweep import _load, _si_step, DT
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
R_ROBOT = 0.2


def pick_navigable(dataset, gamma, cfg):
    """auto-pick a moderately-cluttered episode where the robot meets a few obstacles (interesting, not pathological)."""
    best = None
    for ep in range(40):
        s0, goal, obs, vel = _load(dataset, ep, 80)
        nc = int((~np.isnan(obs[:, :, 0]).all(0)).sum()); travel = float(np.linalg.norm(goal - s0[:2]))
        if 4 <= nc <= 14 and travel > 4.0:
            return ep
        if best is None:
            best = ep
    return best


def rollout(dataset, ep, gamma, cfg, steps=80, dev="cpu"):
    s0, goal, obs, vel = _load(dataset, ep, 80)
    ad = SafeMPPIAdapter(**cfg); state = s0.astype(np.float32).copy(); rec = []; reached = False
    gt = torch.tensor(goal, dtype=torch.float32, device=dev); path = [state[:2].copy()]
    for t in range(steps):
        ob = obs[min(t, obs.shape[0] - 1)]; vl = vel[min(t, vel.shape[0] - 1)]
        ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vl = vl[ok]
        if not reached:
            a, info = ad.plan(torch.tensor(state, dtype=torch.float32, device=dev), gt,
                              torch.tensor(ob, dtype=torch.float32, device=dev), gamma=gamma,
                              obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev),
                              seed=t, return_rollouts=True)
            dr = info["debug_rollouts"]; rate = float(info["infeasibility_rate"]); nrej = int(info["num_barrier_violations"])
            ntot = int(round(nrej / rate)) if rate > 1e-9 else int(cfg["num_samples"])
            # predict-inflated radius (matches the polytope obstacle face retreat) for the dashed-circle viz
            tau = cfg["horizon"] * DT; sm = cfg.get("safety_margin", 0.0); kap = cfg.get("predict_gain", 0.0); infl = []
            for j, (ox, oy, rr) in enumerate(ob):
                rel = np.array([ox, oy]) - state[:2]; d = max(np.linalg.norm(rel), 1e-9); m = rel / d
                vclose = float(m @ (state[2:4] - vl[j])) if j < len(vl) else 0.0
                infl.append(rr + sm + kap * tau * max(0.0, vclose))
            rec.append(dict(p=state[:2].copy(), crowd=ob.copy(), crowd_infl=np.array(infl), traj=dr["states"],
                            feas=np.asarray(dr["feasible"], bool), n_acc=max(0, ntot - nrej), n_rej=nrej,
                            mean=info["mean_control"], smean=info["sample_mean"], sigma=info["sigma"],
                            poly=info["polytope"], cdir=info["centroid_dir"], size=info["polytope_size"]))
            state = _si_step(state, a.detach().cpu().numpy(), DT)
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True
        else:
            rec.append(rec[-1] | {"p": state[:2].copy()})
        path.append(state[:2].copy())
    return rec, np.array(path), goal


def H_grid(poly, GX, GY):
    A, b, c, mr = poly
    pts = np.stack([GX.ravel(), GY.ravel()], 1)
    return (((b[None] - pts @ A.T) / mr[None]).min(1)).reshape(GX.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy"); ap.add_argument("--episode", default="auto")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--centroid-gain", type=float, default=0.1); ap.add_argument("--sigma-volume-gain", type=float, default=0.5)
    ap.add_argument("--centroid-horizon", type=int, default=3); ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--predict-gain", type=float, default=0.4); ap.add_argument("--sensing", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    # MPPI spirit: nominal control = 0 (cold seed) + WARM-START; goal lives in the cost. safety_margin = 0 (keep the
    # per-obstacle predict_gain inflation). mean/cov shift over the first K steps only. 50-eps fine-tuned config.
    cfg = dict(horizon=10, dt=DT, num_samples=128, noise_sigma=(0.5, 0.5), u_min=(-2., -2.), u_max=(2., 2.),
               safety_margin=0.0, temperature=args.temperature, dynamics_type="singleintegrator",
               barrier_activation_radius=args.sensing, use_polytope_barrier=True, use_goal_nominal=False,
               warm_start=True, centroid_gain=args.centroid_gain, centroid_horizon=args.centroid_horizon,
               sigma_volume_gain=args.sigma_volume_gain, predict_gain=args.predict_gain, polytope_nbase=16)
    ep = pick_navigable(args.dataset, args.gamma, cfg) if args.episode == "auto" else int(args.episode)
    print(f"episode {ep} γ={args.gamma}")
    rec, path, goal = rollout(args.dataset, ep, args.gamma, dict(cfg), dev=args.device)
    # pick steps where the mechanism is VISIBLE: polytope healthy (size>0.3) and BOTH accept & reject present, so we
    # actually see the rejection working (not the degenerate all-rejected case).
    cand = [i for i, st in enumerate(rec) if st.get("poly") is not None and st["size"] > 0.3
            and st["n_acc"] > 3 and st["n_rej"] > 3 and i < len(rec) - 1]
    if len(cand) >= 4:
        steps = sorted({cand[int(k)] for k in np.linspace(0, len(cand) - 1, 4)})
    else:
        cand = [i for i, st in enumerate(rec) if st.get("poly") is not None and st["n_acc"] > 0 and i < len(rec) - 1]
        steps = sorted(cand[:4]) if cand else [min(2, len(rec) - 1)]
    print(f"selected steps {steps}")
    vR = 2.6; C = len(steps)
    fig, axes = plt.subplots(2, C, figsize=(3.5 * C, 7.0), squeeze=False)
    for ci, t in enumerate(steps):
        st = rec[t]; p = st["p"]; xl = (p[0] - vR, p[0] + vR); yl = (p[1] - vR, p[1] + vR)
        ax = axes[0][ci]
        if st["poly"] is not None:
            gx = np.linspace(*xl, 90); gy = np.linspace(*yl, 90); GX, GY = np.meshgrid(gx, gy)
            Hh = H_grid(st["poly"], GX, GY)
            lv = sorted({round((1 - args.gamma) ** i, 4) for i in range(8)} | {0.0})
            ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.5, zorder=1)
            ax.contour(GX, GY, Hh, levels=[l for l in lv if l > 0], colors="#2166ac", linewidths=0.4, alpha=0.7, zorder=2)
            ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.6, zorder=3)
        ci_inf = st.get("crowd_infl", None)
        for i, (ox, oy, rr) in enumerate(st["crowd"]):
            ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.6, edgecolor="#7b3294", lw=0.6, zorder=4))
            ir = float(ci_inf[i]) if ci_inf is not None and i < len(ci_inf) else rr   # margin + predict-velocity inflated
            if ir > rr + 1e-3:
                ax.add_patch(Circle((ox, oy), ir, facecolor="none", edgecolor="#d62728", ls="--", lw=0.7, alpha=0.6, zorder=4))
        traj = st["traj"]; feas = st["feas"]
        for k in range(traj.shape[0]):                       # rejected first (under), accepted on top + vivid
            if not feas[k]:
                xy = traj[k, :, :2]; ax.plot(xy[:, 0], xy[:, 1], "-", color="#d62728", lw=0.5, alpha=0.4, zorder=5)
                ax.plot(xy[-1, 0], xy[-1, 1], "x", color="#d62728", ms=4, mew=0.8, zorder=6)
        for k in range(traj.shape[0]):
            if feas[k]:
                xy = traj[k, :, :2]; ax.plot(xy[:, 0], xy[:, 1], "-", color="#00a000", lw=1.0, alpha=0.9, zorder=7)
        if st["cdir"] is not None and np.linalg.norm(st["cdir"]) > 1e-6:
            cd = st["cdir"]; ax.annotate("", xy=(p[0] + 1.1 * cd[0], p[1] + 1.1 * cd[1]), xytext=(p[0], p[1]),
                                         arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=2.4), zorder=9)  # centroid dir
        sm = st["smean"]                                          # sampling-mean velocity (SI: should align w/ centroid)
        ax.annotate("", xy=(p[0] + 0.6 * sm[0], p[1] + 0.6 * sm[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.5, linestyle=":"), zorder=9)
        ax.scatter([p[0]], [p[1]], s=55, c="#00a000", edgecolor="k", zorder=10)
        if xl[0] <= goal[0] <= xl[1] and yl[0] <= goal[1] <= yl[1]:
            ax.scatter([goal[0]], [goal[1]], marker="*", s=130, c="gold", edgecolor="k", zorder=10)
        ax.set_title(f"t={t}  accept {st['n_acc']} / reject {st['n_rej']}", fontsize=9)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # --- control-space proposal panel: the SAMPLING Gaussian (center = steered nominal[0], cov = sigma) ---
        axc = axes[1][ci]; mean = st["smean"]; sig = st["sigma"]; ex = st["mean"]
        axc.axhline(0, color="#bbb", lw=0.6); axc.axvline(0, color="#bbb", lw=0.6)
        axc.add_patch(Ellipse((mean[0], mean[1]), 2 * sig[0], 2 * sig[1], facecolor="#ffd9b3", edgecolor="#ff7f00",
                              lw=1.6, alpha=0.7, zorder=2))
        axc.annotate("", xy=(mean[0], mean[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2), zorder=3)
        axc.scatter([ex[0]], [ex[1]], s=30, c="#08306b", marker="x", zorder=4, label="executed")  # executed (safe-fallback)
        axc.legend(loc="lower left", fontsize=6)
        axc.set_title(f"sampling mean+cov · size={st['size']:.2f} σ={sig[0]:.2f}", fontsize=8)
        axc.set_xlim(-2.2, 2.2); axc.set_ylim(-2.2, 2.2); axc.set_aspect("equal")
        axc.set_xlabel("u_x", fontsize=7); axc.set_ylabel("u_y", fontsize=7); axc.tick_params(labelsize=6)
    fig.suptitle(f"Polytope SafeMPPI explainer · ep{ep} γ={args.gamma} · TOP: polytope+level sets, green=accepted/"
                 f"red x=rejected, orange=centroid dir, dotted-red=sampling-mean velocity (align for SI), dashed=predict-inflated "
                 f"obstacle · BOTTOM: SAMPLING mean(red)+cov, executed=navy x", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, "polytope_explainer.png"); fig.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
