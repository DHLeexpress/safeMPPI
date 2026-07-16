"""TASK A (07_06 viz suite) — trajectory-level rollout viz, pretrained vs expanded.

Two panels: LEFT = pretrained checkpoint, RIGHT = expanded baseline. Per panel a σ-reference GP is built
from the MODEL'S OWN rollouts (two-pass, order-independent): pass 1 = ~40 σ-tilt warmup deploys (β=1.0,
like curriculum_sigma_viz.build_buffer) accumulating a query buffer (every-3rd window, cap 500), featurized
ONCE at the end via GE._buffer_feat(...,384,...) -> unc.set_buffer. Pass 2 = the 50 PLOTTED rollouts,
FAITHFUL mode (tilt=None, temp=1 — exactly what SR/CR measures), 7-8 per γ over the 7 standard γ.

Drawing per rollout: the EXECUTED path as per-step segments colored by σ_t (viridis, vmin/vmax shared
across BOTH panels); every 10 committed steps the full PLANNED H=10 window branches off as a thin dashed
line in the same σ_t color; terminal markers green-circle (reached) / red-x (collided) / grey-square
(timeout or out-of-bounds). σ_t = unc.sigma(pol.phi_s(U_t, ctx_t, s=0.9)) batched over a rollout's steps.

Usage (rollout-heavy -> GPU):
  LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib CUDA_VISIBLE_DEVICES=3 python fm_traj_viz.py \
      --ckpt-left results/hp_repr/pretrained_a32.pt --ckpt-right results/sweep_overnight/a32_unf/final.pt \
      --n 50 --out figures/traj/fm_traj_baseline_v1.png
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import grid_expand as GE
import grid_expand2 as GX2                       # state_from_low5
import grid_metrics2 as GM2
import grid_hp_expt as HP
import sr_cr_eval as SR
from uncertainty import GPUncertainty
from grid_expand_cur import CurConfig
from curriculum_sigma_viz import setup_style     # shared figure style (serif + mathtext-cm fallback)

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


# ------------------------------------------------------------------ pass 1: σ-reference from own rollouts
def build_sigma_ref(pol, env, cfg, dev, n_warmup=40):
    """~n_warmup σ-tilt (β=1.0) deploys accumulate a qbuf (every-3rd window, cap 500); featurize ONCE at the
    end (order-independent σ for the plotted rollouts) -> a frozen GPUncertainty."""
    unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
    qbuf = None
    t0 = time.time()
    for t in range(n_warmup):
        g = GAMMAS[t % len(GAMMAS)]
        qf = GE._buffer_feat(pol, qbuf, "phi_s", cfg.s, cfg.gp_buf, dev) if qbuf is not None else None
        if qf is not None:
            unc.set_buffer(qf)
        out = GR.fm_deploy(pol, env, float(g), T=cfg.T,
                           tilt=dict(unc=unc, beta=1.0, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                     temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter),
                           nfe=cfg.nfe_explore, record=True, verify_fn=GM2.window_label_cheap, device=dev)
        if out["recs"]:
            G, L, H, U = GE._to_t(out["recs"])
            qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=500)
    if qbuf is None:
        unc.set_buffer(None)
        return unc
    unc.set_buffer(GE._buffer_feat(pol, qbuf, "phi_s", 0.9, 384, dev))
    print(f"[traj_viz]   warmup {n_warmup} deploys -> qbuf {qbuf['U'].shape[0]} "
          f"(featurized 384) in {time.time()-t0:.0f}s", flush=True)
    return unc


# ------------------------------------------------------------------ pass 2: the plotted FAITHFUL rollouts
def collect_rollouts(pol, env, unc, dev, n=50, T=250, reach=0.1):
    """n faithful deploys (7-8 per γ), σ_t per committed step batched per rollout. len(recs)=len(path)-1,
    so σ_t aligns with executed segment path[t]->path[t+1]."""
    per = [n // len(GAMMAS) + (1 if i < n % len(GAMMAS) else 0) for i in range(len(GAMMAS))]
    data = []
    t0 = time.time()
    for g, cnt in zip(GAMMAS, per):
        for i in range(cnt):
            torch.manual_seed(i)                                    # like sr_cr_eval (seed0=0)
            out = GR.fm_deploy(pol, env, float(g), T=T, temp=1.0, tilt=None, record=True,
                               verify_fn=GM2.window_label_cheap, reach=reach, device=dev)
            recs = out["recs"]
            if not recs:
                continue
            G, L, H, U = GE._to_t(recs)
            with torch.no_grad():                                   # ONE batched σ call per rollout
                ctx = pol.ctx_from(G.to(dev), L.to(dev), H.to(dev))
                phi = pol.phi_s(U.to(dev), ctx, s=0.9)
                sig = unc.sigma(phi).detach().cpu().numpy()
            coll = SR.path_collides(out["path"], env)
            data.append(dict(path=out["path"], sig=sig, gamma=g, coll=bool(coll),
                             reached=bool(out["reached"]) and not coll,
                             low5=L.numpy(), U=U.numpy()))
    print(f"[traj_viz]   {len(data)} plotted rollouts in {time.time()-t0:.0f}s", flush=True)
    return data


# ------------------------------------------------------------------ drawing
def draw_panel(ax, env, data, nrm, title):
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2], facecolor="#c8c8c8", edgecolor="#888", lw=.4, zorder=1))
    n_reach = n_coll = 0
    for d in data:
        path, sig = d["path"], d["sig"]
        segs = np.stack([path[:-1], path[1:]], axis=1)              # [steps,2,2] ; σ_t colors segment t
        lc = LineCollection(segs, cmap="viridis", norm=nrm, linewidths=1.5, alpha=0.85, zorder=3)
        lc.set_array(sig[:segs.shape[0]])
        ax.add_collection(lc)
        for t in range(0, len(sig), 10):                            # planned H=10 window every 10 steps
            st = GX2.state_from_low5(d["low5"][t])
            w = np.vstack([st[:2][None, :], GR.window_positions(st, d["U"][t], env.dt)])
            ax.plot(w[:, 0], w[:, 1], ls=(0, (4, 2)), lw=0.9, alpha=0.75,
                    color=plt.cm.viridis(nrm(sig[t])), zorder=2)
        e = path[-1]
        if d["coll"]:
            n_coll += 1
            ax.plot(e[0], e[1], "x", color="#d62728", ms=6.5, mew=1.8, zorder=7)
        elif d["reached"]:
            n_reach += 1
            ax.plot(e[0], e[1], "o", color="#2ca02c", ms=5.5, mec="k", mew=.5, zorder=7)
        else:
            ax.plot(e[0], e[1], "s", color="#9a9a9a", ms=4.8, mec="#555", mew=.5, zorder=7)
    ax.scatter([0], [0], marker="s", s=62, c="#222", zorder=8)
    ax.plot(goal[0], goal[1], marker="*", ms=19, c="gold", mec="k", ls="", zorder=8)
    ax.set_xlim(-0.4, 5.4); ax.set_ylim(-0.4, 5.4); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\nreached {n_reach}/{len(data)} · collided {n_coll}/{len(data)}", fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-left", default="results/hp_repr/pretrained_a32.pt")
    ap.add_argument("--ckpt-right", default="results/sweep_overnight/a32_unf/final.pt")
    ap.add_argument("--label-left", default="pretrained (a32)")
    ap.add_argument("--label-right", default="expanded baseline (a32_unf, it5000)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--out", default=os.path.join("figures", "traj", "fm_traj_baseline_v1.png"))
    a = ap.parse_args()
    usetex = setup_style()
    print(f"[traj_viz] usetex={'ON' if usetex else 'OFF (serif/mathtext-cm fallback)'}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = GS.make_grid(); cfg = CurConfig()

    panels = []
    for ck, lab in ((a.ckpt_left, a.label_left), (a.ckpt_right, a.label_right)):
        ckp = ck if os.path.isabs(ck) else os.path.join(HERE, ck)
        pol, meta = HP.load_hp(ckp, device=dev)
        print(f"[traj_viz] {lab}: {ck} (repr {meta['config'].get('repr_dim')})", flush=True)
        unc = build_sigma_ref(pol, env, cfg, dev, n_warmup=a.warmup)
        panels.append((lab, collect_rollouts(pol, env, unc, dev, n=a.n)))

    all_sig = np.concatenate([d["sig"] for _, data in panels for d in data])
    nrm = Normalize(float(all_sig.min()), float(max(all_sig.max(), all_sig.min() + 1e-9)))
    print(f"[traj_viz] shared σ range [{nrm.vmin:.4f}, {nrm.vmax:.4f}]", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.2))
    for ax, (lab, data) in zip(axes, panels):
        draw_panel(ax, env, data, nrm, lab)
    cb = fig.colorbar(ScalarMappable(norm=nrm, cmap="viridis"), ax=axes.tolist(),
                      fraction=0.035, pad=0.02)
    cb.set_label(r"$\sigma_t$ (GP posterior std of the committed window, shared scale)")
    handles = [
        Line2D([], [], color=plt.cm.viridis(0.55), lw=2.2, label=r"executed path (per-step, $\sigma_t$-colored)"),
        Line2D([], [], color=plt.cm.viridis(0.55), lw=0.9, ls=(0, (4, 2)), alpha=0.8,
               label="planned H=10 window (every 10 steps)"),
        Line2D([], [], color="#2ca02c", marker="o", ls="", mec="k", mew=.5, label="reached goal"),
        Line2D([], [], color="#d62728", marker="x", ls="", mew=1.8, label="collided"),
        Line2D([], [], color="#9a9a9a", marker="s", ls="", mec="#555", mew=.5, label="timeout / out-of-bounds"),
        Line2D([], [], color="gold", marker="*", ls="", ms=13, mec="k", label="goal"),
        Line2D([], [], color="#222", marker="s", ls="", label="origin (start)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.44, 0.045), ncol=4)
    fig.suptitle(r"FM policy rollouts from origin, faithful sampling (temp=1, tilt-free — the SR/CR mode) — "
                 fr"{a.n}/panel over $\gamma\in$ {{{', '.join(str(g) for g in GAMMAS)}}}", y=0.98)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj_viz] -> {out}", flush=True)


if __name__ == "__main__":
    main()
