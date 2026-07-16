"""Curriculum easy/mid/frontier + σ visualization (2026-07-07, user "very important"). Loads an expanded
checkpoint, rebuilds a positive buffer by deploying from origin, scores it (σ / SOCP-margin / jerk / goal-
alignment), assigns easy/mid/frontier, and shows: (A) example window-trajectories per pool in the scene,
(B) the σ distribution by pool, (C) the σ-vs-margin structure that defines the pools.

Stacked-iters mode (07_06 viz suite, Task B): `--dbs <p1> <p2> ...` renders ONE ROW of the A/B/C panels
per saved viz_db, row-labeled "it{iter}" on the left, with SHARED σ bins / axis limits across rows so the
pool evolution over expansion iters is directly comparable. Legends live OUTSIDE the axes.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import grid_expand as GE
import grid_expand2 as GX2
import grid_metrics2 as GM2
import grid_hp_expt as HP
from uncertainty import GPUncertainty
from grid_expand_cur import score_positives, curriculum_pools, CurConfig

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "curriculum"); os.makedirs(FIG, exist_ok=True)
COL = {"easy": "tab:green", "mid": "tab:orange", "frontier": "tab:red"}
POOLS = ("easy", "mid", "frontier")


def setup_style():
    """Try text.usetex once (render smoke); fall back to serif + mathtext cm if LaTeX is missing.
    Returns True if usetex works, else False (serif fallback)."""
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130,
        "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10, "figure.titlesize": 15,
        "axes.linewidth": 0.9,
    })
    try:
        mpl.rcParams["text.usetex"] = True
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\sigma_{123}$")
        fig.canvas.draw()
        import io
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["mathtext.fontset"] = "cm"
        return False


def build_buffer(pol, env, cfg, device, n_deploy=80):
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    pos = qbuf = None
    gammas = list(cfg.gammas)
    for t in range(n_deploy):
        g = gammas[t % len(gammas)]
        qf = GE._buffer_feat(pol, qbuf, "phi_s", cfg.s, cfg.gp_buf, device) if qbuf is not None else None
        if qf is not None:
            unc.set_buffer(qf)
        out = GR.fm_deploy(pol, env, float(g), T=cfg.T,
                           tilt=dict(unc=unc, beta=1.0, N=cfg.N, s=cfg.s, broad=0, feature="phi_s",
                                     temp=cfg.temp, churn=cfg.churn, safe_filter=cfg.safe_filter),
                           nfe=cfg.nfe_explore, record=True, verify_fn=GM2.window_label_cheap, device=device)
        if out["recs"]:
            G, L, H, U = GE._to_t(out["recs"])
            qbuf = GE._cat(qbuf, G[::3], L[::3], H[::3], U[::3], cap=cfg.qbuf_cap)
            if out["reached"] and GM2.traj_valid2(out["path"], env, float(g)):
                pos = GE._cat(pos, G, L, H, U, tags=[-1] * G.shape[0], cap=cfg.cap_pos)
    return pos, unc


def win_seg(low5_np, U_np, env, n=6):
    st = GX2.state_from_low5(low5_np)
    p = [st[:2].copy()]
    from di_grid_viz import di_step
    s = st.copy()
    for a in U_np[:n]:
        s = di_step(s, np.asarray(a, np.float32), dt=env.dt); p.append(s[:2].copy())
    return np.array(p)


def load_db(path):
    """Load a saved labeled buffer-DB (grid_expand_cur._save_viz_db): low5,U,label,sigma,margin,jerk,mono,iter."""
    db = torch.load(path, map_location="cpu", weights_only=False)
    return dict(low5=db["low5"].numpy(), U=db["U"].numpy(), label=np.asarray(list(db["label"]), dtype=object),
                sigma=np.asarray(db["sigma"]), margin=np.asarray(db["margin"]),
                jerk=np.asarray(db["jerk"]), mono=np.asarray(db["mono"]), iter=int(db.get("iter", -1)))


def rederive(pol, env, cfg, dev, n_deploy):
    """Re-derive the labeled buffer by deploying from origin + scoring + pooling (fallback when no --db)."""
    pos, unc = build_buffer(pol, env, cfg, dev, n_deploy)
    if pos is None or pos["U"].shape[0] < 30:
        return None
    sc = score_positives(pol, unc, pos, env, cfg, dev)
    easy, mid, frontier = curriculum_pools(sc, cfg)
    idx = sc["idx"]; row_of = {int(v): i for i, v in enumerate(idx)}
    lab = np.array(["mid"] * len(idx), dtype=object)
    for k, arr in (("easy", easy), ("frontier", frontier)):
        for v in arr:
            lab[row_of[int(v)]] = k
    return dict(low5=pos["low5"][idx].numpy(), U=pos["U"][idx].numpy(), label=lab,
                sigma=sc["sigma"], margin=sc["margin"], jerk=sc["jerk"], mono=sc["mono"], iter=None)


def draw_row(axA, axB, axC, D, env, bins, mlim, top_row=True, bottom_row=True, row_label=None):
    """One row of the A/B/C panels for a labeled buffer D. `bins` = shared σ bins; `mlim` = shared margin
    y-lims (comparability across stacked rows). Returns the pool counts. NO in-axes legends (Task B)."""
    lab = D["label"]
    counts = {k: int((lab == k).sum()) for k in POOLS}
    obs = env.obstacles.detach().cpu().numpy(); goal = env.goal.detach().cpu().numpy()
    # ---- A: example window segments per pool, in the scene
    for o in obs:
        axA.add_patch(Circle((o[0], o[1]), o[2], facecolor="#c8c8c8", edgecolor="#888", lw=.4, zorder=1))
    axA.scatter([0], [0], marker="s", s=55, c="#222", zorder=6)
    axA.plot(goal[0], goal[1], marker="*", ms=18, c="gold", mec="k", zorder=6, ls="")
    rng = np.random.default_rng(0)
    for pool_name in POOLS:
        rows = np.where(lab == pool_name)[0]
        for r in (rng.choice(rows, size=min(12, len(rows)), replace=False) if len(rows) else []):
            seg = win_seg(D["low5"][r], D["U"][r], env)
            axA.plot(seg[:, 0], seg[:, 1], "-", color=COL[pool_name], lw=1.4, alpha=0.8, zorder=3)
            axA.plot(seg[0, 0], seg[0, 1], "o", color=COL[pool_name], ms=4, mec="#222", mew=.4, zorder=4)
    axA.set_xlim(-0.4, 5.4); axA.set_ylim(-0.4, 5.4); axA.set_aspect("equal")
    axA.set_ylabel("y (m)")
    # ---- B: sigma histogram by pool — DRAW frontier -> mid -> easy so the small easy pool sits on top
    for pool_name in ("frontier", "mid", "easy"):
        m = lab == pool_name
        if m.any():
            axB.hist(D["sigma"][m], bins=bins, alpha=0.6, color=COL[pool_name],
                     zorder={"frontier": 2, "mid": 3, "easy": 4}[pool_name])
    axB.set_ylabel("count")
    # ---- C: sigma vs margin structure
    for pool_name in POOLS:
        m = lab == pool_name
        axC.scatter(D["sigma"][m], D["margin"][m], s=16, c=COL[pool_name], alpha=0.65, edgecolor="none")
    axC.set_xlim(bins[0] - 0.02 * (bins[-1] - bins[0]), bins[-1] + 0.02 * (bins[-1] - bins[0]))
    axC.set_ylim(*mlim)
    axC.set_ylabel("SOCP margin (m)")
    if top_row:
        axA.set_title("(A) window segments in the scene")
        axB.set_title(r"(B) $\sigma$ distribution by pool")
        axC.set_title(r"(C) $\sigma$ vs SOCP margin (pool structure)")
    if bottom_row:
        axA.set_xlabel("x (m)")
        axB.set_xlabel(r"$\sigma$ (GP posterior std)")
        axC.set_xlabel(r"$\sigma$")
    if row_label:
        axA.text(-0.30, 0.5, row_label, transform=axA.transAxes, rotation=90, ha="center", va="center",
                 fontsize=15, fontweight="bold")
        axA.text(-0.22, 0.5, f"e {counts['easy']} · m {counts['mid']} · f {counts['frontier']}",
                 transform=axA.transAxes, rotation=90, ha="center", va="center", fontsize=10, color="#555")
    return counts


def render(Ds, row_labels, env, out, src, suptitle_extra=""):
    """Ds = list of labeled-buffer dicts -> one A/B/C row each; shared σ bins + margin lims; ONE legend
    OUTSIDE (below the panel grid)."""
    nr = len(Ds)
    sig_all = np.concatenate([D["sigma"] for D in Ds]); mg_all = np.concatenate([D["margin"] for D in Ds])
    lo, hi = float(sig_all.min()), float(max(sig_all.max(), sig_all.min() + 1e-6))
    bins = np.linspace(lo, hi, 26)
    pad = 0.06 * (mg_all.max() - mg_all.min() + 1e-6)
    mlim = (float(mg_all.min() - pad), float(mg_all.max() + pad))
    fig, axes = plt.subplots(nr, 3, figsize=(16.5, 5.2 * nr), squeeze=False)
    for i, (D, rl) in enumerate(zip(Ds, row_labels)):
        c = draw_row(axes[i, 0], axes[i, 1], axes[i, 2], D, env, bins, mlim,
                     top_row=(i == 0), bottom_row=(i == nr - 1), row_label=rl if nr > 1 else None)
        print(f"[cur_viz] {rl or src}: pools easy {c['easy']} mid {c['mid']} frontier {c['frontier']}", flush=True)
    fig.suptitle(f"Curriculum easy/mid/frontier pools — {src}{suptitle_extra}\n"
                 r"easy = low $\sigma$ · high margin · smooth · goal-aligned    |    "
                 r"frontier = high $\sigma$ OR low SOCP margin")
    fig.tight_layout(rect=(0.015, 0.04 / nr, 1, 1))
    fig.legend(handles=[Patch(color=COL[k], alpha=.8, label=k) for k in POOLS],
               loc="upper center", bbox_to_anchor=(0.5, 0.035 / nr), ncol=3)   # OUTSIDE, below the grid
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[cur_viz] -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/sweep_overnight/a32_unf/best.pt")
    ap.add_argument("--db", default=None, help="saved viz_db .pt (plot the ACTUAL labeled buffer); "
                                               "omit to re-derive from --ckpt")
    ap.add_argument("--dbs", nargs="+", default=None,
                    help="MULTIPLE viz_db .pt -> STACKED figure, one A/B/C row per DB (row label = it{iter})")
    ap.add_argument("--n-deploy", type=int, default=80)
    ap.add_argument("--out", default=os.path.join(FIG, "curriculum_pools.png"))
    args = ap.parse_args()
    usetex = setup_style()
    print(f"[cur_viz] usetex={'ON' if usetex else 'OFF (serif/mathtext-cm fallback)'}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid(); cfg = CurConfig()

    if args.dbs:                                            # STACKED mode: one row per saved DB
        Ds = [load_db(p) for p in args.dbs]
        labels = [f"it{D['iter']}" for D in Ds]
        arm = os.path.basename(os.path.dirname(os.path.dirname(args.dbs[0])))
        for p, D in zip(args.dbs, Ds):
            print(f"[cur_viz] DB {p} it{D['iter']} n={len(D['label'])}", flush=True)
        render(Ds, labels, env, args.out, src=f"{arm} · stacked DBs {' / '.join(labels)}")
        return

    if args.db:                                              # plot the ACTUAL saved buffer-DB
        D = load_db(args.db)
        src = f"{os.path.basename(os.path.dirname(os.path.dirname(args.db)))} · DB it{D['iter']} (n={len(D['label'])})"
        print(f"[cur_viz] DB {args.db} it{D['iter']} n={len(D['label'])}", flush=True)
    else:                                                   # re-derive (fallback)
        pol, ck = HP.load_hp(args.ckpt, device=dev)
        print(f"[cur_viz] re-derive {os.path.basename(os.path.dirname(args.ckpt))} "
              f"repr={ck['config'].get('repr_dim')}", flush=True)
        D = rederive(pol, env, cfg, dev, args.n_deploy)
        if D is None:
            print("[cur_viz] too few positives collected; abort"); return
        src = f"{os.path.basename(os.path.dirname(args.ckpt))} · re-derived (n={len(D['label'])})"
    render([D], [None], env, args.out, src=src)


if __name__ == "__main__":
    main()
