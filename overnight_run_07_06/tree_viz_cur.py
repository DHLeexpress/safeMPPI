"""TASK 1 (07_06 viz suite) — labeled σ-tilt exploration tree of a POST-EXPANSION curriculum checkpoint.

Reuses hp_tree_viz.{load_any, build_tree, seg_verdict, roll_states} for the recursive σ-tree (branch
SCHED=[5,4,4,3,3,2,2], importance-resampled p∝exp(σ/β)); and curriculum_sigma_viz.build_buffer +
grid_expand_cur.{score_positives, curriculum_pools} for the easy/mid/frontier buffer overlay.

Drawing (07_06 Task C cleanup):
  (a) failure cause is shown by LINE STYLE ONLY (no text tags at death nodes) — collision/SOCP -> SOLID red,
      out-of-bounds/task-space -> DOTTED red, goal-seeking -> DASHED red; valid/alive segments stay σ-viridis;
  (b) the buffer-sample overlay is GONE (it lives in the curriculum figure now); a saved viz_db (glob
      results/sweep_ac/*/viz_db/it*.pt) or a re-derived buffer is still used — but ONLY as the GP σ-reference;
  (c) all legends sit OUTSIDE the axes.

CLI: --ckpt --gamma --db(optional) --out. Prefix runs with `LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib`
and `CUDA_VISIBLE_DEVICES=<n>`.  SMOKE:
  python tree_viz_cur.py --ckpt results/sweep_overnight/a32_unf/ckpt_2500.pt --gamma 0.5 \
      --out figures/tree/tree_a32_unf_it2500.png
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import _paths  # noqa: F401
import grid_scene as GS
import grid_feats as GF
import grid_rollout as GR
import grid_expand2 as GX2                     # state_from_low5
import hp_tree_viz as TV                        # load_any / build_tree / seg_verdict / roll_states (REUSE)
import curriculum_sigma_viz as CSV              # build_buffer (REUSE)
from grid_expand_cur import score_positives, curriculum_pools, CurConfig
from uncertainty import GPUncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RED = "#d62728"
TERM_LS = {"✗S": "-", "✗T": ":", "✗G": "--"}    # SOCP solid / task-space dotted / goal-seeking dashed
TERM_LABEL = {"✗S": "SOCP / collision fail (solid)", "✗T": "task-space / out-of-bounds fail (dotted)",
              "✗G": "goal-seeking fail (dashed)"}


# ------------------------------------------------------------------ STYLE (professional, LaTeX-like)
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


# ------------------------------------------------------------------ buffer / overlay
def _latest_db(db_arg):
    if db_arg:
        return db_arg
    cands = glob.glob(os.path.join(HERE, "results", "sweep_ac", "*", "viz_db", "it*.pt"))
    if not cands:
        return None

    def _it(p):
        m = re.search(r"it(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cands, key=_it)


def overlay_from_db(path):
    """Load a saved labeled buffer DB (keys grid,low5,U,label,sigma,margin,jerk,mono[,neg_*])."""
    db = torch.load(path, map_location="cpu", weights_only=False)
    return dict(low5=db["low5"].numpy(), U=db["U"].numpy(),
                label=np.asarray(list(db["label"]), dtype=object),
                grid=db["grid"], it=int(db.get("iter", -1)), src=os.path.basename(path))


def unc_from_windows(pol, grid, low5, U, cfg):
    """GP σ-buffer from a set of buffer windows (hist=0; the a32 repr arch ignores hist in ctx)."""
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    with torch.no_grad():
        G = torch.as_tensor(np.asarray(grid)).to(DEV)
        L = torch.as_tensor(np.asarray(low5)).to(DEV)
        Hh = torch.zeros(L.shape[0], GF.K_HIST, 2, device=DEV)
        Ut = torch.as_tensor(np.asarray(U)).to(DEV)
        unc.set_buffer(pol.phi_s(Ut, pol.ctx_from(G, L, Hh), s=cfg.s))
    return unc


def rederive_overlay(pol, env, cfg, n_deploy, per_pool=22, seed=0):
    """Re-derive a labeled buffer by deploying from origin, then stratified-subsample ~64 for the overlay.
    Returns (overlay dict, unc) — unc is build_buffer's own exploration σ-estimator (faithful)."""
    pos, unc = CSV.build_buffer(pol, env, cfg, DEV, n_deploy=n_deploy)
    if pos is None or pos["U"].shape[0] < 30:
        return None, unc
    sc = score_positives(pol, unc, pos, env, cfg, DEV)
    easy, mid, frontier = curriculum_pools(sc, cfg)
    idx = sc["idx"]; row_of = {int(v): i for i, v in enumerate(idx)}
    lab = np.array(["mid"] * len(idx), dtype=object)
    for k, arr in (("easy", easy), ("frontier", frontier)):
        for v in arr:
            lab[row_of[int(v)]] = k
    rng = np.random.default_rng(seed)
    sel_rows = []
    for name in ("easy", "mid", "frontier"):
        rows = np.where(lab == name)[0]
        if len(rows):
            sel_rows += list(rng.choice(rows, size=min(per_pool, len(rows)), replace=False))
    sel_rows = np.array(sel_rows, int)
    buf = idx[sel_rows]
    print(f"[tree_cur] re-derived buffer n_pos={pos['U'].shape[0]} pools "
          f"easy/mid/frontier={len(easy)}/{len(mid)}/{len(frontier)} -> overlay {len(buf)}", flush=True)
    return dict(low5=pos["low5"][buf].numpy(), U=pos["U"][buf].numpy(),
                label=lab[sel_rows], it=None, src="re-derived"), unc


# ------------------------------------------------------------------ drawing (extends hp_tree_viz.draw_tree)
def draw_tree_cur(ax, tree, env, smin, smax):
    """σ-viridis valid branches + cause-styled RED terminal segments (line style = failure cause)."""
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    nrm = Normalize(smin, max(smax, smin + 1e-9))
    for (ox, oy, r) in obs:
        ax.add_patch(Circle((ox, oy), r, facecolor="#d9c8e3", edgecolor="#9b72aa", lw=.4, alpha=.65, zorder=1))
    for (seg, sg, ok, tag, lvl, hit) in tree["segs"]:
        al = max(0.22, 1.0 * (0.86 ** lvl))
        lw = max(0.7, 2.4 * (0.82 ** lvl))
        if ok:
            ax.plot(seg[:, 0], seg[:, 1], "-", color=plt.cm.viridis(nrm(sg)), lw=lw, alpha=al, zorder=3)
            if hit:
                ax.plot(seg[-1, 0], seg[-1, 1], "o", color="#2ca02c", ms=5.0, mec="k", mew=.4, zorder=8)
        else:
            ls = TERM_LS.get(tag, "-")                       # cause -> LINE STYLE ONLY (no text tags)
            ax.plot(seg[:, 0], seg[:, 1], ls, color=RED, lw=max(1.3, lw), alpha=min(1.0, al + .2), zorder=4)
            ax.plot(seg[-1, 0], seg[-1, 1], "x", color=RED, ms=5.0, mew=1.6, zorder=7)
    bp = np.array(tree["branch_pts"]) if tree["branch_pts"] else np.zeros((0, 2))
    if len(bp):
        ax.plot(bp[:, 0], bp[:, 1], "o", color="k", ms=3.0, ls="", zorder=6)
    ax.scatter([0], [0], marker="s", s=62, c="#222", zorder=7)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=240, c="gold", edgecolor="k", lw=.8, zorder=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return nrm


def add_legends(fig, ax):
    """ONE legend, OUTSIDE the axes (below): terminal-branch styles (cause = line style) + markers."""
    term = [Line2D([], [], color=RED, ls=TERM_LS[t], lw=2.2, label=TERM_LABEL[t]) for t in ("✗S", "✗T", "✗G")]
    term.append(Line2D([], [], color=plt.cm.viridis(0.5), lw=2.2, label=r"alive branch ($\sigma$-colored)"))
    term.append(Line2D([], [], color="#2ca02c", marker="o", ls="", mec="k", mew=.4, label="reached goal"))
    term.append(Line2D([], [], color="k", marker="o", ls="", ms=4, label="branch event"))
    leg = fig.legend(handles=term, loc="upper center", bbox_to_anchor=(0.44, 0.02), ncol=3,
                     title="terminal (dying) branch — cause by LINE STYLE")
    leg.get_title().set_fontsize(10)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/sweep_overnight/a32_unf/ckpt_2500.pt")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--db", default=None, help="saved viz_db .pt (else glob results/sweep_ac/*/viz_db/it*.pt, "
                                               "else re-derive from the checkpoint)")
    ap.add_argument("--out", default=os.path.join("figures", "tree", "tree.png"))
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-deploy", type=int, default=30, help="deploys for the re-derive path (no DB)")
    ap.add_argument("--refresh", action="store_true", help="ignore the render cache and recompute")
    a = ap.parse_args()

    usetex = setup_style()
    print(f"[tree_cur] usetex={'ON' if usetex else 'OFF (serif/mathtext-cm fallback)'}", flush=True)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = GS.make_grid()
    cfg = CurConfig()
    db_path = _latest_db(a.db)

    # cache the (expensive) overlay + tree so styling iterations are instant; keyed on ckpt mtime + gamma + db.
    # Kept in a dedicated hidden dir (not figures/) so the deliverable directory stays clean.
    ck_abs = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(HERE, a.ckpt)
    cache_dir = os.path.join(HERE, ".treeviz_cache"); os.makedirs(cache_dir, exist_ok=True)
    key = f"{os.path.basename(os.path.dirname(ck_abs))}_{os.path.basename(ck_abs)}_g{a.gamma}_s{a.seed}"
    key += f"_db{os.path.basename(db_path)}" if db_path else "_rederive"
    cache = os.path.join(cache_dir, f"cache_{key}_{int(os.path.getmtime(ck_abs))}.pt")

    if os.path.exists(cache) and not a.refresh:
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        ov, tree, smin, smax = blob["ov"], blob["tree"], blob["smin"], blob["smax"]
        print(f"[tree_cur] loaded cache {os.path.basename(cache)}", flush=True)
    else:
        pol = TV.load_any(ck_abs)
        if db_path:
            ov = overlay_from_db(db_path)
            unc = unc_from_windows(pol, ov["grid"], ov["low5"], ov["U"], cfg)
            print(f"[tree_cur] DB overlay {ov['src']} it{ov['it']} n={len(ov['label'])}", flush=True)
        else:
            ov, unc = rederive_overlay(pol, env, cfg, a.n_deploy, seed=a.seed)
        tree = TV.build_tree(pol, env, a.gamma, a.temp, a.beta, unc, tilt=True, seed=a.seed)
        sigs = [b[1] for b in tree["segs"]]
        smin, smax = (min(sigs), max(sigs)) if sigs else (0.0, 1.0)
        torch.save(dict(ov=ov, tree=tree, smin=smin, smax=smax), cache)
        nf = sum(1 for b in tree["segs"] if not b[2])
        print(f"[tree_cur] tree branches {len(tree['segs'])} died {nf} reached {tree['n_reached']} "
              f"alive-end {tree['alive_end']}", flush=True)

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    draw_tree_cur(ax, tree, env, smin, smax)              # NO buffer overlay (lives in the curriculum figure)
    add_legends(fig, ax)
    ax.set_xlim(-0.55, 5.55); ax.set_ylim(-0.55, 5.55)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(smin, smax), cmap="viridis"), ax=ax,
                      fraction=0.046, pad=0.02)
    cb.set_label(r"$\sigma$ (GP posterior std, branch novelty)")

    nb = len(tree["segs"]); nf = sum(1 for b in tree["segs"] if not b[2])
    it_txt = (f"$\\sigma$-ref: DB it{ov['it']}" if (ov and ov.get("it")) else
              "$\\sigma$-ref: re-derived buffer") if ov else "$\\sigma$-ref: none"
    name = os.path.join(os.path.basename(os.path.dirname(ck_abs)), os.path.basename(ck_abs))
    fig.suptitle(r"$\sigma$-tilt safe-exploration tree — " + name.replace("_", r"\_") if usetex
                 else r"$\sigma$-tilt safe-exploration tree — " + name, y=0.98)
    ax.set_title(fr"$\gamma$={a.gamma}, temp={a.temp}, $\beta$={a.beta} · {nb} branches, {nf} died, "
                 fr"{tree['n_reached']} reached · {it_txt}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[tree_cur] saved {out}", flush=True)


if __name__ == "__main__":
    main()
