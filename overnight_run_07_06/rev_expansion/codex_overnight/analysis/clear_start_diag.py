"""Cleared-start/goal diagnostic (user 2026-07-14): the pretrained's U-bias is a start-position artifact.
Clear BOTH start (0.3,0.3) and goal (4.7,4.7) to 0.3 m and, on the INITIAL self-generations, measure
(a) how many U-first vs R-first trajectories, (b) the GP novelty sigma of the sampled rollout windows.
-> tells us what beta to use before the big dive. No valid2 here (raw deploy), so goal-move is fine.

  python analysis/clear_start_diag.py --ckpt results/p2/gs3_final/final.pt --M 60 --gamma 0.5
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2))]
import torch
import grid_scene as GS, grid_rollout as GR, grid_hp_expt as HP, grid_metrics as GM
import grid_expand_hardtail as HT
from grid_expand_hardtail import CurConfig, _sigma_of, _executed_horizon_tensors
from uncertainty import GPUncertainty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/p2/gs3_final/final.pt")
    ap.add_argument("--M", type=int, default=60)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--start", type=float, nargs=2, default=[0.3, 0.3])
    ap.add_argument("--goal", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--demo", default="w8d_")
    ap.add_argument("--tag", default="greedy")
    args = ap.parse_args()
    dev = "cuda"
    cfg = CurConfig()
    pol, _ = HP.load_hp(args.ckpt, device=dev); pol.eval()
    env = GS.make_grid(); HT._apply_wall_plugs(env, 8)
    env.x0 = torch.tensor([args.start[0], args.start[1], 0.0, 0.0], dtype=env.x0.dtype)
    env.goal = torch.tensor([args.goal[0], args.goal[1]], dtype=env.goal.dtype)

    # GP buffer from the demo windows (the "known" distribution) -> sigma = novelty vs demos
    d = torch.load(os.path.join(P2, "..", "..", "dataset", f"{args.demo}windows_g{args.gamma}.pt"),
                   map_location="cpu", weights_only=False)
    n = min(cfg.gp_buf, d["U"].shape[0]); idx = torch.randperm(d["U"].shape[0])[:n]
    with torch.no_grad():
        ctx = pol.ctx_from(d["grid"][idx].to(dev), d["low5"][idx].to(dev), d["hist"][idx].to(dev))
        buf = pol.phi_s(d["U"][idx].to(dev), ctx, s=cfg.s)
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.ell, lam=cfg.lam, normalize=True)
    unc.set_buffer(buf)

    # deploy M rollouts, record windows, sigma + U/R
    goal_np = np.asarray(args.goal); paths, sigs, cls = [], [], []
    U = R = 0; words = {}
    for s in range(args.M):
        torch.manual_seed(s)
        out = GR.fm_deploy(pol, env, float(args.gamma), T=250, temp=1.0, nfe=8, record=True,
                           device=dev, reach=0.15)
        p = np.asarray(out["path"], float); paths.append(p)
        reached = np.linalg.norm(p[-1] - goal_np) < 0.15
        if reached:
            w = GM.staircase_id(p, reach=0.15)
            if w:
                words[w] = words.get(w, 0) + 1
                is_u = str(w).startswith("U"); U += is_u; R += not is_u
                cls.append("U" if is_u else "R")
            else:
                cls.append("?")
        else:
            cls.append("fail")
        coh = _executed_horizon_tensors(out["recs"]) if out["recs"] else None
        if coh is not None:
            G, L, H, Uw = coh
            sg = _sigma_of(pol, unc, dict(grid=G, low5=L, hist=H, U=Uw), cfg, dev)
            sigs.append(float(np.mean(sg)))
        else:
            sigs.append(np.nan)

    sigs = np.array(sigs)
    print(f"[{args.tag}] start {tuple(args.start)} goal {tuple(args.goal)} γ{args.gamma} M{args.M}: "
          f"U-first {U} | R-first {R} | reached {U+R}/{args.M} | {len(words)} distinct")
    print(f"  sigma(rollout novelty vs demos): mean {np.nanmean(sigs):.3f} std {np.nanstd(sigs):.3f} "
          f"range [{np.nanmin(sigs):.3f},{np.nanmax(sigs):.3f}]")

    # figure: (A) rollouts colored by sigma, (B) U/R bar, (C) sigma hist by class
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.4))
    for o in env.obstacles.numpy():
        a.add_patch(plt.Circle(o[:2], o[2], color="#cccccc", zorder=1))
    vmin, vmax = np.nanpercentile(sigs, 2), np.nanpercentile(sigs, 98)
    for p, sg, cl in zip(paths, sigs, cls):
        col = plt.cm.viridis((sg - vmin) / (vmax - vmin + 1e-9)) if np.isfinite(sg) else "#bbbbbb"
        a.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.5, alpha=0.85, zorder=3)
    a.plot(*args.start, "ks", ms=8, zorder=6); a.plot(*args.goal, "*", c="gold", mec="k", ms=17, zorder=6)
    a.plot([args.start[0], args.goal[0]], [args.start[1], args.goal[1]], "k:", lw=1, alpha=.5)  # diagonal
    a.set_xlim(-0.45, 5.45); a.set_ylim(-0.45, 5.45); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax)); sm.set_array([])
    fig.colorbar(sm, ax=a, fraction=0.042, pad=0.02, label=r"$\sigma$ (novelty vs demos)")
    a.set_title(f"(A) {args.M} rollouts colored by σ\nstart+goal cleared 0.3m")
    b.bar(["U-first", "R-first", "fail"], [U, R, args.M - U - R], color=["#0072B2", "#D55E00", "#999999"])
    b.set_title(f"(B) mode balance: U {U} / R {R}\n(expert ~59/41, collapsed = all U)"); b.set_ylabel("count")
    for cl_, col in [("U", "#0072B2"), ("R", "#D55E00")]:
        v = [s for s, c2 in zip(sigs, cls) if c2 == cl_ and np.isfinite(s)]
        if v:
            c.hist(v, bins=15, alpha=0.6, color=col, label=f"{cl_}-first ({len(v)})")
    c.set_title("(C) σ distribution by mode\n(is R-first higher-σ = explorable?)")
    c.set_xlabel(r"mean $\sigma$ per rollout"); c.legend()
    fig.suptitle(f"Cleared start+goal diagnostic — {args.tag} (γ{args.gamma})  →  pick β from σ spread & U/R",
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(P2, "grand_final_reports_rev", f"clear_start_diag_{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
