"""Four-panel latent-support diagnosis (START_HERE_LATEST.md visualization-level goal).

Separates, WITHOUT collapsing them into one "ill-conditioning" label:
  (i)   accepted-TARGET conditioning  — what the certified training pool looks like (panel 1);
  (ii)  latent-TAIL conditioning      — which base latents map to bad first actions at a healthy
                                        context (panel 2);
  (iii) empty-STATE support           — contexts where (nearly) every latent maps out of bounds
                                        (panel 3);
  plus the closed-loop consequence     — faithful trajectories of the 11 original failures and the
                                        two s766 gamma-1 regressions, t104 vs s766 (panel 4).

Deterministic: N>=4096 latents from a fixed CPU generator; true NFE8 Euler integration; fixed contexts
taken from the faithful traces themselves. Read-only w.r.t. training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent.parent)]

import seed12_tail_trace as ST  # noqa: E402  (trace_deploy, integrate, _ctx_of, GAMMAS, bounds)

GR = ST.GR
GM = ST.GM

ORIGIN_CASES = [(g, 12) for g in ST.GAMMAS]
NEARGOAL_CASES = list(ST.NEAR_GOAL)                    # [(0.1,22),(0.4,8),(0.5,3),(0.7,5)]
REGRESSION_CASES = [(1.0, 5), (1.0, 14)]               # s766 regressions (fixed_seed_gate_s766)


def det_latents(n, d, seed=20260711):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=gen)


@torch.no_grad()
def raw_u0(policy, ctx, X0, nfe=8):
    """Integrate and return RAW (pre-clamp) first-action y plus the clamped window."""
    n = X0.shape[0]
    ctx_e = policy._expand_ctx(ctx, n)
    x = X0.clone()
    for i in range(nfe):
        tau = torch.full((n,), i / nfe, device=x.device)
        x = x + (1.0 / nfe) * policy.forward(x, tau, ctx_e)
    raw = (x.reshape(n, policy.T, 2) * policy.u_max)
    U = raw.clamp(-policy.u_max, policy.u_max)
    return raw[:, 0, 1].cpu().numpy(), U


def win_oob_frac(policy, step, env, X0, device, goal):
    gT = torch.tensor(step["grid"], device=device); lT = torch.tensor(step["low5"], device=device)
    hT = torch.tensor(step["hist"], device=device)
    ctx = ST._ctx_of(policy, gT, lT, hT)
    _, U = raw_u0(policy, ctx, X0.to(device))
    pos = GR.di_rollout_batch(np.asarray(step["state"], np.float32), U.cpu().numpy(), env.dt)
    oob_t = ((pos < ST.LO) | (pos > ST.HI)).any(axis=2)
    reach_t = np.linalg.norm(pos - goal[None, None, :], axis=2) < 0.1
    H = pos.shape[1]
    first = lambda m: np.where(m.any(axis=1), m.argmax(axis=1), H + 1)
    return float((first(oob_t) < first(reach_t)).mean()), float(U[:, 0, 1].mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-latents", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz-db", default=str(HERE.parent / "results/p2/corrected_mode2_target50_s81_from103_to105/viz_db/it104.pt"))
    ap.add_argument("--probe-json", default=str(HERE / "origin_window_failure_probe.claude2.json"))
    ap.add_argument("--fig", default=str(HERE.parent / "figures/current_goal_latent_support.png"))
    ap.add_argument("--out", default=str(HERE / "latent_support_map.json"))
    args = ap.parse_args()
    dev = args.device
    P2 = HERE.parent

    CKPTS = {
        "t104": P2 / "results/p2/corrected_mode2_target50_s81_from103_to105/ckpt_104.pt",
        "s671": P2 / "results/p2/origin_tightgate_production_s671.pt",
        "s766": P2 / "results/p2/goal_brake_gammaaug_s766.pt",
    }
    pols = {}
    for k, p in CKPTS.items():
        pol, _ = ST.HP.load_hp(str(p), device=dev)
        pol.eval(); pols[k] = pol
    env = ST.GS.make_grid()
    goal = env.goal.detach().cpu().numpy()
    res = {"n_latents": args.n_latents, "ckpts": {k: str(v) for k, v in CKPTS.items()}}

    # ---------- traces: 11 originals (t104 + s766) and 2 regressions (t104 + s766) ----------
    print("[traces]", flush=True)
    traces = {}
    for name in ("t104", "s766"):
        for g, s in ORIGIN_CASES + NEARGOAL_CASES + REGRESSION_CASES:
            traces[(name, g, s)] = ST.trace_deploy(pols[name], env, g, s, device=dev)
    taxa = {f"{n}_g{g}_s{s}": ("success" if traces[(n, g, s)]["reached"] else
                               ("oob" if traces[(n, g, s)]["dead"] else "timeout"))
            for (n, g, s) in traces}
    res["trace_outcomes"] = taxa

    # ---------- panel 2 data: origin latent tail (t104 + comparison OOB rates) ----------
    print("[P2 origin latent tail]", flush=True)
    X0 = det_latents(args.n_latents, pols["t104"].d)
    step0 = traces[("t104", 0.5, 12)]["steps"][0]
    gT = torch.tensor(step0["grid"], device=dev); lT = torch.tensor(step0["low5"], device=dev)
    hT = torch.tensor(step0["hist"], device=dev)
    u0y_raw = {}
    for k, pol in pols.items():
        ctx = ST._ctx_of(pol, gT, lT, hT)
        u0y_raw[k], _ = raw_u0(pol, ctx, X0.to(dev))
    seed12_x0 = torch.tensor(step0["x0"], device=dev)[None]
    seed12_u0y = {k: raw_u0(pols[k], ST._ctx_of(pols[k], gT, lT, hT), seed12_x0)[0][0] for k in pols}
    origin_tail = {}
    for k, pol in pols.items():
        f, _ = win_oob_frac(pol, step0, env, X0, dev, goal)
        origin_tail[k] = f
    res["origin_step0"] = {"seed12_u0y_raw": {k: float(v) for k, v in seed12_u0y.items()},
                           "win_oob_frac": origin_tail}

    # ---------- panel 3 data: near-goal empty-strip support across checkpoints ----------
    print("[P3 near-goal strip support]", flush=True)
    ng_ctx = {
        "g0.5 s3 (orig fail)": traces[("t104", 0.5, 3)]["steps"][-1],
        "g1.0 s5 (s766 reg)": traces[("s766", 1.0, 5)]["steps"][-1],
        "g1.0 s14 (s766 reg)": traces[("s766", 1.0, 14)]["steps"][-1],
    }
    X0s = det_latents(args.n_latents, pols["t104"].d, seed=20260712)
    p3 = {}
    for cname, step in ng_ctx.items():
        p3[cname] = {}
        for k, pol in pols.items():
            f, u0m = win_oob_frac(pol, step, env, X0s, dev, goal)
            p3[cname][k] = {"win_oob": f, "u0y_mean": u0m}
    res["near_goal_support"] = p3

    # ---------- panel 1 data: accepted pool ----------
    db = torch.load(args.viz_db, map_location="cpu", weights_only=False)
    import grid_expand2 as GX2
    posP = np.stack([np.asarray(GX2.state_from_low5(l), float)[:2] for l in db["low5"].numpy()])
    rad = np.linalg.norm(posP, axis=1)
    sig = np.asarray(db["sigma"]); lab = np.asarray(db["label"]).astype(str)
    probe = json.load(open(args.probe_json))
    snap = probe["window_snapshots"]["mode2_it104"]
    cond_near = snap["near_origin"]["control_centered_2d_condition"]["median"]
    cond_away = snap["away_from_origin"]["control_centered_2d_condition"]["median"]
    res["pool"] = {"n": int(len(rad)), "near_share": float((rad < 1).mean()),
                   "cond_median_near": cond_near, "cond_median_away": cond_away}

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    # ================= figure =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    # P1 — accepted-target space
    ax = fig.add_subplot(gs[0, 0])
    em = lab == "easy"
    ax.scatter(rad[em], sig[em], s=6, c="#4477aa", alpha=0.35, label=f"easy ({em.sum()})")
    ax.scatter(rad[~em], sig[~em], s=10, c="#cc3311", alpha=0.75, label=f"frontier ({(~em).sum()})")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlim(0, 7.4); ax.set_ylim(0, 1.02)
    ax.set_xlabel("window start radius ‖p₀‖ (m)"); ax.set_ylabel("GP σ (novelty)")
    ax.set_title("(1) ACCEPTED-TARGET space — t104 certified pool\n"
                 f"origin share (r<1): {(rad<1).mean()*100:.1f}%   "
                 f"SVD cond. median near/away: {cond_near:.2f}/{cond_away:.2f}  → targets are NOT ill-conditioned",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=10)

    # P2 — latent tail at the clean origin context
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-2.4, 2.4, 97)
    colors = {"t104": "#333333", "s671": "#4477aa", "s766": "#009988"}
    for k in ("t104", "s671", "s766"):
        ax.hist(u0y_raw[k], bins=bins, histtype="step", lw=2, color=colors[k], density=True,
                label=f"{k}  (win-OOB tail {origin_tail[k]*100:.1f}%)")
    ax.axvline(-1.0, color="k", ls=":", lw=1)
    ax.axvline(seed12_u0y["t104"], color="crimson", ls="-", lw=2)
    ax.annotate(f"seed-12 fiber\nraw u0_y={seed12_u0y['t104']:.2f} (t104)\n→ {seed12_u0y['s766']:.2f} (s766)",
                xy=(seed12_u0y["t104"], 0.55), xytext=(-2.3, 0.75), fontsize=11, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.set_xlabel("RAW first-action y (pre-clamp), N=%d deterministic NFE8 latents" % args.n_latents)
    ax.set_ylabel("density")
    ax.set_title("(2) LATENT-TAIL conditioning — clean origin context (γ.5)\n"
                 "a rare fiber maps to saturated down; repair retargets the fiber, not the mean", fontsize=12)
    ax.legend(fontsize=10)

    # P3 — empty-state support at near-goal contexts
    ax = fig.add_subplot(gs[1, 0])
    names = list(ng_ctx.keys()); ks = ["t104", "s671", "s766"]
    xx = np.arange(len(names)); w = 0.25
    for j, k in enumerate(ks):
        vals = [p3[n][k]["win_oob"] for n in names]
        ax.bar(xx + (j - 1) * w, vals, w, color=colors[k], label=k)
        for xi, v in zip(xx + (j - 1) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("window-OOB-before-reach fraction")
    ax.set_title("(3) EMPTY-STATE support — last-in-bounds near-goal contexts\n"
                 "≈1.0 for every checkpoint ⇒ the strip is absorbing regardless of latent; only new certified"
                 " data there changes it", fontsize=12)
    ax.legend(fontsize=10)

    # P4 — closed-loop consequence: trajectories
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.18)
    axo = fig.add_subplot(sub[0, 0])
    obs = env.obstacles.detach().cpu().numpy()
    for o in obs:
        axo.add_patch(plt.Circle(o[:2], o[2], color="#ccc", alpha=0.5))
    for g, s in ORIGIN_CASES:
        p = traces[("t104", g, s)]["path"]; axo.plot(p[:, 0], p[:, 1], color="crimson", lw=1.2, alpha=0.8)
        p = traces[("s766", g, s)]["path"]; axo.plot(p[:, 0], p[:, 1], color="#009988", lw=1.0, alpha=0.6)
    axo.axhline(-GM.EPS_TASK, color="k", ls="--", lw=1)
    axo.set_xlim(-0.3, 2.2); axo.set_ylim(-0.35, 2.2); axo.set_aspect("equal")
    axo.set_title("seed-12 ×7γ: t104 (red, all OOB)\nvs s766 (teal, all reach)", fontsize=11)
    axg = fig.add_subplot(sub[0, 1])
    for o in obs:
        axg.add_patch(plt.Circle(o[:2], o[2], color="#ccc", alpha=0.5))
    for g, s in NEARGOAL_CASES:
        p = traces[("t104", g, s)]["path"]; axg.plot(p[:, 0], p[:, 1], color="crimson", lw=1.1, alpha=0.75)
        p = traces[("s766", g, s)]["path"]; axg.plot(p[:, 0], p[:, 1], color="#009988", lw=1.0, alpha=0.6)
    for g, s in REGRESSION_CASES:
        p = traces[("t104", g, s)]["path"]; axg.plot(p[:, 0], p[:, 1], color="#009988", lw=1.6, ls="--", alpha=0.9)
        p = traces[("s766", g, s)]["path"]; axg.plot(p[:, 0], p[:, 1], color="darkorange", lw=1.6, ls="--", alpha=0.9)
    axg.plot(*goal, "*", ms=14, color="green")
    axg.axhline(GM.GRID_M + GM.EPS_TASK, color="k", ls="--", lw=1)
    axg.set_xlim(3.4, 5.4); axg.set_ylim(3.4, 5.4); axg.set_aspect("equal")
    axg.set_title("near-goal: 4 originals (red t104→teal s766 fixed);\n"
                  "dashed = γ1 s5/s14 (teal t104 ok → orange s766 REGRESS)", fontsize=11)
    fig.suptitle("Latent-support diagnosis — three DISTINCT phenomena: certified targets are well-conditioned (1); "
                 "a rare origin latent fiber is mis-mapped (2); boundary strips lack state support entirely (3); "
                 "(4) their closed-loop consequences under t104 vs s766", fontsize=14)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=110, bbox_inches="tight")
    print("wrote", args.fig, "and", args.out, flush=True)


if __name__ == "__main__":
    main()
