"""Validity-tracking report for the AFE-minimal runs (user 2026-07-16: "track the validity of the
expansion results").  Reads probe.jsonl of one or more arms and renders the expansion/validity story:

  (1) V_hat per gamma on the ADVERSE audit slice (the expansion axis: pretrained gamma.1 = 0.00)
  (2) pooled V_hat adverse/rest + V_hat^prog (rest ~ceiling = the no-collapse guard)
  (3) query acceptance a_hat (tilted) vs fallback rate (its decay = the verified set growing under
      the executed distribution) -- kept SEPARATE from model validity by construction
  (4) closed-loop SR/CR + coverage
  (5) solver/telemetry: inner steps, functional step, sigma of drawn queries (acquisition signal
      decay), dithering share of new positives (the measured death-spiral watch)
  (6) mode-collapse audit: raw first-window up-fraction at the start context, final vs pretrained
      (the measured U-collapse failure mode of un-anchored training).

Usage: python analysis/afe_report.py --arms results/afe/A_s910 results/afe/B_s910 --out paper_results/afe_validity_v1.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import plasma

GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]


def load_probe(arm):
    recs = []
    with open(os.path.join(arm, "probe.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def series(recs, key):
    xs, ys = [], []
    for r in recs:
        if r.get(key) is not None:
            xs.append(r["round"])
            ys.append(r[key])
    return np.asarray(xs), np.asarray(ys)


def gamma_color(g):
    return plasma(0.1 + 0.8 * (float(g) - 0.1) / 0.9)


def raw_upfrac(ckpt, device="cpu", n=600):
    """Mode-collapse audit: fraction of raw sampled windows whose first-window displacement is
    up-dominant (dy>dx) at the fixed start context.  Pretrained ~0.14; the measured un-anchored
    collapse drove this to ~0.73 with 100% up-first deploys."""
    import torch
    import grid_feats as GF
    import grid_scene as GS
    import grid_hp_expt as HP
    import grid_expand_hardtail as HT
    import grid_rollout as GR
    pol, _ = HP.load_hp(ckpt, device=device)
    env = HT._apply_wall_plugs(GS.make_grid(), 8)
    env.goal = torch.tensor([4.7, 4.7], dtype=env.goal.dtype)
    st = np.array([0.3, 0.3, 0.0, 0.0], np.float32)
    obs = env.obstacles.detach().cpu().numpy()
    gT = torch.tensor(GF.axis_grid(st[:2], obs, float(env.r_robot)), device=device)
    lT = torch.tensor(GF.low5(st, env.goal.numpy(), 0.5), device=device)
    hT = torch.tensor(GF.hist_pad(np.zeros((0, 2)), GF.K_HIST), device=device)
    torch.manual_seed(0)
    U = pol.sample_window(gT, lT, hT, n=n, temp=1.0, nfe=8).cpu().numpy()
    pos = GR.di_rollout_batch(st, U, env.dt)
    net = pos[:, -1, :] - st[:2]
    return float((net[:, 1] > net[:, 0]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", default="paper_results/afe_validity_v1.png")
    ap.add_argument("--upfrac", action="store_true", help="include the raw-upfrac collapse audit (loads ckpts)")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(a.rstrip("/")) for a in args.arms]
    arms = [(lab, load_probe(a), a) for lab, a in zip(labels, args.arms)]
    ls_by_arm = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("AFE-minimal Safe Flow Expansion — validity tracking (no curriculum; "
                 "σ used once; full verifier BEFORE execution; certified fallback)", fontsize=13)

    # (1) V_hat adverse per gamma (arm 0 solid; arm 1 dashed)
    ax = axes[0, 0]
    for ai, (lab, recs, _) in enumerate(arms):
        for g in GAMMAS:
            xs, ys = [], []
            for r in recs:
                vga = r.get("V_gamma_adverse")
                if vga and g in vga:
                    xs.append(r["round"]); ys.append(vga[g])
            if xs:
                ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color=gamma_color(g), lw=1.6,
                        label=(f"γ{g}" if ai == 0 else None))
    ax.set_title("model validity V̂ per γ — ADVERSE audit slice (untilted, held-out ρ_eval)")
    ax.set_xlabel("round"); ax.set_ylabel("V̂ (certified fraction)"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, ncol=4, loc="lower right"); ax.grid(alpha=0.3)

    # (2) pooled validity
    ax = axes[0, 1]
    for ai, (lab, recs, _) in enumerate(arms):
        for key, c in (("V_adverse", "tab:red"), ("V_rest", "tab:green"), ("Vprog", "tab:blue")):
            xs, ys = series(recs, key)
            if len(xs):
                ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color=c, lw=1.8,
                        label=(f"{key}" if ai == 0 else None))
    ax.set_title("pooled validity (rest ≈ ceiling = no-collapse guard)")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.02, 0.02, " / ".join(f"{ls_by_arm[ai % len(ls_by_arm)]} = {lab}" for ai, (lab, _, _) in enumerate(arms)),
            transform=ax.transAxes, fontsize=9, color="dimgray")

    # (3) acceptance vs fallback (per gamma fallback)
    ax = axes[0, 2]
    for ai, (lab, recs, _) in enumerate(arms):
        xs, ys = series(recs, "a_hat")
        ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color="black", lw=1.8, label=("â (tilted acceptance)" if ai == 0 else None))
        for g in GAMMAS:
            xs, ys = [], []
            for r in recs:
                fg = r.get("fb_g")
                if fg and g in fg and fg[g][1] > 0:
                    xs.append(r["round"]); ys.append(fg[g][0] / fg[g][1])
            if xs and g in ("0.1", "0.2", "0.5", "1.0"):
                ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color=gamma_color(g), lw=1.2, alpha=0.85,
                        label=(f"fallback γ{g}" if ai == 0 else None))
    ax.set_title("query efficiency â vs certified-fallback rate (decay = expansion)")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) closed loop
    ax = axes[1, 0]
    for ai, (lab, recs, _) in enumerate(arms):
        for key, c in (("SR", "tab:green"), ("CR", "tab:red")):
            xs, ys = series(recs, key)
            if len(xs):
                ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color=c, marker="o", ms=3, lw=1.6,
                        label=(key if ai == 0 else None))
        covs = [(r["round"], sum(r["cov"].values())) for r in recs if r.get("cov")]
        if covs:
            xs, ys = zip(*covs)
            ax2 = ax.twinx() if ai == 0 else ax2
            ax2.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color="tab:purple", lw=1.2, alpha=0.7)
            ax2.set_ylabel("Σ coverage (distinct staircases)", color="tab:purple")
    ax.set_title("closed-loop SR / CR (bare policy, no shield) + coverage")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (5) solver + acquisition telemetry
    ax = axes[1, 1]
    for ai, (lab, recs, _) in enumerate(arms):
        xs, ys = series(recs, "sigma_drawn_mean")
        ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color="tab:olive", lw=1.5, label=("σ of drawn queries" if ai == 0 else None))
        xs, ys = series(recs, "fstep")
        ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color="tab:brown", lw=1.5, label=("functional step / round" if ai == 0 else None))
        xs, ys = series(recs, "dither_new")
        ax.plot(xs, ys, ls_by_arm[ai % len(ls_by_arm)], color="tab:pink", lw=1.5, label=("dither share (new D⁺, r<0.05)" if ai == 0 else None))
    ax.set_title("acquisition signal, solver step, dithering-watch")
    ax.set_xlabel("round"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (6) text panel: final numbers + optional collapse audit
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["FINAL / LATEST:"]
    for lab, recs, arm in arms:
        last_v = next((r for r in reversed(recs) if r.get("V_adverse") is not None), {})
        last_m = next((r for r in reversed(recs) if r.get("SR") is not None), {})
        lines.append(f"[{lab}] rounds={recs[-1]['round']}  D={recs[-1].get('n_D')}  D+={recs[-1].get('n_Dpos')}")
        lines.append(f"   V_adv {last_v.get('V_adverse', float('nan')):.3f}  V_rest {last_v.get('V_rest', float('nan')):.3f}"
                     f"  SR {last_m.get('SR', float('nan')):.2f}  CR {last_m.get('CR', float('nan')):.2f}")
        vga = last_v.get("V_gamma_adverse") or {}
        if vga:
            lines.append("   V_adv γ: " + " ".join(f"{g}:{vga[g]:.2f}" for g in GAMMAS if g in vga))
        if args.upfrac:
            fp = os.path.join(arm, "final.pt")
            if os.path.exists(fp):
                uf = raw_upfrac(fp)
                lines.append(f"   raw up-frac @start (collapse audit): {uf:.2f}  (pretrained ≈ 0.14; collapsed ≈ 0.73)")
    if args.upfrac:
        lines.append("")
        lines.append("baseline (pretrained) up-frac: %.2f" %
                     raw_upfrac(os.path.join(_HERE, "..", "..", "results", "hp_repr", "pretrained_a32uni.pt")))
    ax.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=9, va="top")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
