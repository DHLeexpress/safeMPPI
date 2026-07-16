"""sigma-field explanation (CENTERPIECE, user 2026-07-15).

Corrects the intuition "near-origin has HIGH uncertainty early." sigma is a GP posterior std
  sigma^2(x) = k(x,x) - k(x,X)(K+lam I)^-1 k(X,x)          (uncertainty.py; RBF kernel, ell=0.2, lam=1e-2)
over a buffer X of QUERIED window-features phi_s (control ⊕ context). Two facts drive the picture:
  * empty buffer (iter 0) -> k(x,X)=0 -> sigma = sqrt(k(x,x)) = 1 EVERYWHERE (flat).
  * every rollout STARTS at the origin, so near-origin windows are the MOST queried -> buffer densest
    there -> sigma DROPS fastest near the start. High-sigma lives at the frontier (unreached) and the
    avoided OOB strips (never queried).  (Bayesian-linear posterior is the kernel="linear" special case.)

Uses the REAL per-window sigma the trainer logged in viz_db (grid/low5/hist/U/sigma/widx) — no
reconstruction. Shows sigma vs distance-from-start and the spatial sigma map, early vs late.

  python analysis/sigma_field.py --run results/p2/faithful_g47_it100
"""
import argparse, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2)),
                os.path.dirname(os.path.dirname(os.path.dirname(P2)))]
import torch
import grid_expand2 as GX2, grid_metrics2 as GM2


def load_it(run, it, goal, start):
    p = os.path.join(run, "viz_db", f"it{it}.pt")
    if not os.path.exists(p):
        return None
    z = torch.load(p, map_location="cpu", weights_only=False)
    GM2.GOAL_XY = np.array(goal, float)
    pos = np.array([np.asarray(GX2.state_from_low5(l), float)[:2] for l in z["low5"].numpy()])
    r = np.linalg.norm(pos - np.array(start), axis=1)
    return dict(pos=pos, sig=np.asarray(z["sigma"]), r=r,
                widx=np.asarray(z.get("widx", np.zeros(len(pos), int))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--goal-xy", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--start-xy", type=float, nargs=2, default=[0.3, 0.3])
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(P2, args.run)
    its = sorted(int(os.path.basename(f)[2:-3]) for f in glob.glob(os.path.join(run, "viz_db", "it*.pt")))
    early = min(its, key=lambda x: abs(x - 3)); late = max(its)
    de, dl = load_it(run, early, args.goal_xy, args.start_xy), load_it(run, late, args.goal_xy, args.start_xy)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # (1) sigma vs IN-TRAJ window index (widx): the temporally-initial ESCAPE windows are the lowest-σ
    # (most over-queried — every rollout starts there). Spatial distance is NOT the axis; feature novelty is.
    for d, c, lab in ((de, "#4c72b0", f"it{early}"), (dl, "#dd8452", f"it{late}")):
        wmax = int(min(d["widx"].max(), 20))
        xs = np.arange(wmax + 1)
        mu = [d["sig"][d["widx"] == w].mean() if (d["widx"] == w).any() else np.nan for w in xs]
        ax[0].plot(xs, mu, "o-", c=c, label=lab)
    ax[0].set_title("(1) σ vs in-traj window index — ESCAPE windows (widx≈0) lowest")
    ax[0].set_xlabel("window index along rollout (0 = initial escape)"); ax[0].set_ylabel("mean σ")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].annotate("initial-escape windows:\nmost queried → lowest σ\n(NOT high, as one might expect\nfor 'OOD near origin')",
                   xy=(0, np.nanmin([de["sig"][de["widx"] < 1].mean() if (de["widx"] < 1).any() else .45])),
                   xytext=(3, 0.36), fontsize=9, arrowprops=dict(arrowstyle="->"))
    # (2)(3) spatial σ maps, early vs late (viridis = σ; NEVER plasma, that's γ's). σ is ~uniform along the
    # queried corridor (FIFO buffer tracks the policy); the high-σ frontier is feature-space, not spatial.
    for a, d, it in ((ax[1], de, early), (ax[2], dl, late)):
        sc = a.scatter(d["pos"][:, 0], d["pos"][:, 1], c=d["sig"], cmap="viridis", s=22,
                       vmin=0.35, vmax=0.7, edgecolors="none")
        a.plot(*args.start_xy, "s", c="k", ms=8); a.plot(*args.goal_xy, "*", c="gold", mec="k", ms=15)
        a.set_title(f"σ map it{it}: ~uniform ({d['sig'].mean():.2f}) along the queried corridor")
        a.set_xlim(-.3, 5.3); a.set_ylim(-.3, 5.3); a.set_aspect("equal")
        fig.colorbar(sc, ax=a, fraction=.042, pad=.02, label="σ")

    fig.suptitle("σ = GP posterior std over a FIFO buffer of queried windows — roughly uniform (~0.5) once "
                 "filled; escape windows lowest; empty buffer at it0 ⇒ σ≡1 (not 'high near origin')",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P2, "paper_results", f"sigma_field.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote paper_results/sigma_field.png")
    # numbers for the report
    for d, it in ((de, early), (dl, late)):
        w0 = d["sig"][d["widx"] < 2]           # initial-escape windows (near origin, temporally first)
        wl = d["sig"][d["widx"] >= 2]          # later windows
        print(f"  it{it}: σ overall={d['sig'].mean():.3f} (std {d['sig'].std():.3f})  "
              f"escape(widx<2)={w0.mean():.3f}  later={wl.mean():.3f}  "
              f"-> escape LOWEST (over-queried); NOT high-near-origin; σ ~flat spatially")


if __name__ == "__main__":
    main()
