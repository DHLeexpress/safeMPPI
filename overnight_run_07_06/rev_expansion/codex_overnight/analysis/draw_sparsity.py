"""Batch-draw sparsity (CENTERPIECE, user 2026-07-15).

Window-level valid2 gives a HUGE positive pool every iter (~700 windows), but the gradient batch draws
only ~64-120 of them (mix 0.4/0.6 × inner steps). We now "sample efficiently but under-use." This
quantifies that gap and asks whether the sparse random draw is spatially REPRESENTATIVE of the pool.

Reads a run's viz_db/it*.pt (per-iter used_easy/used_frontier masks + window positions) + probe.jsonl.
Outputs paper_results/draw_sparsity.png and prints the numbers.

  python analysis/draw_sparsity.py --run results/p2/faithful_g47_it100
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
sys.path[:0] = [P2, os.path.dirname(P2), os.path.dirname(os.path.dirname(P2)),
                os.path.dirname(os.path.dirname(os.path.dirname(P2)))]
import torch
import grid_expand2 as GX2, grid_metrics2 as GM2


def load_it(run, it, goal):
    p = os.path.join(run, "viz_db", f"it{it}.pt")
    if not os.path.exists(p):
        return None
    z = torch.load(p, map_location="cpu", weights_only=False)
    GM2.GOAL_XY = np.array(goal, float)
    pos = np.array([np.asarray(GX2.state_from_low5(l), float)[:2] for l in z["low5"].numpy()])
    ue = np.asarray(z.get("used_easy", np.zeros(len(z["sigma"]), bool)))
    uf = np.asarray(z.get("used_frontier", np.zeros(len(z["sigma"]), bool)))
    return dict(pos=pos, sig=np.asarray(z["sigma"]), lab=np.asarray(list(z["label"]), dtype=object),
                used=ue | uf, used_e=ue, used_f=uf)


def bin_coverage(pos_pool, pos_used, lo=0.0, hi=5.0, nb=12):
    """Fraction of pool-occupied spatial cells that the USED subset also occupies (0..1)."""
    edges = np.linspace(lo, hi, nb + 1)
    hp = np.histogram2d(pos_pool[:, 0], pos_pool[:, 1], bins=[edges, edges])[0] > 0
    hu = np.histogram2d(pos_used[:, 0], pos_used[:, 1], bins=[edges, edges])[0] > 0
    return float((hp & hu).sum()) / max(int(hp.sum()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--goal-xy", type=float, nargs=2, default=[4.7, 4.7])
    ap.add_argument("--snap-iter", type=int, default=50, help="iter for the spatial pool-vs-used panel")
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(P2, args.run)

    its = sorted(int(os.path.basename(f)[2:-3]) for f in glob.glob(os.path.join(run, "viz_db", "it*.pt")))
    pool, used, fracs, cov = [], [], [], []
    for it in its:
        d = load_it(run, it, args.goal_xy)
        if d is None:
            continue
        npool, nused = len(d["pos"]), int(d["used"].sum())
        pool.append(npool); used.append(nused)
        fracs.append(100.0 * nused / max(npool, 1))
        cov.append(bin_coverage(d["pos"], d["pos"][d["used"]]) if nused else 0.0)
    its = [it for it in its if load_it(run, it, args.goal_xy) is not None]

    # snapshot iter for the spatial panel (nearest available)
    snap = min(its, key=lambda x: abs(x - args.snap_iter))
    ds = load_it(run, snap, args.goal_xy)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    # (1) pool vs used per iter
    ax[0].plot(its, pool, "o-", c="#4c72b0", label="valid pool (window-level)")
    ax[0].plot(its, used, "s-", c="#dd8452", label="drawn into batch")
    ax[0].set_title("(1) Positives GATHERED vs USED per iter")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("# windows"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].fill_between(its, used, pool, color="#4c72b0", alpha=.08)
    mfrac = float(np.mean(fracs))
    ax[0].text(.5, .5, f"mean used = {mfrac:.0f}% of pool\n(≈{np.mean(used):.0f} of {np.mean(pool):.0f})",
               transform=ax[0].transAxes, ha="center", fontsize=12,
               bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))
    # (2) spatial: pool (grey) vs used (colored) at snap iter
    ax[1].scatter(ds["pos"][:, 0], ds["pos"][:, 1], s=14, c="#cccccc", label=f"pool ({len(ds['pos'])})", zorder=2)
    ax[1].scatter(ds["pos"][ds["used_e"], 0], ds["pos"][ds["used_e"], 1], s=34, c="#00b300",
                  edgecolors="k", linewidths=.3, label=f"used-easy ({int(ds['used_e'].sum())})", zorder=3)
    ax[1].scatter(ds["pos"][ds["used_f"], 0], ds["pos"][ds["used_f"], 1], s=34, c="#d62728",
                  edgecolors="k", linewidths=.3, label=f"used-frontier ({int(ds['used_f'].sum())})", zorder=4)
    ax[1].plot(*args.goal_xy, "*", c="gold", mec="k", ms=15, zorder=5)
    ax[1].set_title(f"(2) Spatial draw @ it{snap}: used vs full pool")
    ax[1].set_xlim(-.3, 5.3); ax[1].set_ylim(-.3, 5.3); ax[1].set_aspect("equal")
    ax[1].legend(fontsize=9, loc="lower right"); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    # (3) spatial coverage of the pool by the draw
    ax[2].plot(its, [100 * c for c in cov], "o-", c="#55a868")
    ax[2].set_title("(3) Spatial coverage of pool by the draw")
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("% of pool-occupied cells hit by draw")
    ax[2].set_ylim(0, 105); ax[2].grid(alpha=.3)
    ax[2].axhline(100, ls=":", c="grey")

    fig.suptitle("Batch-draw sparsity — window-level yields a huge valid pool, the draw uses a sparse slice",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    out = os.path.join(P2, "paper_results", "draw_sparsity.png")
    for ext in ("png", "pdf"):
        fig.savefig(out.replace(".png", f".{ext}"), dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  iters={len(its)}  mean pool={np.mean(pool):.0f}  mean used={np.mean(used):.0f}  "
          f"mean used-frac={mfrac:.1f}%  mean spatial-coverage={100*np.mean(cov):.0f}%")
    print(f"  interpretation: {mfrac:.0f}% of the valid pool is drawn each iter; the draw covers "
          f"{100*np.mean(cov):.0f}% of the pool's occupied spatial cells "
          f"({'representative' if np.mean(cov) > .8 else 'SPARSE — misses regions'}).")


if __name__ == "__main__":
    main()
