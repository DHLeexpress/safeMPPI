"""Curriculum diagnostic for the faithful window-level run (results/p2/faithful_g47).

Answers 'why is the pretrained already good from the start?' — it IS (SR starts high); the expansion moves
the SAFETY + DIVERSITY axes, not raw SR. Four panels over 50 iters:
  (1) Reliability      : probe SR50 / CR50 (internal) + milestone eval pooled-SR — SR high from it10, CR->0
  (2) Safety [learning]: milestone pooled clearance rising, vs expert / Kazuki reference lines
  (3) Valid-window supply (window-level): n_easy / n_frontier growing (certifiable region expands)
  (4) Uncertainty      : sig_easy / sig_frontier — FLAT (honest caveat: FIFO novelty tracks the policy)
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
RUN = os.path.join(P2, "results/p2/faithful_g47")
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
MILES = [10, 20, 30, 40, 50]

matplotlib.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                            "axes.spines.top": False, "axes.spines.right": False})


def pooled(evaldir):
    """mean SR & mean clearance over the 7 gammas of an eval dir (None if absent)."""
    srs, clrs = [], []
    for g in GAMMAS:
        fp = os.path.join(P2, evaldir, f"row_g{g}.json")
        if not os.path.exists(fp):
            continue
        r = json.load(open(fp)); srs.append(r["SR"])
        if r.get("clearance_mean") is not None and np.isfinite(r["clearance_mean"]):
            clrs.append(r["clearance_mean"])
    return (np.mean(srs) if srs else np.nan), (np.mean(clrs) if clrs else np.nan)


def main():
    R = [json.loads(l) for l in open(os.path.join(RUN, "probe.jsonl"))]
    it = np.array([r["iter"] for r in R])
    def col(k):
        return np.array([r.get(k) if isinstance(r.get(k), (int, float)) else np.nan for r in R], float)
    sr50, cr50 = col("sr50"), col("cr50")
    ne, nf = col("n_easy"), col("n_frontier")
    se, sf = col("sig_e"), col("sig_f")

    # milestone pooled eval SR + clearance (it10..50), plus expert/kazuki clearance reference
    mi_sr = [pooled(f"results/p2/eval_faithful_it{m}")[0] for m in MILES]
    mi_clr = [pooled(f"results/p2/eval_faithful_it{m}")[1] for m in MILES]
    exp_clr = pooled("results/expert_g47")[1]
    kaz_clr = pooled("results/kazuki_g47")[1]

    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.2))
    (a1, a2), (a3, a4) = ax

    # (1) reliability
    m = np.isfinite(sr50)
    a1.plot(it[m], sr50[m] * 100, "o-", c="#1f77b4", label="probe SR (M50, internal)")
    a1.plot(MILES, np.array(mi_sr) * 100, "s--", c="#2c7fb8", alpha=.7, label="eval SR (pooled 7γ, M40)")
    a1.plot(it[m], cr50[m] * 100, "o-", c="#d62728", label="probe CR")
    a1.axhline(100, ls=":", c="grey", lw=1); a1.set_ylim(-4, 108)
    a1.set_title("(1) Reliability — already high from it10, CR→0")
    a1.set_xlabel("iteration"); a1.set_ylabel("rate [%]"); a1.legend(fontsize=9, loc="center right")

    # (2) safety = the learning axis
    a2.plot(MILES, mi_clr, "*-", c="#238b45", ms=13, label="Ours (pooled 7γ)")
    if np.isfinite(exp_clr): a2.axhline(exp_clr, ls="--", c="#555", lw=1.4, label=f"Expert {exp_clr:.3f}")
    if np.isfinite(kaz_clr): a2.axhline(kaz_clr, ls=":", c="#b30000", lw=1.4, label=f"Kazuki {kaz_clr:.3f}")
    a2.set_title("(2) Safety = the real learning axis (clearance ↑)")
    a2.set_xlabel("iteration"); a2.set_ylabel("min clearance [m]"); a2.legend(fontsize=9, loc="center right")

    # (3) valid-window supply grows
    a3.plot(it, ne, "o-", c="#7570b3", label="n_easy (valid windows)")
    a3.plot(it, nf, "o-", c="#e7298a", label="n_frontier (high-σ valid)")
    a3.set_title("(3) Window-level supply grows — certifiable region expands")
    a3.set_xlabel("iteration"); a3.set_ylabel("# valid windows / iter"); a3.legend(fontsize=9)

    # (4) uncertainty flat (caveat) — sigma uses green/viridis family, never plasma (that's gamma's)
    a4.plot(it, se, "o-", c="#31a354", label="σ easy")
    a4.plot(it, sf, "o-", c="#a1d99b", label="σ frontier")
    a4.set_ylim(0, max(1.0, np.nanmax(sf) * 1.3))
    a4.set_title("(4) Uncertainty stays FLAT (FIFO novelty tracks policy)")
    a4.set_xlabel("iteration"); a4.set_ylabel("mean σ"); a4.legend(fontsize=9)

    fig.suptitle("Faithful window-level curriculum (pretrained → start (0.3,0.3) → goal (4.7,4.7), 50 iters)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"curriculum_faithful.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote curriculum_faithful.png/.pdf")
    print(f"  SR50 it10={sr50[it==10][0]:.2f} -> it50={sr50[it==50][0]:.2f} | "
          f"clr it10={mi_clr[0]:.3f} -> it50={mi_clr[-1]:.3f} (exp {exp_clr:.3f}, kaz {kaz_clr:.3f})")
    print(f"  n_easy {ne[0]:.0f}->{ne[-1]:.0f}  n_frontier {nf[0]:.0f}->{nf[-1]:.0f}  "
          f"σe {np.nanmean(se):.3f} σf {np.nanmean(sf):.3f} (flat)")


if __name__ == "__main__":
    main()
