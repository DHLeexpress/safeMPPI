# _v4 | model: it200 = results/p2/final_b02 (walled emergent-gamma GRAND FINAL, beta .2, mix .4/.6, q .30,
# rollouts 28, gp=qbuf 500, demo .125+LwF .05, start-eps .05, reach .2) | data: probe.jsonl (200 per-iter
# rows: sr50/cr50/cov50, per-gamma gathered-window counts = THE emergent-curriculum evidence, pools,
# batches, fstep/loss, vr, sigma stats) | layout 2x3; gamma colors = plasma (viridis reserved for sigma)
"""Training internals _v4 for the walled it200 run. Panel B is the paper's mechanism plot: per-gamma
certified-window share over training — the low gammas JOIN as the frontier lifts clearance (emergent
curriculum from the certificate, nothing imposed)."""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 11.5, "axes.titlesize": 13})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
RUN = os.path.join(P2, "results/p2/faithful_g47_it100")   # fresh it0→100 window-level run (2026-07-15)
GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
PLA = plt.get_cmap("plasma")
GC = {g: PLA(0.08 + 0.77 * i / 6) for i, g in enumerate(GAMMAS)}


def main():
    rows = [json.loads(l) for l in open(os.path.join(RUN, "probe.jsonl"))]
    T = np.array([r["iter"] for r in rows], float)

    def tr(k, default=np.nan):
        return np.array([r.get(k) if r.get(k) is not None else default for r in rows], float)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.2))

    # A: measure SR/CR/coverage (sr50 only exists at measure iters — plot the valid points)
    a = ax[0, 0]
    for k, c, lab in (("sr50", "#009944", "SR (M50 probe)"), ("cr50", "#cc3311", "CR")):
        v = tr(k); m = np.isfinite(v)
        a.plot(T[m], v[m], "-o", c=c, lw=2.0, ms=4, label=lab)
    cv = tr("cov50"); m = np.isfinite(cv)
    a2 = a.twinx(); a2.plot(T[m], cv[m], "--s", c="#4477aa", lw=1.4, ms=3, label="coverage")
    a2.set_ylabel("coverage (modes)", color="#4477aa")
    a.set_ylim(-0.02, 1.02); a.set_title("(A) probe SR / CR / coverage"); a.legend(loc="center right", fontsize=9)
    a.set_xlabel("iteration"); a.grid(alpha=.3)

    # B: EMERGENT gamma curriculum — per-gamma share of gathered certified windows
    # (gamma_counts keys carry float32 noise, e.g. '0.20000000298023224' — match by round(float(k),2))
    b = ax[0, 1]
    counts = {g: [] for g in GAMMAS}
    for r in rows:
        gc = {round(float(k), 2): v for k, v in (r.get("gamma_counts") or {}).items()}
        tot = max(1.0, float(sum(gc.values())))
        for g in GAMMAS:
            counts[g].append(100.0 * float(gc.get(round(float(g), 2), 0.0)) / tot)
    for g in GAMMAS:
        b.plot(T, np.convolve(counts[g], np.ones(9) / 9, mode="same"), "-", c=GC[g], lw=2.0, label=f"γ{g}")
    b.set_title("(B) per-γ share of gathered valid2 windows"); b.set_xlabel("iteration")
    b.set_ylabel("% of gathered windows (9-it smooth)"); b.legend(fontsize=8, ncol=2); b.grid(alpha=.3)

    # C: pools + batch composition
    c = ax[0, 2]
    c.plot(T, tr("n_easy"), "-", c="#00b300", lw=1.8, label="pool easy")
    c.plot(T, tr("n_frontier"), "-", c="#d62728", lw=1.8, label="pool frontier")
    c.plot(T, tr("batch_e"), "--", c="#00b300", lw=1.2, label="batch e")
    c.plot(T, tr("batch_f"), "--", c="#d62728", lw=1.2, label="batch f")
    c.plot(T, tr("batch_d"), "--", c="#7f7f7f", lw=1.2, label="batch demo")
    c.set_yscale("log"); c.set_title("(C) pools (solid) vs batch (dashed)"); c.set_xlabel("iteration")
    c.legend(fontsize=8, ncol=2); c.grid(alpha=.3)

    # D: update magnitude + loss
    d = ax[1, 0]
    d.plot(T, tr("functional_step"), "-", c="#0072B2", lw=1.8, label="functional step")
    d2 = d.twinx(); d2.plot(T, tr("loss"), "-", c="#D55E00", lw=1.2, alpha=.7, label="FM loss")
    d2.set_ylabel("FM loss", color="#D55E00")
    d.set_title("(D) update magnitude / loss"); d.set_xlabel("iteration"); d.grid(alpha=.3)
    d.legend(loc="upper right", fontsize=9)

    # E: gather valid rate — vr is the COUNT of valid rollouts; att = attempts
    e = ax[1, 1]
    vr, att = tr("vr"), tr("att")
    rate = 100.0 * vr / np.maximum(att, 1)
    e.plot(T, rate, "-", c="#009988", lw=1.8)   # raw (edge-safe); window-level ⇒ ~100% (all 7 γ valid)
    e.set_title("(E) gather valid-rollout rate (vr/att) — window-level ≈ 100%"); e.set_xlabel("iteration")
    e.set_ylabel("valid2 %"); e.set_ylim(0, 105); e.grid(alpha=.3)

    # F: sigma stats (viridis-ish greys per convention: sigma panel, no gamma colors)
    f = ax[1, 2]
    f.plot(T, tr("sig_e"), "-", c="#440154", lw=1.8, label=r"easy $\sigma$")
    f.plot(T, tr("sig_f"), "-", c="#35b779", lw=1.8, label=r"frontier $\sigma$")
    f.plot(T, tr("sigma_plane"), ":", c="k", lw=1.4, label=r"$\sigma_q$ plane")
    f.set_title(r"(F) novelty $\sigma$ by class"); f.set_xlabel("iteration"); f.legend(fontsize=9)
    f.grid(alpha=.3)

    fig.suptitle("Safe Flow Expansion — training internals (window-level, cleared start+goal, it0–100)", fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"internals_v4.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote internals_v4.png/.pdf")


if __name__ == "__main__":
    main()
