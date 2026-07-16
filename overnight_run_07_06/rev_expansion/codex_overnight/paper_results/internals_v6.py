# _v6 | PARADIGM CHANGE (2026-07-16): pure AFE-minimal training internals for the paper. Data =
# results/afe/pure_pi_s910/probe.jsonl (100 rounds: per-gamma tilted acceptance + fallback, D/D+
# growth, distinct trained rows, prox-solver stats, UNTILTED audit V per gamma rest/adverse, sigma of
# drawn queries, dithering watch) + viz_db/round*.pt (fallback LOCATION split: the shield-moral-hazard
# panel). NO curriculum quantities exist in this paradigm (no easy/frontier pools, no mix, no
# quantile) — the mechanism panels are acquisition/verification/solver/audit instead.
"""Training internals _v6 (pure AFE). Panel B is the paper's mechanism plot: the certified fallback
rate by location — mid-route fallbacks are LEARNED AWAY (the verified set expands along the executed
support) while goal-corner fallbacks persist (no certified-positive data can arise there: the shield
carries what the flow cannot certify). Panel E separates model validity (untilted audit V-hat) from
query acceptance a-hat — AFE's central bookkeeping distinction."""
import glob, json, os, re, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 11.5, "axes.titlesize": 13})
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
RUN = os.path.join(P2, "results/afe/pure_pi_s910")
GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
PLA = plt.get_cmap("plasma")
GC = {g: PLA(0.08 + 0.77 * i / 6) for i, g in enumerate(GAMMAS)}


def fallback_location_series(run, goal_r=0.8):
    """Per-round (near-goal, mid-route) certified-fallback step counts from the viz DBs."""
    import torch
    out = {}
    for p in sorted(glob.glob(os.path.join(run, "viz_db", "round*.pt")),
                    key=lambda q: int(re.findall(r"round(\d+)\.pt", q)[0])):
        n = int(re.findall(r"round(\d+)\.pt", p)[0])
        db = torch.load(p, map_location="cpu", weights_only=False)
        goal = np.asarray(db["goal"], float)
        near = far = 0
        for pth, fb in zip(db["ep_paths"], db["ep_fb"]):
            pth = np.asarray(pth, float); fb = np.asarray(fb, bool)
            d = np.linalg.norm(pth[1:len(fb) + 1] - goal[None], axis=1)
            near += int(fb[d < goal_r].sum()); far += int(fb[d >= goal_r].sum())
        out[n] = (near, far)
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(RUN, "probe.jsonl"))]
    T = np.array([r["round"] for r in rows], float)

    def tr(k, default=np.nan):
        return np.array([r.get(k) if r.get(k) is not None else default for r in rows], float)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.2))

    # A: closed-loop SR/CR (bare policy, M=8/gamma) + coverage
    a = ax[0, 0]
    for k, c, lab in (("SR", "#009944", "SR"), ("CR", "#cc3311", "CR")):
        v = tr(k); m = np.isfinite(v)
        a.plot(T[m], v[m], "-o", c=c, lw=2.0, ms=4, label=lab)
    covs = [(r["round"], sum(r["cov"].values())) for r in rows if r.get("cov")]
    if covs:
        a2 = a.twinx(); a2.plot(*zip(*covs), "--s", c="#4477aa", lw=1.4, ms=3)
        a2.set_ylabel(r"$\Sigma$ coverage @M=8 (dominant-mode draw)", color="#4477aa")
        a2.annotate("M=8 undercounts the kept tail:\nfinal M=40 covΣ = 52 (≈ curriculum recipe 51)",
                    xy=(covs[-1][0], covs[-1][1]), xytext=(0.30, 0.42), textcoords="axes fraction",
                    fontsize=8.5, color="#4477aa",
                    arrowprops=dict(arrowstyle="->", color="#4477aa", lw=1.0))
    a.set_ylim(-0.02, 1.02); a.set_title("(A) closed-loop SR / CR / coverage (no shield)")
    a.legend(loc="center right", fontsize=9); a.set_xlabel("round"); a.grid(alpha=.3)

    # B: THE mechanism — certified-fallback rate per gamma + LOCATION split (shield moral hazard)
    b = ax[0, 1]
    for g in GAMMAS:
        xs, ys = [], []
        for r in rows:
            fg = r.get("fb_g")
            if fg and g in fg and fg[g][1] > 0:
                xs.append(r["round"]); ys.append(100.0 * fg[g][0] / fg[g][1])
        if xs:
            b.plot(xs, np.convolve(ys, np.ones(7) / 7, mode="same"), "-", c=GC[g], lw=1.6, label=f"γ{g}")
    loc = fallback_location_series(RUN)
    if loc:
        rn = sorted(loc)
        steps = {r2["round"]: r2.get("ep_steps", np.nan) * 8 for r2 in rows if r2.get("ep_steps")}
        near = [100.0 * loc[n][0] / max(steps.get(n, 1), 1) for n in rn]
        far = [100.0 * loc[n][1] / max(steps.get(n, 1), 1) for n in rn]
        b.plot(rn, np.convolve(near, np.ones(7) / 7, mode="same"), "--", c="k", lw=1.8,
               label="fallback NEAR GOAL (<0.8)")
        b.plot(rn, np.convolve(far, np.ones(7) / 7, mode="same"), ":", c="k", lw=1.8,
               label="fallback mid-route")
    b.set_title("(B) certified fallback: per-γ rate + LOCATION\n(mid-route learned away; goal corner persists)")
    b.set_xlabel("round"); b.set_ylabel("% of executed steps (7-rd smooth)")
    b.legend(fontsize=7.5, ncol=2); b.grid(alpha=.3)

    # C: D_n growth — every drawn plan is verified and stored (pos+neg); D+ trains
    c = ax[0, 2]
    c.plot(T, tr("n_D"), "-", c="#555555", lw=2.0, label=r"$|D_n|$ (all verified queries)")
    c.plot(T, tr("n_Dpos"), "-", c="#00b300", lw=2.0, label=r"$|D_n^+|$ (certified → trains)")
    c.plot(T, tr("n_train_distinct"), "--", c="#d62728", lw=1.4, label="distinct rows trained / round")
    c.set_yscale("log"); c.set_title("(C) cumulative verified store (no eviction)")
    c.set_xlabel("round"); c.legend(fontsize=9); c.grid(alpha=.3)

    # D: the prox solver — functional step per round (self-limited by eta), inner steps, CFM loss
    d = ax[1, 0]
    d.plot(T, tr("fstep"), "-", c="#0072B2", lw=1.8, label="functional step / round")
    d.plot(T, tr("inner_steps") / 40.0 * 0.03, ":", c="#888888", lw=1.2,
           label="inner steps (scaled /40 × .03)")
    d2 = d.twinx(); d2.plot(T, tr("cfm"), "-", c="#D55E00", lw=1.2, alpha=.7)
    d2.set_ylabel("CFM loss", color="#D55E00")
    d.set_title(r"(D) proximal update: $\ell_{CFM}+\|\theta-\theta_n\|^2/2\eta$ (η self-limits)")
    d.set_xlabel("round"); d.grid(alpha=.3); d.legend(loc="upper right", fontsize=9)

    # E: AFE's bookkeeping distinction — model validity (UNTILTED audit) vs query acceptance
    e = ax[1, 1]
    va = tr("V_adverse"); m = np.isfinite(va)
    e.plot(T[m], va[m], "-o", c="#cc3311", lw=2.0, ms=4, label=r"$\hat V$ adverse (pooled)")
    for g in ("0.1", "0.5", "1.0"):
        xs, ys = [], []
        for r in rows:
            vg = r.get("V_gamma_adverse")
            if vg and g in vg:
                xs.append(r["round"]); ys.append(vg[g])
        if xs:
            e.plot(xs, ys, "-", c=GC[g], lw=1.3, alpha=0.9, label=fr"$\hat V$ adv γ{g}")
    vr = tr("V_rest"); m = np.isfinite(vr)
    e.plot(T[m], vr[m], "-", c="#009944", lw=1.8, label=r"$\hat V$ rest")
    ah = tr("a_hat"); m = np.isfinite(ah)
    e.plot(T[m], ah[m], ":", c="k", lw=1.6, label=r"$\hat a$ (tilted acceptance)")
    e.set_ylim(-0.02, 1.05); e.set_title(r"(E) model validity $\hat V$ (untilted ρ_eval) ≠ acceptance $\hat a$")
    e.set_xlabel("round"); e.legend(fontsize=8, ncol=2, loc="center right"); e.grid(alpha=.3)

    # F: acquisition signal + dithering watch (sigma panel → viridis-family colors, never plasma)
    f = ax[1, 2]
    f.plot(T, tr("sigma_drawn_mean"), "-", c="#440154", lw=1.8, label=r"mean $\sigma$ of drawn plans")
    f.plot(T, tr("sigma_drawn_min"), "--", c="#35b779", lw=1.4, label=r"min $\sigma$ of drawn plans")
    f2 = f.twinx()
    f2.plot(T, 100 * tr("dither_new"), "-", c="#ee77aa", lw=1.4)
    f2.set_ylabel("dither share of new $D^+$ [%]", color="#ee77aa"); f2.set_ylim(0, 10)
    f.set_title(r"(F) acquisition signal $\sigma$ (A$_n$ saturates: candidates live in the"
                "\nqueried subspace) + dithering watch")
    f.set_xlabel("round"); f.legend(fontsize=9); f.grid(alpha=.3)

    fig.suptitle("Pure AFE-minimal Safe Flow Expansion — training internals "
                 "(one verified object; no curriculum, no stabilizers; prox-only)", fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"internals_v6.{ext}"), dpi=135, bbox_inches="tight")
    print("wrote internals_v6.png/.pdf")


if __name__ == "__main__":
    main()
