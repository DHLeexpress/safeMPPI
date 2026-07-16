"""Assemble the safe-flow-expansion report (07_06 Task D rewrite): training internals + SR/CR for the
WHITELISTED arms only, one fixed color per arm in every panel, single shared legend OUTSIDE the grid.

Arms (display name = fixed color everywhere):
  BASELINE=sweep_overnight/a32_unf   A=sweep_overnight/a32_unf_hi   A1=sweep_ac/A1
  B=sweep_overnight/a32_unf_long (KILLED -> no history.json; measures parsed from its LOG)
  C1/C2/C3=sweep_ac/C{1,2,3}
Any whitelisted arm without history.json falls back to parsing `<base>/logs/<arm>.log` lines like
  it02500 SR 0.39 CR 0.43 | loss 0.698 gRMS(fld 0.015 enc 0.009) | β 0.50 mix 0.58/0.31/0.11 pools ...

Layout 2x4: row1 = cfm loss · field grad-RMS · encoder grad-RMS · β;
row2 = lr (log-y, nan-safe) · FRONTIER batch portion per arm (easy mirrors it, mid stays ~1/3) · SR · CR.
Output figures/expand_report/internals_v1.png (+ summary_v1.json; the pre-_v1 files are kept untouched).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "expand_report"); os.makedirs(FIG, exist_ok=True)

# ---- the EXACT whitelist (display name, run dir) — every other arm is dropped -------------------------
# ALL uni arms on the a32uni backbone (2026-07-09): which learning was STABLE?
WHITELIST = (
    ("uni_A(s0)", "results/uni_expand/uni_A_b64i121"),     # WINNER seed0: vpf.15 only, fin 0.93
    ("uni_A_s1",  "results/uni_expand/uni_A_s1"),          # same recipe seed1: peak .74 fin 0
    ("uni_A_s2",  "results/uni_expand/uni_A_s2"),          # same recipe seed2: peak .86 fin 0
    ("uni_B_lr5e5", "results/uni_expand/uni_B_lr5e5"),     # lr 5e-5: collapsed
    ("uni_C_fix123", "results/uni_expand/uni_C_full"),     # min-rollouts+dither-gate+loose-thr: starved
    ("uni_base",  "results/uni_expand/uni_base"),          # first try (batch32 inner242, no vpf)
)
ARM_COL = {"uni_A(s0)": "#000000", "uni_A_s1": "#555555", "uni_A_s2": "#999999",   # recipe family = greys
           "uni_B_lr5e5": "#0072B2", "uni_C_fix123": "#D55E00", "uni_base": "#009E73"}


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


# ---- loading: history.json, else parse the run LOG ----------------------------------------------------
_MEAS = re.compile(r"^it(\d+) SR ([\d.]+) CR ([\d.]+) \| loss (nan|[\d.]+) "
                   r"gRMS\(fld ([\d.]+) enc ([\d.]+)\) \| β ([\d.]+) mix ([\d.]+)/([\d.]+)/([\d.]+)")
_BASE = re.compile(r"^it(\d+) SR ([\d.]+) CR ([\d.]+) \| baseline")


def parse_log(path):
    """Rebuild history recs (iter,SR,CR,loss,field/enc grad-RMS,beta,mix; lr absent->nan) from a run log."""
    recs = []
    nan = float("nan")
    for line in open(path):
        m = _MEAS.match(line)
        if m:
            recs.append(dict(iter=int(m[1]), SR=float(m[2]), CR=float(m[3]), loss=float(m[4]),
                             field_grad_rms=float(m[5]), enc_grad_rms=float(m[6]), beta=float(m[7]),
                             mix=[float(m[8]), float(m[9]), float(m[10])], lr=nan))
            continue
        b = _BASE.match(line)
        if b:                                              # it00000 baseline line: SR/CR only
            recs.append(dict(iter=int(b[1]), SR=float(b[2]), CR=float(b[3]), loss=nan,
                             field_grad_rms=nan, enc_grad_rms=nan, beta=nan,
                             mix=[nan, nan, nan], lr=nan))
    return recs


def load_arms():
    arms, src = {}, {}
    for disp, rel in WHITELIST:
        d = os.path.join(HERE, rel)
        h = os.path.join(d, "history.json")
        if os.path.exists(h):
            arms[disp] = json.load(open(h)); src[disp] = "history.json"
            continue
        lg = os.path.join(os.path.dirname(d), "logs", os.path.basename(d) + ".log")
        if os.path.exists(lg):
            H = parse_log(lg)
            if H:
                arms[disp] = H; src[disp] = f"LOG parse ({len(H)} rows)"
                continue
        print(f"[report] WARNING: {disp} ({rel}) has neither history.json nor a parsable log — skipped")
    return arms, src


def _nan_series(H, key):
    it = np.array([r["iter"] for r in H], float)
    v = np.array([float(r.get(key, np.nan)) if r.get(key) is not None else np.nan for r in H], float)
    return it, v


def main():
    usetex = setup_style()
    print(f"[report] usetex={'ON' if usetex else 'OFF (serif/mathtext-cm fallback)'}", flush=True)
    arms, src = load_arms()
    if not arms:
        print("[report] no whitelisted arms found"); return

    # ---- console + summary table ----
    summ = {}
    print(f"{'arm':9} {'SR0':>5} {'SRbest':>6} {'SRfin':>6} | {'CR0':>5} {'CR@best':>7} {'CRfin':>6} | "
          f"{'lossf':>6} {'fld':>5} {'enc':>5} | source")
    for n, H in arms.items():
        sr = np.array([r["SR"] for r in H]); cr = np.array([r["CR"] for r in H])
        ib = int(np.argmax(sr))
        summ[n] = dict(SR0=float(sr[0]), SRbest=float(sr[ib]), SRfin=float(sr[-1]), iter_best=int(H[ib]["iter"]),
                       CR0=float(cr[0]), CRatbest=float(cr[ib]), CRfin=float(cr[-1]),
                       loss_fin=float(H[-1]["loss"]), field_grad=float(H[-1]["field_grad_rms"]),
                       enc_grad=float(H[-1]["enc_grad_rms"]), n_rows=len(H), source=src[n])
        print(f"{n:9} {sr[0]:5.2f} {sr[ib]:6.2f} {sr[-1]:6.2f} | {cr[0]:5.2f} {cr[ib]:7.2f} {cr[-1]:6.2f} | "
              f"{H[-1]['loss']:6.3f} {H[-1]['field_grad_rms']:5.3f} {H[-1]['enc_grad_rms']:5.3f} | {src[n]}")
    json.dump(summ, open(os.path.join(FIG, "summary_rev.json"), "w"), indent=2)

    # ---- internals_v1: 2x4, one color per arm everywhere, ONE legend OUTSIDE (above the grid) ----
    fig = plt.figure(figsize=(22, 9.8))
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.26, top=0.86, bottom=0.07, left=0.045, right=0.985)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)])
    lw = lambda n: 2.6 if n == "uni_A(s0)" else 1.6

    row1 = (("loss", "cfm loss"), ("field_grad_rms", "field grad-RMS (update aggressiveness)"),
            ("enc_grad_rms", "encoder grad-RMS (leakage)"), ("beta", r"$\beta$ ($\sigma$-tilt temp)"))
    for j, (key, title) in enumerate(row1):
        ax = axs[0, j]
        for n, H in arms.items():
            it, v = _nan_series(H, key)
            ok = ~np.isnan(v)
            if ok.any():
                ax.plot(it[ok], v[ok], "-", lw=lw(n), color=ARM_COL[n])
        ax.set_title(title); ax.set_xlabel("expansion iter"); ax.grid(alpha=0.3)

    # (2,1) learning rate — log-y, nan-safe (log-parsed arms have no lr)
    ax = axs[1, 0]; n_lr = 0
    for n, H in arms.items():
        it, v = _nan_series(H, "lr")
        ok = ~np.isnan(v)
        if ok.any():
            n_lr += 1
            ax.plot(it[ok], v[ok], "-", lw=lw(n), color=ARM_COL[n])
    ax.set_title("learning rate (field group)"); ax.set_xlabel("expansion iter"); ax.grid(alpha=0.3, which="both")
    if n_lr:
        ax.set_yscale("log")
        miss = [n for n in arms if not (~np.isnan(_nan_series(arms[n], 'lr')[1])).any()]
        if miss:
            ax.text(0.98, 0.02, "no lr logged: " + ", ".join(miss), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color="#a33", style="italic")
    else:
        ax.text(0.5, 0.5, "no `lr` in any history/log", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#a33", style="italic")

    # (2,2) FRONTIER batch portion per arm — frontier = LAST mix element (2-class: [easy,frontier]; the
    # REF_a32 arm is 3-class [easy,mid,frontier] so last element is frontier for both).
    ax = axs[1, 1]
    for n, H in arms.items():
        it = np.array([r["iter"] for r in H], float)
        f = np.array([(m[-1] if m else np.nan) for m in (r.get("mix") for r in H)], float)
        ok = ~np.isnan(f)
        if ok.any():
            ax.plot(it[ok], f[ok], "-", lw=lw(n), color=ARM_COL[n])
    ax.set_title("FRONTIER batch portion\n(2-class: easy = 1 $-$ frontier)", fontsize=13)
    ax.set_xlabel("expansion iter"); ax.set_ylim(-0.02, 0.6); ax.grid(alpha=0.3)

    # (2,3)/(2,4) SR / CR (merges the old sr_cr_trajectories figure into internals)
    for j, (key, title) in enumerate((("SR", r"SR (origin, reach$\leq$0.1 m)"), ("CR", "CR (collision rate)"))):
        ax = axs[1, 2 + j]
        for n, H in arms.items():
            it, v = _nan_series(H, key)
            ax.plot(it, v, "-", lw=lw(n), color=ARM_COL[n])
        ax.set_title(title); ax.set_xlabel("expansion iter"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)

    for ax in axs.flat:                                    # compare the shared 0-1k regime (REF_a32 runs to 5k)
        ax.set_xlim(-10, 1050)
    handles = [Line2D([], [], color=ARM_COL[n], lw=lw(n), label=n) for n in arms]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=len(arms),
               columnspacing=2.2, handlelength=2.6)
    fig.suptitle("uni arms on a32uni — training internals + SR/CR (greys = uni_A recipe seeds 0/1/2: "
                 "same recipe, endpoint coin-flip)", y=0.985)
    out = os.path.join(FIG, "internals_uni.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
