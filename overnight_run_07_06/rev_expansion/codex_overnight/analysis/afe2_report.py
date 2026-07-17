"""AFE2 diagnostics report (user spec 2026-07-16b): matched prox/afe protocols with one initial
checkpoint and common-random-number streams; learned representations and trajectories diverge.
Panels:
  A pooled expert-free controller SR / NVP / CR      B per-gamma controller SR
  C full-H untilted validity per gamma (audit)       D update: mean CFM loss, per-module grad norms,
                                                       relative parameter change, fixed-probe
                                                       representation cosine drift (NO "encoder loss")
  E uncertainty: all-K vs selected-B sigma medians, sigma IQR, centered feature rank
  F acquisition: ESS/K, normalized entropy, selected-vs-pool uplift
  G data: |D|, |D+|, distinct trained, dither share   H text: final per-gamma table with Wilson CIs
Usage: python analysis/afe2_report.py --arms PROX AFE --pair-manifest PAIR.json
"""
import argparse
import hashlib
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

GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
PLA = plt.get_cmap("plasma")
GC = {g: PLA(0.08 + 0.77 * i / 6) for i, g in enumerate(GAMMAS)}
LS = {0: "-", 1: "--"}


def wilson(p, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - hw), min(1.0, ctr + hw)


def bootstrap_mean_ci(values, seed, n_boot=5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def fmt2(value):
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.2f}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pair-manifest", required=True)
    ap.add_argument("--out", default="paper_results/afe2_report_v1.png")
    args = ap.parse_args()
    arms = []
    metadata = {}
    with open(args.pair_manifest) as stream:
        pair_manifest = json.load(stream)
    if pair_manifest.get("status") != "VALIDATED_MATCHED_AFE2_PAIR":
        raise RuntimeError("pair manifest status is not validated")
    for a in args.arms:
        recs = [json.loads(l) for l in open(os.path.join(a, "probe.jsonl"))]
        label = os.path.basename(a.rstrip("/")).replace("_s910", "")
        with open(os.path.join(a, "recipe.json")) as stream:
            recipe = json.load(stream)
        root = os.path.realpath(a)
        match = next(
            (run for run in pair_manifest["runs"].values() if os.path.realpath(run["root"]) == root),
            None,
        )
        if match is None:
            raise RuntimeError(f"arm is absent from pair manifest: {root}")
        if sha256_file(os.path.join(a, "recipe.json")) != match["recipe_sha256"]:
            raise RuntimeError(f"recipe changed after pair validation: {root}")
        if sha256_file(os.path.join(a, "probe.jsonl")) != match["probe_sha256"]:
            raise RuntimeError(f"probe changed after pair validation: {root}")
        arms.append((label, recs))
        metadata[label] = recipe
    for recipe in metadata.values():
        if recipe["scene"]["sha256"] != pair_manifest["scene_sha256"]:
            raise RuntimeError("pair manifest scene does not match an arm recipe")
        if recipe["source_checkpoint_sha256"] != pair_manifest["source_checkpoint_sha256"]:
            raise RuntimeError("pair manifest checkpoint does not match an arm recipe")

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.6))

    ax = axes[0, 0]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for key, c in (("SR", "#009944"), ("NVP", "#cc7711"), ("CR", "#cc3311")):
            ys = [r["ctrl_pooled"][key] for r in recs]
            ax.plot(R, ys, LS[ai], color=c, lw=2.0, marker="o", ms=3,
                    label=(key if ai == 0 else None))
        M = int(metadata[lab]["M_eval"])
        prefix_required = [
            sum(row["terminal_required_episodes"] for row in r["ctrl"].values())
            / (M * len(GAMMAS))
            for r in recs
        ]
        ax.plot(R, prefix_required, LS[ai], color="#663399", lw=1.6, marker=".",
                label=("prefix-required episode rate" if ai == 0 else None))
    ax.set_title("(A) terminal-aware expert-free controller (pooled)\nSR / NVP / CR")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.5, " / ".join(f"{LS[ai]} = {lab}" for ai, (lab, _) in enumerate(arms)),
            transform=ax.transAxes, fontsize=9, color="dimgray")

    ax = axes[0, 1]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for g in GAMMAS:
            ys = [r["ctrl"][g]["SR"] for r in recs]
            ax.plot(R, ys, LS[ai], color=GC[g], lw=1.3, label=(f"γ{g}" if ai == 0 else None))
    ax.set_title("(B) controller SR per γ"); ax.set_xlabel("round")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 2]
    for ai, (lab, recs) in enumerate(arms):
        R = [r["round"] for r in recs]
        for g in GAMMAS:
            safe = [r.get("V_safe_gamma", r["V_gamma"])[g] for r in recs]
            full = [r.get("V_full_gamma", r.get("Vprog_gamma", r["V_gamma"]))[g]
                    for r in recs]
            ax.plot(R, safe, LS[ai], color=GC[g], lw=1.4,
                    label=(f"γ{g}" if ai == 0 else None))
            ax.plot(R, full, LS[ai], color=GC[g], lw=.8, alpha=.28)
        ax.plot(R, [r.get("V_adverse") for r in recs], LS[ai], color="k", lw=1.8,
                label=("adverse (pooled)" if ai == 0 else None))
    ax.set_title("(C) untilted validity per γ: opaque V_safe, faint V_full")
    ax.set_xlabel("round"); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 3]
    ax2 = ax.twinx()
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("cfm") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["cfm"] for r in rr], LS[ai], color="#D55E00", lw=1.8,
                label=("CFM loss" if ai == 0 else None))
        for kg, c in (("E_g", "#4477aa"), ("trunk", "#009988"), ("head", "#aa3377")):
            ax2.plot(R, [r["grad_norm"].get(kg, np.nan) for r in rr], LS[ai], color=c, lw=1.0,
                     alpha=0.8, label=(f"|grad| {kg}" if ai == 0 else None))
    ax2.set_yscale("log"); ax2.set_ylabel("per-module grad norm (log)")
    ax.set_title("(D) update: mean CFM loss + encoder/trunk/head grad norms")
    ax.set_xlabel("round"); ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

    ax = axes[1, 0]
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("rep_cos") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["rep_cos"] for r in rr], LS[ai], color="#0b3d91", lw=1.8,
                label=("rep cosine vs φ⁰ (fixed probe)" if ai == 0 else None))
        rr2 = [r for r in recs if r.get("rel_param_change")]
        for kg, c in (("E_g", "#4477aa"), ("trunk", "#009988"), ("head", "#aa3377")):
            ax.plot([r["round"] for r in rr2],
                    [10 * r["rel_param_change"].get(kg, np.nan) for r in rr2], LS[ai], color=c,
                    lw=1.0, alpha=0.8, label=(f"10×Δθ/θ {kg}" if ai == 0 else None))
    ax.set_title("(E) representation drift + relative parameter change")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("ess_med") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["ess_med"] for r in rr], LS[ai], color="#117733", lw=1.8,
                label=("median ESS/K" if ai == 0 else None))
        ax.plot(R, [r["ent_med"] for r in rr], LS[ai], color="#88ccee", lw=1.4,
                label=("median entropy/log K" if ai == 0 else None))
        ax.plot(R, [r["uplift_med"] for r in rr], LS[ai], color="#999933", lw=1.4,
                label=("σ uplift (sel−pool)" if ai == 0 else None))
    ax.axhline(0.375, color="#117733", alpha=0.45, lw=1.0, ls=":")
    ax.set_title("(F) acquisition: ESS/K (round-0 target=.375), entropy, uplift")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax2 = ax.twinx()
    for ai, (lab, recs) in enumerate(arms):
        rr = [r for r in recs if r.get("sig_all_med") is not None]
        R = [r["round"] for r in rr]
        ax.plot(R, [r["sig_all_med"] for r in rr], LS[ai], color="#440154", lw=1.6,
                label=("median σ all-K" if ai == 0 else None))
        ax.plot(R, [r["sig_sel_med"] for r in rr], LS[ai], color="#35b779", lw=1.6,
                label=("median σ selected-B" if ai == 0 else None))
        ax.plot(R, [r.get("sig_iqr_med", np.nan) for r in rr], LS[ai],
                color="#31688e", lw=1.0, alpha=.8,
                label=("median σ IQR" if ai == 0 else None))
        ax2.plot(R, [r.get("S_centered_eff_rank", np.nan) for r in rr], LS[ai],
                 color="#888888", lw=1.2)
    ax2.set_ylabel("centered feature effective rank (grey)", color="#666666")
    ax.set_title("(G) uncertainty + centered feature rank (A rebuilt/round)")
    ax.set_xlabel("round"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1, 3]
    ax.axis("off")
    lines = []
    for lab, recs in arms:
        last = recs[-1]
        M = int(metadata[lab]["M_eval"])
        lines.append(f"[{lab}] final round {last['round']}  D {last.get('n_D')}  D+ {last.get('n_Dpos')}")
        lines.append(" γ  SR[95%W]    NVP[95%W]   CR[95%W]")
        lines.append("    Vsafe/Vfull[95%W]/Vadv · q/solve/accept/prefix/train-draw(distinct) · clr/ttg[95%B] · R/Rreq")
        for g in GAMMAS:
            c = last["ctrl"][g]
            sr_lo, sr_hi = wilson(c["SR"], M)
            nv_lo, nv_hi = wilson(c["NVP"], M)
            cr_lo, cr_hi = wilson(c["CR"], M)
            v_counts = last["V_counts_gamma"][g]
            v_lo, v_hi = wilson(last["V_gamma"][g], int(v_counts["n"]))
            v_full = last.get("V_full_gamma", last.get("Vprog_gamma", last["V_gamma"]))[g]
            vf_counts = last.get("V_counts_gamma_full", {}).get(g, v_counts)
            vf_lo, vf_hi = wilson(v_full, int(vf_counts["n"]))
            v_adv = (last.get("V_gamma_adverse") or {}).get(g, np.nan)
            v_adv_counts = (last.get("V_counts_gamma_adverse") or {}).get(g, {})
            va_lo, va_hi = wilson(v_adv, int(v_adv_counts.get("n", 0)))
            gamma_index = GAMMAS.index(g)
            clr_lo, clr_hi = bootstrap_mean_ci(c["clear_values"], 91000 + gamma_index)
            ttg_lo, ttg_hi = bootstrap_mean_ci(c["time_success_values"], 92000 + gamma_index)
            pg = (last.get("per_gamma") or {}).get(g, {})
            lines.append(
                f" {g} {c['SR']:.2f}[{sr_lo:.2f},{sr_hi:.2f}] "
                f"{c['NVP']:.2f}[{nv_lo:.2f},{nv_hi:.2f}] "
                f"{c['CR']:.2f}[{cr_lo:.2f},{cr_hi:.2f}]"
            )
            lines.append(
                f"    {last['V_gamma'][g]:.2f}[{v_lo:.2f},{v_hi:.2f}]/"
                f"{v_full:.2f}[{vf_lo:.2f},{vf_hi:.2f}]/"
                f"{v_adv:.2f}[{va_lo:.2f},{va_hi:.2f}] · "
                f"{pg.get('n_q', '-')}/{pg.get('n_socp_solve', '-')}/"
                f"{pg.get('n_pos', '-')}/{pg.get('n_terminal_rescue', '-')}/"
                f"{last.get('trained_draws_gamma', {}).get(g, 0)}("
                f"{last.get('trained_distinct_gamma', {}).get(g, 0)}) · "
                f"{fmt2(c['clear'])}[{fmt2(clr_lo)},{fmt2(clr_hi)}]/"
                f"{fmt2(c['time'])}[{fmt2(ttg_lo)},{fmt2(ttg_hi)}] · "
                f"{c['terminal_rescue_episodes']}/{c['terminal_required_episodes']}"
            )
        lines.append(
            f"  M={M}/γ pilot; W=plan-level Wilson, B=episode bootstrap (fixed contexts/indices)"
        )
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=5.4, va="top")
    ax.set_title("(H) final per-γ safety/performance + terminal-dependency table")

    fig.suptitle("AFE2 terminal-aware two-arm study — evolving φ_s^(n), rebuilt A, expert-free "
                 "(absorbing goal; full-H D+ only; no fallback), β fixed",
                 fontsize=14)
    provenance = (
        f"scene {pair_manifest['scene_sha256'][:10]} · checkpoint "
        f"{pair_manifest['source_checkpoint_sha256'][:10]} · source "
        f"{str(pair_manifest.get('source_git_commit'))[:10]}"
    )
    fig.text(0.5, 0.006, provenance, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=135)
    print("saved", args.out)


if __name__ == "__main__":
    main()
