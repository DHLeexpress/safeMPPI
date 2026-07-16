#!/usr/bin/env python3
"""Exact v4 2x3 training-internals layout for the challenging sanity stream."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 11.5, "axes.titlesize": 13})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "stage_results/05_sanity/runs/final_v7_ours"
OUT = ROOT / "stage_results/05_sanity/viz"
GAMMAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"]
PLASMA = plt.get_cmap("plasma")
GAMMA_COLOR = {g: PLASMA(0.08 + 0.77 * i / 6) for i, g in enumerate(GAMMAS)}


def smooth(values, width):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    width = max(1, min(int(width), len(values)))
    return np.convolve(values, np.ones(width) / width, mode="same")


def forward_fill(values):
    """Carry the last defined fresh-pool statistic across certification gaps."""
    values = np.asarray(values, dtype=float).copy()
    last = np.nan
    for i, value in enumerate(values):
        if np.isfinite(value):
            last = value
        elif np.isfinite(last):
            values[i] = last
    return values


def main() -> None:
    rows = [json.loads(line) for line in (RUN / "probe.jsonl").read_text().splitlines() if line.strip()]
    history = {int(r["iter"]): r for r in json.loads((RUN / "history.json").read_text())}
    iterations = np.array([r["iter"] for r in rows], dtype=float)

    def trace(key, default=np.nan):
        return np.array([r.get(key) if r.get(key) is not None else default for r in rows], dtype=float)

    # M=2 sanity measurements are used when the expensive SR50 probe is absent.
    sr = np.array([history.get(int(t), {}).get("SR", np.nan) for t in iterations], dtype=float)
    cr = np.array([history.get(int(t), {}).get("CR", np.nan) for t in iterations], dtype=float)
    coverage = np.array([sum(history.get(int(t), {}).get("covered", {}).values())
                         for t in iterations], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2))

    a = axes[0, 0]
    a.plot(iterations, sr, "-o", c="#009944", lw=2.0, ms=4, label="SR (M2 sanity probe)")
    a.plot(iterations, cr, "-o", c="#cc3311", lw=2.0, ms=4, label="CR")
    a2 = a.twinx(); a2.plot(iterations, coverage, "--s", c="#4477aa", lw=1.4, ms=3, label="coverage")
    a2.set_ylabel("coverage (modes)", color="#4477aa")
    a.set_ylim(-0.02, 1.02); a.set_title("(A) probe SR / CR / coverage")
    a.legend(loc="center right", fontsize=9); a.set_xlabel("iteration"); a.grid(alpha=.3)

    b = axes[0, 1]
    counts = {g: [] for g in GAMMAS}
    for row in rows:
        gamma_counts = {round(float(k), 2): v for k, v in (row.get("gamma_counts") or {}).items()}
        total = max(1.0, float(sum(gamma_counts.values())))
        for gamma in GAMMAS:
            counts[gamma].append(100.0 * float(gamma_counts.get(round(float(gamma), 2), 0.0)) / total)
    for gamma in GAMMAS:
        b.plot(iterations, smooth(counts[gamma], 3), "-", c=GAMMA_COLOR[gamma],
               lw=2.0, label=f"γ{gamma}")
    b.set_title("(B) EMERGENT γ-curriculum: share of certified windows")
    b.set_xlabel("iteration"); b.set_ylabel("% of gathered windows (3-it smooth)")
    b.legend(fontsize=8, ncol=2); b.grid(alpha=.3)

    c = axes[0, 2]
    for key, style, color, label in (
        ("n_easy", "-", "#00b300", "pool easy"),
        ("n_frontier", "-", "#d62728", "pool frontier"),
        ("batch_e", "--", "#00b300", "batch e"),
        ("batch_f", "--", "#d62728", "batch f"),
        ("batch_d", "--", "#7f7f7f", "batch demo"),
    ):
        c.plot(iterations, np.maximum(trace(key, 0), 0.5), style, c=color,
               lw=1.8 if style == "-" else 1.2, label=label)
    c.set_yscale("log"); c.set_title("(C) pools (solid) vs batch (dashed)")
    c.set_xlabel("iteration"); c.legend(fontsize=8, ncol=2); c.grid(alpha=.3)

    d = axes[1, 0]
    d.plot(iterations, trace("functional_step"), "-", c="#0072B2", lw=1.8, label="functional step")
    d2 = d.twinx(); d2.plot(iterations, trace("loss"), "-", c="#D55E00", lw=1.2, alpha=.7, label="FM loss")
    d2.set_ylabel("FM loss", color="#D55E00")
    d.set_title("(D) update magnitude / loss"); d.set_xlabel("iteration")
    d.grid(alpha=.3); d.legend(loc="upper right", fontsize=9)

    e = axes[1, 1]
    valid_rate = 100.0 * trace("vr", 0) / np.maximum(trace("att", 0), 1)
    e.plot(iterations, smooth(valid_rate, 3), "-", c="#009988", lw=1.8)
    e.set_title("(E) gather accepted-rollout rate (vr/att, 3-it smooth)")
    e.set_xlabel("iteration"); e.set_ylabel("accepted %"); e.set_ylim(0, 100); e.grid(alpha=.3)

    f = axes[1, 2]
    f.plot(iterations, forward_fill(trace("sig_e")), "-", c="#440154", lw=1.8, label=r"easy $\sigma$")
    f.plot(iterations, forward_fill(trace("sig_f")), "-", c="#35b779", lw=1.8, label=r"frontier $\sigma$")
    f.plot(iterations, trace("sigma_plane"), ":", c="k", lw=1.4, label=r"$\sigma_q$ plane")
    f.set_title(r"(F) novelty $\sigma$ by class"); f.set_xlabel("iteration")
    f.legend(fontsize=9); f.grid(alpha=.3)

    fig.suptitle("Safe Flow Expansion — training internals (challenging sanity, it0–20)", fontsize=15)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"internals_v4.{ext}", dpi=135, bbox_inches="tight")
    # Overwrite the earlier ad-hoc dashboard with this exact v4 grammar.
    fig.savefig(OUT / "stage5_sanity_dashboard.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'internals_v4.png'} and .pdf")


if __name__ == "__main__":
    main()
