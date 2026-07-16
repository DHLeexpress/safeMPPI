#!/usr/bin/env python3
"""Paper-ready internals overlays and final-run probe traces for P2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")


def read_probe(run: Path):
    path = run / "probe.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def series(rows, key):
    return np.asarray([np.nan if r.get(key) is None else r.get(key, np.nan) for r in rows], float)


def parse_arm(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("arm must be LABEL=RUN_DIR")
    label, path = text.split("=", 1)
    return label, Path(path)


def plot_overlay(arms, out: Path):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panels = (
        ("sr50", "Instantaneous SR (M=50)", (0, 1.02)),
        ("cr50", "Instantaneous CR (M=50)", (-.02, .25)),
        ("cov50", "Instantaneous staircase coverage (M=50)", None),
        ("near0_e", "Easy windows starting within 1 m", (0, 1.02)),
        ("loss", "CFM loss", None),
        ("rid_dom", "Largest rollout share in update", (0, 1.02)),
    )
    summary = {}
    for ai, (label, run) in enumerate(arms):
        rows = read_probe(run)
        t = series(rows, "iter")
        color = COLORS[ai % len(COLORS)]
        for ax, (key, title, ylim) in zip(axs.flat, panels):
            y = series(rows, key)
            ok = np.isfinite(y)
            if ok.any():
                ax.plot(t[ok], y[ok], color=color, lw=1.8, alpha=.9, label=label)
            ax.set_title(title); ax.set_xlabel("absolute expansion iteration"); ax.grid(alpha=.25)
            if ylim is not None:
                ax.set_ylim(*ylim)
        summary[label] = {
            "run": str(run), "n_probe_rows": len(rows),
            "last": {k: (None if not np.isfinite(series(rows, k)[-1]) else float(series(rows, k)[-1]))
                     for k in ("sr50", "cr50", "cov50", "near0_e", "loss", "rid_dom")},
            "best_sr50": float(np.nanmax(series(rows, "sr50"))),
            "max_cov50": float(np.nanmax(series(rows, "cov50"))),
        }
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.015),
               ncol=max(1, len(arms)), frameon=False)
    fig.suptitle("Safe Flow Expansion — fixed AND-quantile recipe diagnostics", y=1.065, fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    (out.parent / "p2_diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_final(label, run: Path, out: Path):
    rows = read_probe(run); t = series(rows, "iter")
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(t, series(rows, "sr50"), label="SR50", color="#009E73")
    ax.plot(t, series(rows, "cr50"), label="CR50", color="#D55E00")
    ax.set_ylim(-.02, 1.02); ax.legend(); ax.set_title("Faithful performance")
    ax = axs[0, 1]
    ax.plot(t, series(rows, "cov50"), color="#0072B2"); ax.set_title("Empirical coverage (M=50)")
    ax = axs[0, 2]
    ax.plot(t, series(rows, "n_easy"), label="easy pool", color="#009E73")
    ax.plot(t, series(rows, "n_frontier"), label="frontier pool", color="#D55E00")
    ax.legend(); ax.set_yscale("symlog", linthresh=1); ax.set_title("Gathered class pools")
    ax = axs[1, 0]
    ax.plot(t, series(rows, "near0_e"), label="near-origin easy", color="#CC79A7")
    ax.plot(t, series(rows, "w2_e"), label="first-two-window easy", color="#E69F00")
    ax.set_ylim(-.02, 1.02); ax.legend(); ax.set_title("Ill-conditioned rollout watch")
    ax = axs[1, 1]
    ax.plot(t, series(rows, "sig_e"), label="easy σ", color="#009E73")
    ax.plot(t, series(rows, "sig_f"), label="frontier σ", color="#D55E00")
    ax.legend(); ax.set_title("Novelty by class")
    ax = axs[1, 2]
    ax.plot(t, series(rows, "batch_e"), label="easy", color="#009E73")
    ax.plot(t, series(rows, "batch_f"), label="frontier", color="#D55E00")
    ax.plot(t, series(rows, "batch_d"), label="demo", color="#777777")
    ax.legend(); ax.set_title("Actual update batch")
    for ax in axs.flat:
        ax.set_xlabel("absolute expansion iteration"); ax.grid(alpha=.25)
    fig.suptitle(f"Final P2 run probes — {label}", fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", type=parse_arm, required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--final-label", default=None)
    ap.add_argument("--final-run", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    plot_overlay(args.arm, args.outdir / "p2_internals_overlay.png")
    if args.final_run is not None:
        plot_final(args.final_label or args.final_run.name, args.final_run,
                   args.outdir / "p2_final_probe_traces.png")
    print(f"saved diagnostics to {args.outdir}")


if __name__ == "__main__":
    main()
