#!/usr/bin/env python3
"""Exact v4 phase-plane styling, repointed to the challenging sanity panel.

Marker encodes method; the truncated plasma map encodes safety level gamma.
Viridis is reserved exclusively for uncertainty sigma in the curriculum views.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "stage_results/05_sanity/viz"
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
PLASMA = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(np.linspace(0.04, 0.88, 256)))
NORM = matplotlib.colors.Normalize(vmin=0.1, vmax=1.0)

SERIES = [
    ("Expert", ROOT / "stage_results/06_baselines/results/expert_m6", "o", 95, {}),
    ("Our approach", ROOT / "stage_results/05_sanity/data/eval_final_v7_ours", "*", 240,
     {"edgecolors": "k", "linewidths": 0.9}),
    ("Pretrained", ROOT / "stage_results/04_canonical/data/pretrained_m6", "s", 70,
     {"alpha": 0.55}),
    (r"CFM-MPPI$^{*}$", ROOT / "stage_results/06_baselines/results/kazuki_low_guidance_m6", "v", 95, {}),
    ("NO safety validity check", ROOT / "stage_results/05_sanity/data/eval_final_v7_no_socp", "X", 85, {}),
    ("NO progress check", ROOT / "stage_results/05_sanity/data/eval_final_v7_no_progress", "P", 85, {}),
    ("NO curriculum", ROOT / "stage_results/05_sanity/data/eval_final_v7_no_curriculum", "D", 70, {}),
]


def rows(directory: Path) -> dict[float, dict]:
    out = {}
    for path in glob.glob(str(directory / "row_g*.json")):
        row = json.loads(Path(path).read_text())
        out[round(float(row["gamma"]), 2)] = row
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (reliability, quality) = plt.subplots(1, 2, figsize=(16.2, 5.4))
    for zorder, (name, directory, marker, size, kwargs) in enumerate(SERIES, start=3):
        data = rows(directory)
        for gamma in GAMMAS:
            if gamma not in data:
                continue
            row = data[gamma]
            color = [PLASMA(NORM(gamma))]
            reliability.scatter(row["SR"] * 100, row["CR"] * 100, c=color, marker=marker,
                                s=size, zorder=zorder, **kwargs)
            clearance = row.get("clearance_mean")
            time_s = row.get("time_mean_s")
            if clearance is not None and time_s is not None and np.isfinite(clearance) and np.isfinite(time_s):
                quality.scatter(time_s, clearance, c=color, marker=marker, s=size,
                                zorder=zorder, **kwargs)

    reliability.set_xlabel("success rate SR [%]")
    reliability.set_ylabel("collision rate CR [%]")
    reliability.set_xlim(-5, 105); reliability.set_ylim(-3, 105)
    reliability.grid(alpha=0.3)
    quality.set_xlabel("time to goal [s]")
    quality.set_ylabel("min clearance (successes) [m]")
    quality.grid(alpha=0.3)

    def label(name: str) -> str:
        return r"$\mathbf{Our\ approach}$" if name == "Our approach" else name

    handles = [Line2D([], [], color="#666666", marker=marker, ls="",
                      ms=11 if marker == "*" else 8, label=label(name))
               for name, _, marker, _, _ in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, 1.025), fontsize=10.5)
    scalar = plt.cm.ScalarMappable(cmap=PLASMA, norm=NORM); scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=[reliability, quality], location="right",
                        fraction=0.025, pad=0.015, ticks=GAMMAS)
    cbar.set_label(r"safety level $\gamma$", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"scatter_v4.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'scatter_v4.png'} and .pdf")


if __name__ == "__main__":
    main()
