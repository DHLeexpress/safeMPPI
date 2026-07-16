"""Shared visual semantics for every challenging-run figure.

Gamma and uncertainty deliberately use different colormaps:

* gamma: seven discrete samples of truncated plasma, matching Image #1;
* sigma: continuous viridis, reserved for GP uncertainty/curriculum plots.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps, colors


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
GAMMA_CMAP_NAME = "plasma_trunc"
SIGMA_CMAP_NAME = "viridis"

# These are the exact endpoints used by rev_expansion/plot_uni_trajs.py.
_GAMMA_RGBA = colormaps["plasma"](np.linspace(0.02, 0.90, len(GAMMAS)))
GAMMA_CMAP = colors.ListedColormap(_GAMMA_RGBA, name=GAMMA_CMAP_NAME)
GAMMA_COLORS = {gamma: tuple(_GAMMA_RGBA[i]) for i, gamma in enumerate(GAMMAS)}


def gamma_boundaries(gammas=GAMMAS) -> np.ndarray:
    """Return bin edges centered on the discrete gamma values."""
    values = np.asarray(gammas, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two gamma values are required")
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - (edges[1] - values[0])
    edges[-1] = values[-1] + (values[-1] - edges[-2])
    return edges


GAMMA_NORM = colors.BoundaryNorm(gamma_boundaries(), GAMMA_CMAP.N)

