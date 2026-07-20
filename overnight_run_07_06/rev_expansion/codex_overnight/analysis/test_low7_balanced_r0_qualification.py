from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import low7_balanced_r0_qualification as Q


def test_route_label_separates_mirrored_giant_obstacle_paths() -> None:
    env = Q.EVAL.build_scene(Q.EVAL.get_scene_profile(Q.SCENE_NAME))
    upper = np.asarray(((0.3, 0.3), (1.2, 2.8), (4.7, 4.7)))
    right = upper[:, ::-1].copy()

    assert Q._mode_label(upper, env) == int(Q.RM.MODE_U)
    assert Q._mode_label(right, env) == int(Q.RM.MODE_R)


def test_wilson_interval_contains_empirical_fraction() -> None:
    lower, upper = Q._wilson(50, 100)
    assert lower < 0.5 < upper
    assert Q._wilson(0, 0) == [0.0, 1.0]
