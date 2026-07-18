from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _video_module():
    path = _ROOT / "video_afe2.py"
    spec = importlib.util.spec_from_file_location("afe2_video_schedule_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_long_video_schedule_is_dense_then_decimated() -> None:
    module = _video_module()
    rounds = module.select_video_rounds(
        range(1, 101), dense_until=10, every_after=10
    )
    assert rounds == [*range(1, 11), *range(20, 101, 10)]
    assert len(rounds) == 19


def test_short_video_schedule_keeps_every_round() -> None:
    module = _video_module()
    assert module.select_video_rounds(
        range(1, 6), dense_until=10, every_after=10
    ) == [1, 2, 3, 4, 5]


def test_compact_sweep_viz_inventory_matches_the_19_frame_schedule() -> None:
    module = _video_module()
    rounds = module.expected_viz_rounds({
        "rounds": 100,
        "artifact_profile": "sweep_compact",
    })
    assert rounds == [*range(1, 11), *range(20, 101, 10)]


def test_video_schedule_rejects_half_specified_arguments() -> None:
    module = _video_module()
    with pytest.raises(ValueError, match="supplied together"):
        module.select_video_rounds(range(1, 3), dense_until=10)
