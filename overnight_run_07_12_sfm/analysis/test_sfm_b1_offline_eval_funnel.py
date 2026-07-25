from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_sfm_b1_offline_eval_funnel as FUNNEL  # noqa: E402
import sfm_b1_offline_eval as EVAL  # noqa: E402


def _row(name, round_index, *, sr, cr, validity):
    return {
        "arm": name,
        "round": round_index,
        "checkpoint": f"/tmp/{name}_{round_index}.pt",
        "checkpoint_sha256": f"{name}-{round_index}",
        "SR": sr,
        "CR": cr,
        "timeout": 1.0 - sr - cr,
        "Validity": validity,
        "clearance": 0.1,
        "time_to_goal": 10.0,
    }


def test_selection_rejects_zero_success_low_collision_collapse():
    r0 = _row("pretrained", 0, sr=0.6, cr=0.4, validity=0.5)
    collapsed = _row("a", 2, sr=0.0, cr=0.0, validity=0.9)
    viable = _row("b", 1, sr=0.7, cr=0.2, validity=0.7)
    selected, contract = FUNNEL.choose_candidates(
        [collapsed, viable], r0, top_k=1
    )
    assert selected == [viable]
    assert contract["r0_SR_gate"] == 0.6
    assert not contract["fallback_used"]


def test_selection_fallback_is_highest_success():
    r0 = _row("pretrained", 0, sr=0.8, cr=0.2, validity=0.5)
    first = _row("a", 1, sr=0.4, cr=0.1, validity=0.9)
    second = _row("b", 2, sr=0.7, cr=0.3, validity=0.6)
    selected, contract = FUNNEL.choose_candidates(
        [first, second], r0, top_k=1
    )
    assert selected == [second]
    assert contract["fallback_used"]


def test_evaluator_artifacts_follow_requested_m(tmp_path):
    previous = EVAL.M_PER_GAMMA
    try:
        EVAL.M_PER_GAMMA = 10
        assert EVAL._artifact_prefix() == "raw_m10_offline"
        assert EVAL._status() == "SFM_B1_OFFLINE_RAW_M10_COMPLETE"
    finally:
        EVAL.M_PER_GAMMA = previous


def test_evaluator_rejects_nonpositive_m_before_checkpoint_loading():
    args = SimpleNamespace(m_per_gamma=0)
    try:
        EVAL.run(args)
    except ValueError as error:
        assert "--m-per-gamma must be positive" in str(error)
    else:
        raise AssertionError("nonpositive M must fail")
