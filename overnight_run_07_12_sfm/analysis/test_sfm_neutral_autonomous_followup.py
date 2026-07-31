from types import SimpleNamespace

import json

import run_sfm_neutral_autonomous_followup as A


def test_followup_stops_without_touching_training_when_r50_goal_is_met(tmp_path):
    gamma = tmp_path / "gamma.json"
    gamma.write_text(json.dumps({
        "status": A.GAMMA.STATUS,
        "ci_clean_four_metric_win": True,
    }))
    output = tmp_path / "output"
    result = A.run(SimpleNamespace(
        gamma_delivery=str(gamma),
        output_dir=str(output),
        poll_seconds=1,
        training_gpu=1,
        workers=2,
    ))
    assert result["action"] == "STOP_GOAL_ACHIEVED_AT_R50_OR_EARLIER"
    assert (output / "DELIVERY_COMPLETE.json").is_file()
    assert not (output / "r100_training").exists()
