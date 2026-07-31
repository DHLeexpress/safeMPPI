from types import SimpleNamespace

import json

import run_sfm_neutral_autonomous_followup as A


def test_followup_stops_without_touching_training_when_r50_goal_is_met(
    tmp_path, monkeypatch,
):
    source = "a" * 40
    monkeypatch.setattr(A.GLOBAL, "_source_gate", lambda expected=None: source)
    monkeypatch.setattr(A.GAMMA, "_gpu_inventory", lambda indices: {"devices": []})
    initial = tmp_path / "initial.json"
    initial.write_text("{}")
    gamma = tmp_path / "gamma.json"
    gamma.write_text(json.dumps({
        "status": A.GAMMA.STATUS,
        "ci_clean_four_metric_win": True,
        "objective_achieved": True,
        "source_commit": source,
        "initial_delivery": str(initial),
        "initial_delivery_sha256": A.GLOBAL.FUNNEL.sha256_file(initial),
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


def test_followup_does_not_stop_on_ci_only_without_final_gates(tmp_path, monkeypatch):
    source = "a" * 40
    monkeypatch.setattr(A.GLOBAL, "_source_gate", lambda expected=None: source)
    monkeypatch.setattr(A.GAMMA, "_gpu_inventory", lambda indices: {"devices": []})
    initial = tmp_path / "missing-initial.json"
    initial.write_text("{}")
    gamma = tmp_path / "gamma.json"
    payload = {
        "status": A.GAMMA.STATUS,
        "ci_clean_four_metric_win": True,
        "objective_achieved": False,
        "source_commit": source,
        "initial_delivery": str(initial),
        "initial_delivery_sha256": A.GLOBAL.FUNNEL.sha256_file(initial),
    }
    gamma.write_text(json.dumps(payload))

    def fake_read(path):
        if str(path) == str(gamma):
            return payload
        raise FileNotFoundError(path)

    monkeypatch.setattr(A.GLOBAL, "_read", fake_read)
    try:
        A.run(SimpleNamespace(
            gamma_delivery=str(gamma), output_dir=str(tmp_path / "output"),
            poll_seconds=1, training_gpu=1, workers=2,
        ))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("CI-only result incorrectly stopped the follow-up")
