import json

import pytest

import sfm_b1_alpha_steps_sweep as S


def _cell(value):
    return dict(
        n=50, SR=1-value, CR=value, V_safe=1-value, timeout=0.0,
        successful_clearance=dict(mean=.2), successful_time_to_goal=dict(mean=8.0),
    )


def _record(value):
    return dict(
        round=2,
        temperature_by_gamma={str(g): 1.0 for g in S.CE.SP.GAMMAS},
        summary=dict(pooled=_cell(value), per_gamma={str(g): _cell(value) for g in S.CE.SP.GAMMAS}),
    )


def test_factorial_is_exact_nine_margin_arms():
    arms = S.arm_grid()
    assert len(arms) == 9 and len({arm.name for arm in arms}) == 9
    assert {(arm.alpha, arm.inner_epochs) for arm in arms} == {
        (alpha, epochs) for alpha in S.ALPHAS for epochs in S.INNER_EPOCHS
    }
    assert arms[0].name == "margin_alpha0_inner001"


def test_screening_prefers_metrics_before_update_complexity():
    simple = S.SweepArm(0.0, 1)
    strong = S.SweepArm(.01, 16)
    assert S.screening_key(_record(.1), strong) < S.screening_key(_record(.2), simple)
    assert S.screening_key(_record(.1), simple) < S.screening_key(_record(.1), strong)


def test_preflight_is_bound_to_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "r0.pt"; checkpoint.write_bytes(b"checkpoint")
    payload = dict(
        status="RBF_PREFLIGHT_COMPLETE", checkpoint_sha256=S.SW.sha256_file(checkpoint),
        lengthscale_count=50, lambda_=1e-2,
        selected=dict(
            ess_solved=True, stable_conditioning=True, cap=256, ell=.2,
            ell_multiplier=.5,
        ),
        candidates=[dict(
            ess_solved=True, stable_conditioning=True, cap=512, ell=.2,
            ell_multiplier=.5,
        )],
    )
    path = tmp_path / "preflight.json"; path.write_text(json.dumps(payload))
    expected_sha = S.SW.sha256_file(path)
    loaded = S._load_preflight(path, checkpoint, expected_sha)
    assert loaded["sweep_selected"] == payload["candidates"][0]
    try:
        S._load_preflight(path, checkpoint, "bad")
        assert False
    except RuntimeError as error:
        assert "preflight SHA-256" in str(error)
    payload["checkpoint_sha256"] = "bad"; path.write_text(json.dumps(payload))
    try:
        S._load_preflight(path, checkpoint, S.SW.sha256_file(path))
        assert False
    except RuntimeError as error:
        assert "checkpoint" in str(error)


def test_output_root_is_fail_closed_and_symlink_safe(tmp_path, monkeypatch):
    root = tmp_path / "research1"
    root.mkdir()
    monkeypatch.setattr(S, "OUTPUT_ROOT", str(root))
    assert S._validate_output_root(root / "fresh") == str(root / "fresh")
    with pytest.raises(ValueError, match="fresh directory"):
        S._validate_output_root(root)
    with pytest.raises(ValueError, match="fresh directory"):
        S._validate_output_root(tmp_path / "elsewhere")
