import json

import sfm_b1_alpha_steps_sweep as S


def _cell(value):
    return dict(
        n=50, SR=1-value, CR=value, V_safe=1-value, timeout=0.0,
        successful_clearance=dict(mean=.2), successful_time_to_goal=dict(mean=8.0),
    )


def _record(value):
    return dict(
        round=2, temperature=1.0,
        summary=dict(pooled=_cell(value), per_gamma={str(g): _cell(value) for g in S.CE.SP.GAMMAS}),
    )


def test_factorial_is_exact_nine_margin_arms():
    arms = S.arm_grid()
    assert len(arms) == 9 and len({arm.name for arm in arms}) == 9
    assert {(arm.alpha, arm.optimizer_steps) for arm in arms} == {
        (alpha, steps) for alpha in S.ALPHAS for steps in S.OPTIMIZER_STEPS
    }
    assert arms[0].name == "margin_alpha0_steps001"


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
        selected=dict(ess_solved=True, stable_conditioning=True, cap=256, ell=.2),
    )
    path = tmp_path / "preflight.json"; path.write_text(json.dumps(payload))
    expected_sha = S.SW.sha256_file(path)
    assert S._load_preflight(path, checkpoint, expected_sha) == payload
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
