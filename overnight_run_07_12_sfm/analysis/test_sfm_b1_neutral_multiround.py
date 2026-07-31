from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as M
import sfm_b1_offline_store as OS
import sfm_protocol as SP


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_grid = torch.nn.Linear(1, 1, bias=False)
        self.enc_grid.requires_grad_(False)
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def ctx_from(self, grid, low, hist):
        return low[:, :1]

    def cfm_loss(self, controls, context, weights=None):
        target = controls.reshape(len(controls), -1).mean(dim=1)
        prediction = self.scale * context[:, 0]
        per = (prediction - target).square()
        return per.mean() if weights is None else (per * weights).mean()

    def phi_s_from_x0(self, controls, context, x0, s):
        return x0


def _records(population="D0"):
    holder = M._NeutralHolder(1)
    rows = []
    for index, gamma in enumerate((0.1, 1.0)):
        holder.contexts.append({
            "context_id": index,
            "round": 1,
            "scenario_id": 260000 + index,
            "gamma": gamma,
            "step": index,
            "hp10": np.zeros((10, 16, 12), np.float32),
            "low5": np.asarray([1 + index, 0, 0, 0, gamma], np.float32),
            "hist": np.zeros((16, 2), np.float32),
        })
        row = {
            "query_id": index,
            "window_id": index,
            "context_id": index,
            "controls": np.full((10, 2), 0.2 + index, np.float32),
            "x0": np.zeros(20, np.float32),
            "y": int(population == "Dplus"),
        }
        holder.windows.append(row)
        rows.append((holder, row))
    return rows


def test_study_config_pins_two_scenarios_and_protocol():
    assert M.StudyConfig(name="x").validate().scenarios_per_round == 2
    with pytest.raises(ValueError, match="two scenarios"):
        M.StudyConfig(name="x", scenarios_per_round=8).validate()
    with pytest.raises(ValueError, match="K/B/H/T"):
        M.StudyConfig(name="x", K=64).validate()


def test_population_update_uses_every_row_once_per_inner_pass():
    records = _records()
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam([policy.scale], lr=3.0e-5)
    before = float(policy.scale.detach())
    report = M._population_update(
        policy,
        optimizer,
        records,
        population="D0",
        inner_steps=4,
        batch=1,
        device="cpu",
        seed=17,
    )
    assert report["optimizer_steps"] == 4
    assert report["sample_exposures"] == 8
    assert report["exact_once_per_inner_step"]
    assert len(report["exposure_identity_sha256"]) == 4
    assert float(policy.scale.detach()) != before


def test_restore_optimizer_preserves_global_adam_step(tmp_path):
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam([policy.scale], lr=3.0e-5)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        policy.scale.square().backward()
        optimizer.step()
    path = tmp_path / "optimizer.pt"
    torch.save({"round": 2, "optimizer": optimizer.state_dict()}, path)

    resumed_policy = _TinyPolicy()
    resumed = torch.optim.Adam([resumed_policy.scale], lr=3.0e-5)
    report = M._restore_optimizer(
        resumed,
        str(path),
        [resumed_policy.scale],
        resume_round=2,
        inner_steps=1,
    )
    assert report["expected_adam_step"] == 4
    assert int(resumed.state[resumed_policy.scale]["step"].item()) == 4

    with pytest.raises(RuntimeError, match="round mismatch"):
        M._restore_optimizer(
            torch.optim.Adam([resumed_policy.scale], lr=3.0e-5),
            str(path),
            [resumed_policy.scale],
            resume_round=3,
            inner_steps=1,
        )


def _positive_result():
    return {
        "resolved": True,
        "y": 1,
        "taskspace": True,
        "collision_free": True,
        "certificate": True,
        "full_h": True,
        "terminal_step": 10,
        "train_eligible": True,
        "diagnostics": {},
    }


def _previous_shard(counts):
    shard = OS.ExecutedRoundShard(1)
    for gamma, count in zip(SP.GAMMAS, counts):
        for index in range(int(count)):
            context_id = shard.add_context(
                scenario_id=100000 + index,
                gamma=gamma,
                step=index,
                state=np.zeros(4, np.float32),
                hp10=np.zeros((10, 16, 12), np.float32),
                low5=np.zeros(5, np.float32),
                hist=np.zeros((16, 2), np.float32),
                ped_xy=np.zeros((1, 2), np.float32),
                ped_vel=np.zeros((1, 2), np.float32),
            )
            shard.add_executed_window(
                context_id,
                np.zeros((10, 2), np.float32),
                np.full(20, index + float(gamma), np.float32),
                _positive_result(),
                execution_source="test",
                nvp_context=False,
            )
    return shard


def test_dynamic_gp_uses_equal_supported_quota_without_backfill():
    previous = _previous_shard((3, 5, 5, 5, 5, 5, 5))
    gp, identities, selection = RA._gamma_balanced_gp(
        _TinyPolicy(),
        previous,
        gammas=tuple(map(float, SP.GAMMAS)),
        round_i=2,
        ell=RA.DEFAULT_ELL,
        cap=512,
        lam=1.0e-2,
        phi_s=0.9,
        device="cpu",
        seed=5,
    )
    assert selection["quota"] == 3
    assert selection["effective_cap"] == 22
    assert selection["per_gamma"]["0.1"] == 3
    assert sum(selection["per_gamma"].values()) == 22
    assert len(identities) == len(set(identities)) == 22
    assert gp.diagnostics()["n"] == 22


def test_dynamic_gp_is_empty_on_round_one():
    gp, identities, selection = RA._gamma_balanced_gp(
        _TinyPolicy(),
        None,
        gammas=tuple(map(float, SP.GAMMAS)),
        round_i=1,
        ell=RA.DEFAULT_ELL,
        cap=512,
        lam=1.0e-2,
        phi_s=0.9,
        device="cpu",
        seed=5,
    )
    assert identities == []
    assert selection["effective_cap"] == 0
    assert gp.diagnostics()["n"] == 0


def test_dynamic_gp_fails_if_any_declared_gamma_has_no_support():
    previous = _previous_shard((0, 2, 2, 2, 2, 2, 2))
    with pytest.raises(RuntimeError, match="cannot support all declared"):
        RA._gamma_balanced_gp(
            _TinyPolicy(),
            previous,
            gammas=tuple(map(float, SP.GAMMAS)),
            round_i=2,
            ell=RA.DEFAULT_ELL,
            cap=512,
            lam=1.0e-2,
            phi_s=0.9,
            device="cpu",
            seed=5,
        )


def test_eval_round_parser_is_staged_and_requires_final():
    assert M._parse_eval_rounds(
        "0,1,2,5,10,20,30,40,50", 50,
    ) == (0, 1, 2, 5, 10, 20, 30, 40, 50)
    with pytest.raises(ValueError, match="final round"):
        M._parse_eval_rounds("0,1,2", 50)


def _probe_row(
    anchor_id,
    *,
    population,
    nvp,
    progress,
    admissible,
    rmse,
):
    return {
        "anchor_id": anchor_id,
        "gamma": 0.5,
        "trigger": "finite_B_NVP",
        "population": population,
        "B_orig_NVP": nvp,
        "target_full_rmse": rmse,
        "selected_one_step_progress": progress,
        "all_K_admissible": admissible,
        "B_orig_admissible": int(not nvp),
        "target_latent_admissible": bool(not nvp),
        "_feature": np.asarray([1.0, 0.0], np.float32),
        "_windows": np.zeros((16, 10, 2), np.float32),
    }


def test_probe_classifies_repair_stall_imitation_and_regression():
    before = [
        _probe_row(
            0, population="D0", nvp=True, progress=None,
            admissible=0, rmse=1.0,
        ),
        _probe_row(
            1, population="D0", nvp=True, progress=None,
            admissible=0, rmse=1.0,
        ),
        _probe_row(
            2, population="D0", nvp=True, progress=None,
            admissible=0, rmse=1.0,
        ),
        _probe_row(
            3, population="Dplus", nvp=False, progress=0.1,
            admissible=3, rmse=0.0,
        ),
    ]
    after = [
        _probe_row(
            0, population="D0", nvp=False, progress=0.2,
            admissible=2, rmse=0.4,
        ),
        _probe_row(
            1, population="D0", nvp=False, progress=-0.1,
            admissible=1, rmse=0.4,
        ),
        _probe_row(
            2, population="D0", nvp=True, progress=None,
            admissible=0, rmse=0.4,
        ),
        _probe_row(
            3, population="Dplus", nvp=True, progress=None,
            admissible=0, rmse=0.2,
        ),
    ]
    summary = M._probe_comparison(before, after)["pooled"]
    assert summary["D0_actual_repairs"] == 1
    assert summary["D0_safe_stalls"] == 1
    assert summary["D0_imitation_only"] == 1
    assert summary["Dplus_regressed"] == 1


def test_neutral_record_accepts_explicit_round():
    result = {
        "resolved": True,
        "y": 0,
        "full_h": True,
        "terminal_step": 10,
    }
    chosen = {
        "controls": np.zeros((10, 2), np.float32),
        "x0": np.zeros(20, np.float32),
        "result": result,
        "candidate_id": 16,
        "sigma": 0.1,
        "hp_margin": 0.2,
    }
    prepared = {
        "state": np.zeros(4, np.float32),
        "hp10": torch.zeros(10, 16, 12),
        "low": torch.zeros(5),
        "hist": torch.zeros(16, 2),
        "ped_xy": np.zeros((1, 2), np.float32),
        "ped_vel": np.zeros((1, 2), np.float32),
    }
    record = RA._neutral_record(
        0,
        SimpleNamespace(scenario_id=1, gamma=0.5),
        prepared,
        chosen,
        step=3,
        round_i=2,
        selector="margin",
        repair_trigger="finite_B_NVP",
    )
    assert record["round"] == 2
    assert record["verifier_y"] == 0
    assert not record["gp_eligible"]
