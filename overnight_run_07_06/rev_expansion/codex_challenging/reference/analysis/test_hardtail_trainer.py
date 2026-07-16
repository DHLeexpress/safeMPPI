"""Semantic regression for the hard-tail repair arm.

Runs the FULL 14-gate corrected-trainer harness against `grid_expand_hardtail` (aliased in as
`grid_expand_fixed`), then eight arm-specific gates (22 total):
  15. cfm-equivalence: with no x0 override and identical RNG, `_cfm_loss_x0` reproduces
      `FlowPolicy.cfm_loss` EXACTLY (the arm is a no-op when disabled).
  16. x0-override locality: overriding row r changes ONLY row r's base noise; the interpolation,
      target formula, and every other row are unchanged; sampled recovery starts lie inside their
      configured bands and `_strip_flags` classifies origin/goal contexts correctly.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent
WORK = REV.parent
RUN = WORK.parent
OVERNIGHT_ANALYSIS = WORK / "codex_overnight" / "analysis"
sys.path[:0] = [str(ROOT), str(REV), str(WORK), str(RUN), str(ROOT / "analysis"),
                str(OVERNIGHT_ANALYSIS)]

HT = importlib.import_module("grid_expand_hardtail")
sys.modules["grid_expand_fixed"] = HT          # the 14-gate harness now tests the hard-tail copy

import numpy as np  # noqa: E402
import torch  # noqa: E402


def gate_cfm_equivalence():
    import grid_hp_expt as HP
    pol = HP.GridHPFlowPolicy(width=256, depth=2, u_max=1.0, repr_dim=32, grid_hw=(16, 12))
    pol.eval()
    U = torch.randn(6, pol.T, 2).clamp(-1, 1)
    G = torch.randn(6, 3, 16, 12); L = torch.randn(6, 5); Hh = torch.randn(6, 16, 2)
    ctx = pol.ctx_from(G, L, Hh)
    torch.manual_seed(555)
    ref = pol.cfm_loss(U, ctx)
    torch.manual_seed(555)
    ours = HT._cfm_loss_x0(pol, U, ctx, x0_override=None)
    assert torch.equal(ref, ours), f"loss mismatch: {ref.item()} vs {ours.item()}"
    return "disabled arm reproduces FlowPolicy.cfm_loss bit-exactly"


def gate_x0_override_locality():
    import grid_hp_expt as HP
    pol = HP.GridHPFlowPolicy(width=256, depth=2, u_max=1.0, repr_dim=32, grid_hw=(16, 12))
    pol.eval()
    B = 5
    U = torch.randn(B, pol.T, 2).clamp(-1, 1)
    G = torch.randn(B, 3, 16, 12); L = torch.randn(B, 5); Hh = torch.randn(B, 16, 2)
    ctx = pol.ctx_from(G, L, Hh)
    x1 = (U / pol.u_max).reshape(B, pol.d)
    v = torch.full((pol.d,), 7.0)
    # reproduce the internal draws to verify only row 2 changed
    torch.manual_seed(99)
    x0 = torch.randn_like(x1); x0[2] = v
    tau = torch.rand(B).clamp(1e-4, 1.0)
    x_tau = (1 - tau)[:, None] * x0 + tau[:, None] * x1
    with torch.no_grad():
        pred = pol.forward(x_tau, tau, pol._expand_ctx(ctx, B))
    ref = ((pred - (x1 - x0)) ** 2).mean()
    torch.manual_seed(99)
    with torch.no_grad():
        ours = HT._cfm_loss_x0(pol, U, ctx, x0_override={2: v})
    assert torch.allclose(ref, ours), f"override loss mismatch: {ref.item()} vs {ours.item()}"
    # recovery-start bands + strip flags
    cfg = HT.CurConfig()
    np.random.seed(0)
    for which, band in (("origin", cfg.recovery_origin_band), ("goal", cfg.recovery_goal_band)):
        for _ in range(200):
            s = HT._sample_recovery_start(cfg, which)
            assert band[0] <= s[0] <= band[1] and band[2] <= s[1] <= band[3]
            assert band[4] <= s[2] <= band[5] and band[6] <= s[3] <= band[7]
    import grid_feats as GF
    l5o = GF.low5(np.array([0.05, -0.05, 0.1, -0.2], np.float32), np.array([5.0, 5.0]), 0.5)
    l5g = GF.low5(np.array([4.9, 5.05, 0.0, 0.2], np.float32), np.array([5.0, 5.0]), 0.5)
    l5m = GF.low5(np.array([2.5, 2.5, 0.0, 0.0], np.float32), np.array([5.0, 5.0]), 0.5)
    flags = HT._strip_flags(np.stack([l5o, l5g, l5m]), cfg)
    assert list(flags) == ["origin", "goal", ""], f"strip flags wrong: {list(flags)}"
    return "x0 override is row-local; bands and strip flags correct"


def gate_guard_bands():
    """Interior guard bands must be disjoint from the empty recovery strips."""
    import grid_feats as GF
    cfg = HT.CurConfig()
    goal = np.array([5.0, 5.0])
    states = [
        np.array([0.2, 0.4, 0.2, 0.1], np.float32),   # origin interior guard
        np.array([4.7, 4.3, 0.1, 0.1], np.float32),   # goal interior guard
        np.array([0.2, -0.02, 0.2, -0.1], np.float32),
        np.array([4.8, 5.02, 0.0, 0.1], np.float32),
    ]
    low = np.stack([GF.low5(s, goal, 0.5) for s in states])
    guard = HT._guard_flags(low, cfg)
    strip = HT._strip_flags(low, cfg)
    assert list(guard) == ["origin", "goal", "", ""], list(guard)
    assert list(strip) == ["", "", "origin", "goal"], list(strip)
    assert not np.any((guard != "") & (strip != ""))
    return "interior origin/goal guards are classified and disjoint from recovery strips"


def gate_boundary_adapter_locality():
    import grid_hp_expt as HP
    pol = HP.GridHPFlowPolicy(width=256, depth=2, u_max=1.0, repr_dim=32, grid_hw=(32, 32))
    x = torch.randn(3, pol.d); tau = torch.tensor([0.2, 0.5, 0.8])
    # ctx first two entries encode positions: interior, origin strip, goal strip.
    ctx = torch.randn(3, pol.ctx_dim)
    import grid_feats as GF
    pos = torch.tensor([[2.5, 2.5], [0.2, 0.1], [4.8, 4.9]])
    ctx[:, :2] = (torch.tensor([5.0, 5.0]) - pos) / GF.R_GOAL
    with torch.no_grad():
        base = pol(x, tau, ctx)
    pol.enable_boundary_adapter()
    with torch.no_grad():
        zero = pol(x, tau, ctx)
        pol.adapter_origin.weight.fill_(0.01); pol.adapter_goal.weight.fill_(-0.01)
        moved = pol(x, tau, ctx)
    assert torch.equal(base, zero), "zero-init adapter changed the base policy"
    assert torch.equal(base[0], moved[0]), "compact-support adapter changed an interior context"
    assert not torch.equal(base[1], moved[1]) and not torch.equal(base[2], moved[2])
    assert pol.config()["boundary_adapter"] is True
    return "adapter is zero-init and exactly inactive in the task interior"


def gate_phased_curriculum():
    cfg = HT.CurConfig(phased_curriculum=True, phase_sr_threshold=.85, phase_sr_patience=2)
    assert not HT._phased_frontier_ready([], cfg)
    assert not HT._phased_frontier_ready([{"SR": .9}, {"SR": .84}], cfg)
    history = [{"SR": .2}, {"SR": .85}, {"SR": .9}, {"SR": .1}]
    assert HT._phased_frontier_ready(history, cfg), "activation must remain true after a later regression"
    cfg.phase_uniform_active = True
    assert HT._single_class_active(cfg)
    disabled = HT.CurConfig()
    sig_disabled = HT._resume_signature(disabled, True, 0.0)
    assert "phased_curriculum" not in sig_disabled, "disabled arm broke legacy resume compatibility"
    sig_enabled = HT._resume_signature(cfg, True, 0.0)
    assert sig_enabled["phased_curriculum"] is True
    assert sig_enabled["phase_sr_threshold"] == .85 and sig_enabled["phase_sr_patience"] == 2
    return "competence switch is sustained, irreversible, single-class before activation, and signature-safe"


def gate_coverage_readiness():
    schedule = HT._parse_int_schedule(["160:2", "180:4", "200:8", "220:12", "240:14"])
    assert [HT._int_schedule_at(schedule, t, 0) for t in (159, 160, 199, 200, 239, 240)] == [0, 2, 4, 8, 12, 14]
    cfg = HT.CurConfig(min_modes_per_gamma=0, min_modes_schedule=schedule,
                       mode_hit_gate=True, min_target_hits=2, target_perp_brake=True)
    assert not HT._target_hit_ready({"target_hits": 1}, cfg)
    assert HT._target_hit_ready({"target_hits": 2}, cfg)
    sig = HT._resume_signature(cfg, True, 0.0)
    assert sig["min_modes_schedule"][-1] == [240, 14]
    assert sig["mode_hit_gate"] is True and sig["min_target_hits"] == 2
    assert sig["target_perp_brake"] is True
    disabled_sig = HT._resume_signature(HT.CurConfig(min_modes_per_gamma=0), True, 0.0)
    assert "min_modes_schedule" not in disabled_sig and "mode_hit_gate" not in disabled_sig
    import grid_scene as GS
    env = GS.make_grid(); state = np.array([.5, .9, .5, .8], np.float32)
    np.random.seed(0)
    proposal = HT._perp_braking_targeted(state, [1, 0], env, 64, "cpu").numpy()
    assert proposal[:, 0, 0].mean() > .8 and proposal[:, 0, 1].mean() < -.8
    original = HT.GR.broad_targeted
    with HT._target_proposal_override(True):
        assert HT.GR.broad_targeted is HT._perp_braking_targeted
    assert HT.GR.broad_targeted is original
    return "absolute mode schedule, exact-hit threshold, perpendicular brake, and disabled compatibility pass"


def gate_reduced_acceptance_predicates():
    """The three sanity arms differ only by the named acceptance predicate/class split."""
    path = np.linspace([0.05, 0.05], [5.0, 5.0], 11, dtype=np.float32)
    env = SimpleNamespace(goal=torch.tensor([5.0, 5.0]))

    full = HT.CurConfig(reach=0.1)
    with mock.patch.object(HT.GM2, "traj_valid2", return_value=True) as valid2:
        ok, status = HT._trajectory_acceptance(path, False, env, 0.5, full)
        assert not ok and not status["goal_reach"], "unreached rollout passed the full arm"
        ok, status = HT._trajectory_acceptance(path, True, env, 0.5, full)
        assert ok and all(status["required"].values())
        assert valid2.call_args.kwargs["check_socp"] is True

    no_socp = HT.CurConfig(reach=0.1, ablate_socp=True)
    with mock.patch.object(HT.GM2, "traj_valid2", return_value=True) as valid2, \
         mock.patch.object(HT.GM, "socp_ok", side_effect=AssertionError("-SOCP evaluated SOCP")):
        ok, status = HT._trajectory_acceptance(path, True, env, 0.5, no_socp)
        assert ok and status["socp"] is None and not status["required"]["socp"]
        assert valid2.call_args.kwargs["check_socp"] is False

    no_progress = HT.CurConfig(reach=0.1, ablate_progress=True)
    with mock.patch.object(HT.GM2, "traj_valid2", side_effect=AssertionError("-progress ran progress")), \
         mock.patch.object(HT.GM, "socp_ok", return_value=True):
        ok, status = HT._trajectory_acceptance(path, True, env, 0.5, no_progress)
        assert ok and status["goal_progress"] is None and not status["required"]["goal_progress"]
        assert status["required"]["socp"] and status["required"]["goal_reach"]

    no_curr = HT.CurConfig(ablate_curriculum=True)
    assert HT._single_class_active(no_curr)
    return "full=reach∧task∧progress∧SOCP; -SOCP/-progress remove one gate; -curriculum is single-class"


def gate_curriculum_budget_control():
    """The controlled -Curriculum arm exactly matches full-arm sample identity and volume."""
    n = 9
    fresh = {
        "grid": torch.arange(n)[:, None], "low5": torch.arange(n)[:, None],
        "hist": torch.arange(n)[:, None], "U": torch.arange(n)[:, None],
        "gamma": torch.arange(n), "prog": np.arange(n), "rid": np.arange(n),
        "socp_margin": np.arange(n) + 1.0, "cert_residual": np.arange(n),
        "widx": np.arange(n), "mode": np.asarray(["m"] * n, dtype=object),
        "proposal_target": np.asarray(["p"] * n, dtype=object),
        "rkind": np.asarray(["normal"] * n, dtype=object),
        "paths": [np.zeros((11, 2))], "rollout_gamma": np.asarray([0.5]),
    }
    capped = HT._slice_fresh_rows(fresh, 4)
    assert int(capped["U"].shape[0]) == 4
    for key in ("grid", "low5", "hist", "gamma", "prog", "rid", "socp_margin", "widx", "mode"):
        assert len(capped[key]) == 4, key
    assert capped["paths"] is fresh["paths"] and len(capped["rollout_gamma"]) == 1

    cfg = HT.CurConfig(ablate_curriculum=True)
    env = SimpleNamespace(goal=torch.tensor([5.0, 5.0]))
    with mock.patch.object(HT.GR, "fm_deploy", side_effect=AssertionError("zero budget queried policy")):
        result = HT._gather_fresh(object(), object(), env, cfg, [0.5], 0.2, 1, 12, 0,
                                  None, {0.5: set()}, "cpu", window_budget=0)
    fresh0, _q, _rr, _rc, valid, attempts, audit = result
    assert fresh0 is None and valid == 0 and attempts == 0
    assert audit["window_budget"] == 0 and audit["returned_windows"] == 0

    db = {
        "grid": torch.randn(n, 3, 4, 4), "low5": torch.randn(n, 5),
        "hist": torch.randn(n, 16, 2), "U": torch.randn(n, 10, 2),
        "gamma": torch.full((n,), .5), "prog": np.arange(n, dtype=float),
        "margin": np.arange(n, dtype=float) + 1., "rid": np.zeros(n, dtype=int),
        "widx": np.arange(n, dtype=int), "mode": ["m"] * n,
        "proposal_target": ["ordinary"] * n, "rollout_gamma": np.asarray([.5]),
        "attempted_gamma": np.asarray([.1, .2, .3, .4, .5]), "paths": [],
        "negative": {"grid": torch.randn(3, 3, 4, 4), "low5": torch.randn(3, 5),
                     "hist": torch.randn(3, 16, 2), "U": torch.randn(3, 10, 2)},
        "source_gather_audit": {"rejected_windows": 7},
        "source_reached": [1., 0., 0., 0., 0.], "source_coll": [0., 1., 0., 1., 0.],
    }
    with tempfile.TemporaryDirectory() as td:
        torch.save(db, Path(td) / "it3.pt")
        replay, rr, rc, vr, att, replay_audit = HT._load_accepted_window_replay(td, 3, n, cfg)
        assert torch.equal(replay["U"], db["U"]), "replay changed accepted rows"
        assert torch.equal(replay["negative"]["U"], db["negative"]["U"]), "replay changed rejected rows"
        assert len(replay["socp_margin"]) == n and replay_audit["returned_windows"] == n
        assert replay_audit["replay"] and replay_audit["retained_negative_windows"] == 3
        assert replay_audit["rejected_windows"] == 7 and vr == 1 and att == 5
        assert rr == db["source_reached"] and rc == db["source_coll"]
        try:
            HT._load_accepted_window_replay(td, 3, n - 1, cfg)
        except RuntimeError as exc:
            assert "count mismatch" in str(exc)
        else:
            raise AssertionError("replay accepted a sample-count mismatch")
    return "exact accepted rows replay; count mismatch fails closed; zero budget performs no gather"


def main():
    import test_corrected_trainer as TCT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "analysis" / "test_hardtail_trainer.json"))
    args = ap.parse_args()

    sys.argv = ["test_corrected_trainer", "--json", args.json + ".base14"]
    rc = 0
    try:
        TCT.main()
    except SystemExit as e:
        rc = int(e.code or 0)
    base = json.load(open(args.json + ".base14"))
    results = {"base14_pass": base.get("pass_count"), "base14_fail": base.get("fail_count")}
    extra_pass = 0
    extras = (("hardtail_cfm_equivalence", gate_cfm_equivalence),
              ("hardtail_x0_override_locality", gate_x0_override_locality),
              ("hardtail_guard_bands", gate_guard_bands),
              ("hardtail_boundary_adapter_locality", gate_boundary_adapter_locality),
              ("hardtail_phased_curriculum", gate_phased_curriculum),
              ("hardtail_coverage_readiness", gate_coverage_readiness),
              ("hardtail_reduced_acceptance_predicates", gate_reduced_acceptance_predicates),
              ("hardtail_curriculum_budget_control", gate_curriculum_budget_control))
    for name, fn in extras:
        try:
            msg = fn()
            print(f"[PASS] {name}: {msg}", flush=True)
            results[name] = "PASS"; extra_pass += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}", flush=True)
            results[name] = f"FAIL: {e}"
    results["production"] = "grid_expand_hardtail.py"
    results["total_pass"] = (base.get("pass_count", 0) or 0) + extra_pass
    results["total_fail"] = (base.get("fail_count", 1) or 0) + (len(extras) - extra_pass)
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    sys.exit(0 if (results["total_fail"] == 0 and rc == 0) else 1)


if __name__ == "__main__":
    main()
