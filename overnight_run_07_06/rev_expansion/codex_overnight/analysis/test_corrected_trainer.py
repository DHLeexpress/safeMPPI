#!/usr/bin/env python3
"""Independent semantic regression gates for the corrected P2 trainer.

The tests deliberately exercise public-ish helper boundaries with synthetic data and
``unittest.mock``.  They do not launch training and never write production artifacts.
The only test that reads a historical artifact uses it solely as a bank of realistic
state/control windows for an independent verifier-margin calculation.

Run (physical GPU 2 is optional, and is used only to compare RNG state bytes)::

    OMP_NUM_THREADS=16 LD_LIBRARY_PATH=/home/dohyun/miniforge3/lib \
      CUDA_VISIBLE_DEVICES=2 python analysis/test_corrected_trainer.py
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REV = ROOT.parent
WORK = REV.parent
sys.path[:0] = [str(ROOT), str(REV), str(WORK)]

import grid_expand2 as GX2  # noqa: E402
import grid_expand_fixed as FIX  # noqa: E402
import grid_metrics2 as GM2  # noqa: E402
import grid_rollout as GR  # noqa: E402
import grid_scene as GS  # noqa: E402
import verifier_polytope as VP  # noqa: E402


GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


class SkipGate(RuntimeError):
    """A gate cannot run in this environment, rather than passing silently."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def np_rng_equal(a, b) -> bool:
    return (
        a[0] == b[0]
        and np.array_equal(a[1], b[1])
        and a[2:] == b[2:]
    )


def tensor_tree_equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(tensor_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(tensor_tree_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def rng_context(device: str):
    """Accept either the original or the renamed corrected RNG context API."""
    fn = getattr(FIX, "_preserve_training_rng", None)
    if fn is None:
        fn = getattr(FIX, "_preserve_torch_rng", None)
    if fn is None:
        raise AssertionError("trainer exposes no RNG-preservation context")
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return fn()
    return fn(device)


def tiny_cfg(**overrides):
    values = dict(
        min_rollouts=1,
        gather_attempt_cap=1,
        qbuf_cap=500,
        gp_buf=384,
        T=20,
        N=4,
        s=0.9,
        temp=1.0,
        churn=0.0,
        safe_filter=True,
        nfe_explore=2,
        reach=0.1,
        traj_prog_min=0.0,
        valid_prog_floor=0.15,
        active_quantile=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_rec(value: float):
    return (
        np.full((3, 16, 12), value, np.float32),
        np.full(5, value, np.float32),
        np.full((16, 2), value, np.float32),
        np.full((10, 2), value, np.float32),
    )


def unpack_gather(result):
    """Normalize the original six-field and audited seven-field gather return."""
    require(isinstance(result, tuple) and len(result) in (6, 7),
            f"unexpected gather return shape: {type(result)} / {getattr(result, '__len__', lambda: '?')()}")
    fresh, qbuf, reached, coll, valid, attempts = result[:6]
    audit = result[6] if len(result) == 7 else {}
    return fresh, qbuf, reached, coll, valid, attempts, audit


def test_rng_isolation() -> dict:
    """Evaluation contexts must be observational for NumPy, CPU Torch, and visible CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(3107)
    torch.manual_seed(3107)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3107)
        # Force lazy initialization before taking the baseline states.
        torch.rand(1, device="cuda")
    np_before = np.random.get_state()
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    with rng_context(device):
        np.random.random(19)
        torch.rand(19)
        if torch.cuda.is_available():
            torch.rand(19, device="cuda")
    np_after = np.random.get_state()
    cpu_after = torch.random.get_rng_state()
    cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    require(np_rng_equal(np_before, np_after), "RNG context changed NumPy state")
    require(torch.equal(cpu_before, cpu_after), "RNG context changed Torch CPU state")
    require(len(cuda_before) == len(cuda_after), "visible CUDA device set changed inside context")
    require(all(torch.equal(a, b) for a, b in zip(cuda_before, cuda_after)),
            "RNG context changed a visible CUDA state")
    return dict(device=device, visible_cuda_states=len(cuda_before))


def _rng_snapshot():
    return (
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def _require_rng_snapshot_equal(before, after, where: str) -> None:
    require(np_rng_equal(before[0], after[0]), f"{where} changed NumPy RNG")
    require(torch.equal(before[1], after[1]), f"{where} changed Torch CPU RNG")
    require(len(before[2]) == len(after[2]) and
            all(torch.equal(a, b) for a, b in zip(before[2], after[2])),
            f"{where} changed visible CUDA RNG")


def test_probe_rng_integration() -> dict:
    """Every measurement/probe entry point must actually use the preservation context."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = SimpleNamespace(goal=torch.tensor([5.0, 5.0]))
    cfg = SimpleNamespace(gammas=(0.1, 0.5), M_measure=2, T=20, reach=0.1, nfe_explore=2)

    def consume_rng():
        np.random.random(5)
        torch.rand(5)
        if torch.cuda.is_available():
            torch.rand(5, device="cuda")

    def fake_eval(*_a, **kw):
        consume_rng()
        gs = list(kw.get("gammas", [0.5]))
        rows = {g: {"SR": 1.0, "CR": 0.0} for g in gs}
        paths = {g: [np.array([[0.0, 0.0], [5.0, 5.0]], dtype=float)] for g in gs}
        return rows, {"SR": 1.0, "CR": 0.0, "mean_goal_dist": 0.0}, paths

    def fake_deploy(*_a, **_kw):
        consume_rng()
        return {"path": np.array([[0.0, 0.0], [2.0, 2.0]], dtype=float)}

    checks = []
    with mock.patch.object(FIX.SR, "eval_policy", fake_eval):
        before = _rng_snapshot(); FIX._measure(object(), env, cfg, device); after = _rng_snapshot()
        _require_rng_snapshot_equal(before, after, "_measure"); checks.append("measure")
        before = _rng_snapshot(); FIX._cov_probe(object(), env, cfg, device, M=2, g=0.5); after = _rng_snapshot()
        _require_rng_snapshot_equal(before, after, "_cov_probe"); checks.append("cov_probe")
    with mock.patch.object(FIX.GR, "fm_deploy", fake_deploy):
        before = _rng_snapshot(); FIX._escape_probe(object(), env, cfg, device, M=2, T=2); after = _rng_snapshot()
        _require_rng_snapshot_equal(before, after, "_escape_probe"); checks.append("escape_probe")
    return dict(device=device, isolated=checks)


def _margin_source() -> tuple[dict, Path]:
    candidates = [
        ROOT / "results/p2/finalunit_q50_k14_s15_from_it18/viz_db/it100.pt",
        *sorted((ROOT / "results/p2").glob("**/viz_db/it*.pt")),
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False), path
    raise SkipGate("no local viz_db exists to supply realistic verifier windows")


def test_verifier_face_margin() -> dict:
    """The ranking margin must equal the literal fitted real-face SOCP margin."""
    db, source = _margin_source()
    env = GS.make_grid()
    obs = env.obstacles.detach().cpu().numpy()
    n = min(28, len(db["U"]))
    idx = np.linspace(0, len(db["U"]) - 1, n, dtype=int)
    reported, independent = [], []
    for i in idx:
        low = np.asarray(db["low5"][i], dtype=float)
        U = np.asarray(db["U"][i], dtype=float)
        gamma = float(db["gamma"][i])
        state = GX2.state_from_low5(low)
        seg = GR.window_positions(state, U, env.dt)
        path = np.vstack([np.asarray(state, float)[:2], seg])
        ok, faces, _raw, r_eff = VP.certify_window(
            path, obs, float(env.r_robot), gamma, R=2.5, n_theta=180
        )
        if not ok:
            continue
        real = [float(f.m) for f in faces if f.kind == "real" and f.feasible]
        face_margin = min(real) if real else float(r_eff)
        got = float(GM2.window_socp_margin(state, U, env, gamma, R=2.5, n_theta=180))
        require(np.isfinite(got), f"non-finite feasible margin at source index {i}: {got}")
        require(np.isclose(got, face_margin, rtol=1e-6, atol=1e-7),
                f"frontier margin is not min(real Face.m)/R_eff: got {got}, expected {face_margin}")
        reported.append(got)
        independent.append(face_margin)
    values = np.asarray(reported, dtype=float)
    require(len(values) >= 12, f"only {len(values)} feasible source windows were available")
    require(np.isfinite(values).all(), "feasible verifier margins contain NaN/inf")
    unique = len(np.unique(np.round(values, 8)))
    require(unique >= max(8, int(0.8 * len(values))),
            f"verifier face margin is degenerate: {unique}/{len(values)} unique at 1e-8")
    require(float(np.ptp(values)) > 1e-3, f"verifier margin range is too small: {np.ptp(values)}")
    return dict(source=str(source.relative_to(ROOT)), feasible=len(values), unique_1e8=unique,
                min=float(values.min()), max=float(values.max()),
                max_abs_error=float(np.max(np.abs(values - np.asarray(independent)))))


def test_per_gamma_planes() -> dict:
    """Scale-separated gammas must each receive their own AND planes and both classes."""
    sigma, margin, prog, gamma = [], [], [], []
    expected = []
    for gi, g in enumerate(GAMMAS):
        base = 1000.0 * gi
        for j in range(8):
            sigma.append(base + j)
            margin.append(base + (7 - j))
            prog.append(base + j)
            gamma.append(g)
            expected.append(j >= 4)
    cfg = SimpleNamespace(active_quantile=0.5)
    actual, planes = FIX._front_mask(
        np.asarray(sigma), np.asarray(margin), np.asarray(prog), np.arange(len(sigma)), cfg,
        gamma=np.asarray(gamma), return_planes=True,
    )
    expected = np.asarray(expected, dtype=bool)
    require(np.array_equal(actual, expected),
            "frontier labels do not match per-gamma q=.5 high-sigma/low-margin/high-progress AND")
    counts = {}
    for g in GAMMAS:
        z = np.isclose(gamma, g)
        counts[str(g)] = int(actual[z].sum())
        require(actual[z].any() and (~actual[z]).any(), f"gamma {g} does not contain both classes")
        p = planes.get(str(float(g)))
        require(p is not None and p["n"] == 8, f"missing/wrong plane metadata for gamma {g}")
    return dict(frontier_per_gamma=counts, plane_keys=sorted(planes))


def test_gamma_rollout_balanced_draw() -> dict:
    """A class draw must round-robin gamma first and rollout second without needless repeats."""
    idx, gamma, rid = [], [], []
    k = 0
    for g in GAMMAS:
        for r in (0, 1):
            idx.append(k); gamma.append(g); rid.append(int(round(g * 100)) * 10 + r); k += 1
    np.random.seed(91)
    draw = FIX._draw_gamma_rid_balanced(
        np.asarray(idx), len(idx), np.asarray(gamma), np.asarray(rid)
    )
    require(len(draw) == len(idx), f"balanced draw returned {len(draw)}, expected {len(idx)}")
    require(len(np.unique(draw)) == len(draw), "balanced draw repeated a window despite sufficient unique data")
    counts = {}
    for g in GAMMAS:
        z = np.isclose(np.asarray(gamma)[draw], g)
        counts[str(g)] = int(z.sum())
        require(z.sum() == 2, f"gamma {g} received {z.sum()} draws, expected 2")
        require(len(np.unique(np.asarray(rid)[draw][z])) == 2,
                f"gamma {g} did not cycle its two rollout IDs")
    return dict(draw=draw.tolist(), counts=counts)


def test_gamma_mode_rollout_balanced_draw() -> dict:
    """When staircase modes are known, each gamma draw must cover them before reuse."""
    idx, gamma, rid, mode = [], [], [], []
    k = 0
    for gi, g in enumerate(GAMMAS):
        for mi, m in enumerate(("left", "right")):
            for r in (0, 1):
                idx.append(k); gamma.append(g); rid.append(100 * gi + 10 * mi + r); mode.append(m); k += 1
    np.random.seed(919)
    draw = FIX._draw_gamma_rid_balanced(
        np.asarray(idx), len(idx), np.asarray(gamma), np.asarray(rid), np.asarray(mode, dtype=object)
    )
    require(len(draw) == len(idx) and len(np.unique(draw)) == len(draw),
            "mode-balanced draw omitted/repeated data despite exact finite quota")
    detail = {}
    for g in GAMMAS:
        z = np.isclose(np.asarray(gamma)[draw], g)
        drawn_modes, counts = np.unique(np.asarray(mode, dtype=object)[draw][z], return_counts=True)
        detail[str(g)] = {str(m): int(c) for m, c in zip(drawn_modes, counts)}
        require(detail[str(g)] == {"left": 2, "right": 2},
                f"gamma {g} mode allocation is unbalanced: {detail[str(g)]}")
        require(len(np.unique(np.asarray(rid)[draw][z])) == 4,
                f"gamma {g} reused rollout before covering all four sources")
    return dict(per_gamma_modes=detail)


def test_executed_horizon_targets() -> dict:
    """CFM targets must be consecutive actions actually executed, never proposal tails."""
    recs = []
    for i in range(13):
        U = np.full((10, 2), -1000.0 - i, np.float32)
        U[0] = np.array([i, i + 0.5], np.float32)
        recs.append((
            np.full((3, 16, 12), i, np.float32),
            np.full(5, i, np.float32),
            np.full((16, 2), i, np.float32),
            U,
        ))
    out = FIX._executed_horizon_tensors(recs)
    require(out is not None, "coherent target constructor rejected 13 recs for H=10")
    G, L, H, U = out
    require(tuple(U.shape) == (4, 10, 2), f"unexpected coherent target shape {tuple(U.shape)}")
    expected = np.stack([
        np.stack([np.array([i + j, i + j + 0.5], np.float32) for j in range(10)])
        for i in range(4)
    ])
    require(np.array_equal(U.numpy(), expected),
            "coherent targets contain an unexecuted proposal-tail action or wrong temporal alignment")
    require(np.array_equal(L[:, 0].numpy(), np.arange(4, dtype=np.float32)),
            "coherent target contexts are not aligned to their first executed action")
    return dict(target_shape=list(U.shape), first=U[0].tolist(), last=U[-1].tolist())


def test_nonfinite_margin_unlabelable() -> dict:
    """A failed/NaN verifier result cannot be converted into a frontier score."""
    n = 3
    fresh = dict(
        grid=torch.zeros(n, 3, 16, 12), low5=torch.ones(n, 5),
        hist=torch.zeros(n, 16, 2), U=torch.ones(n, 10, 2),
        gamma=torch.tensor([0.1, 0.1, 0.1]), prog=np.ones(n),
        socp_margin=np.array([0.2, np.nan, 0.4]), widx=np.arange(n),
    )
    with mock.patch.object(FIX, "_sigma_of", return_value=np.ones(n)):
        try:
            FIX.label_fresh(object(), object(), fresh, object(),
                            SimpleNamespace(s=0.9, active_quantile=0.5), "cpu")
        except RuntimeError as exc:
            require("margin" in str(exc).lower(), f"unexpected nonfinite-margin error: {exc}")
            return dict(rejected=True, error=str(exc))
    raise AssertionError("label_fresh accepted a NaN verifier margin")


def test_rejected_queries_enter_qbuf_and_strict_reach() -> dict:
    """Selected proposals are query memory even when their executed path fails Valid2."""
    seen = dict(deploy=[], valid2=0, cat=[])

    def deploy(*_a, **kw):
        seen["deploy"].append(kw)
        return dict(reached=False, path=np.zeros((11, 2), float), recs=[fake_rec(0.0), fake_rec(1.0)])

    def to_t(recs):
        return tuple(torch.tensor(np.asarray([r[j] for r in recs])) for j in range(4))

    def cat(buf, G, L, H, U, cap=None, **_kw):
        seen["cat"].append(dict(n=int(U.shape[0]), cap=cap))
        return dict(grid=G.clone(), low5=L.clone(), hist=H.clone(), U=U.clone(), tag=None)

    def invalid(*_a, **_kw):
        seen["valid2"] += 1
        return False

    cfg = tiny_cfg()
    with mock.patch.object(FIX.GR, "fm_deploy", deploy), \
         mock.patch.object(FIX.SR, "path_collides", return_value=False), \
         mock.patch.object(FIX.GE, "_to_t", to_t), \
         mock.patch.object(FIX.GE, "_cat", cat), \
         mock.patch.object(FIX.GE, "_buffer_feat", return_value=None), \
         mock.patch.object(FIX.GM2, "traj_valid2", invalid):
        result = FIX._gather_fresh(
            object(), object(), SimpleNamespace(goal=torch.tensor([1.0, 1.0])), cfg,
            [0.1], 0.3, 1, 1, 1, None, {0.1: set()}, "cpu"
        )
        fresh, qbuf, _rr, _rc, valid, attempts, audit = unpack_gather(result)
    require(fresh is None and valid == 0 and attempts == 1, "invalid path unexpectedly entered fresh data")
    require(seen["valid2"] == 1, "gather did not invoke the exact executed Valid2 gate")
    require(len(seen["deploy"]) == 1 and seen["deploy"][0].get("reach") == cfg.reach,
            f"deployment did not receive strict reach={cfg.reach}")
    require(len(seen["cat"]) == 1 and qbuf is not None,
            "selected proposals from a Valid2-rejected trajectory were forgotten by qbuf")
    require(seen["cat"][0]["n"] >= 1, "query-buffer update contained no selected proposal")
    return dict(attempts=attempts, qbuf_rows=int(qbuf["U"].shape[0]), reach=cfg.reach,
                audit=audit)


def test_planned_certificate_gate_and_face_cache() -> dict:
    """Only certificate-feasible full-H targets survive, with the face margin cached exactly."""
    recs = [fake_rec(0.0), fake_rec(1.0)]

    def deploy(*_a, **_kw):
        return dict(reached=False, path=np.zeros((11, 2), float), recs=recs)

    def progress(*_a, **_kw):
        return 0.4, np.zeros((11, 2), float), np.linspace(2.0, 1.6, 11)

    def stats(_state, U, *_a, **_kw):
        # Deliberately make face margin and residual different.  The local API is
        # (ok, face_margin, cert_residual); the trainer must cache face_margin.
        return (False, 0.11, -7.0) if float(np.asarray(U)[0, 0]) == 0.0 else (True, 0.42, -3.0)

    def labels(_policy, _unc, fresh, *_a, **_kw):
        return np.arange(len(fresh["U"]), dtype=int), np.array([], dtype=int), {}

    def coherent(recs_):
        return tuple(torch.tensor(np.asarray([r[j] for r in recs_])) for j in range(4))

    cfg = tiny_cfg()
    with mock.patch.object(FIX.GR, "fm_deploy", deploy), \
         mock.patch.object(FIX.SR, "path_collides", return_value=False), \
         mock.patch.object(FIX.GM2, "traj_valid2", return_value=True), \
         mock.patch.object(FIX.GE, "_buffer_feat", return_value=None), \
         mock.patch.object(FIX, "_executed_horizon_tensors", coherent), \
         mock.patch.object(FIX, "_window_progress", progress), \
         mock.patch.object(FIX.GM, "in_taskspace", return_value=True), \
         mock.patch.object(FIX.GM2, "approach_ok", return_value=True), \
         mock.patch.object(FIX.GM2, "window_socp_stats", stats), \
         mock.patch.object(FIX, "label_fresh", labels):
        result = FIX._gather_fresh(
            object(), object(), SimpleNamespace(goal=torch.tensor([1.0, 1.0])), cfg,
            [0.1], 0.3, 1, 0, 0, None, {0.1: set()}, "cpu"
        )
        fresh, qbuf, _rr, _rc, valid, attempts, audit = unpack_gather(result)
    require(fresh is not None and valid == 1 and attempts == 1, "feasible rollout was not retained")
    require(len(fresh["U"]) == 1, f"planned cert gate retained {len(fresh['U'])} targets, expected 1")
    require(float(fresh["U"][0, 0, 0]) == 1.0, "certificate-infeasible full-H target survived")
    require(np.array_equal(np.asarray(fresh["socp_margin"]), np.array([0.42])),
            f"cached frontier margin is not the verifier face margin: {fresh['socp_margin']}")
    require(qbuf is not None and len(qbuf["U"]) >= 1, "selected proposals did not enter query memory")
    return dict(kept=int(len(fresh["U"])), cached_margin=float(fresh["socp_margin"][0]),
                qbuf_rows=int(len(qbuf["U"])), audit=audit)


def test_gather_honors_unique_class_quotas() -> dict:
    """Gather cannot report ready until unique easy/frontier quotas can fill the batch."""
    calls = 0

    def deploy(*_a, **_kw):
        nonlocal calls
        calls += 1
        return dict(reached=False, path=np.zeros((11, 2), float),
                    recs=[fake_rec(float(2 * calls)), fake_rec(float(2 * calls + 1))])

    def progress(*_a, **_kw):
        return 0.4, np.zeros((11, 2), float), np.linspace(2.0, 1.6, 11)

    def labels(_policy, _unc, fresh, *_a, **_kw):
        n = len(fresh["U"])
        nf = min(2, max(1, n // 2))
        frontier = np.arange(nf, dtype=int)
        easy = np.arange(nf, n, dtype=int)
        return easy, frontier, {}

    def coherent(recs_):
        return tuple(torch.tensor(np.asarray([r[j] for r in recs_])) for j in range(4))

    cfg = tiny_cfg(gather_attempt_cap=5)
    with mock.patch.object(FIX.GR, "fm_deploy", deploy), \
         mock.patch.object(FIX.SR, "path_collides", return_value=False), \
         mock.patch.object(FIX.GM2, "traj_valid2", return_value=True), \
         mock.patch.object(FIX.GE, "_buffer_feat", return_value=None), \
         mock.patch.object(FIX, "_executed_horizon_tensors", coherent), \
         mock.patch.object(FIX, "_window_progress", progress), \
         mock.patch.object(FIX.GM, "in_taskspace", return_value=True), \
         mock.patch.object(FIX.GM2, "approach_ok", return_value=True), \
         mock.patch.object(FIX.GM2, "window_socp_stats", return_value=(True, 0.5, -1.0)), \
         mock.patch.object(FIX, "label_fresh", labels):
        result = FIX._gather_fresh(
            object(), object(), SimpleNamespace(goal=torch.tensor([1.0, 1.0])), cfg,
            [0.1], 0.3, 1, 4, 2, None, {0.1: set()}, "cpu"
        )
        fresh, _qbuf, _rr, _rc, valid, attempts, audit = unpack_gather(result)
    require(fresh is not None, "quota gather returned no data")
    easy, frontier, _ = labels(None, None, fresh)
    require(len(easy) >= 4 and len(frontier) >= 2,
            f"gather stopped class-starved at {len(easy)} easy/{len(frontier)} frontier")
    require(attempts == 3 and valid == 3,
            f"quota gather used {attempts} attempts/{valid} valid rollouts, expected 3/3")
    require(bool(fresh.get("classes_ready")) and bool(fresh.get("gamma_ready")),
            "gather readiness metadata is false after satisfying quotas")
    return dict(attempts=attempts, valid_rollouts=valid, easy=len(easy), frontier=len(frontier), audit=audit)


def test_gather_all_gamma_class_quota() -> dict:
    """Readiness requires easy and frontier examples for every conditioning gamma."""
    calls = 0

    def deploy(*_a, **_kw):
        nonlocal calls
        calls += 1
        return dict(reached=False, path=np.zeros((11, 2), float),
                    recs=[fake_rec(float(2 * calls)), fake_rec(float(2 * calls + 1))])

    def coherent(recs_):
        return tuple(torch.tensor(np.asarray([r[j] for r in recs_])) for j in range(4))

    def progress(*_a, **_kw):
        return 0.4, np.zeros((11, 2), float), np.linspace(2.0, 1.6, 11)

    def labels(_policy, _unc, fresh, *_a, **_kw):
        # Each rollout contributes [easy, frontier], hence one of each per gamma.
        n = len(fresh["U"])
        return np.arange(0, n, 2, dtype=int), np.arange(1, n, 2, dtype=int), {}

    cfg = tiny_cfg(gather_attempt_cap=10)
    with mock.patch.object(FIX.GR, "fm_deploy", deploy), \
         mock.patch.object(FIX.SR, "path_collides", return_value=False), \
         mock.patch.object(FIX.GM2, "traj_valid2", return_value=True), \
         mock.patch.object(FIX.GE, "_buffer_feat", return_value=None), \
         mock.patch.object(FIX, "_executed_horizon_tensors", coherent), \
         mock.patch.object(FIX, "_window_progress", progress), \
         mock.patch.object(FIX.GM, "in_taskspace", return_value=True), \
         mock.patch.object(FIX.GM2, "approach_ok", return_value=True), \
         mock.patch.object(FIX.GM2, "window_socp_stats", return_value=(True, 0.5, -1.0)), \
         mock.patch.object(FIX, "label_fresh", labels):
        result = FIX._gather_fresh(
            object(), object(), SimpleNamespace(goal=torch.tensor([1.0, 1.0])), cfg,
            list(GAMMAS), 0.3, 1, 7, 7, None, {g: set() for g in GAMMAS}, "cpu"
        )
        fresh, _qbuf, _rr, _rc, valid, attempts, audit = unpack_gather(result)
    require(fresh is not None and valid == len(GAMMAS) and attempts == len(GAMMAS),
            f"all-gamma gather used {valid} valid/{attempts} attempts, expected {len(GAMMAS)}")
    require(bool(fresh["gamma_ready"]) and bool(fresh["gamma_class_ready"]),
            f"all-gamma class readiness failed: {audit}")
    gamma = fresh["gamma"].numpy()
    easy, frontier, _ = labels(None, None, fresh)
    counts = {}
    for g in GAMMAS:
        ne = int(np.isclose(gamma[easy], g).sum()); nf = int(np.isclose(gamma[frontier], g).sum())
        counts[str(g)] = dict(easy=ne, frontier=nf)
        require(ne >= 1 and nf >= 1, f"gamma {g} class-starved: {counts[str(g)]}")
    return dict(attempts=attempts, per_gamma=counts, audit=audit)


def _state_api():
    """Find the state pack/restore helpers once the production patch lands."""
    pack_names = ("_pack_train_state", "_make_train_state", "_capture_train_state")
    restore_names = ("_apply_train_state", "_restore_train_state", "_load_train_state")
    pack = next((getattr(FIX, n) for n in pack_names if hasattr(FIX, n)), None)
    restore = next((getattr(FIX, n) for n in restore_names if hasattr(FIX, n)), None)
    return pack, restore


def test_resume_state_roundtrip() -> dict:
    """Full-state checkpoint helpers must round-trip every continuation-critical component.

    The state implementation was being patched concurrently when this harness was written.
    This gate is intentionally strict: absence of an explicit pack/restore API is a failure,
    not a model-only-checkpoint pass.
    """
    pack, restore = _state_api()
    require(pack is not None and restore is not None,
            "no explicit full train-state pack/restore helpers are exposed for regression testing")

    # The concrete call adapter is finalized against the production helper signatures.
    sig_pack = str(inspect.signature(pack))
    sig_restore = str(inspect.signature(restore))
    adapter = getattr(FIX, "_train_state_regression_roundtrip", None)
    require(adapter is not None,
            "state helpers exist, but `_train_state_regression_roundtrip()` test adapter is absent")
    result = adapter()
    require(isinstance(result, dict), "state roundtrip adapter did not return diagnostics")
    needed = {"optimizer", "qbuf", "covered", "teacher", "history", "numpy_rng", "torch_rng"}
    missing = needed - set(result)
    require(not missing, f"state roundtrip diagnostics omit: {sorted(missing)}")
    require(all(bool(result[k]) for k in needed), f"state roundtrip failed: {result}")
    if torch.cuda.is_available():
        require(bool(result.get("cuda_rng")), f"CUDA RNG did not round-trip: {result}")
    return dict(pack_signature=sig_pack, restore_signature=sig_restore, checks=result)


class TinyFlowPolicy(torch.nn.Module):
    """Small CFM-compatible policy for an exact uninterrupted-vs-split state test."""

    def __init__(self):
        super().__init__()
        self.d = 20
        self.u_max = 1.0
        self.repr_dim = 8
        self.enc = torch.nn.Linear(2, 3)
        self.trunk = torch.nn.Sequential(torch.nn.Linear(self.d + 3 + 1, 12), torch.nn.SiLU())
        self.head = torch.nn.Linear(12, self.d)

    def encoder_modules(self):
        return list(self.enc.parameters())

    def ctx_from(self, grid, low5, hist):
        del grid, hist
        return self.enc(low5[:, :2].float())

    def _expand_ctx(self, ctx, batch):
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        return ctx if ctx.shape[0] == batch else ctx.expand(batch, -1)

    def forward(self, x, tau, ctx):
        if tau.ndim == 0:
            tau = tau.expand(x.shape[0])
        return self.head(self.trunk(torch.cat([x, ctx, tau[:, None]], dim=1)))

    def cfm_loss(self, U, ctx):
        batch = U.shape[0]
        x1 = (U / self.u_max).reshape(batch, self.d)
        x0 = torch.randn_like(x1)
        tau = torch.rand(batch, device=x1.device).clamp(1e-4, 1.0)
        x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        target = x1 - x0
        pred = self.forward(x_tau, tau, self._expand_ctx(ctx, batch))
        return (pred - target).square().mean()

    def config(self):
        return dict(arch="tiny-regression", width=12, u_max=self.u_max, repr_dim=self.repr_dim)


def _tiny_resume_cfg(iters: int, start_iter: int) -> FIX.CurConfig:
    cfg = FIX.CurConfig(
        iters=iters, start_iter=start_iter, gammas=(0.1, 0.5),
        batch_cap=8, demo_frac=0.25, lwf_eta=0.05,
        fresh_frac=0.5, pile_cap=64, pile_relabel_every=0,
        M_measure=1, measure_every=1, ckpt_every=1, viz_db_every=0,
        probe_escape=0, probe_cov=0, log_comp_every=0,
        rollouts_per_iter=2, gather_attempt_cap=2,
        early_inner=1, inner_steps=1, cooldown_inner=1,
        mix_start=(0.5, 0.5), mix_end=(0.5, 0.5),
        legacy_prime_iters=0, lr=3e-4,
    )
    return cfg


def test_resume_split_equivalence() -> dict:
    """Two uninterrupted iterations must equal one iteration + full-state resume exactly."""
    require(hasattr(FIX, "_capture_train_state") and hasattr(FIX, "_apply_train_state"),
            "full train-state capture/apply helpers are unavailable")

    demo = dict(
        grid=torch.zeros(16, 3, 2, 2),
        low5=torch.linspace(-1.0, 1.0, 16 * 5).reshape(16, 5),
        hist=torch.zeros(16, 2, 2),
        U=torch.linspace(-0.5, 0.5, 16 * 10 * 2).reshape(16, 10, 2),
    )

    def fake_measure(_policy, _env, cfg, _device):
        rows = {g: {"SR": 0.8, "CR": 0.0, "mean_goal_dist": 0.2} for g in cfg.gammas}
        return rows, {"SR": 0.8, "CR": 0.0, "mean_goal_dist": 0.2}

    def fake_label(_policy, _unc, fresh, _env, cfg, _device):
        n = len(fresh["U"])
        easy = np.arange(0, n, 2, dtype=int)
        frontier = np.arange(1, n, 2, dtype=int)
        margin = np.asarray(fresh["socp_margin"], dtype=float)
        prog = np.asarray(fresh["prog"], dtype=float)
        scores = dict(
            sigma=np.linspace(0.1, 0.9, n), margin=margin,
            jerk=np.zeros(n), mono=np.zeros(n), prog=prog,
            quantile=float(cfg.active_quantile), sigma_plane=0.5,
            margin_plane=float(np.median(margin)), prog_plane=float(np.median(prog)),
            planes_by_gamma={},
        )
        return easy, frontier, scores

    def fake_gather(policy, unc, env, cfg, gammas, beta, K, target_e, target_f,
                    qbuf, covered, device, gamma_offset=0):
        del policy, unc, env, beta, K, target_e, target_f, device
        # All four RNG families influence the next update data.  The resumed run
        # deliberately starts from a different command seed, so only state restore
        # can reproduce the uninterrupted second iteration.
        py = random.random()
        nn = float(np.random.normal())
        tn = float(torch.rand(()))
        cn = float(torch.rand((), device="cuda").cpu()) if torch.cuda.is_available() else 0.0
        offset = py + nn + tn + cn + 0.01 * gamma_offset
        n = 8
        grid = torch.full((n, 3, 2, 2), offset, dtype=torch.float32)
        low5 = torch.randn(n, 5) * 0.1 + offset
        hist = torch.randn(n, 2, 2) * 0.1
        U = torch.tanh(torch.randn(n, 10, 2) * 0.2 + offset * 0.01)
        gamma = torch.tensor([gammas[0]] * 4 + [gammas[1]] * 4, dtype=torch.float32)
        fresh = dict(
            grid=grid, low5=low5, hist=hist, U=U, gamma=gamma,
            prog=np.linspace(0.2, 0.5, n), socp_margin=np.linspace(0.1, 0.8, n),
            cert_residual=np.linspace(0.0, 0.01, n), rid=np.arange(n) // 2,
            widx=np.arange(n), mode=np.array(["left", "right"] * 4, dtype=object),
            paths=[], rollout_gamma=np.asarray(gammas), attempted_gamma=np.asarray(gammas),
            gamma_ready=True, classes_ready=True, gamma_class_ready=True,
        )
        qbuf = FIX.GE._cat(qbuf, grid, low5, hist, U, cap=cfg.qbuf_cap)
        for g in gammas:
            covered[g].add(f"mode-{g}-{gamma_offset}")
        audit = dict(ready=True, gamma_ready=True, classes_ready=True, gamma_class_ready=True,
                     valid_rollouts=len(gammas))
        return fresh, qbuf, [1.0, 0.0], [0.0, 0.0], len(gammas), len(gammas), audit

    def seed_all(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    seed_all(4401)
    initial = TinyFlowPolicy()
    initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}

    with tempfile.TemporaryDirectory(prefix="p2_resume_reg_", dir=ROOT / "analysis") as td, \
         mock.patch.object(FIX, "_load_demo", return_value=demo), \
         mock.patch.object(FIX, "_measure", fake_measure), \
         mock.patch.object(FIX, "label_fresh", fake_label), \
         mock.patch.object(FIX, "_gather_fresh", fake_gather):
        td = Path(td)

        seed_all(808)
        policy_a = copy.deepcopy(initial)
        FIX.run_expand_cur(policy_a, object(), _tiny_resume_cfg(2, 0), device="cpu",
                           outdir=str(td / "a"), log=lambda *_a, **_kw: None,
                           freeze_enc=False, enc_lr_mult=0.3, train_seed=808)
        ck_a = torch.load(td / "a/final.pt", map_location="cpu", weights_only=False)

        seed_all(808)
        policy_b1 = copy.deepcopy(initial)
        FIX.run_expand_cur(policy_b1, object(), _tiny_resume_cfg(1, 0), device="cpu",
                           outdir=str(td / "b1"), log=lambda *_a, **_kw: None,
                           freeze_enc=False, enc_lr_mult=0.3, train_seed=808)
        ck_b1 = torch.load(td / "b1/final.pt", map_location="cpu", weights_only=False)

        # Simulate a fresh process and HP.load_hp(...).eval() under an unrelated CLI seed.
        seed_all(999)
        policy_b2 = TinyFlowPolicy().eval()
        policy_b2.load_state_dict(ck_b1["state_dict"])
        FIX.run_expand_cur(policy_b2, object(), _tiny_resume_cfg(1, 1), device="cpu",
                           outdir=str(td / "b2"), log=lambda *_a, **_kw: None,
                           freeze_enc=False, enc_lr_mult=0.3,
                           resume_state=ck_b1["train_state"], train_seed=999)
        ck_b2 = torch.load(td / "b2/final.pt", map_location="cpu", weights_only=False)

    require(ck_a["iter"] == ck_b2["iter"] == 2, "uninterrupted/split completed iteration differs")
    require(tensor_tree_equal(ck_a["state_dict"], ck_b2["state_dict"]),
            "policy tensors differ after split resume")
    sa, sb = ck_a["train_state"], ck_b2["train_state"]
    compared = [
        "optimizer", "qbuf", "covered", "pile", "teacher_state", "history",
        "roll_reached", "roll_coll", "last", "best_sr", "sr0", "best_safe_sr",
        "collapse_ct", "best_probe", "best_probe_cov", "cooled", "numpy_rng",
        "python_rng", "torch_rng", "cuda_rng",
    ]
    mismatched = [k for k in compared if not tensor_tree_equal(sa.get(k), sb.get(k))]
    require(not mismatched, f"split continuation state differs in {mismatched}")
    require(sa["optimizer"]["state"], "Adam moments/steps were not exercised")
    require(sa["qbuf"] is not None and len(sa["qbuf"]["U"]) == 16,
            f"query buffer did not persist both iterations: {None if sa['qbuf'] is None else len(sa['qbuf']['U'])}")
    require(sa["pile"] is not None and sa["pile"]["T"] is not None,
            "bounded pile was not captured/restored")
    require(tensor_tree_equal(sa["teacher_state"], initial_state),
            "LwF teacher drifted from the original pre-update anchor")
    max_err = max(float((ck_a["state_dict"][k] - ck_b2["state_dict"][k]).abs().max())
                  for k in ck_a["state_dict"])
    return dict(iter=2, exact=True, model_max_abs_error=max_err, compared=compared,
                qbuf_rows=len(sa["qbuf"]["U"]), optimizer_slots=len(sa["optimizer"]["state"]),
                cuda_rng_checked=torch.cuda.is_available())


TESTS = [
    ("rng_isolation", test_rng_isolation),
    ("probe_rng_integration", test_probe_rng_integration),
    ("verifier_face_margin", test_verifier_face_margin),
    ("per_gamma_planes", test_per_gamma_planes),
    ("gamma_rollout_balanced_draw", test_gamma_rollout_balanced_draw),
    ("gamma_mode_rollout_balanced_draw", test_gamma_mode_rollout_balanced_draw),
    ("executed_horizon_targets", test_executed_horizon_targets),
    ("nonfinite_margin_unlabelable", test_nonfinite_margin_unlabelable),
    ("rejected_queries_qbuf_strict_reach", test_rejected_queries_enter_qbuf_and_strict_reach),
    ("planned_certificate_gate_face_cache", test_planned_certificate_gate_and_face_cache),
    ("gather_unique_class_quotas", test_gather_honors_unique_class_quotas),
    ("gather_all_gamma_class_quota", test_gather_all_gamma_class_quota),
    ("resume_state_roundtrip", test_resume_state_roundtrip),
    ("resume_split_equivalence", test_resume_split_equivalence),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="optional machine-readable result path")
    ap.add_argument("--only", action="append", default=[], help="run only this named gate (repeatable)")
    ap.add_argument("--allow-skip", action="store_true", help="do not count unavailable-input skips as failures")
    args = ap.parse_args()

    selected = [(n, f) for n, f in TESTS if not args.only or n in set(args.only)]
    unknown = set(args.only) - {n for n, _ in TESTS}
    if unknown:
        ap.error(f"unknown gate(s): {sorted(unknown)}")
    rows = []
    for name, fn in selected:
        try:
            detail = fn()
            row = dict(name=name, status="PASS", detail=detail)
        except SkipGate as exc:
            row = dict(name=name, status="SKIP", error=str(exc))
        except Exception as exc:  # keep running: one command reports the whole semantic surface
            row = dict(name=name, status="FAIL", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        rows.append(row)
        print(f"[{row['status']}] {name}" + (f": {row.get('error')}" if row["status"] != "PASS" else ""),
              flush=True)
    summary = dict(
        production=str((ROOT / "grid_expand_fixed.py").relative_to(ROOT)),
        cuda_available=torch.cuda.is_available(),
        cuda_visible_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        pass_count=sum(r["status"] == "PASS" for r in rows),
        fail_count=sum(r["status"] == "FAIL" for r in rows),
        skip_count=sum(r["status"] == "SKIP" for r in rows),
        results=rows,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    bad_skip = summary["skip_count"] and not args.allow_skip
    return 1 if summary["fail_count"] or bad_skip else 0


if __name__ == "__main__":
    raise SystemExit(main())
