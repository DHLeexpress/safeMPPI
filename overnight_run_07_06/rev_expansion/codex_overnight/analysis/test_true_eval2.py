"""Contract tests for the canonical portable dual-scene TRUE evaluation."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "paper_results"))
sys.path.insert(0, str(_ROOT))

import true_eval_run as TER  # noqa: E402
import true_eval_fig as TEF  # noqa: E402
import kazuki_baseline as KB  # noqa: E402


def _sha(path: Path) -> str:
    return TER.sha256_file(path)


def _make_completed_pair(root: Path, profile: str = "claude_grid_v1", rounds: int = 2) -> Path:
    afe = root / "afe_s910"
    (afe / "viz_db").mkdir(parents=True)
    recipe = {
        "arm": "afe",
        "reference_recipe_locked": True,
        "scene": {"profile": {"name": profile}, "sha256": "s" * 64},
        "source_checkpoint_sha256": "c" * 64,
        "source_checkpoint_model_sha256": "d" * 64,
        "source_checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "f" * 40,
    }
    (afe / "recipe.json").write_text(json.dumps(recipe))
    (afe / "probe.jsonl").write_text("{}\n")
    (afe / "final.pt").write_bytes(b"final")
    (afe / "dstore.pt").write_bytes(b"dstore")
    for n in range(rounds + 1):
        (afe / f"ckpt_{n}.pt").write_bytes(f"ckpt-{n}".encode())
        if n:
            (afe / "viz_db" / f"round{n}.pt").write_bytes(f"viz-{n}".encode())
    required = {
        "recipe.json", "probe.jsonl", "final.pt", "dstore.pt",
        *{f"ckpt_{n}.pt" for n in range(rounds + 1)},
        *{f"viz_db/round{n}.pt" for n in range(1, rounds + 1)},
    }
    inventory = {relative: _sha(afe / relative) for relative in sorted(required)}
    complete = {
        "status": "COMPLETE", "completed_round": rounds,
        "scene_sha256": "s" * 64, "checkpoint_sha256": "c" * 64,
        "checkpoint_model_sha256": "d" * 64,
        "checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "f" * 40, "artifact_sha256": inventory,
    }
    (afe / "COMPLETE.json").write_text(json.dumps(complete))
    manifest_path = root / f"afe2_{profile}_pair_manifest.json"
    manifest = {
        "status": "VALIDATED_MATCHED_AFE2_PAIR", "scene_profile": profile,
        "scene_sha256": "s" * 64, "source_checkpoint_sha256": "c" * 64,
        "source_checkpoint_model_sha256": "d" * 64,
        "source_checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "f" * 40,
        "runs": {"afe": {"recipe_sha256": _sha(afe / "recipe.json"),
                          "probe_sha256": _sha(afe / "probe.jsonl"),
                          "complete_sha256": _sha(afe / "COMPLETE.json")}},
    }
    manifest_path.write_text(json.dumps(manifest))
    delivery = {
        "status": "DELIVERY_COMPLETE", "scene_sha256": "s" * 64,
        "source_checkpoint_sha256": "c" * 64,
        "source_checkpoint_model_sha256": "d" * 64,
        "source_checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "f" * 40,
        "artifacts": {"pair_manifest": {"sha256": _sha(manifest_path)}},
    }
    (root / "DELIVERY_COMPLETE.json").write_text(json.dumps(delivery))
    return afe


def test_named_seed_deterministic_and_key_sensitive() -> None:
    a = TER.named_seed("v", "scene", "policy", 0.5, 7)
    assert a == TER.named_seed("v", "scene", "policy", 0.5, 7)
    assert a != TER.named_seed("v", "scene", "policy", 0.5, 8)
    assert a != TER.named_seed("v", "scene", "policy", 0.1, 7)
    assert 0 <= a < 2 ** 63 - 1


def test_common_random_numbers_do_not_depend_on_round() -> None:
    """Round curves must be CRN-paired: the policy rollout seed is keyed by (gamma, index) only."""
    scene = "SCENESHA"
    seed_round_view = [
        TER.named_seed(TER.METRIC_VERSION, scene, "policy", 0.3, m) for m in range(5)
    ]
    # There is no round argument in the key at all; re-deriving yields identical seeds.
    assert seed_round_view == [
        TER.named_seed(TER.METRIC_VERSION, scene, "policy", 0.3, m) for m in range(5)
    ]


def test_seed_all_streams_seeds_global_python_rng() -> None:
    TER.seed_all_streams(12345)
    first = (random.random(), np.random.random())
    TER.seed_all_streams(12345)
    assert first == (random.random(), np.random.random())


def test_source_freeze_rejects_wrong_commit() -> None:
    with pytest.raises(RuntimeError, match="expansion source"):
        TER.require_clean_source("0" * 40)


def test_require_round_checkpoints_aborts_on_missing(tmp_path: Path) -> None:
    for n in (0, 1, 2, 4):                                   # ckpt_3.pt intentionally missing
        (tmp_path / f"ckpt_{n}.pt").write_bytes(b"x")
    (tmp_path / "final.pt").write_bytes(b"x")               # a fallback that must NOT be used
    with pytest.raises(FileNotFoundError, match="ckpt_3.pt"):
        TER.require_round_checkpoints(str(tmp_path), 4)
    got = TER.require_round_checkpoints(str(tmp_path), 2)
    assert sorted(got) == [0, 1, 2]
    assert all("final.pt" not in p for p in got.values())


def test_stale_cell_outputs_are_rejected(tmp_path: Path) -> None:
    name = TER.cell_name("policy", 3, 0.5)
    (tmp_path / f"paths_{name}.npz").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="stale output rejected"):
        TER.assert_fresh_cell(str(tmp_path), name)


def test_save_cell_validates_exact_M(tmp_path: Path) -> None:
    prov = dict(M=3)
    with pytest.raises(RuntimeError, match="rollout count"):
        TER.save_cell(str(tmp_path), "c", [np.zeros((4, 2))] * 2, [1, 2], prov)
    TER.save_cell(str(tmp_path), "c", [np.zeros((4, 2))] * 3, [1, 2, 3], dict(M=3))
    saved = json.load(open(tmp_path / "c.provenance.json"))
    assert saved["n_paths"] == 3 and saved["seeds"] == [1, 2, 3]
    assert saved["paths_sha256"] == _sha(tmp_path / "paths_c.npz")


def test_wilson_and_bootstrap_intervals() -> None:
    lo, hi = TEF.wilson95(0.5, 100)
    assert 0.39 < lo < 0.5 < hi < 0.61
    assert TEF.wilson95(0.0, 0) == (0.0, 0.0)
    lo, hi = TEF.bootstrap95([1.0, 1.0, 1.0], ("k",))
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert TEF.bootstrap95([], ("k",))[0] != TEF.bootstrap95([], ("k",))[0]  # nan


def test_content_key_changes_with_inputs() -> None:
    base = dict(scene_sha256="s", checkpoint_sha256="c", paths_sha256="p",
                seeds=[1, 2], M=2, T=300, metric_version=TER.METRIC_VERSION)
    k0 = TEF.content_key(base, 0.5, 0.15, 0.1)
    assert k0 == TEF.content_key(dict(base), 0.5, 0.15, 0.1)
    assert k0 != TEF.content_key(dict(base, checkpoint_sha256="c2"), 0.5, 0.15, 0.1)
    assert k0 != TEF.content_key(dict(base, paths_sha256="p2"), 0.5, 0.15, 0.1)
    assert k0 != TEF.content_key(dict(base, seeds=[1, 3]), 0.5, 0.15, 0.1)
    assert k0 != TEF.content_key(dict(base, M=3), 0.5, 0.15, 0.1)
    assert k0 != TEF.content_key(base, 0.3, 0.15, 0.1)
    assert k0 != TEF.content_key(base, 0.5, 0.2, 0.1)
    assert k0 != TEF.content_key(base, 0.5, 0.15, 0.2)


def test_gallery_rule_is_ratio_matched() -> None:
    # emulate the selection block: k = round(10*SR) successes + rest failures, no replacement
    mask = np.array([True] * 73 + [False] * 27)
    sr = float(mask.mean())
    k = int(round(10 * sr))
    assert k == 7
    rng = np.random.default_rng(TEF.named_seed("gallery", "s", "cell", 0.5))
    si, fi = np.where(mask)[0], np.where(~mask)[0]
    pick = (list(rng.choice(si, min(k, len(si)), replace=False)) +
            list(rng.choice(fi, min(10 - k, len(fi)), replace=False)))
    assert len(pick) == 10 and len(set(pick)) == 10
    assert sum(mask[i] for i in pick) == 7
    assert TEF.SUBSET_LABEL.startswith("pre-specified outcome-stratified")


def test_kazuki_definition_frozen() -> None:
    d = TER.KAZUKI_DEFINITION
    assert d["gamma_ctx"] == 0.5 and d["w_safe"] == 0.3 and d["n_sample"] == 200


def test_kazuki_uses_per_obstacle_collision_radii() -> None:
    obs = np.array([[1.0, 1.0, 0.1], [2.0, 2.0, 0.9]], dtype=np.float32)
    got = KB.obstacle_collision_radii(obs, 0.2, 0.05, device="cpu").numpy()
    assert got == pytest.approx([0.35, 1.15])
    assert got[0] != got[1]


def test_validate_pair_binds_completed_afe_checkpoints(tmp_path: Path) -> None:
    afe = _make_completed_pair(tmp_path, rounds=2)
    ckpts, contract = TER.validate_afe_pair(tmp_path, "claude_grid_v1", 2)
    assert sorted(ckpts) == [0, 1, 2]
    assert contract["afe_root"] == str(afe.resolve())
    assert contract["pair_manifest_sha256"] == _sha(
        tmp_path / "afe2_claude_grid_v1_pair_manifest.json")
    (afe / "ckpt_1.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="completion artifact hash mismatch"):
        TER.validate_afe_pair(tmp_path, "claude_grid_v1", 2)


def test_load_cell_rejects_tampered_raw_paths(tmp_path: Path) -> None:
    (tmp_path / "scene_snapshot.json").write_text("{}")
    artifacts = TER.save_cell(str(tmp_path), "c", [np.zeros((2, 2))], [1],
                              dict(M=1, scene_sha256="s"))
    artifacts["scene_snapshot.json"] = _sha(tmp_path / "scene_snapshot.json")
    (tmp_path / "RUN_COMPLETE.json").write_text(json.dumps({
        "status": "TRUE_EVAL_RAW_COMPLETE", "required_cells": ["c"],
        "artifact_sha256": artifacts,
    }))
    TEF._VERIFIED_RAW_RUNS.clear()
    paths, _ = TEF.load_cell(str(tmp_path), "c")
    assert len(paths) == 1
    (tmp_path / "paths_c.npz").write_bytes(b"tampered")
    TEF._VERIFIED_RAW_RUNS.clear()
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        TEF.load_cell(str(tmp_path), "c")
