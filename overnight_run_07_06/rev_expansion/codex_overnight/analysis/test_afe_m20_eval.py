from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_results"))

import afe_m20_eval as EV


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir(parents=True)
    recipe = {
        "algorithm": EV.SUPPORTED_ALGORITHM,
        "arm": "afe",
        "single_arm": True,
        "source_git_commit": "c" * 40,
        "scene": {
            "sha256": "a" * 64,
            "profile": {"name": "codex_radius1_v1"},
        },
        "source_checkpoint_sha256": "b" * 64,
        "source_checkpoint_model_sha256": "d" * 64,
        "source_checkpoint_contract_sha256": "e" * 64,
        "no_curriculum": True,
        "no_anchor": True,
        "no_prox": True,
        "no_fallback": True,
    }
    (run / "recipe.json").write_text(json.dumps(recipe))
    (run / "probe.jsonl").write_text("{}\n")
    required = EV.expected_inventory(EV.SUPPORTED_ALGORITHM, 50)
    for relative in required - {"recipe.json", "probe.jsonl"}:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    inventory = {relative: _sha(run / relative) for relative in required}
    complete = {
        "status": "COMPLETE",
        "algorithm": EV.SUPPORTED_ALGORITHM,
        "completed_round": 50,
        "scene_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_model_sha256": "d" * 64,
        "checkpoint_contract_sha256": "e" * 64,
        "source_git_commit": "c" * 40,
        "artifact_sha256": inventory,
    }
    (run / "COMPLETE.json").write_text(json.dumps(complete))
    delivery = {
        "status": "AFE_ENSEMBLE_DELIVERY_COMPLETE",
        "run": {"path": str(run / "COMPLETE.json"), "sha256": _sha(run / "COMPLETE.json")},
        "recipe": {"path": str(run / "recipe.json"), "sha256": _sha(run / "recipe.json")},
    }
    (root / "DELIVERY_COMPLETE.json").write_text(json.dumps(delivery))
    return run


def test_completed_run_validator_accepts_exact_delivered_inventory(tmp_path: Path):
    run = _fake_run(tmp_path / "delivery")
    contract = EV.validate_completed_run(run, "codex_radius1_v1")
    assert contract["algorithm"] == EV.SUPPORTED_ALGORITHM
    assert sorted(contract["selected_checkpoints"]) == [0, 50]
    assert contract["authenticated_artifact_count"] == len(
        EV.expected_inventory(EV.SUPPORTED_ALGORITHM, 50)
    )


def test_missing_round_checkpoint_never_substitutes_final(tmp_path: Path):
    run = _fake_run(tmp_path / "delivery")
    (run / "ckpt_50.pt").unlink()
    with pytest.raises(FileNotFoundError, match="inventoried artifact is missing"):
        EV.validate_completed_run(run, "codex_radius1_v1")
    assert (run / "final.pt").is_file()


def test_wrong_algorithm_or_scene_is_rejected(tmp_path: Path):
    run = _fake_run(tmp_path / "delivery")
    recipe = json.loads((run / "recipe.json").read_text())
    recipe["algorithm"] = "not-the-declared-algorithm"
    (run / "recipe.json").write_text(json.dumps(recipe))
    with pytest.raises(RuntimeError, match="recipe algorithm"):
        EV.validate_completed_run(run, "codex_radius1_v1")

    run = _fake_run(tmp_path / "other")
    with pytest.raises(RuntimeError, match="scene profile"):
        EV.validate_completed_run(run, "codex_radius03_v1")


def test_paired_seeds_ignore_checkpoint_round_by_construction():
    first = EV.paired_seed("codex_radius1_v1", "raw", 0.3, 7, 11)
    assert first == EV.paired_seed("codex_radius1_v1", "raw", 0.3, 7, 11)
    assert first != EV.paired_seed("codex_radius1_v1", "verified", 0.3, 7, 11)
    assert first != EV.paired_seed("codex_radius1_v1", "raw", 0.3, 7, 12)


def test_wilson95_and_fixed_gallery_contract():
    lo, hi = EV.wilson95(10, 20)
    assert lo == pytest.approx(0.29929800819821234)
    assert hi == pytest.approx(0.7007019918017876)
    assert EV.GALLERY_INDICES == tuple(range(10))
    assert EV.M == 20


def test_terminal_counts_must_partition_each_cell(tmp_path: Path):
    rows = []
    rounds = (0, 50)
    for mode in ("raw", "verified"):
        for round_i in rounds:
            for gamma in EV.GAMMAS:
                binary = {
                    key: {"count": 0, "n": EV.M, "estimate": 0.0, "wilson95": [0, 1]}
                    for key in (
                        "SR", "CR", "NVP", "timeout", "V_safe", "V_full",
                        "collision", "OOB",
                    )
                }
                binary["timeout"]["count"] = EV.M
                rows.append({
                    "mode": mode,
                    "method": "afe",
                    "round": round_i,
                    "scope": "gamma",
                    "gamma": gamma,
                    "n": EV.M,
                    "binary": binary,
                })
            rows.append({
                "mode": mode, "method": "afe", "round": round_i,
                "scope": "pooled", "gamma": None,
            })
    cell_count = 2 * len(rounds) * len(EV.GAMMAS)
    cells = tmp_path / "cells"
    for index in range(cell_count):
        (cells / f"c{index}").mkdir(parents=True)
        (cells / f"c{index}" / "x.npz").write_bytes(b"x")
        (cells / f"c{index}" / "x.provenance.json").write_text("{}")
    EV._authenticate_output_cells(tmp_path, rows)
