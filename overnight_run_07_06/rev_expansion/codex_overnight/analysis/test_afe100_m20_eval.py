from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_results"))

import afe100_m20_eval as EV


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_run(root: Path, method: str) -> Path:
    algorithm = EV.ALGORITHMS[method]
    root.mkdir()
    recipe = {
        "algorithm": algorithm,
        "arm": "afe",
        "single_arm": True,
        "source_git_commit": EV.BASE_SOURCE_COMMIT,
        "scene": {
            "sha256": "a" * 64,
            "profile": {"name": EV.SCENE_PROFILE},
        },
        "source_checkpoint_sha256": "b" * 64,
        "source_checkpoint_model_sha256": "c" * 64,
        "source_checkpoint_contract_sha256": "d" * 64,
        "no_curriculum": True,
        "no_anchor": True,
        "no_prox": True,
        "no_fallback": True,
    }
    (root / "recipe.json").write_text(json.dumps(recipe))
    (root / "probe.jsonl").write_text("{}\n")
    required = EV.expected_inventory(algorithm)
    for relative in required - {"recipe.json", "probe.jsonl"}:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    inventory = {relative: _sha(root / relative) for relative in required}
    complete = {
        "status": "COMPLETE",
        "algorithm": algorithm,
        "completed_round": 100,
        "scene_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_model_sha256": "c" * 64,
        "checkpoint_contract_sha256": "d" * 64,
        "source_git_commit": EV.BASE_SOURCE_COMMIT,
        "artifact_sha256": inventory,
    }
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    return root


@pytest.mark.parametrize("method", ["rbf", "ensemble"])
def test_completed_run_validator_accepts_only_exact_declared_inventory(tmp_path: Path, method: str):
    run = _fake_run(tmp_path / method, method)
    contract = EV.validate_completed_run(run, method)
    assert contract["algorithm"] == EV.ALGORITHMS[method]
    assert sorted(contract["selected_checkpoints"]) == list(EV.ROUNDS)
    assert contract["authenticated_artifact_count"] == len(
        EV.expected_inventory(EV.ALGORITHMS[method])
    )


def test_missing_round_checkpoint_never_substitutes_final(tmp_path: Path):
    run = _fake_run(tmp_path / "rbf", "rbf")
    (run / "ckpt_100.pt").unlink()
    with pytest.raises(FileNotFoundError, match="inventoried artifact is missing"):
        EV.validate_completed_run(run, "rbf")
    assert (run / "final.pt").is_file()


def test_algorithm_cross_loading_is_rejected(tmp_path: Path):
    run = _fake_run(tmp_path / "ensemble", "ensemble")
    with pytest.raises(RuntimeError, match="recipe algorithm"):
        EV.validate_completed_run(run, "rbf")


def test_paired_seeds_ignore_method_and_round_by_construction():
    first = EV.paired_seed("raw", 0.3, 7, 11)
    assert first == EV.paired_seed("raw", 0.3, 7, 11)
    assert first != EV.paired_seed("verified", 0.3, 7, 11)
    assert first != EV.paired_seed("raw", 0.3, 7, 12)


def test_wilson95_and_fixed_gallery_contract():
    lo, hi = EV.wilson95(10, 20)
    assert lo == pytest.approx(0.29929800819821234)
    assert hi == pytest.approx(0.7007019918017876)
    assert EV.GALLERY_INDICES == (0, 1, 2, 3, 4)
    assert EV.M == 20


def test_terminal_counts_must_partition_each_cell(tmp_path: Path):
    rows = []
    for mode in ("raw", "verified"):
        for method in ("rbf", "ensemble"):
            for round_i in EV.ROUNDS:
                for gamma in EV.GAMMAS:
                    binary = {
                        key: {"count": 0, "n": EV.M, "estimate": 0.0, "wilson95": [0, 1]}
                        for key in ("SR", "CR", "NVP", "timeout", "V_safe", "V_full", "collision", "OOB")
                    }
                    binary["timeout"]["count"] = EV.M
                    rows.append({
                        "mode": mode,
                        "method": method,
                        "round": round_i,
                        "scope": "gamma",
                        "gamma": gamma,
                        "n": EV.M,
                        "binary": binary,
                    })
                rows.append({"mode": mode, "method": method, "round": round_i, "scope": "pooled", "gamma": None})
    cells = tmp_path / "cells"
    for index in range(2 * 2 * len(EV.ROUNDS) * len(EV.GAMMAS)):
        (cells / f"c{index}").mkdir(parents=True)
        (cells / f"c{index}" / "x.npz").write_bytes(b"x")
        (cells / f"c{index}" / "x.provenance.json").write_text("{}")
    EV._authenticate_output_cells(tmp_path, rows)
