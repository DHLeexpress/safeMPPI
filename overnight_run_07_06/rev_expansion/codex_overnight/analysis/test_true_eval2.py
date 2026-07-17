"""Contract tests for the portable TRUE evaluation (integration/afe2-terminal-dualscene-v1)."""
from __future__ import annotations

import json
import os
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


def test_wilson_and_bootstrap_intervals() -> None:
    lo, hi = TEF.wilson95(0.5, 100)
    assert 0.39 < lo < 0.5 < hi < 0.61
    assert TEF.wilson95(0.0, 0) == (0.0, 0.0)
    lo, hi = TEF.bootstrap95([1.0, 1.0, 1.0], ("k",))
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert TEF.bootstrap95([], ("k",))[0] != TEF.bootstrap95([], ("k",))[0]  # nan


def test_content_key_changes_with_inputs() -> None:
    base = dict(scene_sha256="s", checkpoint_sha256="c", seeds=[1, 2], M=2)
    k0 = TEF.content_key(base)
    assert k0 == TEF.content_key(dict(base))
    assert k0 != TEF.content_key(dict(base, checkpoint_sha256="c2"))
    assert k0 != TEF.content_key(dict(base, seeds=[1, 3]))
    assert k0 != TEF.content_key(dict(base, M=3))


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
