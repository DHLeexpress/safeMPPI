#!/usr/bin/env python3
"""Independent artifact audit for the bounded Stage 4--6 sanity run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
S4 = HERE / "stage_results/04_canonical"
S5 = HERE / "stage_results/05_sanity"
S6 = HERE / "stage_results/06_baselines"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(directory: Path):
    return [json.loads((directory / f"row_g{float(gamma)}.json").read_text()) for gamma in GAMMAS]


def main():
    checks = {}
    active = HERE / "pretrained_sg_walls8.pt"
    checks["active_pretrained_sha256"] = sha256(active)
    assert checks["active_pretrained_sha256"] == "5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2"
    checkpoint = torch.load(active, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["ctx_dim"] == 37 and not checkpoint["config"]["raw_start_goal"]

    gates = json.loads((S5 / "logs/test_hardtail_trainer.json").read_text())
    signed = json.loads((S5 / "logs/test_signed_negative.json").read_text())
    assert gates["total_pass"] == 20 and gates["total_fail"] == 0
    assert signed["status"] == "PASS" and signed["alpha"] == 5e-4
    checks["trainer_gates"] = "20/20 + signed-negative PASS"

    s4 = rows(S4 / "data/pretrained_m6")
    assert all(row["M"] == 6 and row["n_success"] == 0 and row["CR"] == 1.0 for row in s4)
    checks["stage4"] = "0/42 success; CR=1.0"

    strict_names = ("probe_beta_0.05", "probe_beta_0.1", "probe_unfrozen_b02_a0", "probe_beta_0.4")
    strict_attempts = strict_valid = 0
    for name in strict_names:
        record = json.loads((S5 / "runs" / name / "probe.jsonl").read_text().splitlines()[-1])
        strict_attempts += int(record["att"]); strict_valid += int(record["vr"])
    assert strict_attempts == 44 and strict_valid == 0
    checks["strict_cold_start"] = "0/44 exact-valid"

    seed = torch.load(S5 / "data/canonical_seed_windows.pt", map_location="cpu", weights_only=False)
    assert seed["exact_valid2"] and len(seed["U"]) == 819
    assert set(seed["gamma_id"].unique().tolist()) == set(range(7))
    checks["canonical_seed"] = "819 exact-valid windows; all 7 gamma"

    def valid_series(name):
        history = json.loads((S5 / "runs" / name / "history.json").read_text())[1:]
        return [int(row["valid_rollouts"]) for row in history]

    assert valid_series("hope_exact_b02_a0") == [1, 1, 0]
    assert valid_series("hope_exact_b02_a00005") == [1, 1, 1]
    checks["exact_support"] = {"alpha0": [1, 1, 0], "alpha5e-4": [1, 1, 1]}
    for directory in ("eval_seed_unfrozen_m6", "eval_hope_a0_m6", "eval_hope_atiny_m6"):
        result = rows(S5 / "data" / directory)
        assert all(row["n_success"] == 0 and row["CR"] == 1.0 for row in result)
    checks["faithful_stage5"] = "all reported variants 0/42; CR=1.0"

    expert = rows(S6 / "results/expert_m6")
    assert all(row["M"] == 6 and row["n_success"] == 6 and row["CR"] == 0.0 for row in expert)
    lucky = json.loads((S6 / "results/kazuki_best_row/row_g0.5.json").read_text())
    assert lucky["n_success"] == 1 and lucky["M"] == 6
    checks["stage6"] = "expert 42/42; Kazuki sensitivity 1/6"

    figures = (
        S4 / "viz/canonical_pretrained_m6.png",
        S5 / "viz/stage5_sanity_dashboard.png",
        S6 / "viz/stage6_rollout_comparison.png",
    )
    for figure in figures:
        assert figure.exists() and figure.stat().st_size > 100_000
    checks["figures"] = {str(path.relative_to(HERE)): path.stat().st_size for path in figures}
    for report in (S4 / "REPORT.md", S5 / "SANITY_REPORT.md", S6 / "REPORT.md"):
        assert report.exists() and report.stat().st_size > 300

    payload = {"status": "PASS", "checks": checks}
    output = S5 / "logs/stage456_validation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
