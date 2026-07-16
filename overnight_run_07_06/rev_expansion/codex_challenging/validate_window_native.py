"""Independent artifact audit for the window-native four-arm sanity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "stage_results/05_window_native"
RUNS = STAGE / "runs"
ARMS = ("full", "no_socp", "no_progress", "no_curriculum")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probes(name):
    path = RUNS / f"sanity_v1_{name}" / "probe.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    checks = {}
    original = ROOT / "reference/grid_expand_hardtail.py"
    upstream = ROOT.parent / "codex_overnight/grid_expand_hardtail.py"
    checks["original_trainer_preserved"] = sha256(original) == sha256(upstream)
    assert checks["original_trainer_preserved"]
    checks["original_sha256"] = sha256(original)
    assert checks["original_sha256"] == "941f4890cb7f1fead7635c91b7075eb50d2f9999547f9bb6ff325bcdb8991a11"
    checks["active_pretrained_sha256"] = sha256(ROOT / "pretrained_sg_walls8.pt")
    assert checks["active_pretrained_sha256"] == \
        "5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2"

    tests = json.loads((ROOT / "reference/analysis/test_window_expand.json").read_text())
    assert tests["status"] == "PASS" and len(tests["checks"]) == 9
    checks["semantic_tests"] = "9/9 PASS"

    rows = {name: probes(name) for name in ARMS}
    assert all(len(items) == 6 for items in rows.values())
    assert all(item["functional_step"] > 0 and not item["rollback"]
               for items in rows.values() for item in items)
    checks["updates"] = "24/24 nonzero, 0 rollback"

    full_audits = [row["gather_audit"] for row in rows["full"]]
    assert sum(a["whole_valid2_pass"] for a in full_audits) == 0
    assert sum(a["accepted_windows"] for a in full_audits) == 3378
    assert sum(a["accepted_from_whole_invalid"] for a in full_audits) == 3378
    checks["full_window_native"] = "3378 windows from 42/42 whole-invalid2 rollouts"

    no_socp_audits = [row["gather_audit"] for row in rows["no_socp"]]
    assert sum(a["socp_evaluated"] for a in no_socp_audits) == 0
    assert sum(a["window_failures"].get("safe_space", 0) for a in no_socp_audits) == 13
    checks["no_socp"] = "0 SOCP calls; 13 unsafe local windows rejected"

    no_progress_audits = [row["gather_audit"] for row in rows["no_progress"]]
    assert sum(a["progress_evaluated"] for a in no_progress_audits) == 0
    checks["no_progress"] = "0 progress calls"

    full_budget = json.loads((RUNS / "sanity_v1_full/accepted_window_budget.json").read_text())
    no_curr_budget = json.loads((RUNS / "sanity_v1_no_curriculum/accepted_window_budget.json").read_text())
    assert full_budget == no_curr_budget
    assert all((row["batch_e"], row["batch_f"]) == (16, 0) for row in rows["no_curriculum"])
    assert all((row["batch_e"], row["batch_f"]) == (6, 10) for row in rows["full"])
    checks["no_curriculum"] = {"exact_count_match": True, "budget": full_budget,
                                "batch": "16+0 vs Full 6+10"}

    for name in ARMS:
        card = json.loads((STAGE / f"data/eval_m6/{name}/scorecard.json").read_text())
        assert card["M_per_gamma"] == 6 and len(card["rows"]) == 7
        assert card["aggregate"]["SR"] == 0.0
    checks["faithful_evaluation"] = "4 arms x 7 gamma x M=6; all SR=0"

    for artifact in (STAGE / "viz/prelim_rollouts.png", STAGE / "viz/prelim_training.png"):
        assert artifact.stat().st_size > 100_000
    assert (STAGE / "REPORT.md").stat().st_size > 1500
    checks["artifacts"] = "PASS"

    out = STAGE / "logs/validation.json"
    out.write_text(json.dumps({"status": "PASS", "checks": checks}, indent=2) + "\n")
    print(f"PASS -> {out}")


if __name__ == "__main__":
    main()
