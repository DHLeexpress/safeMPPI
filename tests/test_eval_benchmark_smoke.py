import json
import subprocess
import sys


def test_eval_benchmark_doubleintegrator_smoke(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "cfm_mppi.evaluation.eval_benchmark",
        "--dataset",
        "sfm",
        "--dynamics",
        "doubleintegrator",
        "--methods",
        "safemppi_gamma",
        "--num-episodes",
        "2",
        "--seed",
        "0",
        "--gamma-grid",
        "0.1",
        "--output-root",
        str(tmp_path),
        "--smoke",
        "--device",
        "cpu",
    ]
    subprocess.run(cmd, cwd=".", check=True)
    records = list(tmp_path.glob("*/sfm/doubleintegrator/safemppi_gamma.jsonl"))
    assert records
    rows = [json.loads(line) for line in records[0].read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["safety_guarantee_scope"] == "linear_system_theorem_relevant"
