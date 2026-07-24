import json

import sfm_b1_offline_18arm_compare as C


def _delivery(root, selector):
    aggregate = root / "evaluation" / "aggregate"
    aggregate.mkdir(parents=True)
    rows = []
    prefix = "offline_exec" if selector == "margin" else "offline_exec_safemppi_cost"
    for alpha in (0.0, 0.01, 0.1):
        for exposure in (1, 10, 100):
            arm = (
                f"{prefix}_alpha{str(alpha).replace('.', 'p')}_"
                f"exposures{exposure:03d}"
            )
            for round_i in range(11):
                rows.append({
                    "selector": selector,
                    "arm": arm,
                    "alpha": alpha,
                    "exposure_epochs": exposure,
                    "round": round_i,
                    "SR": .5, "CR": .5, "timeout": 0.,
                    "Validity": .4, "clearance": .1, "time_to_goal": 9.,
                })
    (root / "DELIVERY_COMPLETE.json").write_text(json.dumps({
        "status": "SFM_B1_OFFLINE_9ARM_DELIVERY_COMPLETE",
        "contract": {"execution_selector": selector},
    }))
    (aggregate / "AGGREGATE_COMPLETE.json").write_text(json.dumps({
        "status": "SFM_B1_OFFLINE_9ARM_AGGREGATE_COMPLETE",
        "rows": rows,
    }))


def test_compare_requires_and_combines_paired_99_row_sweeps(tmp_path):
    margin = tmp_path / "margin"
    cost = tmp_path / "cost"
    _delivery(margin, "margin")
    _delivery(cost, "safemppi_cost")
    result = C.compare(margin, cost, tmp_path / "comparison")
    assert result["status"] == C.STATUS
    assert result["rows"] == 198
    assert result["paired_r0"]["CR"] == .5
    assert (tmp_path / "comparison" / "paired_18arm_raw_m50.png").is_file()
