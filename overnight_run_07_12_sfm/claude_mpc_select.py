"""Apply the pre-registered MPC-study M10 selection rule.

Collects every saved checkpoint cell (pre-block and post-block, every arm and
round) from the sweep's per-arm ``m10/`` audit directories, plus the common r0
and the no-distillation control cells, and applies the rule declared in
MPC_STUDY_PREREGISTRATION.json verbatim:

  liveness gate: SR >= SR(r0) - 0.02 and timeout <= timeout(r0) + 0.05;
  among eligible: min CR, then max Validity, then max successful clearance,
  then min successful time-to-goal;
  the winner must be a POST-BLOCK checkpoint to count as a distillation
  effect; its pre-block sibling is always reported alongside.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _pooled(path):
    with open(path) as stream:
        payload = json.load(stream)
    out = []
    for record in payload["records"]:
        cell = record["cell"]
        p = cell["summary"]["pooled"]
        out.append(dict(
            label=record["label"],
            checkpoint=cell["checkpoint"],
            checkpoint_sha256=cell["checkpoint_sha256"],
            SR=float(p["SR"]), CR=float(p["CR"]),
            timeout=float(p["timeout"]),
            Validity=float(p["Validity"]["mean"]),
            clearance=p["successful_clearance"]["mean"],
            time=p["successful_time_to_goal"]["mean"],
        ))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--r0-control-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline = _pooled(args.r0_control_metrics)
    r0 = next(row for row in baseline if row["label"] == "r0")
    control_rows = [
        dict(row, arm="control_no_distill", phase="control",
             round=int(row["label"][1:]))
        for row in baseline if row["label"] != "r0"
    ]
    cells = []
    for arm in args.arms:
        for path in sorted(glob.glob(os.path.join(
            args.sweep_root, arm, "m10", "round_*_*",
            "raw_m10_offline_metrics.json",
        ))):
            phase = "post" if path.split(os.sep)[-2].endswith("_post") else "pre"
            round_i = int(path.split(os.sep)[-2].split("_")[1])
            for row in _pooled(path):
                cells.append(dict(
                    row, arm=arm, phase=phase, round=round_i,
                ))
    gate_sr = r0["SR"] - 0.02
    gate_to = r0["timeout"] + 0.05
    eligible = [
        c for c in cells
        if c["SR"] >= gate_sr and c["timeout"] <= gate_to
        and c["clearance"] is not None and c["time"] is not None
    ]

    def key(c):
        return (
            c["CR"], -c["Validity"],
            -(c["clearance"] if c["clearance"] is not None else -1),
            c["time"] if c["time"] is not None else 1e9,
            c["round"], c["arm"],
        )

    ordered = sorted(eligible, key=key)
    post_ordered = [c for c in ordered if c["phase"] == "post"]
    winner = post_ordered[0] if post_ordered else None
    sibling = None
    if winner:
        sibling = next(
            (c for c in cells
             if c["arm"] == winner["arm"] and c["round"] == winner["round"]
             and c["phase"] == "pre"),
            None,
        )
    payload = dict(
        status="MPC_M10_SELECTION_APPLIED",
        rule="MPC_STUDY_PREREGISTRATION.json verbatim",
        bank=dict(ep0=350000, noise_seed=20260733, m_per_gamma=10),
        r0=r0,
        liveness_gate=dict(SR_min=gate_sr, timeout_max=gate_to),
        n_cells=len(cells), n_eligible=len(eligible),
        winner_post_block=winner,
        pre_block_sibling=sibling,
        top_eligible=ordered[:8],
        control_no_distill=control_rows,
        all_cells=sorted(
            cells + control_rows,
            key=lambda c: (c["arm"], c["round"], c.get("phase", "")),
        ),
    )
    with open(args.out, "w") as stream:
        json.dump(payload, stream, indent=1, allow_nan=False)
    print(json.dumps(dict(
        r0={k: r0[k] for k in ("SR", "CR", "Validity")},
        winner=None if winner is None else {
            k: winner[k]
            for k in ("arm", "round", "phase", "SR", "CR", "Validity",
                      "clearance", "time")
        },
        sibling=None if sibling is None else {
            k: sibling[k]
            for k in ("SR", "CR", "Validity", "clearance", "time")
        },
        eligible=len(eligible),
    ), indent=1))


if __name__ == "__main__":
    main()
