"""Phase 1+2: online privileged-controller qualification and faithful D_MPC.

Phase 1 runs the EXACT historical privileged deterministic SFM controller
(``privileged_sfm_config`` @0e0eca2 driving the unmodified
``kazuki_sfm_deploy``: flow/guidance/MPPI nominal pool, brake/goal/avoidance/
constant-acceleration templates, privileged candidate-specific SFM
simulation, H10 hard margin + recoverability, H20 viability/escalation,
stagnation/escape, always_select=True, execute-one-action-and-replan) on the
fixed CRN teacher bank, reporting SR/CR/timeout/executed-window Validity/
clearance/time.  Compact SOCP is audit-only.

Phase 2 keeps ONLY successful controller episodes and builds ``D_MPC`` from
the sequence of ACTUALLY EXECUTED first actions: contexts are reconstructed
offline exactly as evaluation-time contexts (deterministic pedestrian replay
verified against the rollout's stored arrays, fail-closed), H10 windows are
sliding windows over executed actions, terminal prefixes shorter than H10
are omitted, and every window carries step-filter/selection/clearance
provenance plus an audit-only compact-SOCP label.  D_MPC never enters D,
D+/D-, the GP, acquisition, or beta calibration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os

import numpy as np
import torch

import _paths  # noqa: F401


def _rollout_gamma(payload):
    (checkpoint, ep0, m_per_gamma, gamma, device) = payload
    import grid_feats as GF
    import grid_policy_sfm as GPS
    import claude_mpc_pool as MP
    import sfm_b1_offline_eval as OE_EVAL
    import sfm_hp_history as HH
    import sfm_kazuki as KZ
    import sfm_metrics2 as SM
    import sfm_protocol as SP
    import sfm_scene as SS

    environment = SS.scene_profile("double_density_velocity_ood")
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    config = MP.privileged_sfm_config()
    rows, lineages = [], []
    for episode in range(int(ep0), int(ep0) + int(m_per_gamma)):
        rollout = KZ.kazuki_sfm_deploy(
            policy, episode, float(gamma), cfg=config,
            n_ped=environment["n_ped"], T=int(SP.T), reach=0.5,
            device=device,
            ped_speed_range=tuple(environment["ped_speed_range"]),
            sample_seed=700_000, collect_diagnostics=True,
        )
        success = bool(rollout["success"])
        steps = int(rollout["steps"])
        row = {
            "episode": int(episode), "gamma": float(gamma),
            "status": ("success" if success else
                       "collision" if rollout["collision"] else "timeout"),
            "success": success,
            "collision": bool(rollout["collision"]),
            "timeout": bool(not success and not rollout["collision"]),
            "steps": steps,
            "time_to_goal": steps * SS.DT if success else None,
            "min_clearance": float(rollout["min_clear"]),
            "successful_clearance": (
                float(rollout["min_clear"]) if success else None
            ),
            "states": np.asarray(rollout["states"], np.float32),
            "controls": np.asarray(rollout["controls"], np.float32),
            "ped_xy": np.asarray(rollout["peds"], np.float32),
            "ped_vel": np.asarray(rollout["ped_vels"], np.float32),
        }
        validity = OE_EVAL._verify_executed_episode(row)
        compact = {k: row[k] for k in row if k not in (
            "states", "controls", "ped_xy", "ped_vel",
        )}
        compact.update(validity)
        rows.append(compact)
        if not success or steps < 10:
            continue
        # --- Phase 2: faithful context reconstruction (fail-closed) ---
        humans = SS.make_humans(
            episode, 0, environment["n_ped"],
            tuple(environment["ped_speed_range"]),
        )
        state = np.zeros(4, np.float32)
        history = HH.HpHistory()
        contexts = []
        trace = list(rollout.get("trace") or ())
        for t in range(steps):
            ped_xy, ped_vel = SS.collect_humans(humans)
            if not (
                np.allclose(ped_xy, row["ped_xy"][t], atol=1e-4)
                and np.allclose(state, row["states"][t], atol=1e-4)
            ):
                raise RuntimeError(
                    f"provenance mismatch reconstructing s{episode} "
                    f"g{gamma} t{t}"
                )
            obstacles = np.concatenate([
                ped_xy,
                np.full((len(ped_xy), 1), SS.R_PED, np.float32),
            ], axis=1)
            hp10 = history.append(torch.as_tensor(GF.axis_grid(
                state[:2], obstacles, 0.0, R=SS.R_SENSE,
                sensing=SS.R_SENSE,
            )))
            low = GF.low5(state, SS.GOAL, float(gamma))
            hist = GF.hist_pad(
                row["controls"][max(0, t - 16):t]
                if t else np.zeros((0, 2)), 16,
            )
            diag = trace[t] if t < len(trace) else {}
            step_filter = diag.get("output_filter") or {}
            contexts.append(dict(
                step=t, state=state.copy(),
                hp10=hp10.numpy().astype(np.float32),
                low5=np.asarray(low, np.float32),
                hist=np.asarray(hist, np.float32),
                ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
                selection_reason=step_filter.get("selection_reason"),
                filter_feasible=step_filter.get("filter_feasible"),
                horizon_clear=step_filter.get("selected_horizon_clear"),
            ))
            action = row["controls"][t]
            state = state.copy()
            state[:2] += SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
            state[2:4] += SS.DT * action
            SS.advance_humans(humans, state)
        windows = []
        for start in range(steps - 10 + 1):
            controls = row["controls"][start:start + 10]
            audit = SM.verify_query(
                contexts[start]["state"], controls,
                contexts[start]["ped_xy"], contexts[start]["ped_vel"],
                float(gamma),
            )
            windows.append(dict(
                start=start,
                controls=np.asarray(controls, np.float32),
                context=contexts[start],
                socp_audit_only=dict(
                    resolved=bool(audit.get("resolved")),
                    y=int(audit.get("y", 0)) if audit.get("resolved") else None,
                ),
            ))
        lineages.append(dict(
            episode=int(episode), gamma=float(gamma), steps=steps,
            windows=windows,
        ))
    return rows, lineages


def run(args):
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(outdir)
    os.makedirs(outdir)
    import sfm_b1_offline_eval as OE_EVAL
    import sfm_b1_offline_store as OS
    import sfm_protocol as SP

    payloads = [
        (os.path.abspath(args.checkpoint), int(args.ep0),
         int(args.m_per_gamma), float(gamma), args.device)
        for gamma in SP.GAMMAS
    ]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(7, int(args.workers)), mp_context=context,
    ) as executor:
        results = list(executor.map(_rollout_gamma, payloads))
    rows = [row for cell_rows, _ in results for row in cell_rows]
    lineages = [l for _, cell_lineages in results for l in cell_lineages]
    summary = OE_EVAL.summarize(rows, seed=int(args.ep0))
    OE_EVAL._assert_zero_verifier_errors(summary)
    windows_total = sum(len(l["windows"]) for l in lineages)
    per_gamma = {}
    audit_positive = 0
    for lineage in lineages:
        key = str(lineage["gamma"])
        per_gamma[key] = per_gamma.get(key, 0) + len(lineage["windows"])
        audit_positive += sum(
            1 for w in lineage["windows"]
            if w["socp_audit_only"]["y"] == 1
        )
    buffer = dict(
        status="CORRECTED_D_MPC_BUFFER_COMPLETE",
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=OS.sha256_file(args.checkpoint),
        bank=dict(ep0=int(args.ep0), m_per_gamma=int(args.m_per_gamma)),
        controller="privileged_sfm_config @0e0eca2 via unmodified kazuki_sfm_deploy",
        retention="successful episodes only; H10 windows over EXECUTED actions; terminal prefixes < H10 omitted",
        separation="never enters D, D+/D-, GP, acquisition, or beta calibration",
        lineages=lineages,
        n_lineages=len(lineages),
        n_windows=windows_total,
        windows_per_gamma=per_gamma,
        socp_audit_positive_fraction=(
            audit_positive / windows_total if windows_total else None
        ),
    )
    torch.save(buffer, os.path.join(outdir, "D_MPC_corrected.pt"))
    report = dict(
        status="PRIV_ONLINE_QUALIFICATION_COMPLETE",
        controller_summary=summary,
        rows=rows,
        n_lineages=len(lineages),
        n_windows=windows_total,
        windows_per_gamma=per_gamma,
        socp_audit_positive_fraction=buffer[
            "socp_audit_positive_fraction"
        ],
    )
    OE_EVAL._write_json(os.path.join(outdir, "qualification.json"), report)
    pooled = summary["pooled"]
    print(json.dumps(dict(
        SR=pooled["SR"], CR=pooled["CR"], timeout=pooled["timeout"],
        Validity=pooled["Validity"]["mean"],
        clearance=pooled["successful_clearance"]["mean"],
        time=pooled["successful_time_to_goal"]["mean"],
        lineages=len(lineages), windows=windows_total,
        audit_positive_fraction=buffer["socp_audit_positive_fraction"],
    ), allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ep0", type=int, required=True)
    parser.add_argument("--m-per-gamma", type=int, default=10)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
