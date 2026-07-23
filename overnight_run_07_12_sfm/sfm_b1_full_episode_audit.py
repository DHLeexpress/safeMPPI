"""Diagnostic-only full-episode B1 gathering with explicit post-NVP continuation.

This module does not alter the fail-closed B1 trainer.  It starts from the
pretrained policy, runs the ordinary K=16/B=4 RBF acquisition and max-margin
selector, and records every resolved query.  When the selected B queries contain
no admissible action, an independently sampled raw temperature-one window is
verified and its first action is executed so the simulator can continue.

That post-NVP transition is evidence gathering, not certified deployment.  The
trace keeps the full-H verifier label, nominal-Hp gate, NVP event, progress/trap
event, and episode outcome separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_rbf as BR
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


DEFAULT_SCENARIOS = (250_001, 250_003, 250_007)
DEFAULT_ELL = 0.24210826720721101
DEFAULT_SAMPLE_SEED = 700_000
DEFAULT_AUDIT_SEED = 20260723
TRAP_HORIZON = 10
TRAP_DISPLACEMENT = 0.2


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _save_torch(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip())
    return dict(commit=commit, tracked_worktree_clean=not dirty)


def _raw_windows(policy, live, batch, generators):
    """One raw latent per live cell, preserving raw-evaluator per-cell streams."""
    context = policy.ctx_from(batch["hp10"], batch["low"], batch["hist"])
    latents = torch.stack([
        torch.randn(
            policy.d, generator=generators[(replica.scenario_id, replica.gamma)],
            device=context.device, dtype=context.dtype,
        )
        for replica in live
    ])
    return BE.integrate_latents(policy, latents, context, nfe=8)


def _trap(states, *, horizon=TRAP_HORIZON, displacement=TRAP_DISPLACEMENT):
    if len(states) <= int(horizon):
        return False
    start = np.asarray(states[-int(horizon) - 1], float)[:2]
    end = np.asarray(states[-1], float)[:2]
    return bool(np.linalg.norm(end - start) < float(displacement))


def _result_label(result):
    if not result.get("resolved"):
        return "verifier_error"
    if int(result.get("y", 0)) == 0:
        return "verifier_negative"
    if not bool(result.get("full_h")) or int(result.get("terminal_step", -1)) != SP.H:
        raise RuntimeError("the full-episode audit requires full-H=10 verifier semantics")
    return "verifier_positive"


def _post_action_terminal(replica):
    ped_xy, _ = SS.collect_humans(replica.humans)
    clearance = float(
        np.linalg.norm(ped_xy - replica.state[:2][None], axis=1).min() - SS.R_PED
    )
    replica.minimum_clearance = min(replica.minimum_clearance, clearance)
    collision = clearance < 0.0
    success = bool(
        not collision and float(np.linalg.norm(replica.state[:2] - SS.GOAL)) < 0.5
    )
    if collision:
        replica.alive = False
        replica.status = "collision"
    elif success:
        replica.alive = False
        replica.status = "success"
    return collision, success, clearance


def collect(
        checkpoint, *, scenarios=DEFAULT_SCENARIOS, gammas=SS.GAMMAS,
        scene_profile="double_density_velocity_ood", device="cuda",
        verifier_workers=32, sample_seed=DEFAULT_SAMPLE_SEED,
        audit_seed=DEFAULT_AUDIT_SEED, ell=DEFAULT_ELL, T=SP.T, outdir,
):
    """Collect a fixed scenario-by-gamma full-episode diagnostic bundle."""
    scenarios = tuple(map(int, scenarios))
    gammas = tuple(map(float, gammas))
    if len(scenarios) != 3 or len(set(scenarios)) != 3:
        raise ValueError("the requested audit requires exactly three distinct scenarios")
    if gammas != tuple(map(float, SS.GAMMAS)):
        raise ValueError(f"the requested audit requires all gammas={SS.GAMMAS}")
    if scene_profile != "double_density_velocity_ood":
        raise ValueError("this audit is pinned to the authenticated double-shift OOD")
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse audit output: {outdir}")

    environment = SS.scene_profile(scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)

    replicas = [
        BX.Replica(
            scenario, gamma, n_ped=environment["n_ped"],
            ped_speed_range=tuple(environment["ped_speed_range"]),
        )
        for scenario in scenarios for gamma in gammas
    ]
    cfg = BX.ArmConfig(
        name="A", selector="margin", alpha=0.0, rounds=1,
        scene_profile=scene_profile, verifier_workers=int(verifier_workers),
        seed=int(audit_seed),
    ).validate()
    gp = BR.RBFGP(float(ell), cfg.gp_lam)
    beta, calibrated_ess = BX._initial_beta(
        phi_policy, gp, replicas, cfg, device, int(audit_seed) + 1009,
    )
    audit_generator = torch.Generator(device=device).manual_seed(int(audit_seed) + 2003)
    raw_generators = {
        (replica.scenario_id, replica.gamma):
            torch.Generator(device=device).manual_seed(
                int(sample_seed) + replica.scenario_id * 1000
            )
        for replica in replicas
    }

    traces = []
    counts = Counter()
    ess_values = []
    sigma_pool, sigma_selected = [], []
    trap_active = defaultdict(bool)
    with ProcessPoolExecutor(max_workers=int(verifier_workers)) as executor:
        for step in range(int(T)):
            live = [replica for replica in replicas if replica.alive]
            live, batch = BX._stack_prepared(live, device)
            if not live:
                break
            with torch.no_grad():
                audit_windows = BE.generate_windows(
                    policy, batch["hp10"], batch["low"], batch["hist"],
                    K=cfg.K, nfe=cfg.nfe, temp=cfg.temp,
                    generator=audit_generator,
                )
                raw_windows = _raw_windows(policy, live, batch, raw_generators)
                features = BX._features(phi_policy, audit_windows, batch, cfg.phi_s)

            selected_by_context = []
            acquisition_by_context = []
            for context_index in range(len(live)):
                selected, acquisition = gp.sequential_acquire(
                    features[context_index], cfg.B, beta, generator=audit_generator,
                )
                selected_by_context.append(selected)
                acquisition_by_context.append(acquisition)
                sigma_pool.extend(map(
                    float, gp.acquisition_sigma(features[context_index]).detach().cpu()
                ))
                sigma_selected.extend(float(row["chosen_sigma"]) for row in acquisition)
                ess_values.extend(float(row["ess_norm"]) for row in acquisition)

            tasks = []
            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                for candidate_id in selected_by_context[context_index]:
                    tasks.append((
                        context_index, candidate_id, prepared["state"],
                        audit_windows[context_index, candidate_id].detach().cpu().numpy(),
                        prepared["ped_xy"], prepared["ped_vel"], replica.gamma,
                    ))
            results = list(executor.map(SM.verify_in_worker, tasks))
            by_context = defaultdict(dict)
            for context_index, candidate_id, result in results:
                by_context[int(context_index)][int(candidate_id)] = result

            prepared_contexts = []
            raw_tasks = []
            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                pedestrian_prediction = SM.predict_pedestrians(
                    prepared["ped_xy"], prepared["ped_vel"], cfg.H,
                )
                all_rows = []
                for candidate_id in range(cfg.K):
                    controls = audit_windows[
                        context_index, candidate_id
                    ].detach().cpu().numpy()
                    segment = SM.rollout_positions(prepared["state"], controls)
                    all_rows.append(dict(
                        candidate_id=candidate_id, controls=controls, segment=segment,
                        mode=BE.classify_candidate(segment, pedestrian_prediction),
                    ))
                query_rows = []
                for acquisition_step, candidate_id in enumerate(
                        selected_by_context[context_index]):
                    result = by_context[context_index][candidate_id]
                    query_rows.append(dict(
                        candidate_id=int(candidate_id),
                        controls=all_rows[candidate_id]["controls"],
                        result=result, mode=all_rows[candidate_id]["mode"],
                        acquisition_step=int(acquisition_step),
                        sigma=float(
                            acquisition_by_context[context_index][
                                acquisition_step
                            ]["chosen_sigma"]
                        ),
                    ))
                    counts[f"B_{_result_label(result)}"] += 1
                chosen = BC.select_admissible(
                    query_rows, selector="margin", state=prepared["state"],
                    ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"],
                    gamma=replica.gamma,
                )
                prepared_contexts.append((all_rows, query_rows, chosen))
                if chosen is None:
                    raw_tasks.append((
                        context_index, -1, prepared["state"],
                        raw_windows[context_index].detach().cpu().numpy(),
                        prepared["ped_xy"], prepared["ped_vel"], replica.gamma,
                    ))

            for context_index, candidate_id, result in executor.map(
                    SM.verify_in_worker, raw_tasks):
                by_context[int(context_index)][int(candidate_id)] = result

            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                all_rows, query_rows, chosen = prepared_contexts[context_index]

                raw_controls = raw_windows[context_index].detach().cpu().numpy()
                nvp_context = chosen is None
                if chosen is None:
                    raw_result = by_context[context_index][-1]
                    raw_margin, raw_hp_old, raw_hp_new = BC.nominal_hp_margin(
                        prepared["state"], raw_controls[0], prepared["ped_xy"],
                        replica.gamma,
                    )
                    raw_admissible = bool(
                        raw_result.get("resolved")
                        and int(raw_result.get("y", 0)) == 1
                        and bool(raw_result.get("full_h"))
                        and raw_margin >= -1.0e-9
                    )
                    executed_controls = raw_controls
                    executed_result = raw_result
                    executed_id = None
                    execution_source = (
                        "certified_raw_rescue" if raw_admissible
                        else "uncertified_raw_continuation"
                    )
                    counts["B_NVP_context"] += 1
                    counts[f"raw_continuation_{_result_label(raw_result)}"] += 1
                    raw_candidate = dict(
                        controls=np.asarray(raw_controls, np.float32),
                        result=raw_result, hp_margin=float(raw_margin),
                        hp_old=float(raw_hp_old), hp_new=float(raw_hp_new),
                        admissible=raw_admissible,
                    )
                else:
                    executed_controls = chosen["controls"]
                    executed_result = chosen["result"]
                    executed_id = int(chosen["candidate_id"])
                    execution_source = "verified_max_margin"
                    raw_candidate = None
                executed_label = _result_label(executed_result)
                counts[f"executed_{executed_label}"] += 1
                counts[f"source_{execution_source}"] += 1

                before = prepared["state"].copy()
                BX._advance(replica, executed_controls[0])
                trap_event = _trap(replica.states)
                trap_key = (replica.scenario_id, replica.gamma)
                trap_entry = bool(trap_event and not trap_active[trap_key])
                trap_active[trap_key] = bool(trap_event)
                collision_after, success_after, clearance_after = _post_action_terminal(
                    replica
                )
                if trap_entry:
                    counts["trap_entries"] += 1
                if collision_after:
                    counts["collision_events"] += 1
                if success_after:
                    counts["success_events"] += 1

                negative_reasons = []
                if nvp_context:
                    negative_reasons.append("finite_B_NVP")
                if executed_label == "verifier_negative":
                    negative_reasons.append("executed_full_H_rejected")
                elif executed_label == "verifier_error":
                    negative_reasons.append("executed_verifier_error")
                if (
                    executed_label == "verifier_positive"
                    and execution_source != "verified_max_margin"
                ):
                    if raw_margin < -1.0e-9:
                        negative_reasons.append("executed_nominal_Hp_gate_failure")
                if trap_event:
                    negative_reasons.append("ten_step_progress_below_0p2m")
                if collision_after:
                    negative_reasons.append("collision")

                traces.append(dict(
                    round=1, step=int(step), scenario_id=replica.scenario_id,
                    gamma=replica.gamma, state=before,
                    next_state=replica.state.copy(),
                    ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"],
                    all_K=all_rows,
                    selected_ids=list(map(int, selected_by_context[context_index])),
                    query_rows=query_rows, acquisition=acquisition_by_context[context_index],
                    executed_id=executed_id,
                    executed_controls=np.asarray(executed_controls, np.float32),
                    executed_result=executed_result,
                    executed_label=executed_label,
                    execution_source=execution_source,
                    nvp_context=bool(nvp_context),
                    raw_candidate=raw_candidate,
                    trap_event=bool(trap_event),
                    trap_entry=bool(trap_entry),
                    collision_after_action=bool(collision_after),
                    success_after_action=bool(success_after),
                    clearance_after_action=float(clearance_after),
                    negative_reasons=negative_reasons,
                ))

    for replica in replicas:
        if replica.alive:
            replica.alive = False
            replica.status = "timeout"
    outcomes = [dict(
        scenario_id=replica.scenario_id, gamma=replica.gamma,
        status=replica.status, steps=len(replica.controls),
        success=replica.status == "success",
        collision=replica.status == "collision",
        timeout=replica.status == "timeout",
        minimum_clearance=float(replica.minimum_clearance),
    ) for replica in replicas]
    counts.update(f"outcome_{row['status']}" for row in outcomes)

    source = _source()
    bundle = dict(
        version=1, status="SFM_B1_FULL_EPISODE_LABEL_AUDIT_COMPLETE",
        diagnostic_only=True, enters_training_or_gp=False,
        certified_deployment=False,
        continuation_semantics=(
            "verified max-margin B action when available; otherwise independently "
            "sampled raw temp=1 action is executed after being labeled, even when "
            "uncertified, solely to continue the offline simulator diagnostic"
        ),
        label_semantics=dict(
            safety="executed_label is the independent exact full-H=10 verifier result",
            nvp="finite B=4 acquisition event; not itself a verifier-negative label",
            trap=(
                f"separate performance event: displacement over {TRAP_HORIZON} "
                f"executed actions is below {TRAP_DISPLACEMENT} m"
            ),
            collision="episode event; never retroactively relabels earlier windows",
        ),
        source=source, checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=_sha256_file(checkpoint),
        environment=environment, scenarios=list(scenarios), gammas=list(gammas),
        sample_seed=int(sample_seed), audit_seed=int(audit_seed),
        protocol=dict(
            K=cfg.K, B=cfg.B, H=cfg.H, T=int(T), selector="margin",
            ell=float(ell), gp_buffer=0, beta=float(beta),
            calibrated_ess_over_K=float(calibrated_ess),
            realized_ess_over_K=float(np.mean(ess_values)),
            acquisition=BR.acquisition_diagnostics(sigma_pool, sigma_selected),
        ),
        counts=dict(counts), outcomes=outcomes, traces=traces,
    )
    os.makedirs(outdir)
    trace_path = os.path.join(outdir, "full_episode_label_audit.pt")
    _save_torch(trace_path, bundle)
    manifest = {
        key: value for key, value in bundle.items() if key != "traces"
    }
    manifest["trace_path"] = os.path.abspath(trace_path)
    manifest["trace_sha256"] = _sha256_file(trace_path)
    _write_json(os.path.join(outdir, "full_episode_label_audit.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scenarios", nargs=3, type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verifier-workers", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--ell", type=float, default=DEFAULT_ELL)
    parser.add_argument("--T", type=int, default=SP.T)
    args = parser.parse_args(argv)
    collect(
        args.checkpoint, scenarios=args.scenarios,
        scene_profile=args.scene_profile, device=args.device,
        verifier_workers=args.verifier_workers,
        sample_seed=args.sample_seed, audit_seed=args.audit_seed,
        ell=args.ell, T=args.T, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
