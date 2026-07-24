"""Offline executed-window SFM expansion.

B=4 exact verifier queries are used only to search for the next action.  One
H=10 plan per context enters D: the plan whose first action was executed.  At a
finite-B NVP context, an independent raw temperature-one plan is exact-verified
and executed even when negative so that the offline simulator can continue.

This is an offline data collector, not a certified deployment controller.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


EXPECTED_CHECKPOINT_SHA256 = (
    "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
)
ELL = 0.24210826720721101
CAP = 512
GP_LAMBDA = 1.0e-2
ALPHAS = (0.0, 0.01, 0.1)
EXPOSURE_EPOCHS = (1, 10, 100)
SCENE_PROFILE = "double_density_velocity_ood"


@dataclass(frozen=True)
class OfflineConfig:
    alpha: float
    exposure_epochs: int
    rounds: int = 10
    K: int = 16
    B: int = 4
    T: int = 180
    H: int = 10
    batch: int = 128
    lr: float = 1.0e-4
    ess_target: float = 0.5
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    gp_lam: float = GP_LAMBDA
    verifier_workers: int = 8
    seed: int = 20260724
    scene_profile: str = SCENE_PROFILE
    smoke: bool = False

    def validate(self):
        if float(self.alpha) not in ALPHAS:
            raise ValueError(f"alpha must be one of {ALPHAS}")
        if int(self.exposure_epochs) not in EXPOSURE_EPOCHS:
            raise ValueError(f"exposure_epochs must be one of {EXPOSURE_EPOCHS}")
        if (
            int(self.K), int(self.B), int(self.T), int(self.H),
            int(self.batch), float(self.lr), float(self.ess_target),
            float(self.gp_lam), self.scene_profile,
        ) != (
            16, 4, 180, 10, 128, 1.0e-4, 0.5,
            GP_LAMBDA, SCENE_PROFILE,
        ):
            raise ValueError("offline executed-window scientific contract changed")
        expected_rounds = 1 if self.smoke else 10
        if int(self.rounds) != expected_rounds:
            raise ValueError(
                f"rounds must be {expected_rounds} when smoke={self.smoke}"
            )
        if int(self.verifier_workers) < 1:
            raise ValueError("verifier_workers must be positive")
        return self

    @property
    def arm_name(self):
        alpha = str(float(self.alpha)).replace(".", "p")
        return (
            f"offline_exec_alpha{alpha}_"
            f"exposures{int(self.exposure_epochs):03d}"
        )


def _write_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _source():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True,
    ).strip())
    return dict(commit=commit, tracked_worktree_clean=not dirty)


def _keyed_seed(base, *parts):
    payload = json.dumps(
        [int(base), *parts], separators=(",", ":"), sort_keys=False,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2 ** 63 - 1
    )


@torch.no_grad()
def _keyed_windows(
    policy, live, batch, *, K, round_i, step, source, seed, nfe, temp,
):
    contexts = policy.ctx_from(batch["hp10"], batch["low"], batch["hist"])
    latent_parts = []
    for replica in live:
        generator = np.random.default_rng(_keyed_seed(
            seed, int(round_i), int(replica.scenario_id),
            f"{float(replica.gamma):.8f}", int(step), str(source),
        ))
        latent_parts.append(generator.standard_normal(
            (int(K), int(policy.d)), dtype=np.float32,
        ))
    latents = torch.as_tensor(
        np.stack(latent_parts),
        device=contexts.device,
        dtype=contexts.dtype,
    ) * float(temp)
    expanded = contexts.repeat_interleave(int(K), dim=0)
    windows = BE.integrate_latents(
        policy, latents.reshape(-1, policy.d), expanded, nfe=int(nfe),
    )
    return windows.reshape(
        len(live), int(K), int(policy.H_pred), 2,
    ), contexts


def _gamma_balanced_records(previous, *, cap, round_i, seed):
    if previous is None:
        return [], dict(
            requested_cap=int(cap), selected=0, quota=int(cap) // len(SP.GAMMAS),
            rotating_extra_gamma=None, per_gamma={str(gamma): 0 for gamma in SP.GAMMAS},
            shortfall={str(gamma): int(cap) // len(SP.GAMMAS) for gamma in SP.GAMMAS},
        )
    groups = {}
    for gamma_index, gamma in enumerate(SP.GAMMAS):
        records = [
            (previous, row)
            for row in previous.Dplus
            if round(
                float(previous.contexts[int(row["context_id"])]["gamma"]), 8
            ) == round(float(gamma), 8)
        ]
        groups[float(gamma)] = BS.hierarchical_order(
            records, int(seed) + gamma_index,
        )
    quota = int(cap) // len(SP.GAMMAS)
    rotation = (int(round_i) - 2) % len(SP.GAMMAS)
    extra_gamma = float(SP.GAMMAS[rotation])
    required = {
        float(gamma): quota + int(float(gamma) == extra_gamma)
        for gamma in SP.GAMMAS
    }
    shortfall = {
        str(gamma): max(0, required[float(gamma)] - len(groups[float(gamma)]))
        for gamma in SP.GAMMAS
    }
    if any(shortfall.values()):
        raise RuntimeError(
            "strict gamma-balanced GP quota is unavailable; "
            f"required={required}, shortfall={shortfall}"
        )
    selected = [
        record
        for gamma in SP.GAMMAS
        for record in groups[float(gamma)][:required[float(gamma)]]
    ]
    per_gamma = Counter(
        str(previous.contexts[int(row["context_id"])]["gamma"])
        for _, row in selected
    )
    identities = [
        (int(shard.round_i), int(row["window_id"])) for shard, row in selected
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("GP buffer selection contains duplicates")
    if len(selected) != int(cap):
        raise RuntimeError(
            f"expected {cap} GP records, selected {len(selected)}"
        )
    return selected, dict(
        requested_cap=int(cap),
        selected=len(selected),
        quota=quota,
        rotating_extra_gamma=extra_gamma,
        per_gamma={str(gamma): int(per_gamma[str(gamma)]) for gamma in SP.GAMMAS},
        shortfall=shortfall,
        unique=True,
    )


@torch.no_grad()
def gp_from_previous(
    phi_policy, previous, *, round_i, ell, cap, lam, phi_s, device, seed,
):
    selected, selection = _gamma_balanced_records(
        previous, cap=cap, round_i=round_i, seed=seed,
    )
    gp = BR.RBFGP(float(ell), float(lam))
    if selected:
        feature_parts = []
        for start in range(0, len(selected), 256):
            hp10, low, hist, controls = BX._record_batch(
                selected[start:start + 256], device,
            )
            feature_parts.append(phi_policy.phi_s(
                controls,
                phi_policy.ctx_from(hp10, low, hist),
                s=float(phi_s),
            ))
        gp.set_buffer(torch.cat(feature_parts))
    identities = [
        (int(shard.round_i), int(row["window_id"])) for shard, row in selected
    ]
    return gp, identities, selection


@torch.no_grad()
def _calibrate_beta(
    phi_policy, gp, replicas, cfg, device, *, round_i,
):
    live, batch = BX._stack_prepared(replicas, device)
    windows, _ = _keyed_windows(
        phi_policy, live, batch, K=cfg.K, round_i=round_i, step=-1,
        source="beta_calibration", seed=cfg.seed, nfe=cfg.nfe, temp=cfg.temp,
    )
    features = BX._features(phi_policy, windows, batch, cfg.phi_s)
    vectors = []
    for index, (replica, values) in enumerate(zip(live, features)):
        generator = torch.Generator(device=values.device).manual_seed(
            _keyed_seed(
                cfg.seed, round_i, replica.scenario_id,
                f"{replica.gamma:.8f}", "beta_order",
            )
        )
        order = torch.randperm(
            len(values), generator=generator, device=values.device,
        )
        vectors.extend(gp.sequential_score_vectors(values, order, cfg.B))
    beta, ess = BR.solve_beta(vectors, target=cfg.ess_target)
    return float(beta), float(ess)


def _finalize_alive(replicas):
    for replica in replicas:
        if not replica.alive:
            continue
        terminal_xy, _ = SS.collect_humans(replica.humans)
        clearance = float(
            np.linalg.norm(
                terminal_xy - replica.state[:2][None], axis=1,
            ).min() - SS.R_PED
        )
        replica.minimum_clearance = min(replica.minimum_clearance, clearance)
        if clearance < 0.0:
            replica.status = "collision"
        elif float(np.linalg.norm(replica.state[:2] - SS.GOAL)) < 0.5:
            replica.status = "success"
        else:
            replica.status = "timeout"
        replica.alive = False


def gather_offline_round(
    policy, phi_policy, gp, beta, replicas, cfg, shard, device, executor,
    *, round_i,
):
    timers = Counter()
    counts = Counter()
    sigma_all, sigma_selected, ess_values = [], [], []
    modes = {key: Counter() for key in ("all_K", "selected_B", "Dplus", "Dminus")}
    bounded_traces = []
    trap_active = defaultdict(bool)
    policy_hash = BX.policy_sha256(policy)

    for step in range(int(cfg.T)):
        start = time.perf_counter()
        live = [replica for replica in replicas if replica.alive]
        live, batch = BX._stack_prepared(live, device)
        timers["sfm_stepping"] += time.perf_counter() - start
        if not live:
            break
        counts["contexts"] += len(live)

        start = time.perf_counter()
        with torch.no_grad():
            windows, contexts = _keyed_windows(
                policy, live, batch, K=cfg.K, round_i=round_i, step=step,
                source="K", seed=cfg.seed, nfe=cfg.nfe, temp=cfg.temp,
            )
            raw_windows, _ = _keyed_windows(
                policy, live, batch, K=1, round_i=round_i, step=step,
                source="raw_continuation", seed=cfg.seed,
                nfe=cfg.nfe, temp=cfg.temp,
            )
            raw_windows = raw_windows[:, 0]
            windows_np = windows.detach().cpu().numpy()
            raw_windows_np = raw_windows.detach().cpu().numpy()
        timers["flow_proposal"] += time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            features = BX._features(phi_policy, windows, batch, cfg.phi_s)
            raw_features = BR.l2_normalize(phi_policy.phi_s(
                raw_windows, contexts, s=cfg.phi_s,
            ))
        selected_by_context = []
        acquisitions = []
        for context_index, replica in enumerate(live):
            generator = torch.Generator(device=features.device).manual_seed(
                _keyed_seed(
                    cfg.seed, round_i, replica.scenario_id,
                    f"{replica.gamma:.8f}", step, "acquisition",
                )
            )
            selected, trace = gp.sequential_acquire(
                features[context_index], cfg.B, beta, generator=generator,
            )
            selected_by_context.append(selected)
            acquisitions.append(trace)
            sigma_all.extend(
                float(value)
                for value in trace[0]["scores"].clamp_min(0.0).sqrt()
            )
            sigma_selected.extend(float(row["chosen_sigma"]) for row in trace)
            ess_values.extend(float(row["ess_norm"]) for row in trace)
        timers["phi_rbf"] += time.perf_counter() - start

        tasks = []
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            for candidate_id in selected_by_context[context_index]:
                tasks.append((
                    context_index,
                    candidate_id,
                    prepared["state"],
                    windows_np[context_index, candidate_id],
                    prepared["ped_xy"],
                    prepared["ped_vel"],
                    replica.gamma,
                ))
        start = time.perf_counter()
        results = list(executor.map(SM.verify_in_worker, tasks))
        timers["verifier"] += time.perf_counter() - start
        counts["B_queries"] += len(tasks)
        by_context = defaultdict(dict)
        for context_index, candidate_id, result in results:
            by_context[int(context_index)][int(candidate_id)] = result

        prepared_rows = []
        raw_tasks = []
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            prediction = SM.predict_pedestrians(
                prepared["ped_xy"], prepared["ped_vel"], cfg.H,
            )
            all_rows = []
            for candidate_id in range(cfg.K):
                controls = windows_np[context_index, candidate_id]
                segment = SM.rollout_positions(prepared["state"], controls)
                mode = BE.classify_candidate(segment, prediction)
                modes["all_K"][mode] += 1
                all_rows.append(dict(
                    candidate_id=int(candidate_id),
                    controls=controls,
                    mode=mode,
                ))
            query_rows = []
            for acquisition_step, candidate_id in enumerate(
                selected_by_context[context_index]
            ):
                result = by_context[context_index][candidate_id]
                label = FA._result_label(result)
                counts[f"B_{label}"] += 1
                mode = all_rows[candidate_id]["mode"]
                modes["selected_B"][mode] += 1
                if result.get("resolved"):
                    query_rows.append(dict(
                        candidate_id=int(candidate_id),
                        acquisition_step=int(acquisition_step),
                        controls=all_rows[candidate_id]["controls"],
                        result=result,
                        mode=mode,
                        sigma=float(
                            acquisitions[context_index][
                                acquisition_step
                            ]["chosen_sigma"]
                        ),
                    ))
            chosen = BC.select_admissible(
                query_rows,
                selector="margin",
                state=prepared["state"],
                ped_xy=prepared["ped_xy"],
                ped_vel=prepared["ped_vel"],
                gamma=replica.gamma,
            )
            prepared_rows.append((all_rows, query_rows, chosen))
            if chosen is None:
                raw_tasks.append((
                    context_index,
                    -1,
                    prepared["state"],
                    raw_windows_np[context_index],
                    prepared["ped_xy"],
                    prepared["ped_vel"],
                    replica.gamma,
                ))

        start = time.perf_counter()
        raw_results = list(executor.map(SM.verify_in_worker, raw_tasks))
        timers["verifier"] += time.perf_counter() - start
        counts["raw_continuation_queries"] += len(raw_tasks)
        for context_index, candidate_id, result in raw_results:
            if not result.get("resolved"):
                raise RuntimeError(
                    "executed raw continuation verifier failed; "
                    "aborting instead of executing or omitting an unlabeled context: "
                    f"{result.get('error')}"
                )
            by_context[int(context_index)][int(candidate_id)] = result

        start = time.perf_counter()
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            all_rows, query_rows, chosen = prepared_rows[context_index]
            context_id = shard.add_context(
                scenario_id=replica.scenario_id,
                gamma=replica.gamma,
                step=step,
                state=prepared["state"],
                hp10=prepared["hp10"].numpy(),
                low5=prepared["low"].numpy(),
                hist=prepared["hist"].numpy(),
                ped_xy=prepared["ped_xy"],
                ped_vel=prepared["ped_vel"],
            )
            nvp_context = chosen is None
            if nvp_context:
                controls = raw_windows_np[context_index]
                result = by_context[context_index][-1]
                margin, _, _ = BC.nominal_hp_margin(
                    prepared["state"], controls[0], prepared["ped_xy"],
                    replica.gamma,
                )
                raw_admissible = bool(
                    result.get("resolved")
                    and int(result.get("y", 0)) == 1
                    and bool(result.get("full_h"))
                    and float(margin) >= -1.0e-9
                )
                execution_source = (
                    "certified_raw_rescue"
                    if raw_admissible else "uncertified_raw_continuation"
                )
                candidate_id = None
                acquisition_step = None
                sigma = float(gp.acquisition_sigma(
                    raw_features[context_index:context_index + 1],
                )[0])
                prediction = SM.predict_pedestrians(
                    prepared["ped_xy"], prepared["ped_vel"], cfg.H,
                )
                mode = BE.classify_candidate(
                    SM.rollout_positions(prepared["state"], controls),
                    prediction,
                )
                counts["NVP_contexts"] += 1
            else:
                controls = np.asarray(chosen["controls"], np.float32)
                result = chosen["result"]
                margin = float(chosen["hp_margin"])
                execution_source = "verified_max_margin"
                candidate_id = int(chosen["candidate_id"])
                acquisition_step = int(chosen["acquisition_step"])
                sigma = float(chosen["sigma"])
                mode = chosen["mode"]

            label = FA._result_label(result)
            if label == "verifier_error":
                raise RuntimeError(
                    "resolved executed-window partition requires an exact "
                    "full-H10 binary verifier label"
                )
            counts[f"executed_{label}"] += 1
            counts[f"source_{execution_source}"] += 1
            window_id = shard.add_executed_window(
                context_id,
                controls,
                result,
                execution_source=execution_source,
                nvp_context=nvp_context,
                candidate_id=candidate_id,
                acquisition_step=acquisition_step,
                sigma=sigma,
                hp_margin=margin,
                mode=mode,
            )
            modes["Dplus" if int(result["y"]) == 1 else "Dminus"][mode] += 1

            BX._advance(replica, controls[0])
            trap_event = FA._trap(replica.states)
            trap_key = (replica.scenario_id, replica.gamma)
            trap_entry = bool(trap_event and not trap_active[trap_key])
            trap_active[trap_key] = bool(trap_event)
            collision, success, _ = FA._post_action_terminal(replica)
            counts["trap_steps"] += int(trap_event)
            counts["trap_entries"] += int(trap_entry)
            counts["collision_events"] += int(collision)
            counts["success_events"] += int(success)
            if window_id is not None:
                stored = shard.windows[int(window_id)]
                stored.update(
                    trap_event=bool(trap_event),
                    trap_entry=bool(trap_entry),
                    collision_after_action=bool(collision),
                    success_after_action=bool(success),
                )
            if len(bounded_traces) < 64 or (
                nvp_context and len(bounded_traces) < 128
            ):
                bounded_traces.append(dict(
                    step=int(step),
                    scenario_id=int(replica.scenario_id),
                    gamma=float(replica.gamma),
                    selected_ids=list(map(
                        int, selected_by_context[context_index],
                    )),
                    B_labels=[
                        FA._result_label(
                            by_context[context_index][candidate]
                        )
                        for candidate in selected_by_context[context_index]
                    ],
                    execution_source=execution_source,
                    executed_label=label,
                    nvp_context=bool(nvp_context),
                    window_id=window_id,
                    trap_event=bool(trap_event),
                    collision_after_action=bool(collision),
                    success_after_action=bool(success),
                ))
        timers["sfm_stepping"] += time.perf_counter() - start

    _finalize_alive(replicas)
    if BX.policy_sha256(policy) != policy_hash:
        raise RuntimeError("policy changed during frozen offline macro-round")
    shard_summary = shard.validate()
    if (
        int(shard_summary["D"]) != int(shard_summary["contexts"])
        or int(shard_summary["errors"]) != 0
        or int(shard_summary["unresolved_contexts"]) != 0
    ):
        raise RuntimeError(
            "completed offline round must exactly partition every context "
            "into D+ or D-"
        )
    if int(counts["B_queries"]) != int(counts["contexts"]) * int(cfg.B):
        raise RuntimeError("B verifier query accounting mismatch")
    return dict(
        collector_role="offline_expansion_data_collector_not_safe_controller",
        continuation_semantics=(
            "verified max-margin B action when available; otherwise an "
            "independent raw temperature-one H10 plan is exact-verified and "
            "its first action is executed even when y=0"
        ),
        label_semantics=(
            "D contains one resolved executed proposal per context; "
            "D+=exact full-H10 y=1; D-=exact full-H10 y=0; "
            "NVP/trap/collision are metadata and never retroactively relabel y"
        ),
        timers=dict(timers),
        counts=dict(counts),
        shard=shard_summary,
        beta=float(beta),
        realized_normalized_ess_over_remaining=float(np.mean(ess_values)),
        sigma=BR.acquisition_diagnostics(sigma_all, sigma_selected),
        modes={key: dict(value) for key, value in modes.items()},
        trace_examples=bounded_traces,
        outcomes=[dict(
            scenario_id=replica.scenario_id,
            gamma=replica.gamma,
            status=replica.status,
            success=replica.status == "success",
            collision=replica.status == "collision",
            timeout=replica.status == "timeout",
            steps=len(replica.controls),
            min_clearance=float(replica.minimum_clearance),
        ) for replica in replicas],
    )


def run(checkpoint, outdir, cfg, *, device):
    cfg.validate()
    checkpoint = os.path.abspath(checkpoint)
    outdir = os.path.abspath(outdir)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = OS.sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"got {checkpoint_sha}"
        )
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output directory: {outdir}")
    os.makedirs(outdir)
    environment = SS.scene_profile(cfg.scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    frozen_parameters = BS.configure_expansion_trainability(policy)
    visual_encoder_sha = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [
            parameter for parameter in policy.parameters()
            if parameter.requires_grad
        ],
        lr=cfg.lr,
    )
    BX._save_checkpoint(policy, os.path.join(outdir, "round_00.pt"), dict(
        round=0,
        experiment=cfg.arm_name,
        source_checkpoint=checkpoint,
        source_sha256=checkpoint_sha,
        encoder_sha256=visual_encoder_sha,
        recipe=asdict(cfg),
    ))
    history = []
    previous_shard = None
    with ProcessPoolExecutor(max_workers=cfg.verifier_workers) as executor:
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            scenarios = SP.expansion_scenarios(round_i, smoke=cfg.smoke)
            replicas = [
                BX.Replica(
                    scenario_id,
                    gamma,
                    n_ped=environment["n_ped"],
                    ped_speed_range=tuple(environment["ped_speed_range"]),
                )
                for scenario_id in scenarios for gamma in SP.GAMMAS
            ]
            if len(replicas) != 56:
                raise RuntimeError("offline macro-round requires 56 episodes")
            policy.eval()
            phi_policy = copy.deepcopy(policy).eval()
            for parameter in phi_policy.parameters():
                parameter.requires_grad_(False)
            gp, gp_ids, gp_selection = gp_from_previous(
                phi_policy,
                previous_shard,
                round_i=round_i,
                ell=ELL,
                cap=CAP,
                lam=cfg.gp_lam,
                phi_s=cfg.phi_s,
                device=device,
                seed=cfg.seed + round_i * 101,
            )
            beta, calibrated_ess = _calibrate_beta(
                phi_policy, gp, replicas, cfg, device, round_i=round_i,
            )
            shard = OS.ExecutedRoundShard(round_i)
            gather = gather_offline_round(
                policy,
                phi_policy,
                gp,
                beta,
                replicas,
                cfg,
                shard,
                device,
                executor,
                round_i=round_i,
            )
            shard_path = os.path.join(
                outdir, "round_shards", f"round_{round_i:02d}.pt",
            )
            shard_manifest = shard.save(shard_path)
            replay_start = time.perf_counter()
            replay = OR.replay(
                policy,
                optimizer,
                shard,
                alpha=cfg.alpha,
                exposure_epochs=cfg.exposure_epochs,
                batch=cfg.batch,
                device=device,
                seed=cfg.seed + round_i * 1_000_003,
            )
            gather["timers"]["replay"] = time.perf_counter() - replay_start
            if BS.module_sha256(policy.enc_grid) != visual_encoder_sha:
                raise RuntimeError("visual encoder SHA changed")
            checkpoint_path = os.path.join(
                outdir, f"round_{round_i:02d}.pt",
            )
            BX._save_checkpoint(policy, checkpoint_path, dict(
                round=round_i,
                experiment=cfg.arm_name,
                source_checkpoint=checkpoint,
                source_sha256=checkpoint_sha,
                encoder_sha256=visual_encoder_sha,
                recipe=asdict(cfg),
                ell=ELL,
                cap=CAP,
                beta=float(beta),
            ))
            record = dict(
                round=round_i,
                experiment=cfg.arm_name,
                scenarios=list(map(int, scenarios)),
                environment=environment,
                beta=float(beta),
                calibrated_normalized_ess_over_remaining=float(calibrated_ess),
                verifier=SM.verifier_manifest(),
                gp_buffer_ids=gp_ids,
                gp_selection=gp_selection,
                gp=gp.diagnostics(),
                gather=gather,
                replay=replay,
                shard=shard_manifest,
                checkpoint=os.path.abspath(checkpoint_path),
                checkpoint_sha256=OS.sha256_file(checkpoint_path),
                wall_seconds=time.perf_counter() - round_start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(
                round=round_i,
                experiment=cfg.arm_name,
                D=shard_manifest["D"],
                Dplus=shard_manifest["Dplus"],
                Dminus=shard_manifest["Dminus"],
                beta=float(beta),
                ess_over_remaining=float(
                    gather["realized_normalized_ess_over_remaining"]
                ),
                Adam_steps=int(replay["optimizer_steps"]),
                wall_seconds=record["wall_seconds"],
            )), flush=True)
            previous_shard = shard

    manifest = dict(
        status="SFM_B1_OFFLINE_EXEC_COMPLETE",
        experiment=cfg.arm_name,
        scientific_role="offline_expansion_data_collector_not_safe_controller",
        recipe=asdict(cfg),
        constants=dict(
            ell=ELL,
            gp_buffer_cap=CAP,
            gp_lambda=GP_LAMBDA,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            replay_window_rounds=1,
            gp_quota_semantics=(
                "exactly 73 executed D+ rows per gamma plus one rotating "
                "extra; any support shortage aborts the scientific round"
            ),
            ess_target_semantics=(
                "mean normalized ESS over each sequential remaining pool"
            ),
        ),
        source=_source(),
        source_checkpoint=checkpoint,
        source_checkpoint_sha256=checkpoint_sha,
        environment=environment,
        frozen_parameters=frozen_parameters,
        visual_encoder_sha=visual_encoder_sha,
        history=history,
    )
    _write_json(os.path.join(outdir, "COMPLETE.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--alpha", type=float, choices=ALPHAS, required=True)
    parser.add_argument(
        "--exposure-epochs",
        type=int,
        choices=EXPOSURE_EPOCHS,
        required=True,
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = OfflineConfig(
        alpha=args.alpha,
        exposure_epochs=args.exposure_epochs,
        rounds=args.rounds,
        verifier_workers=args.verifier_workers,
        seed=args.seed,
        smoke=args.smoke,
    )
    run(args.checkpoint, args.outdir, cfg, device=args.device)


if __name__ == "__main__":
    main()
