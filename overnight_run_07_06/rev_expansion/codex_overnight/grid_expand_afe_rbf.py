"""Single-arm RBF Safe Flow Expansion with synchronous parallel rollouts.

This is a task-specific AFE adaptation, not a claim that the main AFE theorem
requires an RBF kernel.  It follows the peptide experiment's RBF choices while
making the control-specific memory semantics explicit:

* exact RBF-GP on at most 512 full-H positives from the previous round;
* append-only D+ with cumulative replay by default and an opt-in round window;
* multiple closed-loop replicas gathered synchronously; the GP is frozen for
  the whole round, so replicas do not depend on an arbitrary execution order;
* B-budget sequential acquisition: only already-selected pending locations,
  never the unqueried remainder of K, condition the next posterior variance;
* fixed pretrained-only beta by default, with an opt-in round-local ESS target;
* one AFE update arm only (batch 128, lr 1e-4, 250 steps, no proximal term);
* deterministic full verifier before execution and expert-free NVP termination.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_WORK = os.path.dirname(_REV)
for _path in (_WORK, _REV, _HERE):
    sys.path.insert(0, _path)

import _paths  # noqa: F401
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_rollout as GR
import grid_hp_expt as HP
import grid_expand_hardtail as HT
from di_grid_viz import di_step

import afe_core as AC
import afe_context as CX
import afe_adaptive as AD
import afe2_calibration as BC
import afe_rbf_core as RC
import grid_expand_afe2 as AFE2
from afe2_scene_profiles import (
    SCENE_PROFILES,
    assert_scene_snapshot,
    build_scene,
    get_scene_profile,
    scene_snapshot,
)


@dataclass
class AFERBFConfig(AFE2.AFE2Config):
    arm: str = "afe"
    replicas: int = 2
    gp_cap: int = 512
    gp_lam: float = 1.0e-2
    verifier_workers: int = 16
    lengthscale_samples: int = 50
    acquisition_mode: str = "sequential"
    adaptive_ess_target: float | None = None
    replay_window: int | None = None
    rbf_offline_sweep: bool = False


def _episode(state, gamma, replica, episode_id, env, cfg):
    obstacles = env.obstacles.detach().cpu().numpy()
    clearance = float(
        (np.linalg.norm(state[:2][None] - obstacles[:, :2], axis=1)
         - obstacles[:, 2] - float(env.r_robot)).min()
    )
    collision = bool(clearance < 0.0)
    oob = bool(
        (state[:2] < -cfg.taskspace_epsilon).any()
        or (state[:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
    )
    goal = env.goal.detach().cpu().numpy()
    status = None
    if collision or oob or np.linalg.norm(state[:2] - goal) < cfg.reach:
        status = "collision" if collision else ("oob" if oob else "reached")
    return {
        "episode_id": int(episode_id),
        "replica": int(replica),
        "gamma": float(gamma),
        "state": state.copy(),
        "hist": [],
        "path": [state[:2].copy()],
        "clear_min": clearance,
        "collision": collision,
        "oob": oob,
        "status": status,
        "term_t": (0 if status is not None else None),
        "step_stats": [],
    }


def _context_arrays(episodes, env, cfg):
    return CX.arrays_for_episodes(episodes, env, cfg.conditioning_schema)


def query_has_socp_error(result):
    """Any full-H or terminal verifier SOCP error makes the query unobserved."""

    return (
        result.get("reason") == "socp_error"
        or result.get("terminal_reason") == "socp_error"
    )


def _proposal_noise(policy, active, cfg, purpose, round_i, control_t, device):
    """Stable per-episode proposal streams, batched only after noise generation."""

    seed_round = 0 if purpose == "controller_eval" else int(round_i)
    chunks = []
    for episode in active:
        generator = torch.Generator(device=device)
        generator.manual_seed(AFE2.named_seed(
            cfg.seed,
            "proposal",
            purpose,
            seed_round,
            episode["episode_id"],
            control_t,
        ))
        chunks.append(torch.randn(
            cfg.K, policy.d, device=device, generator=generator
        ))
    return torch.cat(chunks, dim=0)


def _acquisition_stats(
    sig,
    selected,
    features,
    controls,
    cfg,
    marginal_sigma=None,
    sequential_trace=None,
):
    if sequential_trace:
        ess_by_step = [float(row["ess_norm"]) for row in sequential_trace]
        entropy_by_step = [float(row["entropy_norm"]) for row in sequential_trace]
        ess_norm = float(np.median(ess_by_step))
        ess_first = ess_by_step[0]
        entropy = float(np.median(entropy_by_step))
        pool_vectors = [row["scores"].detach().cpu().numpy() for row in sequential_trace]
        selected_values = np.asarray([
            row["chosen_score"] for row in sequential_trace
        ], dtype=np.float64)
        pool_values = np.concatenate(pool_vectors)
        step_spans = np.asarray([np.ptp(row) for row in pool_vectors])
        step_iqrs = np.asarray([
            np.quantile(row, 0.75) - np.quantile(row, 0.25)
            for row in pool_vectors
        ])
        uplift = float(np.median([
            float(row["chosen_score"]) - float(row["scores"].mean())
            for row in sequential_trace
        ]))
    else:
        weights = torch.exp(((sig - sig.max()) / max(cfg.beta, 1.0e-9)).clamp(-30, 30))
        probability = (weights / weights.sum()).to(torch.float64)
        ess_norm = float(1.0 / (probability.square().sum() * probability.numel()))
        ess_first = ess_norm
        ess_by_step = [ess_norm]
        entropy = float(
            -(probability * (probability + 1.0e-30).log()).sum() / np.log(cfg.K)
        )
        entropy_by_step = [entropy]
        pool_values = sig.detach().cpu().numpy()
        selected_values = sig[selected].detach().cpu().numpy()
        step_spans = np.asarray([np.ptp(pool_values)])
        step_iqrs = np.asarray([
            np.quantile(pool_values, 0.75) - np.quantile(pool_values, 0.25)
        ])
        uplift = float(selected_values.mean() - pool_values.mean())
    quantiles = np.quantile(pool_values, [0.1, 0.25, 0.5, 0.75, 0.9])
    normalized = features.detach().cpu().to(torch.float64)
    cosine_distance = (1.0 - normalized @ normalized.T).clamp_min(0.0)
    pairs = torch.triu_indices(cfg.K, cfg.K, offset=1)
    feature_distance = cosine_distance[pairs[0], pairs[1]].numpy()
    plan_distance = torch.pdist(
        controls.detach().cpu().to(torch.float64).reshape(cfg.K, -1)
    ).numpy()
    correlation = (
        float(np.corrcoef(feature_distance, plan_distance)[0, 1])
        if np.std(feature_distance) > 0.0 and np.std(plan_distance) > 0.0
        else float("nan")
    )
    output = {
        "ess_norm": ess_norm,
        "ess_first": ess_first,
        "ess_by_step": ess_by_step,
        "ent": entropy,
        "entropy_by_step": entropy_by_step,
        "uplift": uplift,
        "sig_span": float(np.median(step_spans)),
        "sig_iqr": float(np.median(step_iqrs)),
        "sig_all": [float(quantiles[index]) for index in (0, 2, 4)],
        "sig_sel": [
            float(value)
            for value in np.quantile(selected_values, [0.1, 0.5, 0.9])
        ],
        "feature_cosine_distance_q": [
            float(value) for value in np.quantile(feature_distance, [0.1, 0.5, 0.9])
        ],
        "feature_plan_distance_corr": correlation,
    }
    if marginal_sigma is not None:
        output["marginal_sigma_med"] = float(marginal_sigma.median())
        output["marginal_sigma_iqr"] = float(
            torch.quantile(marginal_sigma, 0.75) - torch.quantile(marginal_sigma, 0.25)
        )
    return output


@torch.no_grad()
def run_parallel_episodes(
    policy,
    gp,
    env,
    cfg,
    store,
    round_i,
    replicas,
    device,
    executor,
    *,
    collect,
    viz,
    purpose,
    acquisition_mode="sequential",
):
    """Advance all gamma x replica episodes in lockstep with batched GPU proposals."""

    start = env.x0.detach().cpu().numpy().astype(np.float32)
    episodes = []
    for gamma_index, gamma in enumerate(cfg.gammas):
        for replica in range(replicas):
            episode_id = gamma_index * replicas + replica
            episodes.append(_episode(start, gamma, replica, episode_id, env, cfg))
    timings = {"sampling": 0.0, "verifier_wall": 0.0, "bookkeeping": 0.0}
    goal = env.goal.detach().cpu().numpy()
    obstacles = env.obstacles.detach().cpu().numpy()
    robot_radius = float(env.r_robot)

    for control_t in range(cfg.T):
        active = [episode for episode in episodes if episode["status"] is None]
        if not active:
            break
        grid_np, low_np, hist_np = _context_arrays(active, env, cfg)
        grid = torch.as_tensor(grid_np, device=device)
        low = torch.as_tensor(low_np, device=device)
        hist = torch.as_tensor(hist_np, device=device)
        sampling_start = time.perf_counter()
        context = policy.ctx_from(grid, low, hist)
        repeated_context = context.repeat_interleave(cfg.K, dim=0)
        initial_noise = _proposal_noise(
            policy, active, cfg, purpose, round_i, control_t, device
        )
        candidates = policy.sample(
            len(active) * cfg.K,
            repeated_context,
            nfe=cfg.nfe,
            temp=cfg.temp,
            initial_noise=initial_noise,
        ).reshape(len(active), cfg.K, policy.H_pred, 2)
        features = policy.phi_s(
            candidates.reshape(len(active) * cfg.K, policy.H_pred, 2),
            repeated_context,
            s=cfg.s,
        )
        features = RC.l2_normalize(features).reshape(len(active), cfg.K, -1)
        marginal_sigma = gp.sigma(
            features.reshape(len(active) * cfg.K, -1)
        ).reshape(
            len(active), cfg.K
        )
        selected = []
        traces = []
        first_scores = []
        seed_round = 0 if purpose == "controller_eval" else int(round_i)
        for episode_index, episode in enumerate(active):
            with AC.isolated_random_state(AFE2.named_seed(
                cfg.seed,
                "acquisition",
                purpose,
                seed_round,
                episode["episode_id"],
                control_t,
            )):
                if acquisition_mode == "uniform":
                    order = torch.randperm(cfg.K, device=features.device)
                    vectors = gp.sequential_score_vectors(
                        features[episode_index], order, min(cfg.B, cfg.K)
                    )
                    chosen = order[: min(cfg.B, cfg.K)].tolist()
                    pending = list(range(cfg.K))
                    trace = []
                    for step, vector in enumerate(vectors):
                        chosen_global = int(chosen[step])
                        chosen_local = pending.index(chosen_global)
                        trace.append({
                            "scores": vector,
                            "remaining": None,
                            "chosen": chosen_global,
                            "chosen_score": float(vector[chosen_local]),
                            "ess_norm": 1.0,
                            "entropy_norm": 1.0,
                        })
                        pending.pop(chosen_local)
                elif acquisition_mode == "sequential":
                    chosen, trace = gp.sequential_acquire(
                        features[episode_index], min(cfg.B, cfg.K), cfg.beta
                    )
                else:
                    raise ValueError(f"unknown acquisition mode: {acquisition_mode}")
            selected.append(chosen)
            traces.append(trace)
            first_scores.append(trace[0]["scores"])
        sigma = torch.stack(first_scores)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings["sampling"] += time.perf_counter() - sampling_start

        candidate_cpu = candidates.detach().cpu().numpy()
        sigma_cpu = sigma.detach().cpu()
        marginal_sigma_cpu = marginal_sigma.detach().cpu()
        feature_cpu = features.detach().cpu()
        step_context_ids = {}
        if collect:
            for local_index, episode in enumerate(active):
                step_context_ids[episode["episode_id"]] = store.add_step_ctx(
                    episode["state"],
                    grid_np[local_index],
                    low_np[local_index],
                    hist_np[local_index],
                    (round_i, episode["episode_id"], control_t),
                )
        tasks = []
        for local_index, episode in enumerate(active):
            for candidate_id in selected[local_index]:
                tasks.append((
                    episode["episode_id"],
                    candidate_id,
                    episode["state"],
                    candidate_cpu[local_index, candidate_id],
                    episode["gamma"],
                ))
        verifier_start = time.perf_counter()
        results = list(executor.map(RC.verify_in_worker, tasks, chunksize=1))
        timings["verifier_wall"] += time.perf_counter() - verifier_start
        by_episode = {episode["episode_id"]: [] for episode in active}
        for episode_id, candidate_id, result in results:
            by_episode[episode_id].append((candidate_id, result))

        bookkeeping_start = time.perf_counter()
        for local_index, episode in enumerate(active):
            episode_results = by_episode[episode["episode_id"]]
            acquired_scores = {
                int(row["chosen"]): float(row["chosen_score"])
                for row in traces[local_index]
            }
            best = None
            query_rows = []
            verifier_cpu_seconds = 0.0
            for candidate_id, result in episode_results:
                verifier_cpu_seconds += float(result["verifier_seconds"])
                query_id = -1
                controls = candidate_cpu[local_index, candidate_id]
                segment = GR.window_positions(episode["state"], controls, env.dt)
                if not query_has_socp_error(result) and collect:
                    query_id = store.add_query(
                        step_context_ids[episode["episode_id"]],
                        controls,
                        result,
                        acquired_scores[candidate_id],
                        episode["gamma"],
                        round_i,
                        segment,
                    )
                query_rows.append((candidate_id, query_id, result))
                if result["exec_y"] and (
                    best is None or result["exec_prog"] > best[0]
                ):
                    best = (float(result["exec_prog"]), query_id, controls, candidate_id, result)

            drawn = selected[local_index]
            stats = _acquisition_stats(
                sigma_cpu[local_index],
                drawn,
                feature_cpu[local_index],
                torch.from_numpy(candidate_cpu[local_index]),
                cfg,
                marginal_sigma=marginal_sigma_cpu[local_index],
                sequential_trace=traces[local_index],
            )
            full_positive_available = any(row[2]["y"] == 1 for row in query_rows)
            selected_rescue = bool(best is not None and best[4]["terminal_rescue"])
            stats.update(
                n_err=sum(row[2]["reason"] == "socp_error" for row in query_rows),
                n_socp_solve=sum(int(row[2]["n_socp_solve"]) for row in query_rows),
                verifier_seconds=verifier_cpu_seconds,
                n_terminal_error=sum(
                    row[2]["terminal_reason"] == "socp_error" for row in query_rows
                ),
                n_pos=sum(row[2]["y"] == 1 for row in query_rows),
                n_exec_pos=sum(row[2]["exec_y"] == 1 for row in query_rows),
                n_terminal_rescue=sum(bool(row[2]["terminal_rescue"]) for row in query_rows),
                n_terminal_reverify=sum(bool(row[2]["terminal_reverify"]) for row in query_rows),
                selected_terminal_rescue=selected_rescue,
                selected_terminal_required=bool(selected_rescue and not full_positive_available),
                full_positive_available=full_positive_available,
                n_drawn=len(query_rows),
            )
            episode["step_stats"].append(stats)

            if viz is not None:
                segments = GR.di_rollout_batch(
                    episode["state"], candidate_cpu[local_index], env.dt
                ).astype(np.float16)
                admissible = [row[2] for row in query_rows if row[2]["exec_y"]]
                viz.append({
                    "t": control_t,
                    "episode": episode["episode_id"],
                    "replica": episode["replica"],
                    "gamma": episode["gamma"],
                    "state": episode["state"].copy(),
                    "segsK": segments,
                    "drawn": [row[0] for row in query_rows],
                    "y": [(-1 if row[2]["reason"] == "socp_error" else row[2]["y"])
                          for row in query_rows],
                    "exec_y": [row[2]["exec_y"] for row in query_rows],
                    "terminal_rescue": [bool(row[2]["terminal_rescue"]) for row in query_rows],
                    "terminal_tau": [row[2]["terminal_tau"] for row in query_rows],
                    "n_socp_solve": stats["n_socp_solve"],
                    "sel": (-1 if best is None else best[3]),
                    "sig_q": stats["sig_all"],
                    "sigB_q": stats["sig_sel"],
                    "min_margin": (
                        float(np.nanmin([row["exec_margin"] for row in admissible]))
                        if admissible else float("nan")
                    ),
                })

            if best is None:
                episode["status"] = "nvp"
                episode["term_t"] = control_t
                continue
            if collect and best[1] >= 0:
                store.mark_executed(best[1])
            action = np.asarray(best[2][0], dtype=np.float32)
            episode["state"] = di_step(episode["state"], action, dt=env.dt)
            episode["hist"].append(action)
            episode["path"].append(episode["state"][:2].copy())
            episode["clear_min"] = min(
                episode["clear_min"],
                float(
                    (np.linalg.norm(episode["state"][:2][None] - obstacles[:, :2], axis=1)
                     - obstacles[:, 2] - robot_radius).min()
                ),
            )
            episode["collision"] = bool(episode["clear_min"] < 0.0)
            episode["oob"] = bool(
                (episode["state"][:2] < -cfg.taskspace_epsilon).any()
                or (episode["state"][:2] > GM.GRID_M + cfg.taskspace_epsilon).any()
            )
            if episode["collision"] or episode["oob"]:
                episode["status"] = "collision" if episode["collision"] else "oob"
                episode["term_t"] = control_t + 1
            elif np.linalg.norm(episode["state"][:2] - goal) < cfg.reach:
                episode["status"] = "reached"
                episode["term_t"] = control_t + 1
        timings["bookkeeping"] += time.perf_counter() - bookkeeping_start

    output = []
    for episode in episodes:
        if episode["status"] is None:
            episode["status"] = "timeout"
        output.append({
            "episode_id": episode["episode_id"],
            "replica": episode["replica"],
            "gamma": episode["gamma"],
            "path": np.asarray(episode["path"], dtype=np.float32),
            "status": episode["status"],
            "term_t": episode["term_t"],
            "steps": len(episode["path"]) - 1,
            "clear_min": episode["clear_min"],
            "collision": episode["collision"],
            "oob": episode["oob"],
            "step_stats": episode["step_stats"],
        })
    return output, timings


def _per_gamma_episode_stats(episodes, cfg):
    output = {}
    for gamma in cfg.gammas:
        records = [record for record in episodes if record["gamma"] == float(gamma)]
        steps = [item for record in records for item in record["step_stats"]]
        output[str(gamma)] = {
            "episodes": len(records),
            "status_counts": {
                name: sum(record["status"] == name for record in records)
                for name in ("reached", "nvp", "timeout", "collision", "oob")
            },
            "steps": int(sum(record["steps"] for record in records)),
            "ess_med": (float(np.median([item["ess_norm"] for item in steps]))
                        if steps else None),
            "ess_first_med": (float(np.median([item["ess_first"] for item in steps]))
                              if steps else None),
            "ent_med": (float(np.median([item["ent"] for item in steps])) if steps else None),
            "uplift_med": (float(np.median([item["uplift"] for item in steps])) if steps else None),
            "sig_iqr_med": (float(np.median([item["sig_iqr"] for item in steps])) if steps else None),
            "sig_span_med": (float(np.median([item["sig_span"] for item in steps])) if steps else None),
            "n_q": int(sum(item["n_drawn"] for item in steps)),
            "n_pos": int(sum(item["n_pos"] for item in steps)),
            "n_exec_pos": int(sum(item["n_exec_pos"] for item in steps)),
            "n_socp_solve": int(sum(item["n_socp_solve"] for item in steps)),
            "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in steps)),
            "n_err": int(sum(item["n_err"] for item in steps)),
        }
    return output


def _controller_summary(episodes, cfg, env):
    rows = {}
    for gamma in cfg.gammas:
        records = [record for record in episodes if record["gamma"] == float(gamma)]
        count = len(records)
        rows[str(gamma)] = {
            "SR": sum(record["status"] == "reached" for record in records) / count,
            "CR": sum(record["collision"] or record["oob"] for record in records) / count,
            "collision": sum(record["collision"] for record in records) / count,
            "OOB": sum(record["oob"] for record in records) / count,
            "NVP": sum(record["status"] == "nvp" for record in records) / count,
            "TO": sum(record["status"] == "timeout" for record in records) / count,
            "clear": float(np.nanmean([record["clear_min"] for record in records])),
            "time": (
                float(np.mean([
                    record["steps"] * env.dt
                    for record in records if record["status"] == "reached"
                ]))
                if any(record["status"] == "reached" for record in records)
                else float("nan")
            ),
            "clear_values": [float(record["clear_min"]) for record in records],
            "time_success_values": [
                float(record["steps"] * env.dt)
                for record in records if record["status"] == "reached"
            ],
            "status_values": [record["status"] for record in records],
            "nvp_t": [
                int(record["term_t"])
                for record in records if record["status"] == "nvp"
            ],
        }
    pooled = {
        key: float(np.mean([row[key] for row in rows.values()]))
        for key in ("SR", "CR", "NVP")
    }
    return rows, pooled


@torch.no_grad()
def _calibration_score_vectors(policy, gp, store, cfg, device):
    """Build beta-neutral B-step score vectors at disjoint rollout contexts."""

    vectors = []
    pools = []
    gamma_counts = {}
    chunk_size = 16
    for begin in range(0, len(store.ctx_state), chunk_size):
        sids = list(range(begin, min(begin + chunk_size, len(store.ctx_state))))
        grid = store.grid3_of(sids).to(device)
        low = torch.stack([torch.from_numpy(store.ctx_low5[sid]) for sid in sids]).to(device)
        hist = torch.stack([
            torch.from_numpy(store.ctx_hist[sid].astype(np.float32)) for sid in sids
        ]).to(device)
        context = policy.ctx_from(grid, low, hist)
        repeated = context.repeat_interleave(cfg.K, dim=0)
        with AC.isolated_random_state(
            AFE2.named_seed(cfg.seed, "rbf_operational_beta_candidates", begin)
        ):
            controls = policy.sample(
                len(sids) * cfg.K, repeated, nfe=cfg.nfe, temp=cfg.temp
            )
        features = RC.l2_normalize(
            policy.phi_s(controls, repeated, s=cfg.s)
        ).reshape(len(sids), cfg.K, -1)
        for local_index, sid in enumerate(sids):
            pools.append(features[local_index].detach())
            order_rng = np.random.default_rng(
                AFE2.named_seed(cfg.seed, "rbf_operational_beta_order", sid)
            )
            order = torch.as_tensor(
                order_rng.permutation(cfg.K), device=device, dtype=torch.long
            )
            vectors.extend([
                score.detach().cpu().numpy()
                for score in gp.sequential_score_vectors(
                    features[local_index], order, min(cfg.B, cfg.K)
                )
            ])
            gamma = str(round(float(store.ctx_low5[sid][-1]), 2))
            gamma_counts[gamma] = gamma_counts.get(gamma, 0) + 1
    if not vectors:
        raise RuntimeError("operational beta calibration produced no rollout-context scores")
    return vectors, gamma_counts, pools


@torch.no_grad()
def calibrate_rbf(policy, env, cfg, device, executor):
    """Calibrate ell and one fixed beta at the declared operational GP size."""

    calibration_start = time.perf_counter()
    state = env.x0.detach().cpu().numpy().astype(np.float32)
    synthetic = [
        _episode(state, gamma, 0, index, env, cfg)
        for index, gamma in enumerate(cfg.gammas)
    ]
    grid_np, low_np, hist_np = _context_arrays(synthetic, env, cfg)
    grid = torch.as_tensor(grid_np, device=device)
    low = torch.as_tensor(low_np, device=device)
    hist = torch.as_tensor(hist_np, device=device)
    context = policy.ctx_from(grid, low, hist)
    base, extra = divmod(cfg.lengthscale_samples, len(cfg.gammas))
    context_indices = [
        index
        for index in range(len(cfg.gammas))
        for _ in range(base + int(index < extra))
    ]
    context_index = torch.as_tensor(context_indices, device=device)
    with AC.isolated_random_state(AFE2.named_seed(cfg.seed, "rbf_lengthscale")):
        controls = policy.sample(
            cfg.lengthscale_samples,
            context[context_index],
            nfe=cfg.nfe,
            temp=cfg.temp,
        )
        features = RC.l2_normalize(
            policy.phi_s(controls, context[context_index], s=cfg.s)
        )
    lengthscale = RC.mean_pairwise_lengthscale(features)

    # A beta calibrated on the 50-sample length-scale seed is not operational:
    # the expansion GP has `gp_cap` points and sees rollout contexts.  Build a
    # separate pretrained-only archive using uniform B-budget acquisition, then
    # discard it from CFM training and evaluation.
    empty_gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    seed_store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    seed_episodes, seed_timing = run_parallel_episodes(
        policy, empty_gp, env, cfg, seed_store, 0, cfg.replicas,
        device, executor, collect=True, viz=None,
        purpose="rbf_operational_seed", acquisition_mode="uniform",
    )
    seed_ids = RC.previous_round_positive_ids(
        seed_store, 0, cfg.gp_cap, cfg.gammas,
        AFE2.named_seed(cfg.seed, "rbf_operational_seed_buffer"),
    )
    if len(seed_ids) != cfg.gp_cap:
        raise RuntimeError(
            f"operational GP calibration requires exactly {cfg.gp_cap} verified positives; "
            f"found {len(seed_ids)}"
        )
    operational_features = AFE2.embed_queries(
        policy, seed_store, cfg, device, ids=seed_ids
    ).to(device)
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    gp.set_buffer(operational_features)

    # Use an independent uniform-acquisition rollout archive for contexts.  The
    # query plans in this archive are not GP points and never enter D+.
    beta_store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    beta_episodes, beta_timing = run_parallel_episodes(
        policy, gp, env, cfg, beta_store, 0, cfg.replicas,
        device, executor, collect=True, viz=None,
        purpose="rbf_operational_beta_contexts", acquisition_mode="uniform",
    )
    score_vector_start = time.perf_counter()
    score_vectors, context_gamma_counts, feature_pools = _calibration_score_vectors(
        policy, gp, beta_store, cfg, device
    )
    score_vector_seconds = time.perf_counter() - score_vector_start
    target = (
        BC.ESS_TARGET
        if cfg.adaptive_ess_target is None
        else float(cfg.adaptive_ess_target)
    )
    solution = BC.solve_beta_ragged(score_vectors, target=target)
    offline_sweep = None
    if cfg.rbf_offline_sweep:
        offline_sweep = AD.rbf_counterfactual_sweep(
            feature_pools,
            operational_features,
            cfg,
            round_i=0,
            target=target,
            lengthscale=lengthscale,
        )
    seed_steps = [item for row in seed_episodes for item in row["step_stats"]]
    beta_steps = [item for row in beta_episodes for item in row["step_stats"]]

    def verifier_budget(steps):
        return {
            "queries": int(sum(item["n_drawn"] for item in steps)),
            "positives": int(sum(item["n_pos"] for item in steps)),
            "socp_solves": int(sum(item["n_socp_solve"] for item in steps)),
            "socp_errors": int(sum(item["n_err"] for item in steps)),
            "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in steps)),
        }

    seed_budget = verifier_budget(seed_steps)
    beta_budget = verifier_budget(beta_steps)
    calibration_budget = {
        "seed_archive": seed_budget,
        "disjoint_context_archive": beta_budget,
        "total_queries": seed_budget["queries"] + beta_budget["queries"],
        "total_positives": seed_budget["positives"] + beta_budget["positives"],
        "total_socp_solves": seed_budget["socp_solves"] + beta_budget["socp_solves"],
        "total_socp_errors": seed_budget["socp_errors"] + beta_budget["socp_errors"],
        "score_vector_seconds": float(score_vector_seconds),
        "total_wall_seconds": float(time.perf_counter() - calibration_start),
        "enters_training_Dplus": False,
        "enters_round1_GP": "exactly the declared operational_gp_size seed positives",
    }
    return {
        "lengthscale": float(lengthscale),
        "lengthscale_samples": cfg.lengthscale_samples,
        "operational_gp_size": gp.n,
        "operational_seed_queries": len(seed_store),
        "operational_seed_positives": seed_store.n_pos(),
        "operational_seed_status_counts": {
            name: sum(row["status"] == name for row in seed_episodes)
            for name in ("reached", "nvp", "timeout", "collision", "oob")
        },
        "operational_seed_timing": seed_timing,
        "beta_context_queries": len(beta_store),
        "beta_context_count": len(beta_store.ctx_state),
        "beta_context_gamma_counts": context_gamma_counts,
        "beta_context_status_counts": {
            name: sum(row["status"] == name for row in beta_episodes)
            for name in ("reached", "nvp", "timeout", "collision", "oob")
        },
        "beta_context_timing": beta_timing,
        "calibration_budget": calibration_budget,
        "bootstrap_features": operational_features,
        "beta": float(solution["beta"]),
        "beta_solution": solution,
        "score_vector_sha256": BC.score_vectors_sha256(score_vectors),
        "score_vectors": score_vectors,
        "offline_sweep": offline_sweep,
    }


def _gp_from_query_ids(policy, store, query_ids, cfg, device, lengthscale):
    gp = RC.RBFGPSigma(lengthscale, cfg.gp_lam)
    features = AFE2.embed_queries(policy, store, cfg, device, ids=query_ids)
    gp.set_buffer(features.to(device))
    counts = {}
    for query_id in query_ids:
        key = str(round(float(store.q_gamma[query_id]), 2))
        counts[key] = counts.get(key, 0) + 1
    diagnostics = gp.diagnostics()
    diagnostics.update(
        source_query_ids=[int(value) for value in query_ids],
        gamma_counts=counts,
    )
    return gp, diagnostics


def _aggregate_step_stats(episodes, cfg):
    values = [item for record in episodes for item in record["step_stats"]]
    if not values:
        return {}
    correlations = [
        item["feature_plan_distance_corr"]
        for item in values if np.isfinite(item["feature_plan_distance_corr"])
    ]
    return {
        "ess_med": float(np.median([item["ess_norm"] for item in values])),
        "ess_first_med": float(np.median([item["ess_first"] for item in values])),
        "ess_by_step_med": [
            float(np.median([item["ess_by_step"][step] for item in values]))
            for step in range(min(cfg.B, cfg.K))
        ],
        "ent_med": float(np.median([item["ent"] for item in values])),
        "uplift_med": float(np.median([item["uplift"] for item in values])),
        "sig_span_med": float(np.median([item["sig_span"] for item in values])),
        "sig_iqr_med": float(np.median([item["sig_iqr"] for item in values])),
        "sig_all_med": float(np.median([item["sig_all"][1] for item in values])),
        "sig_sel_med": float(np.median([item["sig_sel"][1] for item in values])),
        "feature_plan_distance_corr_med": (
            float(np.median(correlations)) if correlations else None
        ),
        "verifier_cpu_seconds": float(sum(item["verifier_seconds"] for item in values)),
        "marginal_sigma_med": float(np.median([
            item["marginal_sigma_med"] for item in values
        ])),
        "marginal_sigma_iqr_med": float(np.median([
            item["marginal_sigma_iqr"] for item in values
        ])),
    }


def run(policy, env, cfg, device, outdir, checkpoint_path, checkpoint_sha256,
        checkpoint_model_sha256, checkpoint_contract, checkpoint_contract_sha256,
        source_git_state):
    if os.path.exists(outdir) and (not os.path.isdir(outdir) or os.listdir(outdir)):
        raise RuntimeError(f"single-arm run requires a new or empty output directory: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "viz_db"), exist_ok=True)
    profile = get_scene_profile(cfg.scene_profile)
    scene = scene_snapshot(env, profile)
    assert_scene_snapshot(scene)
    store = AC.DStore(
        conditioning_schema=cfg.conditioning_schema,
        condition_dim=cfg.raw_condition_dim,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.afe_lr)
    audit_contexts = AC.build_audit_contexts(
        env,
        cfg.gammas,
        n_pos=cfg.audit_pos,
        conditioning_schema=cfg.conditioning_schema,
    )
    representation_probe = AFE2.rep_probe_build(policy, env, cfg, device)
    goal = env.goal.detach().cpu().numpy()

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=cfg.verifier_workers,
        mp_context=context,
        initializer=RC.initialize_verifier_worker,
        initargs=(cfg.scene_profile, cfg.reach, cfg.n_theta),
    ) as executor:
        calibration = calibrate_rbf(policy, env, cfg, device, executor)
        cfg.beta = calibration["beta"]
        calibration_public = {
            key: AFE2._json_safe(value)
            for key, value in calibration.items()
            if key not in {"bootstrap_features", "score_vectors"}
        }
        calibration_public.update({
            "status": "CALIBRATED_AFE_RBF_SEQUENTIAL_OPERATIONAL_V3",
            "kernel": "RBF on L2-normalized phi_s",
            "lengthscale_rule": (
                "mean pairwise embedding distance of exactly 50 samples from the pretrained model"
            ),
            "gp_buffer_label": "full-H verifier positive only",
            "acquisition_statistic": (
                "normalized GP posterior variance conditioned on the GP buffer and only "
                "the already-selected pending locations in a B-step acquisition"
            ),
            "ess_target": calibration["beta_solution"]["target"],
            "scene_sha256": scene["sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_git_commit": source_git_state["commit"],
        })
        calibration_path = os.path.join(outdir, "rbf_calibration.json")
        with open(calibration_path, "w") as stream:
            json.dump(calibration_public, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

        if cfg.acquisition_mode == "uniform":
            algorithm = "afe_uniform_parallel_v1"
        elif cfg.adaptive_ess_target is not None:
            algorithm = "afe_rbf_adaptive_ess_parallel_v4"
        else:
            algorithm = "afe_rbf_sequential_operational_parallel_v3"
        beta_protocol = (
            "uniform B-without-replacement; RBF beta is diagnostic only"
            if cfg.acquisition_mode == "uniform"
            else (
                f"round-1 operational calibration followed after every round by a beta-neutral "
                f"current-policy calibration targeting median ESS={cfg.adaptive_ess_target:g}; "
                "beta_n is frozen during round n"
                if cfg.adaptive_ess_target is not None
                else (
                    f"one pretrained-only continuous ESS calibration against an operational "
                    f"{cfg.gp_cap}-positive GP and disjoint rollout-context B-step score vectors; "
                    "uniform beta-neutral pending orders; fixed for every expansion round"
                )
            )
        )
        learning_memory = (
            "uniform replay over the complete cumulative full-H D+ archive"
            if cfg.replay_window is None
            else (
                f"uniform replay over full-H positives from the current and previous "
                f"{cfg.replay_window - 1} rounds; cumulative D+ archive is retained"
            )
        )
        recipe = {
            "algorithm": algorithm,
            "arm": "afe",
            "single_arm": True,
            "kernel": "RBF",
            "lengthscale": calibration["lengthscale"],
            "lengthscale_protocol": calibration_public["lengthscale_rule"],
            "beta": cfg.beta,
            "beta_protocol": beta_protocol,
            "adaptive_ess_target": cfg.adaptive_ess_target,
            "acquisition_mode": cfg.acquisition_mode,
            "acquisition_memory": (
                "round 1: verified-positive pretrained calibration seed; later rounds: at most "
                f"{cfg.gp_cap} full-H positives from immediately preceding round, gamma-balanced "
                "random without replacement; re-embedded with current phi; frozen within round"
            ),
            "calibration_budget": calibration_public["calibration_budget"],
            "calibration_scope": (
                "round-0 acquisition-only verifier budget; the seed archive supplies the "
                "declared round-1 GP but neither archive enters cumulative training D+ or audit"
            ),
            "calibration_limitation": (
                "beta is solved on beta-neutral random pending orders; realized first-step "
                "ESS/K and stage-normalized ESS/M_remaining are logged during expansion"
            ),
            "learning_memory": learning_memory,
            "replay_window": cfg.replay_window,
            "rbf_offline_sweep": (
                "one pretrained-policy counterfactual sweep stored in rbf_calibration.json"
                if cfg.rbf_offline_sweep else False
            ),
            "uncertainty_meaning": (
                "RBF posterior variance conditioned on the acquisition buffer and only the "
                "locations already selected within the same B-budget query; not validity "
                "probability and not a safety certificate"
            ),
            "parallel_sampling": (
                f"{cfg.replicas} closed-loop replicas per gamma advanced synchronously; one GPU "
                f"proposal batch per control tick; {cfg.verifier_workers} persistent spawned CPU "
                "verifier workers; no within-round GP update"
            ),
            "execution": (
                "maximum-progress terminal-aware verified plan; execute first action; absorbing "
                "goal prefix allowed only for execution; NVP terminates; no expert/fallback"
            ),
            "update": f"CFM lr {cfg.afe_lr:g}, batch {cfg.batch}, {cfg.afe_steps} steps, no prox",
            "rounds": cfg.rounds,
            "rollout_replicas": cfg.replicas,
            "T": cfg.T,
            "K": cfg.K,
            "B": cfg.B,
            "batch": cfg.batch,
            "afe_lr": cfg.afe_lr,
            "afe_steps": cfg.afe_steps,
            "gp_cap": cfg.gp_cap,
            "gp_lam": cfg.gp_lam,
            "s": cfg.s,
            "nfe": cfg.nfe,
            "M_eval": cfg.M_eval,
            "gammas": list(cfg.gammas),
            "reach": cfg.reach,
            "seed": cfg.seed,
            "scene": scene,
            "source_checkpoint": os.path.abspath(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_checkpoint_model_sha256": checkpoint_model_sha256,
            "source_checkpoint_contract": checkpoint_contract,
            "source_checkpoint_contract_sha256": checkpoint_contract_sha256,
            "source_git_commit": source_git_state["commit"],
            "runtime": AFE2._runtime_provenance(device),
            "methodological_scope": (
                "task-specific peptide-style RBF AFE adaptation; previous-round cap and parallel "
                "frozen acquisition are explicit computational assumptions"
            ),
            "reference_code_semantics": (
                "sequential Schur complements are the B<K budget-consistent adaptation of "
                "the public peptide implementation's batch-conditional covariance"
            ),
            "no_curriculum": True,
            "no_anchor": True,
            "no_prox": True,
            "no_fallback": True,
        }
        recipe_path = os.path.join(outdir, "recipe.json")
        with open(recipe_path, "w") as stream:
            json.dump(AFE2._json_safe(recipe), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

        probe_path = os.path.join(outdir, "probe.jsonl")

        def write_probe(record):
            with open(probe_path, "a") as stream:
                stream.write(json.dumps(AFE2._json_safe(record), allow_nan=False) + "\n")

        bootstrap_gp = RC.RBFGPSigma(calibration["lengthscale"], cfg.gp_lam)
        bootstrap_gp.set_buffer(calibration["bootstrap_features"].to(device))
        audit0 = AC.run_audit(
            policy, audit_contexts, env, goal, device,
            n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
            seed=AFE2.named_seed(cfg.seed, "audit"),
        )
        eval0, eval0_timing = run_parallel_episodes(
            policy, bootstrap_gp, env, cfg, store, 0, cfg.M_eval, device, executor,
            collect=False, viz=None, purpose="controller_eval",
        )
        rows0, pooled0 = _controller_summary(eval0, cfg, env)
        write_probe({
            "round": 0,
            "arm": "afe",
            "acquisition_mode": cfg.acquisition_mode,
            "beta_used": cfg.beta,
            "beta_next": cfg.beta,
            "V": audit0["V"],
            "V_safe": audit0["V_safe"],
            "V_full": audit0["V_full"],
            "V_gamma": audit0["V_gamma"],
            "V_safe_gamma": audit0["V_safe_gamma"],
            "V_full_gamma": audit0["V_full_gamma"],
            "ctrl": rows0,
            "ctrl_pooled": pooled0,
            "n_D": 0,
            "n_Dpos": 0,
            "gp_buffer": bootstrap_gp.diagnostics(),
            "calibration_budget": calibration_public["calibration_budget"],
            "rbf_offline_sweep": calibration_public["offline_sweep"],
            "rep_cos": 1.0,
            "evaluation_timing": eval0_timing,
        })
        HT._save_hp_atomic(
            policy, os.path.join(outdir, "ckpt_0.pt"),
            extra={"iter": 0, "recipe": recipe, "resumable": False},
        )
        print(
            f"[afe-rbf] r000 V {audit0['V']:.3f} ctrl SR {pooled0['SR']:.2f} "
            f"NVP {pooled0['NVP']:.2f} ell {cfg.beta:.4g}/{calibration['lengthscale']:.4g}",
            flush=True,
        )

        gp_for_gather = bootstrap_gp
        gp_start_diagnostics = bootstrap_gp.diagnostics()
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            beta_used = float(cfg.beta)
            policy.eval()
            viz = []
            episodes, gather_timing = run_parallel_episodes(
                policy, gp_for_gather, env, cfg, store, round_i, cfg.replicas,
                device, executor, collect=True, viz=viz, purpose="gather",
                acquisition_mode=cfg.acquisition_mode,
            )
            gather_seconds = time.perf_counter() - round_start
            per_gamma = _per_gamma_episode_stats(episodes, cfg)
            acquisition = _aggregate_step_stats(episodes, cfg)

            update_start = time.perf_counter()
            replay_rng = np.random.default_rng(AFE2.named_seed(cfg.seed, "replay", round_i))
            with AC.isolated_random_state(AFE2.named_seed(cfg.seed, "update", round_i)):
                update = AFE2.update_round(
                    policy, optimizer, store, cfg, device, replay_rng, round_i
                )
            update_seconds = time.perf_counter() - update_start
            policy.eval()

            query_ids = RC.previous_round_positive_ids(
                store, round_i, cfg.gp_cap, cfg.gammas,
                AFE2.named_seed(cfg.seed, "gp_buffer", round_i),
            )
            gp_post, gp_post_diagnostics = _gp_from_query_ids(
                policy, store, query_ids, cfg, device, calibration["lengthscale"]
            )
            adaptive_calibration = None
            beta_next = beta_used
            if cfg.adaptive_ess_target is not None:
                calibration_pools, calibration_gamma_counts = AD.feature_pools(
                    policy, store, cfg, device, round_i
                )
                adaptive_calibration = AD.calibrate_from_pools(
                    gp_post,
                    calibration_pools,
                    cfg,
                    round_i,
                    cfg.adaptive_ess_target,
                )
                adaptive_calibration["context_gamma_counts"] = calibration_gamma_counts
                beta_next = float(adaptive_calibration["beta"])
                cfg.beta = beta_next
            audit = AC.run_audit(
                policy, audit_contexts, env, goal, device,
                n_plans=cfg.audit_plans, nfe=cfg.nfe, n_theta=cfg.n_theta,
                seed=AFE2.named_seed(cfg.seed, "audit"),
            )
            evaluation, evaluation_timing = run_parallel_episodes(
                policy, gp_post, env, cfg, store, round_i, cfg.M_eval,
                device, executor, collect=False, viz=None, purpose="controller_eval",
                acquisition_mode=cfg.acquisition_mode,
            )
            rows, pooled = _controller_summary(evaluation, cfg, env)
            drawn = (update or {}).get("drawn_ids", {})
            trained_gamma = {}
            distinct_gamma = {}
            for query_id, count in drawn.items():
                key = str(round(float(store.q_gamma[query_id]), 2))
                trained_gamma[key] = trained_gamma.get(key, 0) + int(count)
                distinct_gamma[key] = distinct_gamma.get(key, 0) + 1
            record = {
                "round": round_i,
                "arm": "afe",
                "acquisition_mode": cfg.acquisition_mode,
                "beta_used": beta_used,
                "beta_next": beta_next,
                "adaptive_beta_calibration": adaptive_calibration,
                "rbf_offline_sweep": None,
                "n_D": len(store),
                "n_Dpos": store.n_pos(),
                "per_gamma": per_gamma,
                **acquisition,
                "gp_round_start": gp_start_diagnostics,
                "gp_buffer": gp_post_diagnostics,
                "rep_cos": AFE2.rep_cos_drift(policy, representation_probe, cfg),
                "V": audit["V"],
                "V_safe": audit["V_safe"],
                "V_full": audit["V_full"],
                "V_gamma": audit["V_gamma"],
                "V_safe_gamma": audit["V_safe_gamma"],
                "V_full_gamma": audit["V_full_gamma"],
                "V_counts_gamma": audit["counts_gamma"],
                "ctrl": rows,
                "ctrl_pooled": pooled,
                "trained_draws_gamma": trained_gamma,
                "trained_distinct_gamma": distinct_gamma,
                "n_train_distinct": 0 if update is None else update["n_distinct"],
                "t_gather": gather_seconds,
                "t_update": update_seconds,
                "gather_timing": gather_timing,
                "evaluation_timing": evaluation_timing,
            }
            if update is not None:
                record.update({
                    "steps": update["steps"],
                    "stop": update["stop"],
                    "cfm": update["cfm"],
                    "cfm_first": update["cfm_first"],
                    "cfm_last": update["cfm_last"],
                    "fstep_final": update["fstep_final"],
                    "fstep_max": update["fstep_max"],
                    "grad_norm": update["grad_norm"],
                    "rel_param_change": update["rel_param_change"],
                    "replay_window": update["replay_window"],
                    "replay_eligible": update["replay_eligible"],
                    "replay_fresh_draws": update["replay_fresh_draws"],
                    "replay_fresh_distinct": update["replay_fresh_distinct"],
                })
            write_probe(record)
            torch.save({
                "round": round_i,
                "viz": viz,
                "eps": [
                    {key: value for key, value in episode.items() if key != "step_stats"}
                    for episode in episodes
                ],
                "gp_buffer_query_ids": np.asarray(query_ids, dtype=np.int64),
                "gp_diagnostics": gp_post_diagnostics,
                "scene": scene,
                "audit": audit,
                "train_ids": np.asarray(sorted(drawn), dtype=np.int64),
                "train_counts": np.asarray(
                    [drawn[key] for key in sorted(drawn)], dtype=np.int64
                ),
                "goal": goal,
                "x0": env.x0.detach().cpu().numpy(),
            }, os.path.join(outdir, "viz_db", f"round{round_i}.pt"))
            HT._save_hp_atomic(
                policy, os.path.join(outdir, f"ckpt_{round_i}.pt"),
                extra={"iter": round_i, "recipe": recipe, "resumable": False},
            )
            print(
                f"[afe-rbf] r{round_i:03d} D {len(store)} D+ {store.n_pos()} "
                f"GP {gp_post.n}/{cfg.gp_cap} ESS/M {record.get('ess_med', float('nan')):.3f} "
                f"beta {beta_used:.4g}->{beta_next:.4g} "
                f"uplift {record.get('uplift_med', float('nan')):.4f} V {audit['V']:.3f} "
                f"SR {pooled['SR']:.2f} NVP {pooled['NVP']:.2f} "
                f"gather {gather_seconds:.1f}s update {update_seconds:.1f}s",
                flush=True,
            )
            gp_for_gather = gp_post
            gp_start_diagnostics = gp_post_diagnostics

    final_path = os.path.join(outdir, "final.pt")
    store_path = os.path.join(outdir, "dstore.pt")
    HT._save_hp_atomic(
        policy, final_path,
        extra={"iter": cfg.rounds, "recipe": recipe, "resumable": False},
    )
    store.save(store_path)
    required = [
        "recipe.json",
        "rbf_calibration.json",
        "probe.jsonl",
        "final.pt",
        "dstore.pt",
        *[f"ckpt_{index}.pt" for index in range(cfg.rounds + 1)],
        *[f"viz_db/round{index}.pt" for index in range(1, cfg.rounds + 1)],
    ]
    inventory = {}
    for relative in required:
        path = os.path.join(outdir, relative)
        if not os.path.isfile(path):
            raise RuntimeError(f"completion artifact is missing: {relative}")
        inventory[relative] = AFE2._sha256_file(path)
    complete = {
        "status": "COMPLETE",
        "algorithm": recipe["algorithm"],
        "completed_round": cfg.rounds,
        "scene_sha256": scene["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "source_git_commit": source_git_state["commit"],
        "artifact_sha256": inventory,
    }
    with open(os.path.join(outdir, "COMPLETE.json"), "w") as stream:
        json.dump(complete, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[afe-rbf] COMPLETE: {outdir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expected-ckpt-sha256", required=True)
    parser.add_argument("--scene-profile", choices=sorted(SCENE_PROFILES), required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rollout-replicas", type=int, default=2)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--M-eval", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--afe-steps", type=int, default=250)
    parser.add_argument("--afe-lr", type=float, default=1.0e-4)
    parser.add_argument("--gp-cap", type=int, default=512)
    parser.add_argument("--gp-lam", type=float, default=1.0e-2)
    parser.add_argument(
        "--acquisition-mode",
        choices=("sequential", "uniform"),
        default="sequential",
    )
    parser.add_argument("--adaptive-ess-target", type=float, default=None)
    parser.add_argument("--replay-window", type=int, default=None)
    parser.add_argument("--rbf-offline-sweep", action="store_true")
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=910)
    args = parser.parse_args()
    if args.K != 64 or args.B != 8 or args.batch != 128:
        raise ValueError("the first RBF study holds K=64, B=8, and batch=128 fixed")
    if args.afe_steps != 250 or args.afe_lr != 1.0e-4:
        raise ValueError("the first RBF study holds the AFE update at 250 steps and lr=1e-4")
    if args.rounds < 1 or args.rollout_replicas < 1 or args.M_eval < 1:
        raise ValueError("rounds, rollout replicas, and M-eval must be positive")
    if args.verifier_workers < 1:
        raise ValueError("verifier worker count must be positive")
    if args.adaptive_ess_target is not None and not 0.0 < args.adaptive_ess_target < 1.0:
        raise ValueError("adaptive ESS target must lie strictly between zero and one")
    if args.acquisition_mode == "uniform" and args.adaptive_ess_target is not None:
        raise ValueError("uniform acquisition does not use adaptive beta")
    if args.replay_window is not None and args.replay_window < 1:
        raise ValueError("replay window must be at least one round")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_sha256 = AFE2._sha256_file(args.ckpt)
    if checkpoint_sha256 != args.expected_ckpt_sha256.lower():
        raise ValueError(
            f"checkpoint hash {checkpoint_sha256} != expected {args.expected_ckpt_sha256.lower()}"
        )
    policy, checkpoint = HP.load_hp(args.ckpt, device="cpu")
    policy = policy.to(device)
    profile = get_scene_profile(args.scene_profile)
    checkpoint_model_sha256, checkpoint_contract, checkpoint_contract_sha256 = (
        AFE2.validate_checkpoint_contract(
            profile.name, policy, checkpoint, checkpoint_sha256
        )
    )
    if any(not parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("all encoder, trunk, and head parameters must remain trainable")
    source_git_state = AFE2._git_state()
    if (
        source_git_state["commit"] is None
        or source_git_state["tracked_dirty"] is not False
        or source_git_state["untracked_runtime_sources"] != []
    ):
        raise RuntimeError(
            "AFE-RBF requires committed clean source; "
            f"untracked runtime sources={source_git_state['untracked_runtime_sources']}"
        )
    env = build_scene(profile)
    GM2.GOAL_XY = np.asarray(profile.goal, dtype=float)
    cfg = AFERBFConfig(
        rounds=args.rounds,
        T=args.T,
        K=args.K,
        B=args.B,
        arm="afe",
        batch=args.batch,
        afe_steps=args.afe_steps,
        afe_lr=args.afe_lr,
        M_eval=args.M_eval,
        wall_plugs=profile.wall_plugs,
        start_eps=profile.start[0],
        goal_xy=profile.goal,
        scene_profile=profile.name,
        seed=args.seed,
        replicas=args.rollout_replicas,
        gp_cap=args.gp_cap,
        gp_lam=args.gp_lam,
        verifier_workers=args.verifier_workers,
        acquisition_mode=args.acquisition_mode,
        adaptive_ess_target=args.adaptive_ess_target,
        replay_window=args.replay_window,
        rbf_offline_sweep=args.rbf_offline_sweep,
    )
    print(
        f"[afe-rbf] scene={profile.name} rounds={cfg.rounds} replicas/gamma={cfg.replicas} "
        f"K={cfg.K} B={cfg.B} GPcap={cfg.gp_cap} workers={cfg.verifier_workers} "
        f"acquisition={cfg.acquisition_mode} adaptive_ESS={cfg.adaptive_ess_target} "
        f"replay_W={cfg.replay_window}",
        flush=True,
    )
    run(
        policy, env, cfg, device, args.outdir,
        args.ckpt, checkpoint_sha256, checkpoint_model_sha256,
        checkpoint_contract, checkpoint_contract_sha256, source_git_state,
    )


if __name__ == "__main__":
    main()
