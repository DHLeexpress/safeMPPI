#!/usr/bin/env python3
"""Stage 02: real, balanced SafeMPPI *planned-window* demonstrations.

This stage deliberately does not reuse any legacy training target.  A legacy
seed/signature census may only influence the order in which seeds are tried.
For every receding-horizon step the data path is exactly::

    SafeMPPI H=10 proposal -> full verifier -> select safe by progress
        -> save that exact H=10 plan -> execute only plan[0]

The candidate episodes are resumable.  The final dataset contains 12 real
R-first and 12 real U-first successful trajectories for every gamma (unless a
smoke run explicitly disables the quota requirement).  There is no reflection,
trajectory padding, executed-window reconstruction, or legacy checkpoint use.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from .deps import assert_no_legacy_expansion_imports, sha256_file, write_dependency_manifest
from .dynamics import execute_first_action
from .fallback import BackupProposal, SafeMPPIBackup
from .scene import GAMMAS, GOAL, START, context_from_state, make_id_scene
from .schemas import QueryContext, query_content_hash
from .verifier import PlanVerification, verify_plan


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = PACKAGE_ROOT / "stage_results/02_planned_demos"
LEGACY_HINT_PATH = (
    PACKAGE_ROOT.parent
    / "giant_obstacle_ood/stage_results/02b_balanced_id/data/balanced_id_paths_all_gamma.npz"
)
SCHEMA_VERSION = "afe_planned_demo_v1"


@dataclass(frozen=True)
class DemoRunConfig:
    """Rollout and exact-balancing settings recorded in the manifest."""

    max_steps: int = 240
    reach_m: float = 0.20
    smooth_weight: float = 8.0
    retreat_weight: float = 1.0
    max_debug_candidates: int = 6
    max_proposals_per_step: int = 8
    quota_per_direction: int = 12
    max_candidate_seeds_per_gamma: int = 256
    seed0: int = 72_000

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reach_m <= 0.0:
            raise ValueError("reach_m must be positive")
        if self.smooth_weight < 0.0 or self.retreat_weight < 0.0:
            raise ValueError("SafeMPPI cost weights must be nonnegative")
        if self.max_debug_candidates < 0 or self.max_proposals_per_step <= 0:
            raise ValueError("proposal counts must be positive")
        if self.quota_per_direction <= 0:
            raise ValueError("quota_per_direction must be positive")
        if self.max_candidate_seeds_per_gamma <= 0:
            raise ValueError("max_candidate_seeds_per_gamma must be positive")


def gamma_tag(gamma: float) -> str:
    return f"{float(gamma):g}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(path)


def _np_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _environment_clearance(path: np.ndarray, env: Any) -> np.ndarray:
    obstacles = env.obstacles.detach().cpu().numpy().astype(np.float64, copy=False)
    if len(obstacles) == 0:
        return np.full(len(path), np.inf, dtype=np.float64)
    return (
        np.linalg.norm(path[:, None, :] - obstacles[None, :, :2], axis=2)
        - obstacles[None, :, 2]
        - float(env.r_robot)
    ).min(axis=1)


def _first_crossing_time(values: np.ndarray, threshold: float = 1.0) -> float:
    for index in range(1, len(values)):
        left, right = float(values[index - 1]), float(values[index])
        if left < threshold <= right:
            denominator = right - left
            fraction = 1.0 if abs(denominator) <= 1.0e-12 else (threshold - left) / denominator
            return float(index - 1 + fraction)
    return math.inf


def direction_class(path: np.ndarray, *, tie_tolerance_steps: float = 1.0e-5) -> str:
    """Classify a real successful path by its first x=1/y=1 crossing.

    Linear interpolation avoids an x-loop ordering bias.  Ties and missing
    crossings are explicitly unclassified and can never fill either quota.
    """

    xy = np.asarray(path, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"path must have shape [N,2], got {xy.shape}")
    right_time = _first_crossing_time(xy[:, 0])
    up_time = _first_crossing_time(xy[:, 1])
    if not np.isfinite(right_time) or not np.isfinite(up_time):
        return "unclassified"
    if abs(right_time - up_time) <= float(tie_tolerance_steps):
        return "unclassified"
    return "R-first" if right_time < up_time else "U-first"


def _proposal_metrics(result: Any) -> dict[str, Any]:
    """Serialize the structured verifier output without changing its label."""

    return {
        "safe": bool(result.safe),
        "in_bounds": bool(getattr(result, "in_bounds", result.safe)),
        "socp_ok": bool(getattr(result, "socp_ok", result.safe)),
        "bounds_margin_m": _np_float(getattr(result, "bounds_margin_m", math.nan)),
        "physical_clearance_m": _np_float(
            getattr(result, "physical_clearance_m", math.nan)
        ),
        "face_margin_m": _np_float(getattr(result, "face_margin_m", math.nan)),
        "certificate_residual": _np_float(
            getattr(result, "certificate_residual", math.nan)
        ),
        "certificate_worst_step": int(
            getattr(result, "certificate_worst_step", -1)
        ),
        "progress_m": _np_float(getattr(result, "progress_m", -math.inf)),
        "start_goal_distance_m": _np_float(
            getattr(result, "start_goal_distance_m", math.nan)
        ),
        "terminal_goal_distance_m": _np_float(
            getattr(result, "terminal_goal_distance_m", math.nan)
        ),
    }


def _context_arrays(contexts: Sequence[QueryContext]) -> dict[str, np.ndarray]:
    if not contexts:
        # Empty smoke/fail-closed episodes still receive explicit arrays.  The
        # final balanced dataset never contains an empty successful episode.
        return {
            "context_grid": np.empty((0, 0), dtype=np.float32),
            "context_low5": np.empty((0, 0), dtype=np.float32),
            "context_hist": np.empty((0, 0), dtype=np.float32),
        }
    return {
        "context_grid": np.asarray([item.grid for item in contexts], dtype=np.float32),
        "context_low5": np.asarray([item.low5 for item in contexts], dtype=np.float32),
        "context_hist": np.asarray([item.hist for item in contexts], dtype=np.float32),
    }


def _default_backup(config: DemoRunConfig) -> SafeMPPIBackup:
    return SafeMPPIBackup(
        smooth_weight=config.smooth_weight,
        retreat_weight=config.retreat_weight,
        max_debug_candidates=config.max_debug_candidates,
    )


@torch.inference_mode()
def run_expert_rollout(
    *,
    env: Any,
    gamma: float,
    seed: int,
    device: torch.device,
    config: DemoRunConfig,
    backup: Any | None = None,
    verify_fn: Callable[..., Any] = verify_plan,
    context_fn: Callable[..., QueryContext] = context_from_state,
) -> dict[str, Any]:
    """Generate one real receding-horizon trajectory under the clean contract.

    The return object keeps all queried H=10 plans and their step indices, so
    every verifier call can be reconstructed as ``(context[step], plan)``.
    ``training_plans`` is an exact view of selected verified-safe query plans.
    """

    if backup is None:
        backup = _default_backup(config)
    state = np.asarray(env.x0.detach().cpu().numpy(), dtype=np.float64).copy()
    goal = np.asarray(env.goal.detach().cpu().numpy(), dtype=np.float64).reshape(-1)[:2]
    initial_state = state.copy()
    states: list[np.ndarray] = [state.copy()]
    executed_actions: list[np.ndarray] = []
    contexts: list[QueryContext] = []
    query_plans: list[np.ndarray] = []
    query_steps: list[int] = []
    query_hashes: list[str] = []
    query_kinds: list[str] = []
    query_internal_feasible: list[int] = []
    query_metrics: list[dict[str, Any]] = []
    selected_query_indices: list[int] = []
    telemetry_rows: list[dict[str, Any]] = []
    dead_reason: str | None = None
    started = time.perf_counter()

    # Never query/execute after the episode is already at its goal.
    if float(np.linalg.norm(state[:2] - goal)) < config.reach_m:
        dead_reason = None
    else:
        for step in range(config.max_steps):
            context = context_fn(state, goal, gamma, executed_actions, env)
            if not isinstance(context, QueryContext):
                raise TypeError("context_fn must return QueryContext")
            contexts.append(context)
            proposals, telemetry = backup.propose(
                state,
                goal,
                env,
                float(gamma),
                seed=int(seed) * 10_000 + step,
                device=device,
            )
            proposals = list(proposals)[: config.max_proposals_per_step]
            telemetry_rows.append({"step": step, **dict(telemetry)})
            safe_at_step: list[tuple[float, int]] = []
            for proposal in proposals:
                if not isinstance(proposal, BackupProposal):
                    # Tests and alternative proposal sources may use any object
                    # exposing the same immutable fields.
                    plan = np.asarray(proposal.plan, dtype=np.float32).copy()
                    kind = str(getattr(proposal, "kind", "proposal"))
                    internal = getattr(proposal, "internal_feasible", None)
                else:
                    plan = np.asarray(proposal.plan, dtype=np.float32).copy()
                    kind = proposal.kind
                    internal = proposal.internal_feasible
                if plan.shape != (10, 2) or not np.isfinite(plan).all():
                    raise ValueError(f"proposal must be a finite H=10 plan, got {plan.shape}")

                generated_hash = query_content_hash(context, gamma, plan)
                verifier_plan = plan.copy()
                result = verify_fn(state, verifier_plan, env, gamma, goal=goal)
                # A verifier must never mutate the object it was asked about.
                if not np.array_equal(verifier_plan, plan):
                    raise RuntimeError("full verifier mutated its planned-window input")
                verifier_hash = query_content_hash(context, gamma, verifier_plan)
                if generated_hash != verifier_hash:
                    raise RuntimeError("generated and fully verified plan identities differ")

                query_index = len(query_plans)
                query_plans.append(plan)
                query_steps.append(step)
                query_hashes.append(generated_hash)
                query_kinds.append(kind)
                query_internal_feasible.append(-1 if internal is None else int(bool(internal)))
                metrics = _proposal_metrics(result)
                query_metrics.append(metrics)
                if metrics["safe"]:
                    safe_at_step.append((metrics["progress_m"], query_index))

            if not safe_at_step:
                # Fail closed: no state transition and no target is emitted.
                dead_reason = "no_certified_plan"
                break

            # Progress ranks already-safe proposals only.  Stable query index
            # tie-breaking makes the selection exactly reproducible.
            _progress, selected_index = max(
                safe_at_step, key=lambda pair: (pair[0], -pair[1])
            )
            selected_plan = query_plans[selected_index]
            selected_hash = query_hashes[selected_index]
            if selected_hash != query_content_hash(context, gamma, selected_plan):
                raise RuntimeError("selected training target identity changed after verification")
            selected_query_indices.append(selected_index)

            executed_action = np.asarray(selected_plan[0], dtype=np.float64).copy()
            next_state = execute_first_action(state, selected_plan)
            # The action log is the literal first row of the saved target.
            if not np.array_equal(
                executed_action.astype(selected_plan.dtype, copy=False), selected_plan[0]
            ):
                raise RuntimeError("executed action is not selected_plan[0]")
            executed_actions.append(executed_action.astype(np.float32))
            state = next_state
            states.append(state.copy())

            position = state[:2]
            clearance = float(_environment_clearance(position[None], env)[0])
            if clearance < 0.0:
                dead_reason = "collision_after_verified_action"
                break
            if bool(np.any(position < 0.0) or np.any(position > 5.0)):
                dead_reason = "out_of_bounds_after_verified_action"
                break
            if float(np.linalg.norm(position - goal)) < config.reach_m:
                break
        else:
            dead_reason = "timeout"

    states_array = np.asarray(states, dtype=np.float32)
    path = states_array[:, :2]
    actions_array = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 2)
    query_plan_array = np.asarray(query_plans, dtype=np.float32).reshape(-1, 10, 2)
    selected_indices_array = np.asarray(selected_query_indices, dtype=np.int64)
    training_plans = (
        query_plan_array[selected_indices_array]
        if len(selected_indices_array)
        else np.empty((0, 10, 2), dtype=np.float32)
    )
    training_hashes = [query_hashes[index] for index in selected_query_indices]
    selected_context_steps = [query_steps[index] for index in selected_query_indices]
    if selected_context_steps != list(range(len(training_plans))):
        raise RuntimeError("each executed action must select exactly one query at its current step")
    if len(training_plans) != len(actions_array):
        raise RuntimeError("one exact planned target is required for every executed action")
    if len(training_plans) and not np.array_equal(training_plans[:, 0], actions_array):
        raise RuntimeError("saved target first actions differ from executed action log")

    endpoint_distance = float(np.linalg.norm(path[-1] - goal))
    reached = endpoint_distance < config.reach_m
    clearance = _environment_clearance(path.astype(np.float64), env)
    collision = bool(np.min(clearance, initial=np.inf) < 0.0)
    in_bounds = bool(np.all((path >= 0.0) & (path <= 5.0)))
    success = bool(reached and not collision and in_bounds and dead_reason is None)
    if dead_reason is None and not reached:
        dead_reason = "timeout"
    route_class = direction_class(path) if success else "unclassified"
    status = "success" if success else str(dead_reason)

    metric_arrays: dict[str, np.ndarray] = {}
    metric_fields = (
        "safe",
        "in_bounds",
        "socp_ok",
        "bounds_margin_m",
        "physical_clearance_m",
        "face_margin_m",
        "certificate_residual",
        "certificate_worst_step",
        "progress_m",
        "start_goal_distance_m",
        "terminal_goal_distance_m",
    )
    for field in metric_fields:
        values = [row[field] for row in query_metrics]
        if field in {"safe", "in_bounds", "socp_ok"}:
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=bool)
        elif field == "certificate_worst_step":
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=np.int16)
        else:
            metric_arrays[f"query_{field}"] = np.asarray(values, dtype=np.float64)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gamma": float(gamma),
        "seed": int(seed),
        "status": status,
        "success": success,
        "reached": bool(reached),
        "collision": collision,
        "in_bounds": in_bounds,
        "dead_reason": dead_reason,
        "direction_class": route_class,
        "steps": len(actions_array),
        "queries": len(query_plans),
        "safe_queries": int(sum(row["safe"] for row in query_metrics)),
        "query_acceptance": float(np.mean([row["safe"] for row in query_metrics]))
        if query_metrics
        else 0.0,
        "endpoint_distance_m": endpoint_distance,
        "min_clearance_m": float(np.min(clearance, initial=np.inf)),
        "path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "wall_seconds": time.perf_counter() - started,
        "initial_state": initial_state.astype(np.float32),
        "states": states_array,
        "path": path,
        "executed_actions": actions_array,
        "contexts": contexts,
        "query_plans": query_plan_array,
        "query_steps": np.asarray(query_steps, dtype=np.int32),
        "query_hashes": query_hashes,
        "query_kinds": query_kinds,
        "query_internal_feasible": np.asarray(query_internal_feasible, dtype=np.int8),
        "selected_query_indices": selected_indices_array,
        "training_plans": training_plans,
        "training_hashes": training_hashes,
        "telemetry": telemetry_rows,
        **metric_arrays,
    }
    return result


def _episode_stem(gamma: float, seed: int) -> str:
    return f"g{gamma_tag(gamma)}_seed{int(seed)}"


def save_episode(episode: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Persist one candidate with enough data to revalidate every query hash."""

    directory.mkdir(parents=True, exist_ok=True)
    stem = _episode_stem(episode["gamma"], episode["seed"])
    array_path = directory / f"{stem}.npz"
    meta_path = directory / f"{stem}.json"
    contexts = _context_arrays(episode["contexts"])
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            states=episode["states"],
            executed_actions=episode["executed_actions"],
            query_plans=episode["query_plans"],
            query_steps=episode["query_steps"],
            query_hashes=np.asarray(episode["query_hashes"], dtype="U64"),
            query_kinds=np.asarray(episode["query_kinds"], dtype="U32"),
            query_internal_feasible=episode["query_internal_feasible"],
            selected_query_indices=episode["selected_query_indices"],
            **contexts,
            **{
                key: value
                for key, value in episode.items()
                if key.startswith("query_")
                and isinstance(value, np.ndarray)
                and key
                not in {
                    "query_plans",
                    "query_steps",
                    "query_internal_feasible",
                }
            },
        )
    temporary.replace(array_path)
    omitted = {
        "initial_state",
        "states",
        "path",
        "executed_actions",
        "contexts",
        "query_plans",
        "query_steps",
        "query_hashes",
        "query_kinds",
        "query_internal_feasible",
        "selected_query_indices",
        "training_plans",
        "training_hashes",
        "telemetry",
    }
    omitted.update(key for key in episode if key.startswith("query_") and isinstance(episode[key], np.ndarray))
    metadata = {key: value for key, value in episode.items() if key not in omitted}
    metadata.update(
        {
            "array_file": array_path.name,
            "array_sha256": sha256_file(array_path),
            "training_hashes": episode["training_hashes"],
            "telemetry": episode["telemetry"],
            "identity_contract": {
                "generated_equals_verifier_input_equals_training_target": True,
                "executed_action_equals_training_plan_first_action": True,
            },
        }
    )
    _atomic_json(meta_path, metadata)
    return array_path, meta_path


def load_episode(meta_path: Path, *, validate: bool = True) -> dict[str, Any]:
    metadata = json.loads(meta_path.read_text())
    array_path = meta_path.parent / metadata["array_file"]
    if sha256_file(array_path) != metadata["array_sha256"]:
        raise RuntimeError(f"candidate array checksum mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    contexts = [
        QueryContext(arrays["context_grid"][i], arrays["context_low5"][i], arrays["context_hist"][i])
        for i in range(len(arrays["context_grid"]))
    ]
    query_hashes = arrays["query_hashes"].astype(str).tolist()
    selected = arrays["selected_query_indices"].astype(np.int64)
    training_plans = arrays["query_plans"][selected]
    training_hashes = [query_hashes[index] for index in selected]
    episode: dict[str, Any] = {
        **metadata,
        **arrays,
        "contexts": contexts,
        "path": arrays["states"][:, :2],
        "query_hashes": query_hashes,
        "query_kinds": arrays["query_kinds"].astype(str).tolist(),
        "training_plans": training_plans,
        "training_hashes": training_hashes,
    }
    if validate:
        gamma = float(metadata["gamma"])
        steps = arrays["query_steps"].astype(int)
        plans = arrays["query_plans"]
        if len(steps) != len(plans) or len(query_hashes) != len(plans):
            raise RuntimeError(f"query array length mismatch: {array_path}")
        for index, (step, plan, stored_hash) in enumerate(zip(steps, plans, query_hashes)):
            if not 0 <= step < len(contexts):
                raise RuntimeError(f"query {index} has invalid context step {step}")
            actual = query_content_hash(contexts[step], gamma, plan)
            if actual != stored_hash:
                raise RuntimeError(f"query identity mismatch at {array_path}:{index}")
        if len(selected) != len(arrays["executed_actions"]):
            raise RuntimeError("selected queries/actions length mismatch")
        if len(selected) and not np.array_equal(
            plans[selected, 0], arrays["executed_actions"]
        ):
            raise RuntimeError("loaded executed actions differ from selected plan[0]")
        if training_hashes != metadata["training_hashes"]:
            raise RuntimeError("loaded training-target identity list differs from manifest")
    return episode


def legacy_seed_hints(gamma: float, path: Path = LEGACY_HINT_PATH) -> tuple[list[int], dict[str, Any]]:
    """Read only legacy ``gammas/seeds/signatures`` as candidate-order hints."""

    if not path.exists():
        return [], {"available": False, "path": str(path)}
    with np.load(path, allow_pickle=False) as payload:
        # Do not access legacy paths, states, controls, windows, or checkpoints.
        old_gammas = payload["gammas"].astype(float)
        old_seeds = payload["seeds"].astype(np.int64)
        old_signatures = payload["signatures"].astype(str)
    mask = np.isclose(old_gammas, float(gamma), atol=1.0e-7)
    right = [int(seed) for seed, word in zip(old_seeds[mask], old_signatures[mask]) if word.startswith("R")]
    up = [int(seed) for seed, word in zip(old_seeds[mask], old_signatures[mask]) if word.startswith("U")]
    ordered: list[int] = []
    for index in range(max(len(right), len(up))):
        if index < len(right):
            ordered.append(right[index])
        if index < len(up):
            ordered.append(up[index])
    # Stable de-duplication; a seed is evaluated once under the new contract.
    ordered = list(dict.fromkeys(ordered))
    return ordered, {
        "available": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "fields_read": ["gammas", "seeds", "signatures"],
        "fields_explicitly_not_reused": [
            "paths",
            "states",
            "controls",
            "legacy training windows",
            "legacy checkpoints",
        ],
        "matched_hint_count": len(ordered),
    }


def candidate_seed_order(gamma: float, config: DemoRunConfig) -> tuple[list[int], dict[str, Any]]:
    hints, provenance = legacy_seed_hints(gamma)
    fallback = range(config.seed0, config.seed0 + config.max_candidate_seeds_per_gamma * 4)
    ordered = list(dict.fromkeys([*hints, *fallback]))
    return ordered[: config.max_candidate_seeds_per_gamma], provenance


def _candidate_meta_path(directory: Path, gamma: float, seed: int) -> Path:
    return directory / f"{_episode_stem(gamma, seed)}.json"


def _quality_score(episode: dict[str, Any]) -> tuple[float, float, int]:
    path = np.asarray(episode["path"], dtype=np.float64)
    goal_distance = np.linalg.norm(path - GOAL.astype(np.float64)[None], axis=1)
    retreat = float(np.maximum(np.diff(goal_distance), 0.0).sum())
    # Prefer less backtracking, then higher clearance, then shorter execution.
    return (
        retreat,
        -float(episode["min_clearance_m"]),
        int(episode["steps"]),
    )


def select_exact_balance(
    episodes: Iterable[dict[str, Any]], quota: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = {
        label: [
            episode
            for episode in episodes
            if episode["success"] and episode["direction_class"] == label
        ]
        for label in ("R-first", "U-first")
    }
    for label in groups:
        groups[label].sort(key=lambda item: (_quality_score(item), int(item["seed"])))
    if any(len(groups[label]) < quota for label in groups):
        raise RuntimeError(
            "insufficient real balanced trajectories: "
            + ", ".join(f"{label}={len(groups[label])}/{quota}" for label in groups)
        )
    selected = groups["R-first"][:quota] + groups["U-first"][:quota]
    selected.sort(key=lambda item: (item["direction_class"], int(item["seed"])))
    audit = {
        "R-first": len([item for item in selected if item["direction_class"] == "R-first"]),
        "U-first": len([item for item in selected if item["direction_class"] == "U-first"]),
        "real_successful_trajectories_only": bool(all(item["success"] for item in selected)),
        "synthetic_reflections": 0,
        "trajectory_padding": 0,
        "target_padding": 0,
    }
    return selected, audit


def build_dataset(
    selected_by_gamma: dict[float, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    grids: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    plans: list[np.ndarray] = []
    hashes: list[str] = []
    gammas: list[float] = []
    seeds: list[int] = []
    trajectory_ids: list[int] = []
    trajectory_steps: list[int] = []
    directions: list[int] = []
    query_progress: list[float] = []
    query_clearance: list[float] = []
    target_safe: list[bool] = []
    target_in_bounds: list[bool] = []
    target_socp_ok: list[bool] = []
    trajectory_rows: list[dict[str, Any]] = []

    trajectory_id = 0
    for gamma in sorted(selected_by_gamma):
        for episode in selected_by_gamma[gamma]:
            selected = np.asarray(episode["selected_query_indices"], dtype=np.int64)
            query_steps = np.asarray(episode["query_steps"], dtype=np.int64)
            query_plans = np.asarray(episode["query_plans"], dtype=np.float32)
            query_hashes = list(episode["query_hashes"])
            contexts: list[QueryContext] = episode["contexts"]
            if len(selected) != int(episode["steps"]):
                raise RuntimeError("successful trajectory is missing selected planned targets")
            for local_step, query_index in enumerate(selected):
                context_step = int(query_steps[query_index])
                if context_step != local_step:
                    raise RuntimeError("selected target/context step mismatch")
                context = contexts[context_step]
                plan = query_plans[query_index]
                identity = query_content_hash(context, gamma, plan)
                if identity != query_hashes[query_index]:
                    raise RuntimeError("training target differs from fully verified query")
                grids.append(np.asarray(context.grid, dtype=np.float32))
                lows.append(np.asarray(context.low5, dtype=np.float32))
                histories.append(np.asarray(context.hist, dtype=np.float32))
                plans.append(plan.copy())
                hashes.append(identity)
                gammas.append(float(gamma))
                seeds.append(int(episode["seed"]))
                trajectory_ids.append(trajectory_id)
                trajectory_steps.append(local_step)
                directions.append(0 if episode["direction_class"] == "R-first" else 1)
                query_progress.append(float(episode["query_progress_m"][query_index]))
                query_clearance.append(
                    float(episode["query_physical_clearance_m"][query_index])
                )
                target_safe.append(bool(episode["query_safe"][query_index]))
                target_in_bounds.append(bool(episode["query_in_bounds"][query_index]))
                target_socp_ok.append(bool(episode["query_socp_ok"][query_index]))
            trajectory_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "gamma": float(gamma),
                    "seed": int(episode["seed"]),
                    "direction_class": episode["direction_class"],
                    "steps": int(episode["steps"]),
                    "min_clearance_m": float(episode["min_clearance_m"]),
                    "path_length_m": float(episode["path_length_m"]),
                    "query_acceptance": float(episode["query_acceptance"]),
                }
            )
            trajectory_id += 1

    if not plans:
        raise RuntimeError("cannot build a training dataset with no verified planned targets")
    if not all(target_safe) or not all(target_in_bounds) or not all(target_socp_ok):
        raise RuntimeError("a selected training target is not bounds+SOCP verified-safe")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "grid": torch.from_numpy(np.asarray(grids, dtype=np.float32)),
        "low5": torch.from_numpy(np.asarray(lows, dtype=np.float32)),
        "hist": torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        "U": torch.from_numpy(np.asarray(plans, dtype=np.float32)),
        "gamma": torch.tensor(gammas, dtype=torch.float32),
        "window_seeds": torch.tensor(seeds, dtype=torch.long),
        "window_trajectory_ids": torch.tensor(trajectory_ids, dtype=torch.long),
        "source_trajectory_ids": torch.tensor(trajectory_ids, dtype=torch.long),
        "window_steps": torch.tensor(trajectory_steps, dtype=torch.int32),
        "window_direction": torch.tensor(directions, dtype=torch.int8),
        "query_progress_m": torch.tensor(query_progress, dtype=torch.float32),
        "query_physical_clearance_m": torch.tensor(query_clearance, dtype=torch.float32),
        "target_safe": torch.tensor(target_safe, dtype=torch.bool),
        "target_in_bounds": torch.tensor(target_in_bounds, dtype=torch.bool),
        "target_socp_ok": torch.tensor(target_socp_ok, dtype=torch.bool),
        "query_hashes": hashes,
        "target_query_hash": list(hashes),
        "generated_hash": list(hashes),
        "verifier_input_hash": list(hashes),
        "training_target_hash": list(hashes),
        "generated_hashes": list(hashes),
        "verifier_input_hashes": list(hashes),
        "training_target_hashes": list(hashes),
        "trajectory_rows": trajectory_rows,
        "start": torch.from_numpy(START.copy()),
        "goal": torch.from_numpy(GOAL.copy()),
        "contract": {
            "generated_equals_verified_equals_training": True,
            "planned_horizon": 10,
            "only_first_action_executed": True,
            "all_targets_pre_execution_fully_verified": True,
            "progress_not_in_safety_label": True,
            "synthetic_reflections": 0,
            "padding": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return payload


def _draw_scene(axis: Any, env: Any) -> None:
    obstacles = env.obstacles.detach().cpu().numpy()
    for x, y, radius in obstacles:
        axis.add_patch(Circle((float(x), float(y)), float(radius), color="0.72", zorder=1))
    axis.scatter(*START, marker="s", s=34, color="black", zorder=6)
    axis.scatter(*GOAL, marker="*", s=120, color="#ffd21f", edgecolor="black", zorder=6)
    axis.set(xlim=(-0.15, 5.15), ylim=(-0.15, 5.15), aspect="equal")
    axis.set_xticks([])
    axis.set_yticks([])


def render_selected(
    env: Any, selected_by_gamma: dict[float, list[dict[str, Any]]], output: Path
) -> None:
    gamma_values = sorted(selected_by_gamma)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(gamma_values)))
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4))
    for axis, gamma, color in zip(axes.ravel(), gamma_values, colors):
        _draw_scene(axis, env)
        for episode in selected_by_gamma[gamma]:
            linestyle = "-" if episode["direction_class"] == "R-first" else "--"
            path = np.asarray(episode["path"])
            axis.plot(path[:, 0], path[:, 1], color=color, lw=1.0, alpha=0.66, ls=linestyle)
        r_count = sum(item["direction_class"] == "R-first" for item in selected_by_gamma[gamma])
        u_count = sum(item["direction_class"] == "U-first" for item in selected_by_gamma[gamma])
        axis.set_title(rf"$\gamma={gamma:g}$: R={r_count}, U={u_count}")
    for axis in axes.ravel()[len(gamma_values):]:
        axis.axis("off")
    fig.suptitle(
        "Real fully verified planned-window SafeMPPI demonstrations\n"
        "solid: R-first, dashed: U-first (no reflection or padding)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_target_audit(payload: dict[str, Any], output: Path) -> None:
    gamma = payload["gamma"].cpu().numpy()
    progress = payload["query_progress_m"].cpu().numpy()
    clearance = payload["query_physical_clearance_m"].cpu().numpy()
    direction = payload["window_direction"].cpu().numpy()
    gammas = sorted(set(float(value) for value in gamma))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for code, label, marker in ((0, "R-first", "o"), (1, "U-first", "^")):
        mask = direction == code
        axes[0].scatter(gamma[mask], progress[mask], s=7, alpha=0.24, marker=marker, label=label)
        axes[1].scatter(gamma[mask], clearance[mask], s=7, alpha=0.24, marker=marker, label=label)
    axes[0].axhline(0.0, color="black", lw=0.7, ls=":")
    axes[0].set(title="Verified training plans: progress (ranking only)", ylabel="H=10 progress [m]")
    axes[1].axhline(0.0, color="black", lw=0.7, ls=":")
    axes[1].set(title="Verified training plans: physical clearance", ylabel="clearance [m]")
    for axis in axes:
        axis.set_xlabel(r"safety level $\gamma$")
        axis.set_xticks(gammas)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("The exact planned samples passed to pretraining")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _summarize_candidates(episodes: Sequence[dict[str, Any]], gamma: float) -> dict[str, Any]:
    return {
        "gamma": float(gamma),
        "candidates": len(episodes),
        "successes": sum(bool(item["success"]) for item in episodes),
        "R-first_successes": sum(
            bool(item["success"]) and item["direction_class"] == "R-first" for item in episodes
        ),
        "U-first_successes": sum(
            bool(item["success"]) and item["direction_class"] == "U-first" for item in episodes
        ),
        "unclassified_successes": sum(
            bool(item["success"]) and item["direction_class"] == "unclassified" for item in episodes
        ),
        "fail_closed": sum(item.get("dead_reason") == "no_certified_plan" for item in episodes),
        "queries": sum(int(item["queries"]) for item in episodes),
        "safe_queries": sum(int(item["safe_queries"]) for item in episodes),
    }


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    gammas = tuple(float(value) for value in args.gammas)
    unknown = sorted(set(gammas) - set(float(value) for value in GAMMAS))
    if unknown:
        raise ValueError(f"unsupported gammas: {unknown}")
    config = DemoRunConfig(
        max_steps=args.max_steps,
        reach_m=args.reach,
        smooth_weight=args.smooth_weight,
        retreat_weight=args.retreat_weight,
        max_debug_candidates=args.max_debug_candidates,
        max_proposals_per_step=args.max_proposals,
        quota_per_direction=args.quota,
        max_candidate_seeds_per_gamma=args.max_candidate_seeds,
        seed0=args.seed0,
    )
    outdir = args.outdir.resolve()
    candidate_dir = outdir / "data/candidates"
    for directory in (candidate_dir, outdir / "data", outdir / "logs", outdir / "tables", outdir / "viz"):
        directory.mkdir(parents=True, exist_ok=True)
    dependency_manifest = write_dependency_manifest(outdir / "logs/dependencies.json")
    assert_no_legacy_expansion_imports()
    env = make_id_scene()
    env.T = int(config.max_steps)
    started = time.perf_counter()
    selected_by_gamma: dict[float, list[dict[str, Any]]] = {}
    all_by_gamma: dict[float, list[dict[str, Any]]] = {}
    hint_rows: dict[str, Any] = {}

    for gamma in gammas:
        seed_order, hint_info = candidate_seed_order(gamma, config)
        hint_rows[gamma_tag(gamma)] = hint_info
        episodes: list[dict[str, Any]] = []
        for ordinal, seed in enumerate(seed_order, start=1):
            meta_path = _candidate_meta_path(candidate_dir, gamma, seed)
            if meta_path.exists():
                episode = load_episode(meta_path)
            else:
                backup = _default_backup(config)
                episode = run_expert_rollout(
                    env=env,
                    gamma=gamma,
                    seed=seed,
                    device=device,
                    config=config,
                    backup=backup,
                )
                save_episode(episode, candidate_dir)
            episodes.append(episode)
            counts = _summarize_candidates(episodes, gamma)
            print(
                f"[planned-demo] gamma={gamma:g} seed={seed} ({ordinal}/{len(seed_order)}) "
                f"status={episode['status']} class={episode['direction_class']} "
                f"R={counts['R-first_successes']} U={counts['U-first_successes']} "
                f"queries={episode['queries']}",
                flush=True,
            )
            if args.smoke:
                if ordinal >= args.smoke_seeds:
                    break
            elif (
                counts["R-first_successes"] >= config.quota_per_direction
                and counts["U-first_successes"] >= config.quota_per_direction
            ):
                break
        all_by_gamma[gamma] = episodes
        if not args.smoke:
            selected, balance = select_exact_balance(episodes, config.quota_per_direction)
            selected_by_gamma[gamma] = selected
            print(f"[balance] gamma={gamma:g} {balance}", flush=True)

    candidate_rows = [_summarize_candidates(all_by_gamma[gamma], gamma) for gamma in gammas]
    with (outdir / "tables/candidate_census.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    dataset_path: Path | None = None
    dataset_hash: str | None = None
    window_count = 0
    if not args.smoke:
        dataset_path = outdir / "data/planned_id_balanced.pt"
        payload = build_dataset(selected_by_gamma, dataset_path)
        dataset_hash = sha256_file(dataset_path)
        window_count = len(payload["U"])
        render_selected(env, selected_by_gamma, outdir / "viz/selected_real_paths.png")
        render_target_audit(payload, outdir / "viz/training_target_audit.png")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "SMOKE_COMPLETE" if args.smoke else "PLANNED_DEMOS_COMPLETE",
        "created_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "cuda_visible_devices": str(__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scene": {
            "name": "ordinary_symmetric_4x4_ID_stadium",
            "start": START.tolist(),
            "goal": GOAL.tolist(),
            "gammas": list(gammas),
        },
        "config": asdict(config),
        "contract": {
            "generated_object": "SafeMPPI planned H=10 control window",
            "queried_object": "same planned H=10 control window",
            "verified_object": "same planned H=10 control window",
            "training_object": "same planned H=10 control window",
            "executed_object": "only first action of selected verified-safe plan",
            "safe_selection": "maximum progress among full-verifier-safe plans",
            "no_safe_behavior": "fail closed; no action and no target",
            "progress_is_safety_label": False,
            "synthetic_reflections": 0,
            "target_padding": 0,
            "legacy_training_targets_reused": False,
            "legacy_checkpoints_reused": False,
        },
        "legacy_seed_hints": hint_rows,
        "candidate_census": candidate_rows,
        "balance": None
        if args.smoke
        else {
            gamma_tag(gamma): {
                "R-first": sum(item["direction_class"] == "R-first" for item in selected_by_gamma[gamma]),
                "U-first": sum(item["direction_class"] == "U-first" for item in selected_by_gamma[gamma]),
            }
            for gamma in gammas
        },
        "training_windows": window_count,
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "dataset_sha256": dataset_hash,
        "dependency_manifest": dependency_manifest,
    }
    _atomic_json(outdir / "manifest.json", manifest)
    _atomic_json(outdir / "logs/stage_summary.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "candidate_census": candidate_rows,
        "training_windows": window_count,
        "manifest": str(outdir / "manifest.json"),
    }, indent=2), flush=True)
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke"))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gammas", nargs="+", type=float, default=list(GAMMAS))
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--reach", type=float, default=0.20)
    parser.add_argument("--smooth-weight", type=float, default=8.0)
    parser.add_argument("--retreat-weight", type=float, default=1.0)
    parser.add_argument("--max-debug-candidates", type=int, default=6)
    parser.add_argument("--max-proposals", type=int, default=8)
    parser.add_argument("--quota", type=int, default=12)
    parser.add_argument("--max-candidate-seeds", type=int, default=256)
    parser.add_argument("--seed0", type=int, default=72_000)
    parser.add_argument("--smoke-seeds", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    args.smoke = args.command == "smoke"
    if args.smoke_seeds <= 0:
        raise ValueError("smoke-seeds must be positive")
    run_stage(args)


if __name__ == "__main__":
    main()
