"""SafeMPPI as a proposal source for same-verifier certified backup plans."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import grid_scene
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter


@dataclass(frozen=True)
class BackupProposal:
    plan: np.ndarray
    kind: str
    internal_feasible: bool | None

    def __post_init__(self) -> None:
        plan = np.asarray(self.plan, dtype=np.float32)
        if plan.shape != (10, 2) or not np.isfinite(plan).all():
            raise ValueError(f"backup plan must be finite [10,2], got {plan.shape}")
        object.__setattr__(self, "plan", plan.copy())


class SafeMPPIBackup:
    """Generate proposals only; this class makes no certification claim."""

    def __init__(
        self,
        *,
        smooth_weight: float = 8.0,
        retreat_weight: float = 1.0,
        max_debug_candidates: int = 24,
    ) -> None:
        config = grid_scene.mode1_config()
        config["smooth_weight"] = float(smooth_weight)
        config["goal_retreat_exp_weight"] = float(retreat_weight)
        config["debug_max_rollouts"] = max(int(max_debug_candidates), 1)
        self.adapter = SafeMPPIAdapter(**config)
        self.max_debug_candidates = int(max_debug_candidates)

    @torch.inference_mode()
    def propose(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        env,
        gamma: float,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[list[BackupProposal], dict[str, object]]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=device)
        planner_obstacles = grid_scene.planner_obstacles(env).to(device=device, dtype=torch.float32)
        _action, info = self.adapter.plan(
            state_tensor,
            goal_tensor,
            planner_obstacles,
            gamma=float(gamma),
            seed=int(seed),
            return_rollouts=True,
        )
        proposals: list[BackupProposal] = []
        seen: set[bytes] = set()

        def add(plan: object, kind: str, feasible: bool | None) -> None:
            value = np.asarray(plan, dtype=np.float32)
            if value.shape != (10, 2) or not np.isfinite(value).all():
                return
            key = value.tobytes()
            if key in seen:
                return
            seen.add(key)
            proposals.append(BackupProposal(value, kind, feasible))

        add(info.get("mean_sequence"), "weighted_mean", None)
        add(info.get("best_sequence"), "internal_best", bool(info.get("best_feasible_internal", False)))
        debug = info.get("debug_rollouts") or {}
        controls = np.asarray(debug.get("controls", []), dtype=np.float32)
        feasibility = np.asarray(debug.get("feasible", []), dtype=bool)
        # Internal feasibility only determines query order.  The external full
        # verifier remains authoritative for every proposal.
        order = np.argsort(~feasibility, kind="stable") if len(feasibility) == len(controls) else range(len(controls))
        for position in order:
            if len(proposals) >= self.max_debug_candidates + 2:
                break
            internal = bool(feasibility[position]) if len(feasibility) == len(controls) else None
            add(controls[position], "debug_candidate", internal)
        telemetry = {
            "internal_all_infeasible": bool(info.get("all_samples_infeasible_internal", False)),
            "internal_best_feasible": bool(info.get("best_feasible_internal", False)),
            "proposal_count": len(proposals),
            "polytope_size": info.get("polytope_size"),
        }
        return proposals, telemetry

