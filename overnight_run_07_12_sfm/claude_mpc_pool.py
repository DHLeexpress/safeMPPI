"""Privileged MPC candidate-pool harvesting into a separate D_MPC+ buffer.

Imports the original Codex privileged candidate-pool logic (read-only source:
``agent/sfm-adhoc-controller-overlay-20260726`` @ 0e0eca2,
``sfm_adhoc_controller_compare.privileged_sfm_config`` and the pool machinery
in ``sfm_kazuki``) without modifying either.  At declared hard contexts of an
ordinary B1 gathering round this module:

1. reconstructs the LIVE reactive SFM crowd by deterministic prefix replay of
   the episode's executed actions (verified against the stored context);
2. regenerates the original Codex MPC pool at that context: guided flow
   sampling (n_sample=200) + MPPI refinement + brake/goal/avoidance templates
   + 25 constant-acceleration escapes, exactly as
   ``exact_sfm_horizon_filter_action`` constructs it;
3. labels every candidate with BOTH the privileged exact-SFM look-ahead
   feasibility (recoverable AND horizon clearance >= the per-gamma hard
   margin) and our canonical exact full-H10 SOCP verifier;
4. keeps only the intersection, ranks it by the native frozen SafeMPPI
   proposal cost, and stores the top ``KEEP_PER_CONTEXT`` (context, U) pairs
   in a fresh per-round ``D_MPC+`` buffer.

``D_MPC+`` NEVER enters the ordinary D/D+, the GP buffer, or acquisition.
Records carry no x0: distillation uses ``policy.cfm_loss`` which draws fresh
Gaussian CFM bases at every step.  The privileged controller itself is never
used at evaluation time.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_cost as BC
import sfm_b1_store as BS
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS

KEEP_PER_CONTEXT = 2
POOL_SEED = 700_000
H = 10


def privileged_sfm_config():
    """Verbatim import of the historical v3 wrapper recipe.

    Source: agent/sfm-adhoc-controller-overlay-20260726 @ 0e0eca2,
    sfm_adhoc_controller_compare.privileged_sfm_config (read-only).
    """
    gammas = tuple(map(float, SS.GAMMAS))
    return KZ.KazukiConfig(
        safe_coefs=(0.3,),
        goal_coef=0.5,
        n_sample=200,
        n_elite=10,
        n_copy=200,
        exact_sfm_step_filter=True,
        step_filter_margin=0.22,
        step_filter_horizon=10,
        step_filter_goal_plans=12,
        step_filter_avoid_plans=18,
        step_filter_always_select=True,
        step_filter_min_progress=0.05,
        step_filter_goal_score_weight=1.0,
        step_filter_clearance_weight=0.05,
        step_filter_escape_patience=5,
        step_filter_escape_burst=3,
        step_filter_viability_lookahead=20,
        step_filter_viability_band=0.05,
        step_filter_viability_escalate=True,
        step_filter_viability_escalation_band=1.0,
        step_filter_viability_escalation_min_progress=2.0,
        step_filter_viability_escalation_entry_progress=5.0,
        step_filter_viability_escalation_burst=40,
        step_filter_stagnation_gamma_max=0.1,
        step_filter_stagnation_window=20,
        step_filter_stagnation_progress=0.1,
        step_filter_stagnation_horizon=20,
        step_filter_stagnation_burst=4,
        controller_gammas=gammas,
        safe_coef_by_gamma=(1.0, 0.3, 1.0, 0.3, 0.3, 0.3, 0.1),
        goal_coef_by_gamma=(2.0, 0.5, 2.0, 0.5, 0.5, 0.5, 3.0),
        step_filter_margin_by_gamma=(0.24, 0.22, 0.24, 0.22, 0.22, 0.22, 0.22),
        step_filter_goal_score_weight_by_gamma=(2.0, 1.0, 2.0, 1.0, 1.0, 6.0, 2.0),
        step_filter_clearance_weight_by_gamma=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0),
        step_filter_clearance_target_weight_by_gamma=(0.0,) * len(gammas),
    ).validate()


def replay_prefix_humans(shard, scenario, gamma, upto_step, environment):
    """Deterministically reconstruct the live SFM crowd at a stored context."""
    windows_by_step = {}
    for window in shard.windows:
        context = shard.contexts[int(window["context_id"])]
        if (
            int(context["scenario_id"]) == int(scenario)
            and round(float(context["gamma"]), 8) == round(float(gamma), 8)
        ):
            windows_by_step[int(context["step"])] = window
    humans = SS.make_humans(
        int(scenario), 0, int(environment["n_ped"]),
        tuple(environment["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    for step in range(int(upto_step)):
        window = windows_by_step.get(step)
        if window is None:
            raise KeyError(
                f"missing executed window for s{scenario} g{gamma} step {step}"
            )
        action = np.asarray(window["controls"], np.float32)[0]
        state[:2] = state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] = state[2:4] + SS.DT * action
        SS.advance_humans(humans, state)
    return humans, state


def _recoverable(inside, terminal, reach_step, horizon):
    stop_hi = terminal[:, :2] + np.maximum(terminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
    stop_lo = terminal[:, :2] - np.maximum(-terminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
    reached = reach_step <= horizon
    return inside & (
        reached
        | ((stop_hi <= SS.TASK_HI).all(axis=1) & (stop_lo >= SS.TASK_LO).all(axis=1))
    )


@torch.no_grad()
def build_codex_pool(policy, context, humans, *, device, seed_step):
    """Regenerate the original Codex MPC pool at one stored context."""
    gamma = float(context["gamma"])
    base = privileged_sfm_config()
    cfg = KZ._gamma_controller_config(base, gamma).validate()
    guidance_cfg = KZ._gamma_guidance_config(cfg, gamma)
    state = np.asarray(context["state"], np.float32)
    ped_xy = np.asarray(context["ped_xy"], np.float32)
    ped_vel = np.asarray(context["ped_vel"], np.float32)
    hp10 = torch.as_tensor(context["hp10"], device=device)[None].float()
    low = torch.as_tensor(context["low5"], device=device)[None].float()
    hist = torch.as_tensor(context["hist"], device=device)[None].float()
    ctx = policy.ctx_from(hp10, low, hist)
    goal = torch.tensor(SS.GOAL, dtype=torch.float32, device=device)
    torch.manual_seed(
        POOL_SEED + int(context["scenario_id"]) * 1000 + int(seed_step)
    )
    z = torch.randn(int(cfg.n_sample), int(policy.d), device=device)
    taus = torch.tensor(cfg.ode_times, dtype=torch.float32, device=device)
    ped_pred = KZ.predict_pedestrians_t(ped_xy, ped_vel, H, SS.DT, device)
    ped_vel_t = torch.tensor(ped_vel, dtype=torch.float32, device=device)
    z1, _, _ = KZ.guided_generate(
        policy, ctx, state, goal, ped_pred, ped_vel_t,
        SS.R_PED + cfg.collision_margin, z, taus, guidance_cfg,
        collect_diagnostics=False,
    )
    u_gen = torch.clamp(
        z1.reshape(int(cfg.n_sample), H, 2) * float(policy.u_max),
        -float(policy.u_max), float(policy.u_max),
    )
    u_best, refine_diag = KZ.flow_mppi_refine(
        policy, state, goal, ped_pred, SS.R_PED + cfg.collision_margin,
        u_gen, None, guidance_cfg, collect_diagnostics=True,
    )
    refined_pool = refine_diag.pop("_refined_controls")
    nominal = u_best.detach().cpu().numpy().astype(np.float32)
    plans = [nominal]
    plans.extend(np.asarray(refined_pool, np.float32))
    plans.append(KZ._brake_control_plan(state, H))
    plans.extend(KZ._goal_control_plans(
        state, int(cfg.step_filter_goal_plans), H,
    ))
    plans.extend(KZ._avoidance_control_plans(
        humans, state, int(cfg.step_filter_avoid_plans), H,
    ))
    for ux in np.linspace(-SS.U_MAX, SS.U_MAX, 5):
        for uy in np.linspace(-SS.U_MAX, SS.U_MAX, 5):
            plans.append(np.repeat(np.array([[ux, uy]], np.float32), H, axis=0))
    unique = []
    for plan in plans:
        plan = KZ._extend_plan_with_goal(state, plan, H)
        plan = np.clip(np.asarray(plan, np.float32)[:H], -SS.U_MAX, SS.U_MAX)
        if not any(np.allclose(plan, old, atol=1e-7) for old in unique):
            unique.append(plan)
    stacked = np.stack(unique)
    clear, inside, terminal, _, reach_step = KZ._simulate_sfm_plans(
        humans, state, stacked, H,
    )
    margin = KZ._adaptive_step_filter_margin(cfg, gamma)
    privileged = _recoverable(inside, terminal, reach_step, H) & (
        clear >= float(margin)
    )
    return dict(
        plans=stacked,
        privileged_feasible=privileged,
        privileged_clearance=clear,
        margin=float(margin),
        pool_manifest=dict(
            nominal=1, refined=len(refined_pool),
            brake=1, goal_plans=int(cfg.step_filter_goal_plans),
            avoid_plans=int(cfg.step_filter_avoid_plans),
            const_accel=25, unique=len(unique),
        ),
    )


def harvest_round(
    policy, shard, hard_windows, executor, *, device, environment,
    max_contexts=None,
):
    """Build the per-round D_MPC+ from the round's declared hard contexts."""
    by_lineage = {}
    for window in hard_windows:
        context = shard.contexts[int(window["context_id"])]
        key = (int(context["scenario_id"]), round(float(context["gamma"]), 8))
        by_lineage.setdefault(key, []).append(int(window["context_id"]))
    records, audit_rows = [], []
    counts = dict(
        hard_contexts=0, replay_mismatch=0, pool_candidates=0,
        privileged_feasible=0, socp_positive=0, intersection=0, kept=0,
    )
    context_ids_all = sorted(
        cid for ids in by_lineage.values() for cid in ids
    )
    if max_contexts is not None and len(context_ids_all) > int(max_contexts):
        stride = len(context_ids_all) / float(max_contexts)
        context_ids_all = [
            context_ids_all[int(i * stride)] for i in range(int(max_contexts))
        ]
    chosen = set(context_ids_all)
    for (scenario, gamma), context_ids in sorted(by_lineage.items()):
        for context_id in sorted(context_ids):
            if context_id not in chosen:
                continue
            context = shard.contexts[context_id]
            counts["hard_contexts"] += 1
            humans, replay_state = replay_prefix_humans(
                shard, scenario, gamma, int(context["step"]), environment,
            )
            ped_xy_live, _ = SS.collect_humans(humans)
            if not np.allclose(
                ped_xy_live, np.asarray(context["ped_xy"], np.float32),
                atol=1e-4,
            ) or not np.allclose(
                replay_state, np.asarray(context["state"], np.float32),
                atol=1e-4,
            ):
                counts["replay_mismatch"] += 1
                continue
            pool = build_codex_pool(
                policy, context, humans, device=device,
                seed_step=int(context["step"]),
            )
            plans = pool["plans"]
            counts["pool_candidates"] += len(plans)
            privileged = pool["privileged_feasible"]
            counts["privileged_feasible"] += int(privileged.sum())
            tasks = [
                (index, 0, context["state"], plans[index],
                 context["ped_xy"], context["ped_vel"], gamma)
                for index in range(len(plans)) if privileged[index]
            ]
            results = {i: r for i, _, r in executor.map(
                SM.verify_in_worker, tasks,
            )}
            certified = [
                index for index in results
                if results[index].get("resolved")
                and int(results[index].get("y", 0)) == 1
            ]
            counts["socp_positive"] += len(certified)
            counts["intersection"] += len(certified)
            if not certified:
                continue
            controls = torch.as_tensor(
                np.stack([plans[i] for i in certified]), dtype=torch.float32,
            )
            costs = BC.safemppi_proposal_cost(
                context["state"], controls, SS.GOAL,
                context["ped_xy"], context["ped_vel"],
            ).cpu().numpy()
            order = sorted(
                range(len(certified)), key=lambda j: (float(costs[j]), j),
            )
            for rank, j in enumerate(order[:KEEP_PER_CONTEXT]):
                index = certified[j]
                counts["kept"] += 1
                records.append(dict(
                    context_id=int(context_id),
                    controls=np.asarray(plans[index], np.float32),
                    y=1,
                    query_id=len(records),
                    source="codex_privileged_mpc_pool",
                    privileged_clearance=float(
                        pool["privileged_clearance"][index],
                    ),
                    privileged_margin=pool["margin"],
                    safemppi_cost=float(costs[j]),
                    rank=int(rank),
                    verifier_diagnostics=dict(
                        results[index]["diagnostics"],
                    ),
                ))
                audit_rows.append(dict(
                    context_id=int(context_id), scenario=int(scenario),
                    gamma=float(gamma), step=int(context["step"]),
                    rank=int(rank), cost=float(costs[j]),
                    privileged_clearance=float(
                        pool["privileged_clearance"][index],
                    ),
                    socp_slack=float(
                        results[index]["diagnostics"]["slack"],
                    ),
                ))
    per_gamma = {}
    for record in records:
        gamma = str(shard.contexts[record["context_id"]]["gamma"])
        per_gamma[gamma] = per_gamma.get(gamma, 0) + 1
    return records, dict(counts=counts, per_gamma=per_gamma, rows=audit_rows)


class MPCView:
    """Duck-typed positive-only view over D_MPC+ for the mass machinery."""

    def __init__(self, shard, records):
        self.round_i = int(shard.round_i)
        self.contexts = shard.contexts
        self.windows = [dict(record) for record in records]
        for index, row in enumerate(self.windows):
            row["window_id"] = index
            row["query_id"] = index

    @property
    def Dplus(self):
        return list(self.windows)


def distill_block(policy, optimizer, shard, records, *, epochs, batch, seed):
    """Dedicated CFM distillation on D_MPC+ only (fresh Gaussian bases)."""
    if not records:
        return dict(steps=0, records=0, losses=None)
    view = MPCView(shard, records)
    pairs = [(view, row) for row in view.windows]
    mass, accounting = BS.hierarchy_mass(pairs)
    policy.train()
    losses = []
    steps = 0
    generator = np.random.default_rng(int(seed))
    device = next(policy.parameters()).device
    for epoch in range(int(epochs)):
        order = list(generator.permutation(len(pairs)))
        for start in range(0, len(order), int(batch)):
            chunk = [pairs[i] for i in order[start:start + int(batch)]]
            grid, low, hist, controls = BS._tensor_batch(chunk, device)
            context = policy.ctx_from(grid, low, hist)
            weights = torch.as_tensor([
                len(pairs) * mass[(id(view), int(row["query_id"]))]
                for _, row in chunk
            ], dtype=controls.dtype, device=device)
            torch.manual_seed(int(seed) + epoch * 100_003 + start)
            loss = policy.cfm_loss(controls, context, weights=weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite MPC distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
    policy.eval()
    return dict(
        steps=steps, records=len(pairs), epochs=int(epochs),
        loss_first=losses[0], loss_last=losses[-1],
        loss_mean=float(np.mean(losses)),
        mass_gamma=accounting["gamma"],
    )
