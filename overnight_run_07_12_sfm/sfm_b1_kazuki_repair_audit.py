"""Training-ready audit of same-latent Kazuki repair during B1 gathering.

The collector keeps three deliberately separate stores:

* ``executed_round.pt``: one exact full-H window per executed context, matching
  the existing offline replay contract;
* ``query_sidecar.pt``: every resolved base B=4 and repair B=4 query, retained
  for audit only unless a later experiment explicitly opts into all-query
  replay.
* ``neutral_round.pt``: opt-in guided verifier-negative executions, isolated
  from training D+/D-, GP, and replay while remaining exact-negative in the
  audit query sidecar.
* ``guided_positive_round.pt``: opt-in (``guided_collect_until_step > 0``)
  proactively harvested guided windows that the exact verifier certified
  positive at early steps.  This ``G+`` population is collection-only: it
  never changes execution, selection, repair triggering, D0, or the GP.

No independent raw continuation and no privileged MPC proposal are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import copy
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair as KR
import sfm_b1_offline_store as OS
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_metrics2 as SM
import sfm_scene as SS


STATUS = "SFM_B1_KAZUKI_REPAIR_AUDIT_COMPLETE"
GUIDED_POSITIVE_STATUS = "SFM_B1_GUIDED_POSITIVE_ROUND_COMPLETE"
GUIDED_POSITIVE_STORE = "guided_positive_round.pt"
GUIDED_GENERATORS = ("same_latent", "kazuki_full")
DEFAULT_GUIDED_GENERATOR = "same_latent"
DEFAULT_GUIDED_TOPK = 2
GUIDED_SOURCES = dict(
    same_latent="guided_proactive",
    kazuki_full="kazuki_full_proactive",
)
DEFAULT_SCENARIOS = (250_001, 250_003)
DEFAULT_GAMMAS = (0.1, 0.5, 1.0)
DEFAULT_ELL = 0.24210826720721101
DEFAULT_SAMPLE_SEED = 700_000
DEFAULT_AUDIT_SEED = 20260730
EXPECTED_CHECKPOINT_SHA256 = (
    "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
)


def _query_row(
    candidate_id, controls, x0, result, *, acquisition_step, sigma,
    mode, source, parent_candidate_id=None,
):
    return dict(
        candidate_id=int(candidate_id),
        parent_candidate_id=(
            None if parent_candidate_id is None else int(parent_candidate_id)
        ),
        controls=np.asarray(controls, np.float32),
        x0=None if x0 is None else np.asarray(x0, np.float32),
        result=result,
        acquisition_step=int(acquisition_step),
        sigma=float(sigma),
        mode=str(mode),
        source=str(source),
    )


def _add_sidecar_query(shard, context_id, row):
    result = row["result"]
    if not result.get("resolved"):
        shard.add_error(
            context_key=(
                shard.contexts[int(context_id)]["scenario_id"],
                shard.contexts[int(context_id)]["gamma"],
                shard.contexts[int(context_id)]["step"],
            ),
            candidate_id=row["candidate_id"],
            error=result.get("error"),
        )
        return None
    query_id = shard.add_resolved_query(
        context_id,
        row["candidate_id"],
        row["controls"],
        row["sigma"],
        result,
        acquisition_step=row["acquisition_step"],
        hp_margin=row.get("hp_margin"),
        expert_cost=row.get("expert_cost"),
        mode=row["mode"],
    )
    stored = shard.queries[int(query_id)]
    stored.update(
        x0=np.asarray(row["x0"], np.float32),
        query_source=row["source"],
        parent_candidate_id=row["parent_candidate_id"],
        audit_only=True,
        replay_default=False,
    )
    row["query_id"] = int(query_id)
    return int(query_id)


def _admissible(rows, selector, prepared, gamma):
    return BC.select_admissible(
        [row for row in rows if row["result"].get("resolved")],
        selector=selector,
        state=prepared["state"],
        ped_xy=prepared["ped_xy"],
        ped_vel=prepared["ped_vel"],
        gamma=float(gamma),
    )


def _all_full_h_negative(rows, expected=4):
    return len(rows) == int(expected) and all(
        row["result"].get("resolved")
        and int(row["result"].get("y", -1)) == 0
        and bool(row["result"].get("full_h"))
        and int(row["result"].get("terminal_step", -1)) == 10
        for row in rows
    )


def _select_neutral(rows, selector, prepared, gamma):
    """Rank four exact-negative guided rows without relabeling them."""
    if not _all_full_h_negative(rows):
        return None
    for row in rows:
        margin, hp_old, hp_new = BC.nominal_hp_margin(
            prepared["state"],
            row["controls"][0],
            prepared["ped_xy"],
            gamma,
        )
        row["hp_margin"] = float(margin)
        row["hp_old"] = float(hp_old)
        row["hp_new"] = float(hp_new)
    if selector == "margin":
        return max(
            rows,
            key=lambda row: (
                row["hp_margin"],
                -int(row["candidate_id"]),
            ),
        )
    if selector == "progress_gated_margin":
        return BC.select_progress_gated_margin(
            rows, state=prepared["state"],
        )
    if selector != "safemppi_cost":
        raise ValueError(f"unknown neutral selector: {selector}")
    controls = torch.as_tensor(
        np.stack([row["controls"] for row in rows]),
        dtype=torch.float32,
    )
    costs = BC.safemppi_proposal_cost(
        prepared["state"],
        controls,
        SS.GOAL,
        prepared["ped_xy"],
        prepared["ped_vel"],
    ).cpu().numpy()
    for row, cost in zip(rows, costs):
        row["expert_cost"] = float(cost)
    return min(
        rows,
        key=lambda row: (
            row["expert_cost"],
            int(row["candidate_id"]),
        ),
    )


def _neutral_record(
    neutral_id,
    replica,
    prepared,
    chosen,
    *,
    step,
    round_i=1,
    selector,
    repair_trigger,
):
    result = chosen["result"]
    if not _all_full_h_negative([chosen], expected=1):
        raise ValueError("neutral execution requires one exact full-H negative")
    controls = np.asarray(chosen["controls"], np.float32)
    x0 = np.asarray(chosen["x0"], np.float32)
    if tuple(controls.shape) != (10, 2) or not np.isfinite(controls).all():
        raise ValueError("neutral controls must be finite [10,2]")
    if tuple(x0.shape) != (20,) or not np.isfinite(x0).all():
        raise ValueError("neutral x0 must be finite [20]")
    return dict(
        neutral_id=int(neutral_id),
        population="D0",
        semantic_label="neutral",
        round=int(round_i),
        scenario_id=int(replica.scenario_id),
        gamma=float(replica.gamma),
        step=int(step),
        state=np.asarray(prepared["state"], np.float32),
        hp10=np.asarray(prepared["hp10"].numpy(), np.float32),
        low5=np.asarray(prepared["low"].numpy(), np.float32),
        hist=np.asarray(prepared["hist"].numpy(), np.float32),
        ped_xy=np.asarray(prepared["ped_xy"], np.float32),
        ped_vel=np.asarray(prepared["ped_vel"], np.float32),
        controls=controls,
        x0=x0,
        verifier_result=result,
        verifier_y=int(result["y"]),
        candidate_id=int(chosen["candidate_id"]),
        parent_candidate_id=chosen.get("parent_candidate_id"),
        query_id=chosen.get("query_id"),
        sigma=float(chosen["sigma"]),
        hp_margin=float(chosen["hp_margin"]),
        expert_cost=(
            None
            if chosen.get("expert_cost") is None
            else float(chosen["expert_cost"])
        ),
        selector=str(selector),
        repair_trigger=str(repair_trigger),
        execution_source=(
            f"kazuki_repair_neutral_{repair_trigger}_{selector}"
        ),
        train_eligible=False,
        replay_default=False,
        gp_eligible=False,
    )


def _save_neutral_records(path, records, *, round_i=1):
    path = os.fspath(path)
    for expected, record in enumerate(records):
        if int(record["neutral_id"]) != expected:
            raise AssertionError("neutral IDs are not dense")
        if (
            record["semantic_label"] != "neutral"
            or record["population"] != "D0"
            or record["train_eligible"]
            or record["replay_default"]
            or record["gp_eligible"]
            or int(record["verifier_y"]) != 0
        ):
            raise AssertionError("invalid neutral execution record")
    payload = dict(
        version=1,
        status="SFM_B1_NEUTRAL_ROUND_COMPLETE",
        round=int(round_i),
        records=records,
        summary=dict(D0=len(records), train_eligible=0, gp_eligible=0),
    )
    FA._save_torch(path, payload)
    marker = dict(
        status=payload["status"],
        file=os.path.abspath(path),
        sha256=FA._sha256_file(path),
        **payload["summary"],
    )
    FA._write_json(path + ".COMPLETE.json", marker)
    return marker


def _guided_positive_record(
    guided_id,
    replica,
    prepared,
    row,
    *,
    step,
    round_i=1,
    generator=DEFAULT_GUIDED_GENERATOR,
    repair_trigger,
    reused_from_repair,
    selected_for_execution,
):
    """One proactively harvested, exact-verifier-certified guided window."""
    if generator not in GUIDED_GENERATORS:
        raise ValueError(f"generator must be one of {GUIDED_GENERATORS}")
    result = row["result"]
    if (
        not result.get("resolved")
        or int(result.get("y", -1)) != 1
        or not bool(result.get("full_h"))
        or int(result.get("terminal_step", -1)) != 10
    ):
        raise ValueError("G+ requires an exact full-H verifier positive")
    components = (
        bool(result.get("taskspace")),
        bool(result.get("collision_free")),
        bool(result.get("certificate")),
    )
    if int(result["y"]) != int(all(components)):
        raise ValueError("verifier y disagrees with its exact components")
    controls = np.asarray(row["controls"], np.float32)
    if tuple(controls.shape) != (10, 2) or not np.isfinite(controls).all():
        raise ValueError("guided-positive controls must be finite [10,2]")
    x0 = row.get("x0")
    if x0 is None:
        # The full external controller returns MPPI-refined elite modes; they
        # are averages of perturbed plans and therefore have no single
        # generating latent.  Recording a fabricated x0 would be a lie, so the
        # field stays empty and the row is never GP- or latent-replayed.
        if generator != "kazuki_full":
            raise ValueError("same-latent G+ rows must retain their x0")
    else:
        x0 = np.asarray(x0, np.float32)
        if tuple(x0.shape) != (20,) or not np.isfinite(x0).all():
            raise ValueError("guided-positive x0 must be finite [20]")
    return dict(
        guided_id=int(guided_id),
        population="Gplus",
        semantic_label="guided_positive",
        round=int(round_i),
        scenario_id=int(replica.scenario_id),
        gamma=float(replica.gamma),
        step=int(step),
        state=np.asarray(prepared["state"], np.float32),
        hp10=np.asarray(prepared["hp10"].numpy(), np.float32),
        low5=np.asarray(prepared["low"].numpy(), np.float32),
        hist=np.asarray(prepared["hist"].numpy(), np.float32),
        ped_xy=np.asarray(prepared["ped_xy"], np.float32),
        ped_vel=np.asarray(prepared["ped_vel"], np.float32),
        controls=controls,
        x0=x0,
        x0_available=bool(x0 is not None),
        generator=str(generator),
        verifier_result=result,
        verifier_y=1,
        candidate_id=int(row["candidate_id"]),
        parent_candidate_id=(
            None
            if row.get("parent_candidate_id") is None
            else int(row["parent_candidate_id"])
        ),
        acquisition_step=int(row["acquisition_step"]),
        sigma=float(row["sigma"]),
        mode=str(row["mode"]),
        query_id=row.get("query_id"),
        source=GUIDED_SOURCES[str(generator)],
        repair_trigger=(
            None if repair_trigger is None else str(repair_trigger)
        ),
        reused_from_repair=bool(reused_from_repair),
        selected_for_execution=bool(selected_for_execution),
        train_eligible=True,
        replay_default=False,
        gp_eligible=False,
    )


def _save_guided_positive_records(
    path, records, *, round_i=1, until_step,
    generator=DEFAULT_GUIDED_GENERATOR,
):
    path = os.fspath(path)
    contexts = set()
    for expected, record in enumerate(records):
        if int(record["guided_id"]) != expected:
            raise AssertionError("guided-positive IDs are not dense")
        if (
            record["semantic_label"] != "guided_positive"
            or record["population"] != "Gplus"
            or record["generator"] != str(generator)
            or record["source"] != GUIDED_SOURCES[str(generator)]
            or not record["train_eligible"]
            or record["replay_default"]
            or record["gp_eligible"]
            or int(record["verifier_y"]) != 1
            or int(record["step"]) >= int(until_step)
        ):
            raise AssertionError("invalid guided-positive record")
        contexts.add((
            int(record["scenario_id"]),
            round(float(record["gamma"]), 8),
            int(record["step"]),
        ))
    payload = dict(
        version=1,
        status=GUIDED_POSITIVE_STATUS,
        round=int(round_i),
        guided_collect_until_step=int(until_step),
        guided_generator=str(generator),
        records=records,
        summary=dict(
            Gplus=len(records),
            contexts=len(contexts),
            train_eligible=len(records),
            gp_eligible=0,
        ),
    )
    FA._save_torch(path, payload)
    marker = dict(
        status=payload["status"],
        file=os.path.abspath(path),
        sha256=FA._sha256_file(path),
        guided_collect_until_step=int(until_step),
        guided_generator=str(generator),
        **payload["summary"],
    )
    FA._write_json(path + ".COMPLETE.json", marker)
    return marker


def _gamma_balanced_gp(
    phi_policy,
    previous,
    *,
    gammas,
    round_i,
    ell,
    cap,
    lam,
    phi_s,
    device,
    seed,
):
    """Build a previous-round-only GP with an equal supported gamma quota."""
    gp = BR.RBFGP(float(ell), float(lam))
    empty = {
        str(gamma): 0 for gamma in gammas
    }
    if previous is None:
        return gp, [], dict(
            requested_cap=int(cap),
            effective_cap=0,
            quota=0,
            rotating_extra_gamma=None,
            per_gamma=empty,
            source_round=None,
            population="previous executed D+ only",
        )

    groups = {}
    for gamma_index, gamma in enumerate(gammas):
        records = [
            (previous, row)
            for row in previous.Dplus
            if round(
                float(previous.contexts[int(row["context_id"])]["gamma"]),
                8,
            ) == round(float(gamma), 8)
        ]
        groups[float(gamma)] = BS.hierarchical_order(
            records, int(seed) + gamma_index,
        )
    quota = min(
        int(cap) // len(gammas),
        min(len(groups[float(gamma)]) for gamma in gammas),
    )
    if quota == 0:
        counts = {
            str(gamma): len(groups[float(gamma)])
            for gamma in gammas
        }
        raise RuntimeError(
            "previous-round D+ cannot support all declared gammas; "
            f"per_gamma={counts}"
        )
    selected = [
        record
        for gamma in gammas
        for record in groups[float(gamma)][:quota]
    ]
    rotation = (int(round_i) - 2) % len(gammas)
    extra_gamma = None
    if quota > 0 and len(selected) < int(cap):
        for offset in range(len(gammas)):
            gamma = float(gammas[(rotation + offset) % len(gammas)])
            if len(groups[gamma]) > quota:
                selected.append(groups[gamma][quota])
                extra_gamma = gamma
                break

    if selected:
        feature_parts = []
        for start in range(0, len(selected), 256):
            values = selected[start:start + 256]
            hp10, low, hist, controls = BX._record_batch(values, device)
            x0 = torch.as_tensor(
                np.stack([row["x0"] for _, row in values]),
                device=device,
            ).float()
            feature_parts.append(phi_policy.phi_s_from_x0(
                controls,
                phi_policy.ctx_from(hp10, low, hist),
                x0,
                s=float(phi_s),
            ))
        gp.set_buffer(torch.cat(feature_parts))
    identities = [
        (int(shard.round_i), int(row["window_id"]))
        for shard, row in selected
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("dynamic gamma-balanced GP contains duplicates")
    per_gamma = Counter(
        str(previous.contexts[int(row["context_id"])]["gamma"])
        for _, row in selected
    )
    return gp, identities, dict(
        requested_cap=int(cap),
        effective_cap=len(selected),
        quota=int(quota),
        rotating_extra_gamma=extra_gamma,
        per_gamma={
            str(gamma): int(per_gamma[str(gamma)])
            for gamma in gammas
        },
        source_round=int(previous.round_i),
        population="previous executed D+ only; D0 excluded",
    )


@torch.no_grad()
def _calibrate_gp_beta(
    phi_policy, gp, replicas, cfg, device, *, round_i,
):
    live, batch = BX._stack_prepared(replicas, device)
    windows, contexts, x0 = FA._keyed_windows(
        phi_policy,
        live,
        batch,
        K=cfg.K,
        round_i=int(round_i),
        step=-1,
        source="beta_calibration",
        seed=cfg.seed,
        nfe=cfg.nfe,
        temp=cfg.temp,
    )
    features = FA._features_from_x0(
        phi_policy, windows, contexts, x0, cfg.phi_s,
    )
    score_vectors = []
    for replica, values in zip(live, features):
        generator = torch.Generator(device=values.device).manual_seed(
            FA._keyed_seed(
                cfg.seed,
                int(round_i),
                replica.scenario_id,
                f"{replica.gamma:.8f}",
                "beta_order",
            )
        )
        order = torch.randperm(
            len(values), generator=generator, device=values.device,
        )
        score_vectors.extend(
            gp.sequential_score_vectors(values, order, cfg.B)
        )
    beta, ess = BR.solve_beta(
        score_vectors, target=cfg.ess_target,
    )
    return float(beta), float(ess)


def _repair_trigger(replica, chosen):
    current_trap = FA._trap(replica.states)
    if current_trap:
        return "trap_streak"
    if chosen is None:
        return "finite_B_NVP"
    if KR.predicted_trap(replica.states, chosen["controls"][0]):
        return "predicted_trap"
    return None


def collect(
    checkpoint,
    *,
    scenarios=DEFAULT_SCENARIOS,
    gammas=DEFAULT_GAMMAS,
    scene_profile="double_density_velocity_ood",
    selector="margin",
    device="cuda",
    verifier_workers=16,
    sample_seed=DEFAULT_SAMPLE_SEED,
    audit_seed=DEFAULT_AUDIT_SEED,
    ell=DEFAULT_ELL,
    neutral_continuation=False,
    guided_collect_until_step=0,
    guided_collect_store=None,
    guided_generator=DEFAULT_GUIDED_GENERATOR,
    guided_topk=DEFAULT_GUIDED_TOPK,
    crunch_full_pool_until_step=0,
    round_i=1,
    expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
    previous_executed_path=None,
    gp_cap=512,
    ess_target=0.5,
    verifier_executor=None,
    T=180,
    outdir,
):
    """Collect immutable traces plus executed/query shards for one frozen model."""
    scenarios = tuple(map(int, scenarios))
    gammas = tuple(map(float, gammas))
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be distinct and nonempty")
    if (
        not gammas
        or len(set(gammas)) != len(gammas)
        or any(value not in tuple(map(float, SS.GAMMAS)) for value in gammas)
    ):
        raise ValueError(f"gammas must be a distinct subset of {SS.GAMMAS}")
    if selector not in (
        "margin", "progress_gated_margin", "safemppi_cost",
    ):
        raise ValueError(
            "selector must be margin, progress_gated_margin, or "
            "safemppi_cost"
        )
    if scene_profile != "double_density_velocity_ood":
        raise ValueError("repair audit is pinned to double-shift OOD")
    if int(T) != 180:
        raise ValueError("repair audit is scientifically pinned to T=180")
    guided_collect_until_step = int(guided_collect_until_step)
    if guided_collect_until_step < 0:
        raise ValueError("guided_collect_until_step must be non-negative")
    if guided_collect_until_step > int(T):
        raise ValueError("guided_collect_until_step cannot exceed T")
    guided_collect = guided_collect_until_step > 0
    if guided_collect_store is not None and not guided_collect:
        raise ValueError(
            "guided_collect_store requires guided_collect_until_step > 0"
        )
    if guided_generator not in GUIDED_GENERATORS:
        raise ValueError(f"guided_generator must be one of {GUIDED_GENERATORS}")
    guided_topk = int(guided_topk)
    if not 1 <= guided_topk <= 16:
        raise ValueError("guided_topk must lie in [1, K]")
    crunch_full_pool_until_step = int(crunch_full_pool_until_step)
    if crunch_full_pool_until_step < 0:
        raise ValueError("crunch_full_pool_until_step must be non-negative")
    if crunch_full_pool_until_step > int(T):
        raise ValueError("crunch_full_pool_until_step cannot exceed T")
    crunch_full_pool = crunch_full_pool_until_step > 0
    checkpoint = os.path.abspath(checkpoint)
    outdir = os.path.abspath(outdir)
    guided_positive_path = None
    if guided_collect:
        guided_positive_path = (
            os.path.join(outdir, GUIDED_POSITIVE_STORE)
            if guided_collect_store is None
            else os.path.abspath(
                guided_collect_store
                if os.path.isabs(guided_collect_store)
                else os.path.join(outdir, guided_collect_store)
            )
        )
    round_i = int(round_i)
    if round_i < 1:
        raise ValueError("round_i must be positive")
    if int(gp_cap) < len(gammas):
        raise ValueError("gp_cap must permit at least one row per gamma")
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output directory: {outdir}")
    checkpoint_sha = FA._sha256_file(checkpoint)
    if (
        expected_checkpoint_sha256 is None
        or checkpoint_sha != str(expected_checkpoint_sha256)
    ):
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected "
            f"{expected_checkpoint_sha256}, "
            f"observed {checkpoint_sha}"
        )
    previous = (
        None
        if previous_executed_path is None
        else OS.ExecutedRoundShard.load(previous_executed_path)
    )
    if round_i == 1 and previous is not None:
        raise ValueError("round 1 cannot have previous GP support")
    if round_i > 1 and previous is None:
        raise ValueError("round >1 requires previous executed D+ support")
    if previous is not None and int(previous.round_i) != round_i - 1:
        raise ValueError(
            "previous executed shard must be from the immediately "
            "preceding round"
        )

    environment = SS.scene_profile(scene_profile)
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    policy_hash = BX.policy_sha256(policy)
    replicas = [
        BX.Replica(
            scenario,
            gamma,
            n_ped=environment["n_ped"],
            ped_speed_range=tuple(environment["ped_speed_range"]),
        )
        for scenario in scenarios for gamma in gammas
    ]
    cfg = BX.ArmConfig(
        name="diagnostic",
        selector=selector,
        alpha=0.0,
        rounds=1,
        K=16,
        B=4,
        T=int(T),
        H=10,
        W=2,
        batch=128,
        lr=1.0e-5,
        ess_target=float(ess_target),
        nfe=8,
        temp=1.0,
        phi_s=0.9,
        gp_lam=1.0e-2,
        verifier_workers=int(verifier_workers),
        smoke=False,
        seed=int(audit_seed),
        scene_profile=scene_profile,
    ).validate()
    gp, gp_ids, gp_selection = _gamma_balanced_gp(
        phi_policy,
        previous,
        gammas=gammas,
        round_i=round_i,
        ell=float(ell),
        cap=int(gp_cap),
        lam=cfg.gp_lam,
        phi_s=cfg.phi_s,
        device=device,
        seed=int(audit_seed) + round_i * 101,
    )
    beta, calibrated_ess = _calibrate_gp_beta(
        phi_policy, gp, replicas, cfg, device, round_i=round_i,
    )

    executed_shard = OS.ExecutedRoundShard(round_i)
    query_shard = BS.RoundShard(round_i)
    neutral_records = []
    guided_positive_records = []
    traces = []
    counts = Counter()
    sigma_pool, sigma_selected, ess_values = [], [], []
    trap_streaks = defaultdict(int)

    executor_scope = (
        ProcessPoolExecutor(max_workers=int(verifier_workers))
        if verifier_executor is None
        else nullcontext(verifier_executor)
    )
    with executor_scope as executor:
        for step in range(int(T)):
            live = [replica for replica in replicas if replica.alive]
            live, batch = BX._stack_prepared(live, device)
            if not live:
                break
            with torch.no_grad():
                windows, contexts, x0 = FA._keyed_windows(
                    policy,
                    live,
                    batch,
                    K=cfg.K,
                    round_i=round_i,
                    step=step,
                    source="K",
                    seed=int(sample_seed),
                    nfe=cfg.nfe,
                    temp=cfg.temp,
                )
                features = FA._features_from_x0(
                    phi_policy, windows, contexts, x0, cfg.phi_s,
                )
            windows_np = windows.detach().cpu().numpy()
            x0_np = x0.detach().cpu().numpy()

            selected_by_context = []
            acquisition_by_context = []
            pool_sigma_by_context = []
            for context_index, replica in enumerate(live):
                generator = torch.Generator(
                    device=features.device,
                ).manual_seed(FA._keyed_seed(
                    int(audit_seed),
                    round_i,
                    replica.scenario_id,
                    f"{replica.gamma:.8f}",
                    step,
                    "acquisition",
                ))
                selected, acquisition = gp.sequential_acquire(
                    features[context_index],
                    cfg.B,
                    beta,
                    generator=generator,
                )
                selected_by_context.append(list(map(int, selected)))
                acquisition_by_context.append(acquisition)
                pool_sigma = [
                    float(value) for value in
                    gp.acquisition_sigma(features[context_index]).detach().cpu()
                ]
                pool_sigma_by_context.append(pool_sigma)
                sigma_pool.extend(pool_sigma)
                sigma_selected.extend(
                    float(row["chosen_sigma"]) for row in acquisition
                )
                ess_values.extend(float(row["ess_norm"]) for row in acquisition)

            # Crunch-band arm: verify the whole K pool instead of only the
            # B=4 GP picks.  The executed selector is unchanged; the admissible
            # set is simply no longer truncated by the acquisition budget.
            full_pool_active = (
                crunch_full_pool and step < crunch_full_pool_until_step
            )
            extra_by_context = [
                (
                    [
                        candidate_id for candidate_id in range(cfg.K)
                        if candidate_id not in set(selected_by_context[index])
                    ]
                    if full_pool_active else []
                )
                for index in range(len(live))
            ]
            base_tasks = []
            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                for candidate_id in (
                    *selected_by_context[context_index],
                    *extra_by_context[context_index],
                ):
                    base_tasks.append((
                        context_index,
                        candidate_id,
                        prepared["state"],
                        windows_np[context_index, candidate_id],
                        prepared["ped_xy"],
                        prepared["ped_vel"],
                        replica.gamma,
                    ))
            if full_pool_active:
                counts["crunch_full_pool_verifier_queries"] += sum(
                    len(values) for values in extra_by_context
                )
                counts["crunch_full_pool_contexts"] += len(live)
            base_results = list(executor.map(SM.verify_in_worker, base_tasks))
            counts["base_verifier_queries"] += len(base_tasks)
            base_by_context = defaultdict(dict)
            for context_index, candidate_id, result in base_results:
                base_by_context[int(context_index)][int(candidate_id)] = result

            prepared_contexts = []
            repair_indices = []
            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                prediction = SM.predict_pedestrians(
                    prepared["ped_xy"], prepared["ped_vel"], cfg.H,
                )
                all_rows = []
                for candidate_id in range(cfg.K):
                    controls = windows_np[context_index, candidate_id]
                    segment = SM.rollout_positions(prepared["state"], controls)
                    all_rows.append(dict(
                        candidate_id=int(candidate_id),
                        controls=controls,
                        x0=x0_np[context_index, candidate_id],
                        segment=segment,
                        mode=BE.classify_candidate(segment, prediction),
                    ))
                base_rows = []
                for acquisition_step, candidate_id in enumerate(
                    selected_by_context[context_index]
                ):
                    source = all_rows[candidate_id]
                    base_rows.append(_query_row(
                        candidate_id,
                        source["controls"],
                        source["x0"],
                        base_by_context[context_index][candidate_id],
                        acquisition_step=acquisition_step,
                        sigma=acquisition_by_context[context_index][
                            acquisition_step
                        ]["chosen_sigma"],
                        mode=source["mode"],
                        source="base_B",
                    ))
                for candidate_id in extra_by_context[context_index]:
                    source = all_rows[candidate_id]
                    base_rows.append(_query_row(
                        candidate_id,
                        source["controls"],
                        source["x0"],
                        base_by_context[context_index][candidate_id],
                        acquisition_step=-1,
                        sigma=pool_sigma_by_context[context_index][
                            candidate_id
                        ],
                        mode=source["mode"],
                        source="base_full_pool",
                    ))
                base_choice = _admissible(
                    base_rows, selector, prepared, replica.gamma,
                )
                trigger = _repair_trigger(replica, base_choice)
                prepared_contexts.append(dict(
                    all_rows=all_rows,
                    base_rows=base_rows,
                    base_choice=base_choice,
                    trigger=trigger,
                    repair_rows=[],
                    repair_diagnostics=None,
                    proactive_rows=[],
                    proactive_diagnostics=None,
                ))
                if trigger is not None:
                    repair_indices.append(context_index)

            repair_tasks = []
            for context_index in repair_indices:
                replica = live[context_index]
                selected = selected_by_context[context_index]
                guided, diagnostics = KR.same_latent_guided_controls(
                    policy,
                    contexts[context_index],
                    replica.prepared["state"],
                    replica.prepared["ped_xy"],
                    replica.prepared["ped_vel"],
                    replica.gamma,
                    x0[context_index, selected],
                    nfe=cfg.nfe,
                    collect_diagnostics=True,
                )
                guided_np = guided.detach().cpu().numpy()
                prepared_contexts[context_index][
                    "repair_diagnostics"
                ] = diagnostics
                for acquisition_step, (candidate_id, controls) in enumerate(
                    zip(selected, guided_np)
                ):
                    repair_id = cfg.K + int(candidate_id)
                    repair_tasks.append((
                        context_index,
                        repair_id,
                        replica.prepared["state"],
                        controls,
                        replica.prepared["ped_xy"],
                        replica.prepared["ped_vel"],
                        replica.gamma,
                    ))
                    prepared_contexts[context_index]["repair_rows"].append(
                        dict(
                            repair_id=repair_id,
                            parent_candidate_id=int(candidate_id),
                            acquisition_step=int(acquisition_step),
                            controls=controls,
                        )
                    )
            repair_results = list(executor.map(SM.verify_in_worker, repair_tasks))
            counts["repair_verifier_queries"] += len(repair_tasks)
            repair_result_lookup = {
                (int(context_index), int(candidate_id)): result
                for context_index, candidate_id, result in repair_results
            }

            # Proactive early-step guided harvest (collection only).  Nothing
            # below this point may change execution, selection, repair
            # triggering, D0, or the GP; the block is skipped entirely when
            # guided_collect_until_step == 0.
            proactive_active = (
                guided_collect and step < guided_collect_until_step
            )
            proactive_result_lookup = {}
            if proactive_active:
                proactive_tasks = []
                for context_index, replica in enumerate(live):
                    values = prepared_contexts[context_index]
                    if (
                        guided_generator == "same_latent"
                        and values["trigger"] is not None
                    ):
                        # The trigger path already regenerated exactly these
                        # latents through the identical frozen policy, so its
                        # rows are reused instead of generating them twice.
                        continue
                    selected = selected_by_context[context_index]
                    if guided_generator == "same_latent":
                        guided, diagnostics = KR.same_latent_guided_controls(
                            policy,
                            contexts[context_index],
                            replica.prepared["state"],
                            replica.prepared["ped_xy"],
                            replica.prepared["ped_vel"],
                            replica.gamma,
                            x0[context_index, selected],
                            nfe=cfg.nfe,
                            collect_diagnostics=False,
                        )
                        plans = guided.detach().cpu().numpy()
                        parents = [int(value) for value in selected]
                        offsets = list(map(int, selected))
                        steps = list(range(len(parents)))
                        sigmas = [
                            float(
                                acquisition_by_context[context_index][index][
                                    "chosen_sigma"
                                ]
                            )
                            for index in steps
                        ]
                    else:
                        plans, diagnostics = KR.locked_full_kazuki_plans(
                            policy,
                            contexts[context_index],
                            replica.prepared["state"],
                            replica.prepared["ped_xy"],
                            replica.prepared["ped_vel"],
                            replica.gamma,
                            topk=guided_topk,
                            seed=FA._keyed_seed(
                                int(audit_seed),
                                round_i,
                                replica.scenario_id,
                                f"{replica.gamma:.8f}",
                                step,
                                "kazuki_full_proactive",
                            ),
                        )
                        parents = [None] * len(plans)
                        offsets = list(range(len(plans)))
                        steps = [-1] * len(plans)
                        sigmas = [0.0] * len(plans)
                    values["proactive_diagnostics"] = diagnostics
                    for offset, parent, acquisition_step, sigma, controls in zip(
                        offsets, parents, steps, sigmas, plans,
                    ):
                        proactive_id = 2 * cfg.K + int(offset)
                        proactive_tasks.append((
                            context_index,
                            proactive_id,
                            replica.prepared["state"],
                            controls,
                            replica.prepared["ped_xy"],
                            replica.prepared["ped_vel"],
                            replica.gamma,
                        ))
                        values["proactive_rows"].append(dict(
                            proactive_id=proactive_id,
                            parent_candidate_id=parent,
                            acquisition_step=int(acquisition_step),
                            sigma=float(sigma),
                            controls=controls,
                        ))
                proactive_results = list(
                    executor.map(SM.verify_in_worker, proactive_tasks)
                )
                counts["guided_proactive_verifier_queries"] += len(
                    proactive_tasks
                )
                proactive_result_lookup = {
                    (int(context_index), int(candidate_id)): result
                    for context_index, candidate_id, result in proactive_results
                }

            for context_index, replica in enumerate(live):
                prepared = replica.prepared
                values = prepared_contexts[context_index]
                sidecar_context_id = query_shard.add_context(
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
                for row in values["base_rows"]:
                    _add_sidecar_query(query_shard, sidecar_context_id, row)
                    prefix = (
                        "base" if row["source"] == "base_B"
                        else "crunch_full_pool"
                    )
                    counts[f"{prefix}_{FA._result_label(row['result'])}"] += 1

                repair_rows = []
                for repair in values["repair_rows"]:
                    parent_id = int(repair["parent_candidate_id"])
                    source = values["all_rows"][parent_id]
                    repair_id = int(repair["repair_id"])
                    row = _query_row(
                        repair_id,
                        repair["controls"],
                        source["x0"],
                        repair_result_lookup[(context_index, repair_id)],
                        acquisition_step=repair["acquisition_step"],
                        sigma=values["base_rows"][
                            repair["acquisition_step"]
                        ]["sigma"],
                        mode=BE.classify_candidate(
                            SM.rollout_positions(
                                prepared["state"], repair["controls"],
                            ),
                            SM.predict_pedestrians(
                                prepared["ped_xy"],
                                prepared["ped_vel"],
                                cfg.H,
                            ),
                        ),
                        source="kazuki_guided_B",
                        parent_candidate_id=parent_id,
                    )
                    repair_rows.append(row)
                    _add_sidecar_query(query_shard, sidecar_context_id, row)
                    counts[f"repair_{FA._result_label(row['result'])}"] += 1
                values["repair_rows"] = repair_rows
                repaired_choice = (
                    _admissible(
                        repair_rows, selector, prepared, replica.gamma,
                    )
                    if values["trigger"] is not None else None
                )
                neutral_choice = (
                    _select_neutral(
                        repair_rows, selector, prepared, replica.gamma,
                    )
                    if (
                        bool(neutral_continuation)
                        and values["trigger"] is not None
                        and repaired_choice is None
                        and _all_full_h_negative(repair_rows)
                    )
                    else None
                )
                for row in repair_rows:
                    if row.get("query_id") is None:
                        continue
                    stored = query_shard.queries[int(row["query_id"])]
                    if "hp_margin" in row:
                        stored["hp_margin"] = float(row["hp_margin"])
                    if "expert_cost" in row:
                        stored["expert_cost"] = float(row["expert_cost"])
                chosen = (
                    repaired_choice or neutral_choice
                    if values["trigger"] is not None else values["base_choice"]
                )

                trace = dict(
                    round=round_i,
                    step=int(step),
                    scenario_id=int(replica.scenario_id),
                    gamma=float(replica.gamma),
                    state=prepared["state"].copy(),
                    next_state=prepared["state"].copy(),
                    ped_xy=prepared["ped_xy"].copy(),
                    ped_vel=prepared["ped_vel"].copy(),
                    all_K=values["all_rows"],
                    selected_ids=selected_by_context[context_index],
                    query_rows=values["base_rows"],
                    guided_query_rows=repair_rows,
                    acquisition=acquisition_by_context[context_index],
                    repair_trigger=values["trigger"],
                    repair_diagnostics=values["repair_diagnostics"],
                    repair_selected_id=None,
                    neutral_execution=False,
                    neutral_id=None,
                    executed_id=None,
                    executed_controls=None,
                    executed_x0=None,
                    executed_result=None,
                    execution_source=None,
                    trap_streak_before=int(trap_streaks[
                        (replica.scenario_id, replica.gamma)
                    ]),
                    trap_event=False,
                    trap_fail_closed=False,
                    negative_reasons=[],
                )
                if proactive_active:
                    reused = (
                        guided_generator == "same_latent"
                        and values["trigger"] is not None
                    )
                    if not reused:
                        harvested = []
                        for proactive in values["proactive_rows"]:
                            parent = proactive["parent_candidate_id"]
                            proactive_id = int(proactive["proactive_id"])
                            row = _query_row(
                                proactive_id,
                                proactive["controls"],
                                (
                                    None if parent is None
                                    else values["all_rows"][int(parent)]["x0"]
                                ),
                                proactive_result_lookup[
                                    (context_index, proactive_id)
                                ],
                                acquisition_step=proactive["acquisition_step"],
                                sigma=proactive["sigma"],
                                mode=BE.classify_candidate(
                                    SM.rollout_positions(
                                        prepared["state"],
                                        proactive["controls"],
                                    ),
                                    SM.predict_pedestrians(
                                        prepared["ped_xy"],
                                        prepared["ped_vel"],
                                        cfg.H,
                                    ),
                                ),
                                source=GUIDED_SOURCES[guided_generator],
                                parent_candidate_id=parent,
                            )
                            harvested.append(row)
                            counts[
                                "guided_proactive_"
                                f"{FA._result_label(row['result'])}"
                            ] += 1
                        values["proactive_rows"] = harvested
                    else:
                        harvested = repair_rows
                        counts["guided_proactive_reused_repair_rows"] += len(
                            harvested
                        )
                    for row in harvested:
                        outcome = row["result"]
                        if (
                            not outcome.get("resolved")
                            or int(outcome.get("y", -1)) != 1
                            or not bool(outcome.get("full_h"))
                            or int(outcome.get("terminal_step", -1)) != 10
                        ):
                            continue
                        guided_positive_records.append(
                            _guided_positive_record(
                                len(guided_positive_records),
                                replica,
                                prepared,
                                row,
                                step=step,
                                round_i=round_i,
                                generator=guided_generator,
                                repair_trigger=values["trigger"],
                                reused_from_repair=bool(reused),
                                selected_for_execution=row is chosen,
                            )
                        )
                        counts["guided_positive_records"] += 1
                    trace["guided_proactive_rows"] = (
                        [] if reused else values["proactive_rows"]
                    )
                    trace["guided_proactive_generator"] = str(guided_generator)
                    trace["guided_proactive_reused_repair"] = bool(reused)
                if chosen is None:
                    replica.alive = False
                    replica.status = "repair_nvp"
                    trace["negative_reasons"].append("repair_no_admissible_B")
                    counts["repair_fail_closed"] += 1
                    traces.append(trace)
                    continue

                is_repair = values["trigger"] is not None
                is_neutral = (
                    neutral_choice is not None and chosen is neutral_choice
                )
                selected_x0 = np.asarray(chosen["x0"], np.float32)
                controls = np.asarray(chosen["controls"], np.float32)
                result = chosen["result"]
                if (
                    trap_streaks[(replica.scenario_id, replica.gamma)]
                    >= KR.TRAP_PATIENCE - 1
                    and KR.predicted_trap(replica.states, controls[0])
                ):
                    replica.alive = False
                    replica.status = "trap_fail_closed"
                    trace.update(
                        trap_fail_closed=True,
                        negative_reasons=["predicted_third_consecutive_trap"],
                    )
                    counts["trap_fail_closed"] += 1
                    traces.append(trace)
                    continue

                execution_source = (
                    (
                        f"kazuki_repair_neutral_"
                        f"{values['trigger']}_{selector}"
                    )
                    if is_neutral
                    else (
                        f"kazuki_repair_{values['trigger']}_{selector}"
                        if is_repair else f"verified_{selector}"
                    )
                )
                window_id = None
                neutral_record = None
                if is_neutral:
                    neutral_record = _neutral_record(
                        len(neutral_records),
                        replica,
                        prepared,
                        chosen,
                        step=step,
                        round_i=round_i,
                        selector=selector,
                        repair_trigger=values["trigger"],
                    )
                    neutral_records.append(neutral_record)
                else:
                    executed_context_id = executed_shard.add_context(
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
                    window_id = executed_shard.add_executed_window(
                        executed_context_id,
                        controls,
                        selected_x0,
                        result,
                        execution_source=execution_source,
                        nvp_context=values["trigger"] == "finite_B_NVP",
                        candidate_id=chosen["candidate_id"],
                        acquisition_step=chosen["acquisition_step"],
                        sigma=chosen["sigma"],
                        hp_margin=chosen["hp_margin"],
                        mode=chosen["mode"],
                    )
                sidecar_query_id = chosen.get("query_id")
                if sidecar_query_id is not None:
                    query_shard.mark_executed(
                        sidecar_query_id,
                        hp_margin=chosen["hp_margin"],
                        expert_cost=chosen.get("expert_cost"),
                    )
                    query_stored = query_shard.queries[
                        int(sidecar_query_id)
                    ]
                    query_stored["execution_role"] = (
                        "neutral_continuation"
                        if is_neutral else "verified_execution"
                    )
                    query_stored["neutral_id"] = (
                        int(neutral_record["neutral_id"])
                        if is_neutral else None
                    )

                BX._advance(replica, controls[0])
                trap_event = FA._trap(replica.states)
                trap_key = (replica.scenario_id, replica.gamma)
                streak, stop = KR.next_trap_streak(
                    trap_streaks[trap_key], trap_event,
                )
                trap_streaks[trap_key] = streak
                collision, success, clearance = FA._post_action_terminal(replica)
                if stop and replica.alive:
                    replica.alive = False
                    replica.status = "trap_fail_closed"
                    counts["trap_fail_closed"] += 1
                if is_neutral:
                    neutral_record.update(
                        next_state=replica.state.copy(),
                        trap_event=bool(trap_event),
                        trap_streak=int(streak),
                        collision_after_action=bool(collision),
                        success_after_action=bool(success),
                        clearance_after_action=float(clearance),
                    )
                    counts["neutral_executions"] += 1
                else:
                    stored = executed_shard.windows[int(window_id)]
                    stored.update(
                        trap_event=bool(trap_event),
                        trap_streak=int(streak),
                        collision_after_action=bool(collision),
                        success_after_action=bool(success),
                    )
                    counts["executed_training_windows"] += 1
                trace.update(
                    next_state=replica.state.copy(),
                    repair_selected_id=(
                        int(chosen["candidate_id"]) if is_repair else None
                    ),
                    neutral_execution=bool(is_neutral),
                    neutral_id=(
                        int(neutral_record["neutral_id"])
                        if is_neutral else None
                    ),
                    executed_id=int(chosen["candidate_id"]),
                    executed_controls=controls,
                    executed_x0=selected_x0,
                    executed_result=result,
                    executed_label=(
                        "neutral"
                        if is_neutral else FA._result_label(result)
                    ),
                    executed_verifier_label=FA._result_label(result),
                    execution_source=execution_source,
                    window_id=(
                        None if window_id is None else int(window_id)
                    ),
                    trap_event=bool(trap_event),
                    trap_streak_after=int(streak),
                    trap_fail_closed=bool(stop),
                    collision_after_action=bool(collision),
                    success_after_action=bool(success),
                    clearance_after_action=float(clearance),
                )
                counts["executed_windows"] += 1
                counts[f"source_{execution_source}"] += 1
                traces.append(trace)

    for replica in replicas:
        if replica.alive:
            replica.alive = False
            replica.status = "timeout"
    if BX.policy_sha256(policy) != policy_hash:
        raise RuntimeError("policy changed during frozen repair collection")
    executed_summary = executed_shard.validate()
    query_summary = query_shard.validate()
    if int(executed_summary["Dminus"]) != 0:
        raise RuntimeError("repair collector executed a verifier-negative window")
    if len(neutral_records) != int(counts["neutral_executions"]):
        raise RuntimeError("neutral execution accounting mismatch")
    if int(counts["executed_training_windows"]) != int(
        executed_summary["D"]
    ):
        raise RuntimeError("executed training-window accounting mismatch")
    if int(counts["executed_windows"]) != (
        int(executed_summary["D"]) + len(neutral_records)
    ):
        raise RuntimeError("total executed-action accounting mismatch")
    if not bool(neutral_continuation) and neutral_records:
        raise RuntimeError("default fail-closed arm produced neutral records")
    if int(counts["base_verifier_queries"]) != (
        len(query_shard.contexts) * cfg.B
        + int(counts["crunch_full_pool_verifier_queries"])
    ):
        raise RuntimeError("base B=4 accounting mismatch")
    if not crunch_full_pool and int(
        counts["crunch_full_pool_verifier_queries"]
    ):
        raise RuntimeError("full-pool queries issued while disabled")
    if crunch_full_pool and int(
        counts["crunch_full_pool_verifier_queries"]
    ) != int(counts["crunch_full_pool_contexts"]) * (cfg.K - cfg.B):
        raise RuntimeError("full-pool query accounting mismatch")
    if not guided_collect and guided_positive_records:
        raise RuntimeError("guided-positive rows collected while disabled")
    if guided_collect:
        if len(guided_positive_records) != int(
            counts["guided_positive_records"]
        ):
            raise RuntimeError("guided-positive accounting mismatch")
        if int(counts["guided_proactive_verifier_queries"]) != (
            int(counts["guided_proactive_verifier_positive"])
            + int(counts["guided_proactive_verifier_negative"])
            + int(counts["guided_proactive_verifier_error"])
        ):
            raise RuntimeError("proactive guided query accounting mismatch")

    os.makedirs(outdir)
    executed_path = os.path.join(outdir, "executed_round.pt")
    query_path = os.path.join(outdir, "query_sidecar.pt")
    neutral_path = os.path.join(outdir, "neutral_round.pt")
    executed_manifest = executed_shard.save(executed_path)
    query_manifest = query_shard.save(query_path)
    neutral_manifest = _save_neutral_records(
        neutral_path, neutral_records, round_i=round_i,
    )
    guided_positive_manifest = (
        _save_guided_positive_records(
            guided_positive_path,
            guided_positive_records,
            round_i=round_i,
            until_step=guided_collect_until_step,
            generator=guided_generator,
        )
        if guided_collect else None
    )
    outcomes = [dict(
        scenario_id=int(replica.scenario_id),
        gamma=float(replica.gamma),
        status=str(replica.status),
        success=replica.status == "success",
        collision=replica.status == "collision",
        timeout=replica.status == "timeout",
        repair_nvp=replica.status == "repair_nvp",
        trap_fail_closed=replica.status == "trap_fail_closed",
        steps=len(replica.controls),
        minimum_clearance=float(replica.minimum_clearance),
    ) for replica in replicas]
    bundle = dict(
        version=1,
        status=STATUS,
        round=round_i,
        source=FA._source(),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        scene_profile=scene_profile,
        environment=environment,
        scenarios=list(scenarios),
        gammas=list(gammas),
        selector=selector,
        neutral_continuation=bool(neutral_continuation),
        protocol=dict(
            K=cfg.K,
            B=cfg.B,
            H=cfg.H,
            T=int(T),
            ell=float(ell),
            gp_cap=int(gp_cap),
            gp_buffer_ids=gp_ids,
            gp_selection=gp_selection,
            beta=float(beta),
            calibrated_ess_over_K=float(calibrated_ess),
            realized_ess_over_K=float(np.mean(ess_values)),
            gp_diagnostics=gp.diagnostics(),
            acquisition=BR.acquisition_diagnostics(
                sigma_pool, sigma_selected,
            ),
            repair=KR.manifest(),
            D_exec=(
                "one exact-positive executed H10 window per executed context"
            ),
            D_query=(
                "all resolved base and repair B labels; audit-only by default"
            ),
            D_neutral=(
                "guided full-H verifier-negative actions actually executed "
                "after an NVP/trap repair trigger; isolated from "
                "training D+/D-/GP/replay and retained as exact-negative "
                "in the audit query sidecar"
            ),
            GP_population="unchanged: prior executed D+ only",
        ),
        sample_seed=int(sample_seed),
        audit_seed=int(audit_seed),
        counts=dict(counts),
        outcomes=outcomes,
        executed_shard=executed_manifest,
        query_sidecar=query_manifest,
        neutral_shard=neutral_manifest,
        traces=traces,
    )
    guided_collect_manifest = None
    if guided_collect:
        guided_collect_manifest = dict(
            until_step=int(guided_collect_until_step),
            store=os.path.abspath(guided_positive_path),
            generator=str(guided_generator),
            topk=(
                int(cfg.B) if guided_generator == "same_latent"
                else int(guided_topk)
            ),
            candidate_id_base=int(2 * cfg.K),
            population="Gplus",
            policy=(
                "at every live step t < until_step the selected guided "
                "generator proposes plans at the current context and the "
                "exact verifier labels them; certified positives are stored "
                "as G+ and never touch execution, selection, repair "
                "triggering, D0, or the GP"
            ),
            trigger_reuse=(
                "same_latent: contexts whose genuine repair trigger fired at "
                "the same step reuse the identical trigger rows (same "
                "latents, same frozen policy) instead of regenerating them; "
                "kazuki_full: an independent external controller, so every "
                "live context is harvested and no trigger row is reused"
            ),
            generator_manifest=(
                KR.manifest() if guided_generator == "same_latent"
                else KR.full_controller_manifest()
            ),
        )
        bundle["guided_collect"] = guided_collect_manifest
        bundle["guided_positive_shard"] = guided_positive_manifest
    crunch_manifest = None
    if crunch_full_pool:
        crunch_manifest = dict(
            until_step=int(crunch_full_pool_until_step),
            K=int(cfg.K),
            B=int(cfg.B),
            extra_queries=int(counts["crunch_full_pool_verifier_queries"]),
            policy=(
                "for t < until_step every one of the K proposals is verified "
                "and joins the admissible set; the GP still records its B "
                "picks and the executed selector is unchanged, so this arm "
                "removes only the acquisition-budget truncation"
            ),
            external_teacher=False,
        )
        bundle["crunch_full_pool"] = crunch_manifest
    trace_path = os.path.join(outdir, "repair_trace.pt")
    FA._save_torch(trace_path, bundle)
    marker = dict(
        status=STATUS,
        source=bundle["source"],
        trace_path=os.path.abspath(trace_path),
        trace_sha256=FA._sha256_file(trace_path),
        checkpoint_sha256=checkpoint_sha,
        round=round_i,
        selector=selector,
        neutral_continuation=bool(neutral_continuation),
        counts=dict(counts),
        outcomes=outcomes,
        executed_shard=executed_manifest,
        query_sidecar=query_manifest,
        neutral_shard=neutral_manifest,
        gp_buffer_ids=gp_ids,
        gp_selection=gp_selection,
        gp_diagnostics=gp.diagnostics(),
        acquisition=bundle["protocol"]["acquisition"],
    )
    if guided_collect:
        marker["guided_collect"] = guided_collect_manifest
        marker["guided_positive_shard"] = guided_positive_manifest
    if crunch_full_pool:
        marker["crunch_full_pool"] = crunch_manifest
    FA._write_json(os.path.join(outdir, "COMPLETE.json"), marker)
    return trace_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--scenarios", type=int, nargs="+", default=DEFAULT_SCENARIOS,
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS,
    )
    parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    parser.add_argument(
        "--selector", choices=(
            "margin", "progress_gated_margin", "safemppi_cost",
        ), default="margin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verifier-workers", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--ell", type=float, default=DEFAULT_ELL)
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--neutral-continuation", action="store_true")
    parser.add_argument(
        "--guided-collect-until-step", type=int, default=0,
        help=(
            "0 disables the proactive guided harvest entirely (byte-identical "
            "control); a positive value runs the locked guidance on every "
            "live context at steps t < value and stores the exact-verifier "
            "certified positives as the collection-only G+ population"
        ),
    )
    parser.add_argument(
        "--guided-collect-store", default=None,
        help=(
            "optional G+ store path; relative names resolve inside --outdir "
            f"(default {GUIDED_POSITIVE_STORE})"
        ),
    )
    parser.add_argument(
        "--guided-generator", choices=GUIDED_GENERATORS,
        default=DEFAULT_GUIDED_GENERATOR,
        help=(
            "same_latent regenerates the selected B=4 latents under the "
            "locked guidance; kazuki_full runs the complete locked external "
            "Kazuki controller (200 samples, 10 elites, 200 MPPI copies) at "
            "the context and keeps its top --guided-topk refined modes"
        ),
    )
    parser.add_argument(
        "--guided-topk", type=int, default=DEFAULT_GUIDED_TOPK,
        help="kazuki_full only: refined elite modes verified per context",
    )
    parser.add_argument(
        "--crunch-full-pool-until-step", type=int, default=0,
        help=(
            "0 disables the arm (byte-identical control); a positive value "
            "verifies all K=16 proposals instead of the GP's B=4 for every "
            "step below it, removing the finite-B acquisition truncation "
            "without any external teacher"
        ),
    )
    args = parser.parse_args(argv)
    collect(
        args.checkpoint,
        scenarios=args.scenarios,
        gammas=args.gammas,
        scene_profile=args.scene_profile,
        selector=args.selector,
        device=args.device,
        verifier_workers=args.verifier_workers,
        sample_seed=args.sample_seed,
        audit_seed=args.audit_seed,
        ell=args.ell,
        ess_target=args.ess_target,
        neutral_continuation=args.neutral_continuation,
        guided_collect_until_step=args.guided_collect_until_step,
        guided_collect_store=args.guided_collect_store,
        guided_generator=args.guided_generator,
        guided_topk=args.guided_topk,
        crunch_full_pool_until_step=args.crunch_full_pool_until_step,
        T=180,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
