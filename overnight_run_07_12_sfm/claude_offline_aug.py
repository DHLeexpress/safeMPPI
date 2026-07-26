"""Opt-in replay-data interventions for offline executed-window SFM expansion.

Everything in this module is additive and OFF by default: with
``replay_mode="original"`` the caller uses ``sfm_b1_offline_replay.replay`` on
the untouched :class:`ExecutedRoundShard` and no code here runs.  The exact
B1 gathering, GP acquisition, B querying, verifier, and raw evaluation are
never modified here; this module only re-composes which resolved executed
records (plus optional exact-certified synthetic recovery records) the replay
step trains on.

Declared population rules (fixed BEFORE evaluating their effect):

Population A - near-obstacle verified positives.  An executed window with
``y=1`` whose plan-time constant-velocity predicted minimum time-indexed
pedestrian clearance is at most ``NEAR_CLEARANCE_MAX`` metres and whose
10-step displacement is at least ``MIN_DISPLACEMENT`` metres (successful
avoidance = certified, close to a constraint boundary, and actually moving;
the displacement floor equals the existing declared trap displacement).

Population B - collision/trap/no-progress negatives.  An executed window with
``y=0`` satisfying at least one of: actual collision after the executed
action (``collision_after_action``); plan-time predicted window collision
(``collision_free`` false); the existing declared trap rule fired at this
step (``trap_event``: closed-loop displacement < 0.2 m over the last 10
executed steps); or window displacement below ``MIN_DISPLACEMENT`` (the same
declared 10-step/0.2 m insufficient-progress rule applied to the executed
window).  A certified-but-slow window (y=1) is never relabeled negative.

Population C - deterministic certified recovery positives.  For each hard
context (the context of a population-B window), solve the deterministic
control problem written out in :func:`recovery_candidates`:

    minimize   J(u) = || p_10(u) - GOAL ||_2
    over       the fixed finite deterministic family below
    subject to |u_t|_inf <= U_MAX, double-integrator dynamics with DT,
               and the EXACT full-H10 verifier certificate y=1
               (task-space bounds, time-indexed CV collision avoidance,
               GREEN moving-window SOCP certificate).

Family: closed-loop brake steps b in {0,2,4} (u_t = clip(-v_t/DT)) followed
by bang-bang steering u = a*(cos t_k, sin t_k) for the first half of the
remaining steps and -a*(...) for the second half, over K_DIR=16 world
directions and magnitudes a in {0.7, 1.4, 2.0}; plus the pure 10-step brake.
Candidates are pre-filtered by the cheap exact numpy task-space and
CV-collision checks, ranked by J, and at most ``PREVERIFY_CAP`` are submitted
to the exact SOCP verifier; the first ``RECOVERY_KEEP`` certified candidates
(in increasing J) enter the positive set.  A failed candidate is discarded,
never relabeled.  Recovery rows join the positive replay population at their
parent context, so the existing hierarchical mass (gamma -> episode ->
context -> query) automatically splits the parent context's mass across
them; they carry ``x0 = zeros(20)`` for schema compatibility and are NEVER
eligible for the GP buffer (the GP reads only the real ExecutedRoundShard).
The recovery controller itself is never deployed at evaluation time.
"""
from __future__ import annotations

import math

import numpy as np

import _paths  # noqa: F401
import sfm_b1_offline_replay as OR
import sfm_b1_offline_store as OS
import sfm_metrics2 as SM
import sfm_scene as SS

NEAR_CLEARANCE_MAX = 0.35
MIN_DISPLACEMENT = 0.2
BRAKE_STEPS = (0, 2, 4)
K_DIR = 16
ACCELS = (0.7, 1.4, 2.0)
PREVERIFY_CAP = 24
RECOVERY_KEEP = 2
REPLAY_MODES = (
    "original", "hard", "hard_recovery", "orig_plus_recovery",
    "orig_plus_recovery_v2",
)

# --- Recovery family v2 ("dodge-then-cruise"), declared 2026-07-26 before
# any evaluation of its effect.  Motivation (user hypothesis + Stage-A
# measurement): the v1 brake/bang-bang family certifies CONSERVATIVE escapes
# that end near-stationary; training on them creates slowdown.  v2 candidates
# end moving TOWARD the goal at cruise speed:
#   phase 1 (d in {0,2,3} steps): dodge with u = a*(cos t_k, sin t_k),
#     a in {1.4, 2.0}, t_k over K_DIR world directions (d=0 skips the dodge);
#   phase 2 (remaining steps): deterministic saturated velocity servo
#     u_t = clip(KP_CRUISE * (v_des(p_t) - v_t), +/-U_MAX),
#     v_des(p) = v_c * unit(GOAL - p), v_c in {1.0, 1.5}.
# Objective (v2): J2 = ||p_10 - GOAL|| - 0.5 * (v_10 . unit(GOAL - p_10)) —
# prefer end states that are close to AND moving toward the goal.  The same
# cheap exact prefilter, PREVERIFY_CAP, RECOVERY_KEEP, and the exact full-H10
# SOCP certificate gate apply unchanged.
DODGE_STEPS = (0, 2, 3)
CRUISE_SPEEDS = (1.0, 1.5)
KP_CRUISE = 4.0
V2_ACCELS = (1.4, 2.0)


def declared_rules():
    return dict(
        population_A=dict(
            requires="y=1",
            predicted_min_clearance_max=NEAR_CLEARANCE_MAX,
            window_displacement_min=MIN_DISPLACEMENT,
        ),
        population_B=dict(
            requires="y=0",
            any_of=[
                "collision_after_action",
                "not collision_free (plan-time predicted window collision)",
                "trap_event (declared closed-loop 10-step/0.2m rule)",
                f"window displacement < {MIN_DISPLACEMENT} m over H=10",
            ],
        ),
        population_C=dict(
            objective="min ||p_10(u) - GOAL||_2 over the fixed family",
            family=dict(
                brake_steps=list(BRAKE_STEPS), k_dir=K_DIR,
                accels=list(ACCELS), plus="pure 10-step closed-loop brake",
            ),
            preverify_cap=PREVERIFY_CAP,
            keep_per_context=RECOVERY_KEEP,
            certificate="exact full-H10 SM.verify_query y=1 only",
        ),
    )


def _window_geometry(context, controls):
    segment = SM.rollout_positions(context["state"], controls)
    prediction = SM.predict_pedestrians(
        context["ped_xy"], context["ped_vel"], H=len(controls),
    )
    clearance = float(
        np.linalg.norm(segment[:, None, :] - prediction, axis=2).min()
        - SS.R_PED
    )
    displacement = float(np.linalg.norm(segment[-1] - segment[0]))
    return clearance, displacement


def tag_populations(shard):
    """Classify every executed window; returns (popA, popB, stats)."""
    pop_a, pop_b = [], []
    reasons = dict(actual_collision=0, predicted_collision=0, trap=0,
                   no_progress=0)
    for window in shard.windows:
        context = shard.contexts[int(window["context_id"])]
        clearance, displacement = _window_geometry(
            context, window["controls"],
        )
        if int(window["y"]) == 1:
            if (
                clearance <= NEAR_CLEARANCE_MAX
                and displacement >= MIN_DISPLACEMENT
            ):
                pop_a.append(window)
        else:
            actual = bool(window.get("collision_after_action"))
            predicted = not bool(window["collision_free"])
            trap = bool(window.get("trap_event"))
            slow = displacement < MIN_DISPLACEMENT
            if actual or predicted or trap or slow:
                pop_b.append(window)
                reasons["actual_collision"] += int(actual)
                reasons["predicted_collision"] += int(predicted)
                reasons["trap"] += int(trap)
                reasons["no_progress"] += int(slow)
    stats = dict(
        D=len(shard.windows), Dplus=len(shard.Dplus),
        Dminus=len(shard.Dminus), popA=len(pop_a), popB=len(pop_b),
        popB_reasons=reasons, rules=declared_rules(),
    )
    return pop_a, pop_b, stats


def _closed_loop_brake(state, steps):
    """Deterministic max-effort brake controls for ``steps`` steps."""
    velocity = np.asarray(state, np.float32).reshape(4)[2:4].copy()
    controls = []
    for _ in range(steps):
        action = np.clip(-velocity / SS.DT, -SS.U_MAX, SS.U_MAX)
        controls.append(action.astype(np.float32))
        velocity = velocity + SS.DT * action
    return controls, velocity


def recovery_candidates(state):
    """The fixed deterministic candidate family for one context."""
    candidates = []
    full_brake, _ = _closed_loop_brake(state, 10)
    candidates.append((
        np.asarray(full_brake, np.float32),
        dict(kind="brake10", brake=10, theta=None, accel=None),
    ))
    for brake in BRAKE_STEPS:
        prefix, _ = _closed_loop_brake(state, brake)
        remaining = 10 - brake
        forward = math.ceil(remaining / 2)
        for k in range(K_DIR):
            theta = 2.0 * math.pi * k / K_DIR
            direction = np.array(
                [math.cos(theta), math.sin(theta)], np.float32,
            )
            for accel in ACCELS:
                steer = [accel * direction] * forward
                steer += [-accel * direction] * (remaining - forward)
                controls = np.asarray(prefix + steer, np.float32)
                if controls.shape != (10, 2):
                    raise AssertionError("recovery candidate must be H=10")
                candidates.append((
                    np.clip(controls, -SS.U_MAX, SS.U_MAX),
                    dict(kind="brake_steer", brake=brake,
                         theta=round(theta, 6), accel=accel),
                ))
    return candidates


def _cruise_controls(position, velocity, steps, v_cruise):
    """Deterministic saturated velocity servo toward the goal."""
    position = np.asarray(position, np.float32).copy()
    velocity = np.asarray(velocity, np.float32).copy()
    controls = []
    for _ in range(steps):
        offset = SS.GOAL - position
        norm = float(np.linalg.norm(offset))
        v_des = (
            v_cruise * offset / norm if norm > 1e-6
            else np.zeros(2, np.float32)
        )
        action = np.clip(
            KP_CRUISE * (v_des - velocity), -SS.U_MAX, SS.U_MAX,
        ).astype(np.float32)
        controls.append(action)
        position = position + SS.DT * velocity + 0.5 * SS.DT ** 2 * action
        velocity = velocity + SS.DT * action
    return controls


def recovery_candidates_v2(state, ped_xy=None, ped_vel=None):
    """Goal-directed dodge-then-cruise family (state-only, deterministic)."""
    del ped_xy, ped_vel  # verifier inputs; unused by this state-only family
    state = np.asarray(state, np.float32).reshape(4)
    candidates = []
    for v_cruise in CRUISE_SPEEDS:
        controls = _cruise_controls(state[:2], state[2:4], 10, v_cruise)
        candidates.append((
            np.asarray(controls, np.float32),
            dict(kind="cruise", dodge=0, theta=None, accel=None,
                 v_cruise=v_cruise),
        ))
    for dodge in DODGE_STEPS:
        if dodge == 0:
            continue
        for k in range(K_DIR):
            theta = 2.0 * math.pi * k / K_DIR
            direction = np.array(
                [math.cos(theta), math.sin(theta)], np.float32,
            )
            for accel in V2_ACCELS:
                prefix = [
                    np.clip(accel * direction, -SS.U_MAX, SS.U_MAX)
                    .astype(np.float32)
                ] * dodge
                position = np.asarray(state[:2], np.float32).copy()
                velocity = np.asarray(state[2:4], np.float32).copy()
                for action in prefix:
                    position = (
                        position + SS.DT * velocity
                        + 0.5 * SS.DT ** 2 * action
                    )
                    velocity = velocity + SS.DT * action
                for v_cruise in CRUISE_SPEEDS:
                    controls = prefix + _cruise_controls(
                        position, velocity, 10 - dodge, v_cruise,
                    )
                    controls = np.asarray(controls, np.float32)
                    if controls.shape != (10, 2):
                        raise AssertionError("v2 candidate must be H=10")
                    candidates.append((
                        controls,
                        dict(kind="dodge_cruise", dodge=dodge,
                             theta=round(theta, 6), accel=accel,
                             v_cruise=v_cruise),
                    ))
    return candidates


def _objective_v2(context, controls):
    segment = SM.rollout_positions(context["state"], controls)
    state = np.asarray(context["state"], np.float32).reshape(4)
    velocity = state[2:4].copy()
    for action in np.asarray(controls, np.float32):
        velocity = velocity + SS.DT * action
    offset = SS.GOAL - segment[-1]
    norm = float(np.linalg.norm(offset))
    toward = float(velocity @ (offset / norm)) if norm > 1e-6 else 0.0
    return norm - 0.5 * toward


def _prefilter(context, controls):
    """Cheap exact numpy feasibility check + objective J."""
    segment = SM.rollout_positions(context["state"], controls)
    if not SM.taskspace_ok(segment):
        return None
    prediction = SM.predict_pedestrians(
        context["ped_xy"], context["ped_vel"], H=10,
    )
    if not SM.collision_free_time_indexed(segment, prediction):
        return None
    return float(np.linalg.norm(segment[-1] - SS.GOAL))


def build_recovery_records(shard, hard_windows, executor, family="v1"):
    """Exact-certified recovery positives for the given hard windows.

    Returns (records, audit).  Every returned record passed the exact
    full-H10 verifier inside ``executor`` (the same worker pool and
    ``SM.verify_in_worker`` entry as B1 queries).  ``family`` selects the
    declared deterministic candidate family and ranking objective:
    v1 = brake/bang-bang, J = final goal distance;
    v2 = dodge-then-cruise, J2 = goal distance - 0.5 * toward-goal speed.
    """
    if family not in ("v1", "v2"):
        raise ValueError("recovery family must be v1 or v2")
    context_ids = sorted({int(w["context_id"]) for w in hard_windows})
    tasks, meta = [], []
    per_context_pool = {}
    for context_id in context_ids:
        context = shard.contexts[context_id]
        generator_fn = (
            recovery_candidates if family == "v1"
            else recovery_candidates_v2
        )
        scored = []
        for controls, provenance in generator_fn(context["state"]):
            feasible = _prefilter(context, controls)
            if feasible is None:
                continue
            objective = (
                feasible if family == "v1"
                else _objective_v2(context, controls)
            )
            scored.append((objective, controls, provenance))
        scored.sort(key=lambda row: (row[0], str(row[2])))
        pool = scored[:PREVERIFY_CAP]
        per_context_pool[context_id] = len(pool)
        for rank, (objective, controls, provenance) in enumerate(pool):
            tasks.append((
                context_id, rank, context["state"], controls,
                context["ped_xy"], context["ped_vel"], context["gamma"],
            ))
            meta.append((context_id, rank, objective, controls, provenance))
    results = list(executor.map(SM.verify_in_worker, tasks))
    verified = {}
    for (context_id, rank, result), (_, _, objective, controls, provenance) \
            in zip(results, meta):
        verified.setdefault(int(context_id), []).append(
            (int(rank), float(objective), controls, provenance, result),
        )
    records, audit_rows = [], []
    certified_total = queried_total = 0
    for context_id in context_ids:
        rows = sorted(verified.get(context_id, []), key=lambda r: r[0])
        queried_total += len(rows)
        kept = 0
        for rank, objective, controls, provenance, result in rows:
            if kept >= RECOVERY_KEEP:
                break
            if not result.get("resolved"):
                continue
            if int(result.get("y", 0)) != 1 or not bool(result.get("full_h")):
                continue
            certified_total += 1
            kept += 1
            records.append(dict(
                window_id=None, query_id=None,
                context_id=int(context_id),
                controls=np.asarray(controls, np.float32),
                x0=np.zeros(20, np.float32),
                y=1, taskspace=True, collision_free=True, certificate=True,
                full_h=True, terminal_step=10, train_eligible=True,
                execution_source="synthetic_certified_recovery",
                nvp_context=False, candidate_id=None, acquisition_step=None,
                sigma=None, hp_margin=None, mode="recovery",
                verifier_diagnostics=dict(result["diagnostics"]),
                recovery_provenance=dict(
                    parent_round=int(shard.round_i),
                    parent_context_id=int(context_id),
                    family=str(family),
                    generator=provenance,
                    objective=float(objective),
                    prefilter_rank=int(rank),
                ),
            ))
            audit_rows.append(dict(
                context_id=int(context_id), rank=int(rank),
                generator=provenance, J=float(objective),
                verifier=dict(
                    y=int(result["y"]), taskspace=bool(result["taskspace"]),
                    collision_free=bool(result["collision_free"]),
                    certificate=bool(result["certificate"]),
                    slack=float(result["diagnostics"]["slack"]),
                ),
            ))
    audit = dict(
        hard_contexts=len(context_ids),
        exact_verifier_queries=queried_total,
        certified_kept=len(records),
        certified_total_seen=certified_total,
        keep_per_context=RECOVERY_KEEP,
        preverify_cap=PREVERIFY_CAP,
        per_context_preverified_pool_mean=(
            float(np.mean(list(per_context_pool.values())))
            if per_context_pool else 0.0
        ),
        family=str(family),
        rows=audit_rows,
    )
    return records, audit


class ShardView:
    """Duck-typed shard exposing a re-composed training population.

    Shares the parent shard's contexts; windows are the selected subset plus
    optional synthetic records, re-indexed with dense window/query ids so the
    untouched ``sfm_b1_offline_replay.replay`` machinery (hierarchy mass,
    stratified batches, exact-once accounting) applies verbatim.
    """

    def __init__(self, parent, windows):
        self.round_i = int(parent.round_i)
        self.contexts = parent.contexts
        self.windows = []
        for index, window in enumerate(windows):
            row = dict(window)
            row["window_id"] = index
            row["query_id"] = index
            self.windows.append(row)

    @property
    def D(self):
        return list(self.windows)

    @property
    def Dplus(self):
        return [row for row in self.windows if row["y"] == 1]

    @property
    def Dminus(self):
        return [row for row in self.windows if row["y"] == 0]


def build_replay_view(shard, mode, executor=None):
    """Compose the replay population for ``mode``; returns (view, report)."""
    if mode not in REPLAY_MODES:
        raise ValueError(f"replay mode must be one of {REPLAY_MODES}")
    if mode == "original":
        return shard, dict(mode=mode, note="untouched ExecutedRoundShard")
    pop_a, pop_b, stats = tag_populations(shard)
    report = dict(mode=mode, populations=stats)
    if mode in ("orig_plus_recovery", "orig_plus_recovery_v2"):
        # Declared BEFORE evaluation (Stage-A log 2026-07-26): keep the FULL
        # original positive and negative populations and only APPEND the
        # exact-certified recovery positives at their parent (hard) contexts.
        # The _v2 variant uses the declared dodge-then-cruise family.
        if executor is None:
            raise ValueError("orig_plus_recovery needs the verifier executor")
        recovery, audit = build_recovery_records(
            shard, pop_b, executor,
            family="v2" if mode.endswith("_v2") else "v1",
        )
        report["recovery_audit"] = audit
        windows = list(shard.windows) + recovery
    else:
        windows = list(pop_a) + list(pop_b)
    if mode == "hard_recovery":
        if executor is None:
            raise ValueError("hard_recovery needs the verifier executor")
        recovery, audit = build_recovery_records(shard, pop_b, executor)
        report["recovery_audit"] = audit
        windows = windows + recovery
    if not any(int(row["y"]) == 1 for row in windows):
        # Fail open to the untouched population rather than training on a
        # positive-free set (the replay contract requires positives).
        report["fallback"] = "no positives in composed set; using original"
        return shard, report
    view = ShardView(shard, windows)
    report["view"] = dict(
        D=len(view.windows), Dplus=len(view.Dplus), Dminus=len(view.Dminus),
    )
    return view, report


def replay_with_mode(
    policy, optimizer, shard, *, mode, alpha, exposure_epochs, batch,
    device, seed, executor=None,
):
    """Opt-in wrapper: ``original`` delegates verbatim to OR.replay."""
    view, report = build_replay_view(shard, mode, executor=executor)
    result = OR.replay(
        policy, optimizer, view,
        alpha=alpha, exposure_epochs=exposure_epochs, batch=batch,
        device=device, seed=seed,
    )
    result["replay_intervention"] = report
    return result
