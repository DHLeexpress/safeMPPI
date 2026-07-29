"""Exact-SOCP-faithful AFE over the historical privileged candidate pool.

Golden rule: the exact full-H SOCP verifier is the ONLY execution/replay
authority (resolved AND y=1 AND full_h AND terminal_step=10).  The privileged
SFM simulation generates and ranks proposals but never certifies; NVP
terminates fail-closed; privileged-safe/SOCP-negative is never relabeled.
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch

import _paths  # noqa: F401
import claude_corrected_distill as CD
import claude_mpc_pool as MP
import grid_feats as GF
import sfm_b1_offline_exec as OE
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_hp_history as HH
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_scene as SS

SEED = 20260729
ELL, LAM, CAP = 0.3259987743470518, 1.0e-2, 512
B_BUDGET = 8
S_PHI = 0.9


def certified(result):
    return bool(
        result.get("resolved") and int(result.get("y", 0)) == 1
        and bool(result.get("full_h"))
        and int(result.get("terminal_step", -1)) == 10
    )


def keyed_rng(*parts):
    payload = ":".join(str(p) for p in parts).encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"),
    )


class VerifierCache:
    """Shared read-only-safe cache keyed by (context, U, gamma, version)."""

    def __init__(self):
        self._data = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(context, controls, gamma):
        h = hashlib.sha256()
        for arr in (context["state"], context["ped_xy"], context["ped_vel"]):
            h.update(np.ascontiguousarray(arr, np.float32).tobytes())
        h.update(np.ascontiguousarray(controls, np.float32).tobytes())
        h.update(f"{float(gamma):.8f}".encode())
        h.update(str(SM.verifier_manifest()).encode())
        return h.digest()

    def verify_many(self, context, plans, indices, executor):
        results = {}
        tasks = []
        for index in indices:
            k = self.key(context, plans[index], context["gamma"])
            if k in self._data:
                self.hits += 1
                results[index] = self._data[k]
            else:
                tasks.append((index, k))
        if tasks:
            payloads = [
                (i, 0, context["state"], plans[i], context["ped_xy"],
                 context["ped_vel"], float(context["gamma"]))
                for i, _ in tasks
            ]
            outs = {i: r for i, _, r in executor.map(
                SM.verify_in_worker, payloads,
            )}
            for i, k in tasks:
                self.misses += 1
                slim = {key: outs[i][key] for key in (
                    "resolved", "y", "taskspace", "collision_free",
                    "certificate", "full_h", "terminal_step",
                ) if key in outs[i]}
                slim["diagnostics"] = dict(
                    slack=float(outs[i]["diagnostics"]["slack"]),
                ) if outs[i].get("resolved") else {}
                self._data[k] = slim
                results[i] = slim
        return results


def controller_scores(plans, context, cfg_gamma):
    """Faithful replication of the committed step-filter score."""
    clear, inside, terminal, _, reach = KZ._simulate_sfm_plans(
        _humans_placeholder(context), context["state"], np.stack(plans), 10,
    )
    goal = np.asarray(SS.GOAL, np.float32)
    gsw = float(cfg_gamma.step_filter_goal_score_weight)
    cw = float(cfg_gamma.step_filter_clearance_weight)
    nominal_u0 = np.asarray(plans[0][0], np.float32)
    scores = []
    for i, plan in enumerate(plans):
        goal_cost = (
            0.04 * float(reach[i]) if reach[i] <= 10
            else float(np.linalg.norm(terminal[i, :2] - goal))
        )
        scores.append(
            gsw * goal_cost + 0.015 * float(np.mean(plan * plan))
            + 0.02 * float(np.sum((plan[0] - nominal_u0) ** 2))
            - cw * min(float(clear[i]), 1.0)
        )
    return np.asarray(scores, np.float64), clear, inside, reach


_HUMANS = {}


def _humans_placeholder(context):
    return _HUMANS["current"]


def build_pool(policy, context, humans, device):
    """Complete historical pool + keyed x0 + controller scores J."""
    _HUMANS["current"] = humans
    pool = MP.build_codex_pool(
        policy, context, humans, device=device,
        seed_step=int(context["step"]),
    )
    plans = pool["plans"]
    cfg_gamma = KZ._gamma_controller_config(
        MP.privileged_sfm_config(), float(context["gamma"]),
    ).validate()
    scores, clear, inside, reach = controller_scores(
        list(plans), context, cfg_gamma,
    )
    x0 = np.stack([
        keyed_rng(SEED, "afe_x0", context["scenario_id"],
                  f"{float(context['gamma']):.8f}", context["step"], i)
        .standard_normal(int(policy.d)).astype(np.float32)
        for i in range(len(plans))
    ])
    return dict(
        plans=plans, x0=x0, J=scores,
        privileged_feasible=pool["privileged_feasible"],
        privileged_clearance=clear,
    )


@torch.no_grad()
def pool_features(policy, context, plans, x0, device):
    hp10 = torch.as_tensor(context["hp10"], device=device)[None].float()
    low = torch.as_tensor(context["low5"], device=device)[None].float()
    hist = torch.as_tensor(context["hist"], device=device)[None].float()
    ctx = policy.ctx_from(hp10, low, hist)
    features = policy.phi_s_from_x0(
        torch.as_tensor(np.stack(plans), device=device).float(),
        ctx.repeat_interleave(len(plans), dim=0),
        torch.as_tensor(x0, device=device).float(), s=S_PHI,
    )
    return BR.l2_normalize(features)


class CertifiedQueryShard:
    """Set-valued certified store: D = resolved B queries, D+ = ALL positives."""

    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.windows = []

    def add_context(self, **kw):
        kw["context_id"] = len(self.contexts)
        kw["round"] = self.round_i
        self.contexts.append(kw)
        return kw["context_id"]

    def add_query(self, context_id, controls, x0, result, *, J, sigma,
                  executed, source):
        if int(result.get("y", -1)) == 1 and not certified(result):
            raise ValueError("positive query must satisfy the golden rule")
        row = dict(
            window_id=len(self.windows), query_id=len(self.windows),
            context_id=int(context_id),
            controls=np.asarray(controls, np.float32),
            x0=np.asarray(x0, np.float32),
            y=int(result.get("y", 0)) if result.get("resolved") else 0,
            resolved=bool(result.get("resolved")),
            train_eligible=certified(result),
            J=float(J), sigma=None if sigma is None else float(sigma),
            executed=bool(executed), source=str(source),
        )
        self.windows.append(row)
        return row["window_id"]

    @property
    def Dplus(self):
        return [r for r in self.windows if r["train_eligible"]]

    @property
    def Dminus(self):
        return [r for r in self.windows if r["resolved"] and not r["y"]]

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(dict(round=self.round_i, contexts=self.contexts,
                        windows=self.windows), path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        shard = cls(payload["round"])
        shard.contexts = payload["contexts"]
        shard.windows = payload["windows"]
        return shard


def run_certified_episode(policy, scenario, gamma, *, mode, gp, beta, device,
                          executor, cache, shard=None, T=180, reach=0.5):
    """Closed-loop F (verify all) or B8 (AFE budget) certified controller."""
    environment = SS.scene_profile("double_density_velocity_ood")
    humans = SS.make_humans(int(scenario), 0, environment["n_ped"],
                            tuple(environment["ped_speed_range"]))
    state = np.zeros(4, np.float32)
    history = HH.HpHistory()
    controls_list = []
    status, min_clear = None, float("inf")
    stats = dict(steps=0, nvp=False, pool_positive_counts=[],
                 b_positive_counts=[], sigma_all=[], sigma_selected=[],
                 oracle_rejected=0)
    states, peds, pvels = [state.copy()], [], []
    for t in range(int(T)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        clearance = float(np.linalg.norm(
            ped_xy - state[:2][None], axis=1,
        ).min() - SS.R_PED)
        min_clear = min(min_clear, clearance)
        if clearance < 0.0:
            status = "collision"
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            status = "success"
            break
        obstacles = np.concatenate([
            ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32),
        ], axis=1)
        hp10 = history.append(torch.as_tensor(GF.axis_grid(
            state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE,
        )))
        context = dict(
            scenario_id=int(scenario), gamma=float(gamma), step=int(t),
            state=state.copy(),
            hp10=hp10.numpy().astype(np.float32),
            low5=np.asarray(GF.low5(state, SS.GOAL, gamma), np.float32),
            hist=np.asarray(GF.hist_pad(
                np.asarray(controls_list[-16:]) if controls_list
                else np.zeros((0, 2)), 16,
            ), np.float32),
            ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
        )
        pool = build_pool(policy, context, humans, device)
        n = len(pool["plans"])
        if mode == "full":
            queried = list(range(n))
            sigmas = [None] * n
        else:
            features = pool_features(
                policy, context, pool["plans"], pool["x0"], device,
            )
            generator = torch.Generator(device=features.device)
            generator.manual_seed(int(keyed_rng(
                SEED, "acq", scenario, f"{gamma:.8f}", t,
            ).integers(0, 2**62)))
            selected, trace = gp.sequential_acquire(
                features, min(B_BUDGET, n), beta, generator=generator,
            )
            queried = list(map(int, selected))
            stats["sigma_all"].extend(
                float(v) for v in
                trace[0]["scores"].clamp_min(0).sqrt()[:64]
            )
            stats["sigma_selected"].extend(
                float(r["chosen_sigma"]) for r in trace
            )
            sigmas = {q: float(r["chosen_sigma"])
                      for q, r in zip(queried, trace)}
        results = cache.verify_many(context, pool["plans"], queried, executor)
        positives = [i for i in queried if certified(results[i])]
        stats["pool_positive_counts"].append(
            len(positives) if mode == "full" else None,
        )
        stats["b_positive_counts"].append(len(positives))
        context_id = None
        if shard is not None:
            context_id = shard.add_context(**context)
            for i in queried:
                if results[i].get("resolved"):
                    shard.add_query(
                        context_id, pool["plans"][i], pool["x0"][i],
                        results[i], J=pool["J"][i],
                        sigma=(sigmas[i] if isinstance(sigmas, dict)
                               else None),
                        executed=False, source="pool",
                    )
        if not positives:
            status = "nvp"
            stats["nvp"] = True
            break
        best = min(positives, key=lambda i: (pool["J"][i], i))
        oracle_best = int(np.argmin(pool["J"]))
        if oracle_best not in positives:
            stats["oracle_rejected"] += 1
        if shard is not None:
            for row in shard.windows[::-1]:
                if row["context_id"] != context_id:
                    break
                if np.array_equal(row["controls"], pool["plans"][best]):
                    row["executed"] = True
                    break
        action = np.asarray(pool["plans"][best][0], np.float32)
        controls_list.append(action)
        state = state.copy()
        state[:2] += SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
        state[2:4] += SS.DT * action
        states.append(state.copy())
        peds.append(ped_xy)
        pvels.append(ped_vel)
        SS.advance_humans(humans, state)
    if status is None:
        status = "timeout"
    stats.update(
        status=status, steps=len(controls_list),
        min_clearance=float(min_clear),
        states=np.asarray(states, np.float32),
        controls=np.asarray(controls_list, np.float32),
        ped_xy=(np.asarray(peds, np.float32) if peds
                else np.zeros((0, 0, 2), np.float32)),
        ped_vel=(np.asarray(pvels, np.float32) if pvels
                 else np.zeros((0, 0, 2), np.float32)),
    )
    return stats


def build_round_gp(phi_policy, previous_shard, *, device, ess_target,
                   round_seed):
    """GP from previous round's certified positives; graceful gamma quota."""
    gp = BR.RBFGP(float(ELL), float(LAM))
    identities = []
    if previous_shard is not None:
        by_gamma = {}
        for row in previous_shard.Dplus:
            context = previous_shard.contexts[int(row["context_id"])]
            by_gamma.setdefault(
                round(float(context["gamma"]), 8), [],
            ).append(row)
        quota = CAP // 7
        chosen = []
        for gamma in sorted(by_gamma):
            rows = sorted(
                by_gamma[gamma],
                key=lambda r: (int(r["context_id"]), int(r["query_id"])),
            )
            chosen.extend(rows[:quota])
        parts = []
        for start in range(0, len(chosen), 256):
            values = chosen[start:start + 256]
            records = [(previous_shard, row) for row in values]
            grid, low, hist, controls = BS._tensor_batch(records, device)
            x0 = torch.as_tensor(np.stack([
                np.asarray(row["x0"], np.float32) for row in values
            ]), device=device)
            parts.append(phi_policy.phi_s_from_x0(
                controls, phi_policy.ctx_from(grid, low, hist), x0, s=S_PHI,
            ))
        if parts:
            gp.set_buffer(BR.l2_normalize(torch.cat(parts)))
        identities = [int(row["query_id"]) for row in chosen]
    return gp, identities


def calibrate_tau_j(shard):
    """tau_J so the within-context median ESS/|P_c| is 0.5 (bisection)."""
    groups = {}
    for row in shard.Dplus:
        groups.setdefault(int(row["context_id"]), []).append(float(row.get("J", 0.0)))
    multi = [np.asarray(v) for v in groups.values() if len(v) > 1]
    if not multi:
        return None, 1.0

    def median_ess(tau):
        values = []
        for J in multi:
            q = np.exp(-(J - J.min()) / max(tau, 1e-9))
            q = q / q.sum()
            values.append(1.0 / (np.square(q).sum() * len(q)))
        return float(np.median(values))

    lo, hi = 1e-6, 1e6
    if median_ess(hi) < 0.5:
        return hi, median_ess(hi)
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if median_ess(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return hi, median_ess(hi)


def objective_weights(shards, objective):
    """Per-record weights over the W=2 certified union; sums to 1."""
    records = [(s, row) for s in shards for row in s.Dplus]
    hmass, accounting = BS.hierarchy_mass(records)
    if objective == "U":
        weights = {k: float(v) for k, v in hmass.items()}
        tau = None
    else:
        weights = {}
        context_mass = {}
        context_rows = {}
        for shard, row in records:
            key = (id(shard), int(row["context_id"]))
            context_mass[key] = context_mass.get(key, 0.0) + float(
                hmass[(id(shard), int(row["query_id"]))],
            )
            context_rows.setdefault(key, []).append((shard, row))
        tau = None
        taus = [calibrate_tau_j(s)[0] for s in shards]
        taus = [t for t in taus if t is not None]
        tau = float(np.median(taus)) if taus else 1.0
        for key, rows in context_rows.items():
            J = np.asarray([float(r.get("J", 0.0)) for _, r in rows])
            q = np.exp(-(J - J.min()) / max(tau, 1e-9))
            q = q / q.sum()
            for (shard, row), qi in zip(rows, q):
                weights[(id(shard), int(row["query_id"]))] = (
                    context_mass[key] * float(qi)
                )
    residual = abs(sum(weights.values()) - 1.0)
    return records, weights, dict(
        objective=objective, tau_J=tau, mass_residual=residual,
        gamma=accounting["gamma"],
    )


def certified_replay(policy, optimizer, shards, *, objective, epochs, batch,
                     device, seed):
    """Whole-dataset accumulated replay; one Adam step per epoch."""
    records, weights, info = objective_weights(shards, objective)
    if info["mass_residual"] > 1e-6:
        raise RuntimeError(f"hierarchy-mass residual {info['mass_residual']}")
    encoder = BS.module_sha256(policy.enc_grid)
    policy.train()
    losses = []
    n = len(records)
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, n, int(batch)):
            values = records[start:start + int(batch)]
            grid, low, hist, controls = BS._tensor_batch(values, device)
            ctx = policy.ctx_from(grid, low, hist)
            w = torch.as_tensor([
                len(values) * weights[(id(h), int(r["query_id"]))]
                for h, r in values
            ], dtype=controls.dtype, device=device)
            torch.manual_seed(int(seed) + epoch * 1_000_003 + start)
            loss = policy.cfm_loss(controls, ctx, weights=w)
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        losses.append(total)
    policy.eval()
    if BS.module_sha256(policy.enc_grid) != encoder:
        raise RuntimeError("visual encoder changed")
    info.update(adam_steps=int(epochs), losses=losses, records=n,
                coverage=n)
    return info
