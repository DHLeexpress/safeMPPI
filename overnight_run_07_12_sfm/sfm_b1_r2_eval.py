"""Canonical raw temperature-one evaluation for SFM rounds 0, 1, and 2.

This module is intentionally independent of expansion-time acquisition.  It
uses one fixed M=50/scenario/gamma bank and one fixed latent-noise bank for
every checkpoint, samples one raw flow window per context at temperature one,
and executes its first action.  No RBF tilt, verifier selection, fallback,
guidance, or temperature search is present.

The archived double-shift M100 baseline is carried only as a labeled reference
in the result and figure footer.  It is never inserted as an evaluation point
and is never used to alter a measured M50 value.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_hp_history as HH
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


VERSION = "sfm_b1_r2_raw_m50_v1"
M_PER_GAMMA = 50
T = int(SP.T)
H = int(SP.H)
NFE = 8
TEMPERATURE = 1.0
DEFAULT_EP0 = 260_000
DEFAULT_NOISE_SEED = 2_026_072_3

# Historical reference only.  These values were measured with M=100/gamma on
# scenarios 250000:250099 and are not a target for the disjoint M50 run.
ARCHIVED_M100_REFERENCE = {
    "role": "separate_historical_reference_not_a_curve_point",
    "scene_profile": "double_density_velocity_ood",
    "ep0": 250_000,
    "M_per_gamma": 100,
    "checkpoint_sha256": (
        "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
    ),
    "source_commit": "ca7f0d718f8d70cf74833b1c75157caf7f1b13f2",
    "SR": 0.7000000000,
    "CR": 0.3000000000,
    "successful_clearance": 0.1310315136398588,
    "successful_time_to_goal": 8.692857142857143,
    "note": (
        "Archived raw temp=1/NFE=8 M100 result. It is reported separately and "
        "must never be substituted for an independently measured M50 value."
    ),
}


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _checkpoint_specs(checkpoints: list[str], labels: list[str]) -> list[dict]:
    if len(checkpoints) != len(labels) or not checkpoints:
        raise ValueError("--checkpoints and --labels must have the same nonzero length")
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    specs = []
    for checkpoint, label in zip(checkpoints, labels):
        match = re.fullmatch(r"r([0-9]+)", str(label))
        if match is None:
            raise ValueError(f"checkpoint label {label!r} must have form r0, r1, ...")
        path = os.path.abspath(checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        specs.append(dict(label=str(label), round=int(match.group(1)), checkpoint=path))
    rounds = [spec["round"] for spec in specs]
    if rounds != sorted(rounds) or len(rounds) != len(set(rounds)):
        raise ValueError("checkpoint labels must be unique and increasing")
    return specs


def _assert_disjoint_from_archive(ep0: int) -> None:
    current = set(range(int(ep0), int(ep0) + M_PER_GAMMA))
    archived = set(range(
        int(ARCHIVED_M100_REFERENCE["ep0"]),
        int(ARCHIVED_M100_REFERENCE["ep0"])
        + int(ARCHIVED_M100_REFERENCE["M_per_gamma"]),
    ))
    if current & archived:
        raise ValueError(
            "the M50 qualification bank must remain disjoint from the archived M100 bank"
        )


def _noise_bank(*, ep0: int, d: int, seed: int) -> tuple[np.ndarray, dict]:
    contract = {
        "version": VERSION,
        "ep0": int(ep0),
        "M_per_gamma": M_PER_GAMMA,
        "gammas": list(map(float, SP.GAMMAS)),
        "T": T,
        "d": int(d),
        "seed": int(seed),
        "temperature": TEMPERATURE,
        "NFE": NFE,
    }
    generator = np.random.default_rng(int(seed))
    values = generator.standard_normal(
        (len(SP.GAMMAS), M_PER_GAMMA, T, int(d)), dtype=np.float32,
    )
    metadata = {
        **contract,
        "dtype": "float32",
        "shape": list(values.shape),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "CRN": (
            "same (gamma,scenario,step) latent across checkpoints; paired scenario "
            "IDs across gamma, independent latent slices across gamma"
        ),
    }
    return values, metadata


@dataclass
class _Episode:
    gamma_index: int
    rollout_index: int
    episode: int
    gamma: float
    humans: list
    state: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    history: HH.HpHistory = field(default_factory=HH.HpHistory)
    controls: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(
        default_factory=lambda: [np.zeros(4, np.float32)]
    )
    context_states: list[np.ndarray] = field(default_factory=list)
    planned_controls: list[np.ndarray] = field(default_factory=list)
    ped_xy: list[np.ndarray] = field(default_factory=list)
    ped_vel: list[np.ndarray] = field(default_factory=list)
    status: str | None = None
    minimum_clearance: float = float("inf")


def _clearance(state: np.ndarray, ped_xy: np.ndarray) -> float:
    if not len(ped_xy):
        return float("inf")
    return float(
        np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED
    )


def _terminal_check(episode: _Episode, ped_xy: np.ndarray) -> bool:
    clearance = _clearance(episode.state, ped_xy)
    episode.minimum_clearance = min(episode.minimum_clearance, clearance)
    if clearance < 0.0:
        episode.status = "collision"
    elif float(np.linalg.norm(episode.state[:2] - SS.GOAL)) < 0.5:
        episode.status = "success"
    return episode.status is not None


@torch.no_grad()
def run_batched_raw(
    policy,
    *,
    scene_profile: str,
    ep0: int,
    noise: np.ndarray,
    device: str,
) -> list[dict]:
    """Evaluate all 7xM cells with one flow batch per closed-loop tick."""
    environment = SS.scene_profile(scene_profile)
    expected = (len(SP.GAMMAS), M_PER_GAMMA, T, int(policy.d))
    if tuple(noise.shape) != expected or noise.dtype != np.float32:
        raise ValueError(f"noise bank {noise.shape}/{noise.dtype} != {expected}/float32")
    episodes = [
        _Episode(
            gamma_index=gamma_index,
            rollout_index=rollout_index,
            episode=int(ep0) + rollout_index,
            gamma=float(gamma),
            humans=SS.make_humans(
                int(ep0) + rollout_index,
                0,
                environment["n_ped"],
                tuple(environment["ped_speed_range"]),
            ),
        )
        for gamma_index, gamma in enumerate(SP.GAMMAS)
        for rollout_index in range(M_PER_GAMMA)
    ]

    for step in range(T):
        active, hp10, lows, histories, latents = [], [], [], [], []
        for episode in episodes:
            if episode.status is not None:
                continue
            ped_xy, ped_vel = SS.collect_humans(episode.humans)
            if _terminal_check(episode, ped_xy):
                continue
            obstacles = np.concatenate([
                ped_xy,
                np.full((len(ped_xy), 1), SS.R_PED, np.float32),
            ], axis=1)
            raw_grid = torch.as_tensor(
                GF.axis_grid(
                    episode.state[:2],
                    obstacles,
                    0.0,
                    R=SS.R_SENSE,
                    sensing=SS.R_SENSE,
                )
            )
            active.append((episode, ped_xy.copy(), ped_vel.copy()))
            hp10.append(episode.history.append(raw_grid))
            lows.append(torch.as_tensor(
                GF.low5(episode.state, SS.GOAL, episode.gamma)
            ))
            histories.append(torch.as_tensor(GF.hist_pad(
                np.asarray(episode.controls[-16:])
                if episode.controls else np.zeros((0, 2)),
                16,
            )))
            latents.append(noise[
                episode.gamma_index,
                episode.rollout_index,
                step,
            ])
        if not active:
            break

        hp10_tensor = torch.stack(hp10).to(device)
        low_tensor = torch.stack(lows).to(device)
        history_tensor = torch.stack(histories).to(device)
        context = policy.ctx_from(hp10_tensor, low_tensor, history_tensor)
        windows = BE.integrate_latents(
            policy,
            torch.as_tensor(np.asarray(latents), device=device),
            context,
            nfe=NFE,
        ).reshape(len(active), H, 2)
        windows = windows.detach().cpu().numpy().astype(np.float32)

        for (episode, ped_xy, ped_vel), window in zip(active, windows):
            if tuple(window.shape) != (H, 2):
                raise RuntimeError(f"generated plan {window.shape} != {(H, 2)}")
            episode.context_states.append(episode.state.copy())
            episode.planned_controls.append(window.copy())
            episode.ped_xy.append(ped_xy)
            episode.ped_vel.append(ped_vel)
            action = window[0].copy()
            episode.controls.append(action)
            episode.state = BE._step(episode.state, action)
            episode.states.append(episode.state.copy())
            SS.advance_humans(episode.humans, episode.state)

    rows = []
    for episode in episodes:
        if episode.status is None:
            ped_xy, _ = SS.collect_humans(episode.humans)
            if not _terminal_check(episode, ped_xy):
                episode.status = "timeout"
        success = episode.status == "success"
        rows.append({
            "episode": episode.episode,
            "gamma": episode.gamma,
            "status": episode.status,
            "success": success,
            "collision": episode.status == "collision",
            "timeout": episode.status == "timeout",
            "steps": len(episode.controls),
            "time_to_goal": (
                len(episode.controls) * SS.DT if success else None
            ),
            "min_clearance": float(episode.minimum_clearance),
            "successful_clearance": (
                float(episode.minimum_clearance) if success else None
            ),
            "states": np.asarray(episode.states, np.float32),
            "context_states": np.asarray(episode.context_states, np.float32),
            "planned_controls": np.asarray(episode.planned_controls, np.float32),
            "ped_xy": np.asarray(episode.ped_xy, np.float32),
            "ped_vel": np.asarray(episode.ped_vel, np.float32),
        })
    return rows


def _verify_episode(row: dict) -> dict:
    n_steps = int(row["steps"])
    states = np.asarray(row["states"], np.float32)
    context_states = np.asarray(row["context_states"], np.float32)
    planned_controls = np.asarray(row["planned_controls"], np.float32)
    ped_xy = np.asarray(row["ped_xy"], np.float32)
    ped_vel = np.asarray(row["ped_vel"], np.float32)
    expected_lengths = (
        len(states) == n_steps + 1
        and len(context_states) == n_steps
        and len(planned_controls) == n_steps
        and len(ped_xy) == n_steps
        and len(ped_vel) == n_steps
    )
    if not expected_lengths or (
        n_steps and tuple(planned_controls.shape[1:]) != (H, 2)
    ):
        return {"v_safe": False, "verifier_errors": 1, "certified_windows": 0}

    physical_safe = (
        not bool(row["collision"])
        and SM.taskspace_ok(states[:, :2])
        and n_steps > 0
    )
    if not physical_safe:
        return {"v_safe": False, "verifier_errors": 0, "certified_windows": 0}

    certified_windows = 0
    for state, controls, current_xy, current_vel in zip(
        context_states, planned_controls, ped_xy, ped_vel
    ):
        result = SM.verify_query(
            state, controls, current_xy, current_vel, float(row["gamma"])
        )
        if not result.get("resolved", False):
            return {
                "v_safe": False,
                "verifier_errors": 1,
                "certified_windows": certified_windows,
            }
        if not result.get("full_h", False) or int(result.get("terminal_step", -1)) != H:
            return {
                "v_safe": False,
                "verifier_errors": 1,
                "certified_windows": certified_windows,
            }
        certified_windows += 1
        if not bool(result["y"]):
            return {
                "v_safe": False,
                "verifier_errors": 0,
                "certified_windows": certified_windows,
            }
    return {
        "v_safe": True,
        "verifier_errors": 0,
        "certified_windows": certified_windows,
    }


def _attach_validity(rows: list[dict], executor) -> list[dict]:
    futures = [executor.submit(_verify_episode, row) for row in rows]
    compact = []
    omitted = {
        "states", "context_states", "planned_controls", "ped_xy", "ped_vel",
    }
    for row, future in zip(rows, futures):
        value = {key: item for key, item in row.items() if key not in omitted}
        value.update(future.result())
        compact.append(value)
    return compact


def _cluster_bootstrap_interval(
    rows: list[dict],
    key: str,
    *,
    seed: int,
    draws: int = 2_000,
) -> list[float | None]:
    episode_ids = sorted({int(row["episode"]) for row in rows})
    sums, counts = [], []
    for episode in episode_ids:
        values = [row.get(key) for row in rows if int(row["episode"]) == episode]
        finite = [
            float(value) for value in values
            if value is not None and math.isfinite(float(value))
        ]
        sums.append(sum(finite))
        counts.append(len(finite))
    if not episode_ids or not sum(counts):
        return [None, None]
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(episode_ids), size=(int(draws), len(episode_ids))
    )
    numerator = np.asarray(sums, float)[indices].sum(axis=1)
    denominator = np.asarray(counts, float)[indices].sum(axis=1)
    samples = numerator[denominator > 0] / denominator[denominator > 0]
    if not len(samples):
        return [None, None]
    return list(map(float, np.quantile(samples, [.025, .975])))


def _summarize_one(rows: list[dict], seed: int) -> dict:
    n = len(rows)
    if n < 1:
        raise ValueError("cannot summarize an empty cell")
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    valid = sum(bool(row["v_safe"]) for row in rows)
    if successes + collisions + timeouts != n:
        raise RuntimeError("success, collision, and timeout must partition a cell")
    return {
        "n": n,
        "SR": successes / n,
        "SR_wilson95": BE.wilson(successes, n),
        "CR": collisions / n,
        "CR_wilson95": BE.wilson(collisions, n),
        "timeout": timeouts / n,
        "timeout_wilson95": BE.wilson(timeouts, n),
        "V_safe": valid / n,
        "V_safe_wilson95": BE.wilson(valid, n),
        "successful_clearance": BE.bootstrap_mean(
            [row["successful_clearance"] for row in rows], seed=seed
        ),
        "successful_time_to_goal": BE.bootstrap_mean(
            [row["time_to_goal"] for row in rows], seed=seed + 1
        ),
        "verifier_errors": sum(int(row["verifier_errors"]) for row in rows),
        "certified_windows": sum(int(row["certified_windows"]) for row in rows),
    }


def summarize(rows: list[dict], *, seed: int) -> dict:
    per_gamma = {
        str(gamma): _summarize_one(
            [row for row in rows if float(row["gamma"]) == float(gamma)],
            seed + index * 10,
        )
        for index, gamma in enumerate(SP.GAMMAS)
    }
    pooled = _summarize_one(rows, seed + 100)
    for metric, key in (
        ("SR", "success"),
        ("CR", "collision"),
        ("timeout", "timeout"),
        ("V_safe", "v_safe"),
    ):
        pooled[f"{metric}_cluster_bootstrap95"] = _cluster_bootstrap_interval(
            rows, key, seed=seed + 200 + len(metric)
        )
    pooled["successful_clearance"]["cluster_bootstrap95"] = (
        _cluster_bootstrap_interval(
            rows, "successful_clearance", seed=seed + 300
        )
    )
    pooled["successful_time_to_goal"]["cluster_bootstrap95"] = (
        _cluster_bootstrap_interval(
            rows, "time_to_goal", seed=seed + 301
        )
    )
    pooled["ci_method"] = (
        "scenario-cluster bootstrap across seven paired gamma rows"
    )
    return {"pooled": pooled, "per_gamma": per_gamma}


def _assert_zero_verifier_errors(summary: dict) -> None:
    cells = [summary["pooled"], *summary["per_gamma"].values()]
    if any(int(cell["verifier_errors"]) != 0 for cell in cells):
        raise RuntimeError("evaluation contains verifier errors")


def _cell_key(
    *,
    checkpoint_sha256: str,
    scene_profile: str,
    ep0: int,
    noise_meta: dict,
) -> str:
    return _sha256_json({
        "version": VERSION,
        "evaluator_sha256": _sha256_file(__file__),
        "checkpoint_sha256": checkpoint_sha256,
        "scene_profile": scene_profile,
        "ep0": int(ep0),
        "M_per_gamma": M_PER_GAMMA,
        "noise_bank": noise_meta,
        "temperature": TEMPERATURE,
        "NFE": NFE,
        "T": T,
        "H": H,
        "verifier": SM.verifier_manifest(),
    })


def _evaluate_checkpoint(
    checkpoint: str,
    *,
    scene_profile: str,
    ep0: int,
    noise: np.ndarray,
    noise_meta: dict,
    device: str,
    cache_dir: str,
    executor,
) -> dict:
    checkpoint_sha = _sha256_file(checkpoint)
    key = _cell_key(
        checkpoint_sha256=checkpoint_sha,
        scene_profile=scene_profile,
        ep0=ep0,
        noise_meta=noise_meta,
    )
    cache_path = os.path.join(
        cache_dir, f"cell_{checkpoint_sha[:12]}_{key[:12]}.json"
    )
    if os.path.isfile(cache_path):
        with open(cache_path) as stream:
            payload = json.load(stream)
        if (
            payload.get("status") != "SFM_B1_R2_RAW_CELL_COMPLETE"
            or payload.get("cell_key") != key
        ):
            raise RuntimeError(f"stale evaluation cache: {cache_path}")
        _assert_zero_verifier_errors(payload["summary"])
        return payload

    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    if int(policy.d) != int(noise.shape[-1]):
        raise ValueError("checkpoint latent dimension does not match the fixed noise bank")
    rows = run_batched_raw(
        policy,
        scene_profile=scene_profile,
        ep0=ep0,
        noise=noise,
        device=device,
    )
    del policy
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    compact = _attach_validity(rows, executor)
    summary = summarize(
        compact,
        seed=int(ep0) + int(checkpoint_sha[:8], 16) % 100_000,
    )
    _assert_zero_verifier_errors(summary)
    payload = {
        "status": "SFM_B1_R2_RAW_CELL_COMPLETE",
        "cell_key": key,
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "scene_profile": scene_profile,
        "ep0": int(ep0),
        "M_per_gamma": M_PER_GAMMA,
        "summary": summary,
        "rows": compact,
        "metric_semantics": {
            "policy": (
                "canonical unguided raw flow, temperature=1, NFE=8, one "
                "generated H=10 window per context, execute first action"
            ),
            "V_safe": (
                "episode is physically collision/task-space safe and every "
                "generated plan at every executed context passes the exact "
                "full-H=10 moving-pedestrian verifier"
            ),
            "clearance": (
                "mean of each successful trajectory's minimum pedestrian "
                "clearance; failures are excluded"
            ),
            "time": "successful trajectories only",
            "outcome_partition": "SR + CR + timeout = 1",
        },
    }
    _write_json(cache_path, payload)
    return payload


def _metric_value(cell: dict, metric: str) -> float:
    if metric in ("CR", "V_safe"):
        return float(cell[metric])
    key = (
        "successful_clearance"
        if metric == "clearance"
        else "successful_time_to_goal"
    )
    value = cell[key]["mean"]
    return float("nan") if value is None else float(value)


def _pooled_interval(cell: dict, metric: str) -> list[float]:
    if metric in ("CR", "V_safe"):
        value = cell[f"{metric}_cluster_bootstrap95"]
    else:
        key = (
            "successful_clearance"
            if metric == "clearance"
            else "successful_time_to_goal"
        )
        value = cell[key]["cluster_bootstrap95"]
    return [
        float("nan") if item is None else float(item)
        for item in value
    ]


def render(records: list[dict], output_dir: str) -> list[str]:
    """Render the four requested metrics in the B1 paper-curve style."""
    colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.92, len(SP.GAMMAS))
    )
    specs = (
        ("CR", "Collision rate"),
        ("V_safe", r"$V_{\mathrm{safe}}$"),
        ("clearance", "Successful min. clearance [m]"),
        ("time", "Successful time-to-goal [s]"),
    )
    rounds = [int(record["round"]) for record in records]
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9))
    for axis, (metric, title) in zip(axes.flat, specs):
        for gamma, color in zip(SP.GAMMAS, colors):
            cells = [
                record["cell"]["summary"]["per_gamma"][str(gamma)]
                for record in records
            ]
            axis.plot(
                rounds,
                [_metric_value(cell, metric) for cell in cells],
                color=color,
                lw=1.5,
                marker="o",
                ms=5,
                alpha=0.72,
                label=rf"$\gamma={gamma:g}$",
            )
        pooled = [
            record["cell"]["summary"]["pooled"] for record in records
        ]
        values = [_metric_value(cell, metric) for cell in pooled]
        intervals = [_pooled_interval(cell, metric) for cell in pooled]
        axis.plot(
            rounds,
            values,
            color="black",
            lw=3.0,
            marker="o",
            ms=6,
            label=r"pooled ($7\gamma$)",
            zorder=4,
        )
        axis.fill_between(
            rounds,
            [value[0] for value in intervals],
            [value[1] for value in intervals],
            color="black",
            alpha=0.11,
            lw=0,
            zorder=1,
        )
        axis.set_title(title, pad=8)
        axis.set_xlabel("expansion round")
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
        axis.set_xlim(min(rounds) - 0.15, max(rounds) + 0.15)
        if metric in ("CR", "V_safe"):
            axis.set_ylim(-0.03, 1.03)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    reference = ARCHIVED_M100_REFERENCE
    figure.text(
        0.5,
        0.012,
        (
            "Separate archived M100 reference (not plotted): "
            f"SR {reference['SR']:.3f}, CR {reference['CR']:.3f}, "
            f"successful clearance {reference['successful_clearance']:.3f} m, "
            f"successful time {reference['successful_time_to_goal']:.3f} s."
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.35",
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.91))
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = os.path.join(output_dir, f"raw_m50_r0_r2_curves.{suffix}")
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def run(args) -> dict:
    specs = _checkpoint_specs(args.checkpoints, args.labels)
    _assert_disjoint_from_archive(args.ep0)
    output_dir = os.path.abspath(args.output_dir)
    cache_dir = os.path.abspath(args.cache_dir or os.path.join(output_dir, "cache"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    probe, _ = GPS.load_sfm_policy(specs[0]["checkpoint"], device="cpu")
    noise, noise_meta = _noise_bank(
        ep0=args.ep0, d=int(probe.d), seed=args.noise_seed
    )
    del probe
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=context
    ) as executor:
        for spec in specs:
            cell = _evaluate_checkpoint(
                spec["checkpoint"],
                scene_profile=args.scene_profile,
                ep0=args.ep0,
                noise=noise,
                noise_meta=noise_meta,
                device=args.device,
                cache_dir=cache_dir,
                executor=executor,
            )
            records.append({
                "label": spec["label"],
                "round": spec["round"],
                "cell": cell,
            })

    outputs = render(records, output_dir)
    result = {
        "status": "SFM_B1_R2_RAW_M50_COMPLETE",
        "version": VERSION,
        "scene_profile": args.scene_profile,
        "environment": SS.scene_profile(args.scene_profile),
        "bank": {
            "ep0": int(args.ep0),
            "M_per_gamma": M_PER_GAMMA,
            "scenario_ids": list(range(
                int(args.ep0), int(args.ep0) + M_PER_GAMMA
            )),
            "same_scenario_ids_for_every_gamma": True,
            "disjoint_from_archived_M100": True,
        },
        "noise_bank": noise_meta,
        "records": records,
        "archived_M100_reference": ARCHIVED_M100_REFERENCE,
        "reference_policy": (
            "The archived M100 result is provenance only. No measured M50 "
            "value is replaced, shifted, selected, or calibrated against it."
        ),
        "outputs": outputs,
    }
    result_path = os.path.join(output_dir, "raw_m50_r0_r2_metrics.json")
    _write_json(result_path, result)
    result["metrics_json"] = result_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--scene-profile",
        default="double_density_velocity_ood",
        choices=SS.SCIENTIFIC_EVAL_PROFILES,
    )
    parser.add_argument("--ep0", type=int, default=DEFAULT_EP0)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(result["metrics_json"])
    for path in result["outputs"]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
