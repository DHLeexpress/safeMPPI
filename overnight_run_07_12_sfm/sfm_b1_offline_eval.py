"""Raw SFM evaluation with terminal-truncated executed-window Validity.

Every checkpoint uses one fixed M=50/scenario/gamma seed and latent bank.  The
controller is the unguided raw flow at temperature one: it samples one H=10
plan per context and executes only its first action.  Acquisition, verifier
selection, fallback, guidance, and temperature search are absent.

For an executed trajectory with ``N_tau`` controls, Validity is the mean of
the ``N_tau`` exact GREEN-verifier indicators.  The window at start ``t`` uses
the actions actually executed from ``t`` onward and has
``H_t=min(10, N_tau-t)``.  Terminal tails are neither dropped nor padded.
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


VERSION = "sfm_b1_offline_executed_window_v1"
M_PER_GAMMA = 50
T = int(SP.T)
H = int(SP.H)
NFE = 8
TEMPERATURE = 1.0
DEFAULT_EP0 = 260_000
DEFAULT_NOISE_SEED = 2_026_072_3
Z95 = 1.959963984540054
PLOT_SPECS = (
    ("CR", "Collision rate", (-0.03, 1.03)),
    ("Validity", "Validity", (-0.03, 1.03)),
    ("clearance", "Min. clearance [m]", None),
    ("time", "Time-to-goal [s]", None),
)


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
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
        raise ValueError("--checkpoints and --labels need the same nonzero length")
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
            "same (gamma,scenario,step) latent across checkpoints; paired "
            "scenario IDs across gamma"
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
    """Evaluate all 7xM cells and retain the controls actually executed."""
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
            raw_grid = torch.as_tensor(GF.axis_grid(
                episode.state[:2],
                obstacles,
                0.0,
                R=SS.R_SENSE,
                sensing=SS.R_SENSE,
            ))
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
            action = window[0].copy()
            episode.ped_xy.append(ped_xy)
            episode.ped_vel.append(ped_vel)
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
            "time_to_goal": len(episode.controls) * SS.DT if success else None,
            "min_clearance": float(episode.minimum_clearance),
            "successful_clearance": (
                float(episode.minimum_clearance) if success else None
            ),
            "states": np.asarray(episode.states, np.float32),
            "controls": np.asarray(episode.controls, np.float32),
            "ped_xy": np.asarray(episode.ped_xy, np.float32),
            "ped_vel": np.asarray(episode.ped_vel, np.float32),
        })
    return rows


def _verify_executed_episode(row: dict) -> dict:
    """Return the fractional GREEN validity of all executed window starts."""
    n_steps = int(row["steps"])
    states = np.asarray(row["states"], np.float32)
    controls = np.asarray(row["controls"], np.float32)
    ped_xy = np.asarray(row["ped_xy"], np.float32)
    ped_vel = np.asarray(row["ped_vel"], np.float32)
    expected = (
        len(states) == n_steps + 1
        and len(controls) == n_steps
        and len(ped_xy) == n_steps
        and len(ped_vel) == n_steps
    )
    if not expected or (n_steps and tuple(controls.shape[1:]) != (2,)):
        return {
            "validity": 0.0,
            "valid_windows": 0,
            "evaluated_windows": 0,
            "verifier_errors": 1,
        }
    if n_steps == 0:
        return {
            "validity": 0.0,
            "valid_windows": 0,
            "evaluated_windows": 0,
            "verifier_errors": 0,
        }

    valid_windows = 0
    for start in range(n_steps):
        stop = min(start + H, n_steps)
        result = SM.verify_executed_window(
            states[start],
            controls[start:stop],
            ped_xy[start],
            ped_vel[start],
            float(row["gamma"]),
        )
        if not result.get("resolved", False):
            return {
                "validity": valid_windows / n_steps,
                "valid_windows": valid_windows,
                "evaluated_windows": start,
                "verifier_errors": 1,
            }
        if int(result.get("window_horizon", -1)) != stop - start:
            return {
                "validity": valid_windows / n_steps,
                "valid_windows": valid_windows,
                "evaluated_windows": start,
                "verifier_errors": 1,
            }
        valid_windows += int(bool(result["y"]))
    return {
        "validity": valid_windows / n_steps,
        "valid_windows": valid_windows,
        "evaluated_windows": n_steps,
        "verifier_errors": 0,
    }


def _attach_validity(rows: list[dict], executor) -> list[dict]:
    futures = [executor.submit(_verify_executed_episode, row) for row in rows]
    compact = []
    omitted = {"states", "controls", "ped_xy", "ped_vel"}
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
        values = [
            row.get(key) for row in rows if int(row["episode"]) == episode
        ]
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
    if successes + collisions + timeouts != n:
        raise RuntimeError("success, collision, and timeout must partition a cell")
    validity = BE.bootstrap_mean(
        [row["validity"] for row in rows], seed=seed + 2
    )
    valid_windows = sum(int(row["valid_windows"]) for row in rows)
    evaluated_windows = sum(int(row["evaluated_windows"]) for row in rows)
    validity.update(
        valid_windows=valid_windows,
        evaluated_windows=evaluated_windows,
        window_weighted_fraction=(
            valid_windows / evaluated_windows if evaluated_windows else 0.0
        ),
    )
    return {
        "n": n,
        "SR": successes / n,
        "SR_wilson95": BE.wilson(successes, n),
        "CR": collisions / n,
        "CR_wilson95": BE.wilson(collisions, n),
        "timeout": timeouts / n,
        "timeout_wilson95": BE.wilson(timeouts, n),
        "Validity": validity,
        "successful_clearance": BE.bootstrap_mean(
            [row["successful_clearance"] for row in rows], seed=seed
        ),
        "successful_time_to_goal": BE.bootstrap_mean(
            [row["time_to_goal"] for row in rows], seed=seed + 1
        ),
        "verifier_errors": sum(int(row["verifier_errors"]) for row in rows),
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
    ):
        pooled[f"{metric}_cluster_bootstrap95"] = _cluster_bootstrap_interval(
            rows, key, seed=seed + 200 + len(metric)
        )
    pooled["Validity"]["cluster_bootstrap95"] = _cluster_bootstrap_interval(
        rows, "validity", seed=seed + 208
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
        "validity": "executed sliding windows H_t=min(10,N_tau-t)",
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
        cache_dir, f"offline_cell_{checkpoint_sha[:12]}_{key[:12]}.json"
    )
    if os.path.isfile(cache_path):
        with open(cache_path) as stream:
            payload = json.load(stream)
        if (
            payload.get("status") != "SFM_B1_OFFLINE_RAW_CELL_COMPLETE"
            or payload.get("cell_key") != key
        ):
            raise RuntimeError(f"stale evaluation cache: {cache_path}")
        _assert_zero_verifier_errors(payload["summary"])
        return payload

    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    if int(policy.d) != int(noise.shape[-1]):
        raise ValueError("checkpoint latent dimension does not match the noise bank")
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
        "status": "SFM_B1_OFFLINE_RAW_CELL_COMPLETE",
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
            "Validity": (
                "mean per-trajectory fraction of all executed window starts "
                "whose terminal-truncated actual-action window is task-space "
                "valid, collision-free, and exact GREEN-certified; "
                "H_t=min(10,N_tau-t)"
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
    if metric == "CR":
        return float(cell[metric])
    key = {
        "Validity": "Validity",
        "clearance": "successful_clearance",
        "time": "successful_time_to_goal",
    }[metric]
    value = cell[key]["mean"]
    return float("nan") if value is None else float(value)


def _metric_interval(cell: dict, metric: str, *, pooled: bool) -> list[float]:
    if metric == "CR":
        value = (
            cell["CR_cluster_bootstrap95"]
            if pooled else cell["CR_wilson95"]
        )
    else:
        key = {
            "Validity": "Validity",
            "clearance": "successful_clearance",
            "time": "successful_time_to_goal",
        }[metric]
        value = (
            cell[key]["cluster_bootstrap95"]
            if pooled else cell[key]["interval95"]
        )
    return [
        float("nan") if item is None else float(item)
        for item in value
    ]


def render(records: list[dict], output_dir: str) -> list[str]:
    """Render the four metrics in the ball-evaluator paper style."""
    colors = {
        gamma: plt.get_cmap("plasma")(
            0.08 + 0.84 * index / max(len(SP.GAMMAS) - 1, 1)
        )
        for index, gamma in enumerate(SP.GAMMAS)
    }
    rounds = [int(record["round"]) for record in records]
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14.6, 10.8), squeeze=False)
    for axis, (metric, title, ylim) in zip(axes.flat, PLOT_SPECS):
        for gamma in SP.GAMMAS:
            cells = [
                record["cell"]["summary"]["per_gamma"][str(gamma)]
                for record in records
            ]
            values = [_metric_value(cell, metric) for cell in cells]
            intervals = [
                _metric_interval(cell, metric, pooled=False) for cell in cells
            ]
            axis.plot(
                rounds, values, color=colors[gamma], lw=1.35, alpha=0.75
            )
            axis.fill_between(
                rounds,
                [value[0] for value in intervals],
                [value[1] for value in intervals],
                color=colors[gamma],
                alpha=0.18,
                linewidth=0,
            )
        pooled = [
            record["cell"]["summary"]["pooled"] for record in records
        ]
        values = [_metric_value(cell, metric) for cell in pooled]
        intervals = [
            _metric_interval(cell, metric, pooled=True) for cell in pooled
        ]
        axis.plot(rounds, values, color="black", lw=3.0)
        axis.fill_between(
            rounds,
            [value[0] for value in intervals],
            [value[1] for value in intervals],
            color="black",
            alpha=0.14,
            linewidth=0,
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_xlabel("Expansion round")

    handles = [
        plt.Line2D(
            [0], [0], color=colors[gamma], lw=2.2,
            label=rf"$\gamma={gamma:g}$",
        )
        for gamma in SP.GAMMAS
    ]
    handles.append(
        plt.Line2D([0], [0], color="black", lw=3.0, label="pooled")
    )
    figure.legend(
        handles=handles, ncol=5, loc="upper center", frameon=False
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = os.path.join(output_dir, f"raw_m50_offline_curves.{suffix}")
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    manifest = os.path.join(output_dir, "raw_m50_offline_curves.figure.json")
    _write_json(manifest, {
        "status": "SFM_B1_OFFLINE_FIGURE_COMPLETE",
        "rounds": rounds,
        "gammas": list(map(float, SP.GAMMAS)),
        "claim": (
            "fixed raw temperature-1 rollouts; Validity is the trajectory-mean "
            "fraction over every executed window start; terminal horizons use "
            "H_t=min(10,N_tau-t); every indicator requires task-space bounds, "
            "time-indexed collision avoidance, and the exact GREEN certificate"
        ),
        "confidence_bands": (
            "per-gamma Wilson for CR and trajectory bootstrap for continuous "
            "metrics; pooled scenario-cluster bootstrap"
        ),
        "style_source": "safeMPPI_demo_3d/scripts/evaluate_ball_expansion.py",
    })
    outputs.append(manifest)
    return outputs


def run(args) -> dict:
    specs = _checkpoint_specs(args.checkpoints, args.labels)
    output_dir = os.path.abspath(args.output_dir)
    cache_dir = os.path.abspath(
        args.cache_dir or os.path.join(output_dir, "cache")
    )
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
        "status": "SFM_B1_OFFLINE_RAW_M50_COMPLETE",
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
        },
        "noise_bank": noise_meta,
        "records": records,
        "outputs": outputs,
    }
    result_path = os.path.join(output_dir, "raw_m50_offline_metrics.json")
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
