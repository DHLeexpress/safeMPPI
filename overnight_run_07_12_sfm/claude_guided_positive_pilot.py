"""Guided-positive early-collection pilot (fastlab, isolated from the study).

Motivation.  Guided-repair rescues concentrate at early episode steps, where
the OOD SFM crowd is fast and the learned policy is weakest; late-episode
executed positives are easy and nearly straight.  The canonical loop only
invokes the locked-Kazuki guidance when a repair trigger fires, so the
exact-verifier-certified early-step dodges that the guidance *can* produce are
never observed, let alone trained on.

This pilot changes exactly one thing.  During gathering it asks
``sfm_b1_kazuki_repair_audit.collect`` to run a guided generator at every live
context with ``step < --guided-until-step``, independently of any trigger, and
to keep the windows the exact verifier certifies positive as a new
collection-only population ``G+`` (``guided_positive_round.pt``).  Execution,
selection, repair triggering, the D0 neutral store, and the GP are untouched:
``G+`` rows are never executed and never enter the GP.

Two generators are available.  ``same_latent`` is the collector's existing
weak repair operator (kept only as an ablation: it certifies ~1% of the time
in the early band, so it harvests almost nothing there).  ``kazuki_full``, the
default, runs the *complete* locked external Kazuki controller at the context
-- 200 generated samples, 10 elites, 200 MPPI perturbations per elite, locked
goal .5 / safe .3 -- and verifies its top ``--guided-topk`` refined modes.

``--crunch-full-pool-until-step`` is a separate, independently launchable arm
with no external teacher at all: below that step every one of the K=16
proposals is verified instead of only the B=4 the GP acquires, which attacks
the finite-B NVP budget artifact directly.  It changes the admissible set and
therefore the executed trajectory, so it is a different arm, not a control.

The pilot then runs ``--update-passes`` whole-buffer Adam steps over ``G+``
(and, by default, a separate equally sized pass block over the ordinary
executed ``D+``), each pass being one optimizer step over the whole population
with the canonical ``gamma -> (round, scenario) -> context -> window``
hierarchy mass.  ``D0`` is collected and audited as usual but is deliberately
*not* trained on, so the pilot isolates the ``G+`` signal.

Nothing here is a confirmation.  It is a single-round, unreplicated pilot whose
only job is to measure the early-step ``G+`` yield and whether a massive update
on it moves the paired offline metrics at all.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as MR
import sfm_b1_neutral_teacher_sanity as NS
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_protocol as SP


STATUS = "CLAUDE_GUIDED_POSITIVE_PILOT_COMPLETE"
ROUND_STATUS = "CLAUDE_GUIDED_POSITIVE_PILOT_ROUND_COMPLETE"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENARIO_EP0 = MR.DEFAULT_SCENARIO_EP0
DEFAULT_EVAL_EP0 = MR.DEFAULT_EVAL_EP0
DEFAULT_GUIDED_UNTIL_STEP = 50
DEFAULT_UPDATE_PASSES = 4
DEFAULT_LR = MR.DEFAULT_LR
DEFAULT_BATCH = 128
STEP_BIN = 10


class _GuidedHolder:
    """Minimal shard-like holder so G+ reuses the canonical replay helpers."""

    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.windows = []


def _guided_positive_records(path):
    """Authenticate one ``guided_positive_round.pt`` and expose replay rows.

    Windows harvested at the same ``(scenario, gamma, step)`` share one context
    entry, so the hierarchy mass keeps equal weight per *context* rather than
    silently up-weighting steps where more than one guided candidate was
    certified.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != RA.GUIDED_POSITIVE_STATUS:
        raise ValueError("not an authenticated guided-positive payload")
    holder = _GuidedHolder(payload["round"])
    context_ids = {}
    records = []
    for expected, source in enumerate(payload["records"]):
        result = source.get("verifier_result", {})
        if (
            int(source["guided_id"]) != expected
            or source["population"] != "Gplus"
            or source["semantic_label"] != "guided_positive"
            or source["generator"] not in RA.GUIDED_GENERATORS
            or source["source"] != RA.GUIDED_SOURCES[source["generator"]]
            or int(source["verifier_y"]) != 1
            or not result.get("resolved")
            or int(result.get("y", -1)) != 1
            or not result.get("full_h")
            or int(result.get("terminal_step", -1)) != SP.H
            or not source["train_eligible"]
            or source["replay_default"]
            or source["gp_eligible"]
        ):
            raise RuntimeError("G+ semantics changed")
        key = (
            int(source["scenario_id"]),
            round(float(source["gamma"]), 8),
            int(source["step"]),
        )
        if key not in context_ids:
            context_ids[key] = len(holder.contexts)
            holder.contexts.append({
                "context_id": context_ids[key],
                "round": int(payload["round"]),
                "scenario_id": int(source["scenario_id"]),
                "gamma": float(source["gamma"]),
                "step": int(source["step"]),
                "state": np.asarray(source["state"], np.float32),
                "hp10": np.asarray(source["hp10"], np.float32),
                "low5": np.asarray(source["low5"], np.float32),
                "hist": np.asarray(source["hist"], np.float32),
                "ped_xy": np.asarray(source["ped_xy"], np.float32),
                "ped_vel": np.asarray(source["ped_vel"], np.float32),
            })
        row = {
            "window_id": expected,
            "query_id": expected,
            "context_id": context_ids[key],
            "controls": np.asarray(source["controls"], np.float32),
            "x0": (
                None if source["x0"] is None
                else np.asarray(source["x0"], np.float32)
            ),
            "y": 1,
            "semantic_label": "guided_positive",
            "generator": str(source["generator"]),
            "step": int(source["step"]),
            "gamma": float(source["gamma"]),
            "scenario_id": int(source["scenario_id"]),
            "reused_from_repair": bool(source["reused_from_repair"]),
            "selected_for_execution": bool(source["selected_for_execution"]),
        }
        if tuple(row["controls"].shape) != (SP.H, 2):
            raise RuntimeError("invalid G+ controls shape")
        if row["x0"] is not None and tuple(row["x0"].shape) != (2 * SP.H,):
            raise RuntimeError("invalid G+ x0 shape")
        holder.windows.append(row)
        records.append((holder, row))
    if len(records) != int(payload["summary"]["Gplus"]):
        raise RuntimeError("G+ payload count mismatch")
    return holder, records


def _step_bins(records, *, until_step, bin_size=STEP_BIN):
    """G+ yield by early-step bin, with per-gamma and context breakdowns."""
    bins = defaultdict(lambda: dict(
        windows=0, contexts=set(), reused_from_repair=0, per_gamma=Counter(),
    ))
    for _, row in records:
        index = int(row["step"]) // int(bin_size)
        start = index * int(bin_size)
        cell = bins[start]
        cell["windows"] += 1
        cell["contexts"].add(
            (int(row["scenario_id"]), round(float(row["gamma"]), 8),
             int(row["step"]))
        )
        cell["reused_from_repair"] += int(bool(row["reused_from_repair"]))
        cell["per_gamma"][str(row["gamma"])] += 1
    ordered = []
    for start in range(0, int(until_step), int(bin_size)):
        cell = bins.get(start)
        ordered.append({
            "step_start": int(start),
            "step_end": int(min(start + int(bin_size), int(until_step))),
            "windows": 0 if cell is None else int(cell["windows"]),
            "contexts": 0 if cell is None else len(cell["contexts"]),
            "reused_from_repair": (
                0 if cell is None else int(cell["reused_from_repair"])
            ),
            "per_gamma": (
                {} if cell is None
                else {key: int(value) for key, value in cell["per_gamma"].items()}
            ),
        })
    return ordered


def _population_summary(records):
    # D+ rows (ExecutedRoundShard windows) carry gamma on the context;
    # G+ rows carry it on the record itself.
    per_gamma = Counter(
        str(row["gamma"] if "gamma" in row else ctx["gamma"])
        for ctx, row in records
    )
    return dict(
        windows=len(records),
        per_gamma={key: int(value) for key, value in per_gamma.items()},
    )


def _read_json(path):
    with open(path) as stream:
        return json.load(stream)


def _run_paired_eval(args, *, checkpoints, labels, output_dir):
    command = [
        sys.executable,
        os.path.join(HERE, "sfm_b1_offline_eval.py"),
        "--checkpoints", *checkpoints,
        "--labels", *labels,
        "--scene-profile", args.scene_profile,
        "--ep0", str(int(args.eval_ep0)),
        "--noise-seed", str(int(args.eval_noise_seed)),
        "--m-per-gamma", str(int(args.eval_m)),
        "--device", str(args.device),
        "--workers", str(int(args.workers)),
        "--output-dir", output_dir,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=HERE, text=True, capture_output=True, check=False,
    )
    log_path = os.path.join(output_dir, "offline_eval.log")
    os.makedirs(output_dir, exist_ok=True)
    with open(log_path, "w") as stream:
        stream.write(completed.stdout)
        stream.write("\n--- stderr ---\n")
        stream.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"paired offline eval failed; see {log_path}")
    return dict(
        command=command,
        returncode=int(completed.returncode),
        log=log_path,
        seconds=float(time.perf_counter() - started),
        stdout_tail=completed.stdout.strip().splitlines()[-8:],
    )


def run(args):
    gammas = tuple(map(float, args.gammas))
    if (
        not gammas
        or len(set(gammas)) != len(gammas)
        or any(value not in tuple(map(float, SP.GAMMAS)) for value in gammas)
    ):
        raise ValueError(f"--gammas must be a distinct subset of {SP.GAMMAS}")
    if int(args.T) != 180:
        raise ValueError("the collector protocol is pinned to T=180")
    if not 0 <= int(args.guided_until_step) <= int(args.T):
        raise ValueError("--guided-until-step must lie in [0, T]")
    if not 0 <= int(args.crunch_full_pool_until_step) <= int(args.T):
        raise ValueError("--crunch-full-pool-until-step must lie in [0, T]")
    collect_gplus = int(args.guided_until_step) > 0
    include_dplus = str(args.include_dplus) == "yes"
    if not collect_gplus and int(args.crunch_full_pool_until_step) == 0:
        raise ValueError("enable --guided-until-step, the crunch arm, or both")
    if not collect_gplus and not include_dplus:
        raise ValueError(
            "the crunch-only arm has no G+ population, so D+ must be trained"
        )
    if int(args.update_passes) < 1:
        raise ValueError("--update-passes must be positive")
    if int(args.rounds) < 1:
        raise ValueError("--rounds must be positive")
    if int(args.batch) != DEFAULT_BATCH:
        raise ValueError("canonical microbatch size is 128")

    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root):
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    checkpoint = os.path.abspath(args.checkpoint)
    source_sha = FA._sha256_file(checkpoint)
    if source_sha != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source pretrained checkpoint SHA changed")
    source = FA._source()
    if args.require_clean_worktree and not source["tracked_worktree_clean"]:
        raise RuntimeError("pilot was asked for a clean frozen worktree")

    os.makedirs(output_root)
    checkpoints_dir = os.path.join(output_root, "checkpoints")
    rounds_dir = os.path.join(output_root, "rounds")
    os.makedirs(checkpoints_dir)
    os.makedirs(rounds_dir)

    policy, _ = GPS.load_sfm_policy(checkpoint, device=args.device)
    frozen = BS.configure_expansion_trainability(policy)
    encoder_sha_start = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [
            parameter for parameter in policy.parameters()
            if parameter.requires_grad
        ],
        lr=float(args.lr),
    )
    config = dict(
        status=STATUS,
        source=source,
        checkpoint=checkpoint,
        checkpoint_sha256=source_sha,
        scene_profile=str(args.scene_profile),
        selector=str(args.selector),
        gammas=list(gammas),
        scenario_ep0=int(args.scenario_ep0),
        rounds=int(args.rounds),
        T=int(args.T),
        guided_until_step=int(args.guided_until_step),
        guided_generator=str(args.guided_generator),
        guided_topk=int(args.guided_topk),
        crunch_full_pool_until_step=int(args.crunch_full_pool_until_step),
        update_passes=int(args.update_passes),
        lr=float(args.lr),
        batch=int(args.batch),
        include_dplus=bool(include_dplus),
        train_D0=False,
        ell=float(args.ell),
        gp_cap=int(args.gp_cap),
        ess_target=float(args.ess_target),
        device=str(args.device),
        workers=int(args.workers),
        sample_seed=int(args.sample_seed),
        audit_seed=int(args.audit_seed),
        train_seed=int(args.train_seed),
        eval_m=int(args.eval_m),
        eval_ep0=int(args.eval_ep0),
        eval_noise_seed=int(args.eval_noise_seed),
        frozen_parameters=frozen,
        encoder_sha256_start=encoder_sha_start,
    )
    FA._write_json(os.path.join(output_root, "pilot_config.json"), config)

    round0_path = os.path.join(checkpoints_dir, "round_00.pt")
    BX._save_checkpoint(policy, round0_path, {
        "study": STATUS,
        "round": 0,
        "phase": "pretrained",
        "source_checkpoint": checkpoint,
        "source_sha256": source_sha,
        "pilot_config": config,
    })

    history = []
    current_checkpoint = checkpoint
    previous_executed_path = None
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=spawn,
    ) as executor:
        for round_i in range(1, int(args.rounds) + 1):
            started = time.perf_counter()
            round_dir = os.path.join(rounds_dir, f"round_{round_i:02d}")
            os.makedirs(round_dir)
            gather_dir = os.path.join(round_dir, "gather")
            scenarios = tuple(range(
                int(args.scenario_ep0) + (round_i - 1) * 2,
                int(args.scenario_ep0) + round_i * 2,
            ))
            current_sha = FA._sha256_file(current_checkpoint)
            gather_started = time.perf_counter()
            RA.collect(
                current_checkpoint,
                scenarios=scenarios,
                gammas=gammas,
                scene_profile=str(args.scene_profile),
                selector=str(args.selector),
                device=args.device,
                verifier_workers=int(args.workers),
                sample_seed=int(args.sample_seed),
                audit_seed=int(args.audit_seed),
                ell=float(args.ell),
                neutral_continuation=True,
                guided_collect_until_step=int(args.guided_until_step),
                guided_generator=str(args.guided_generator),
                guided_topk=int(args.guided_topk),
                crunch_full_pool_until_step=int(
                    args.crunch_full_pool_until_step
                ),
                round_i=round_i,
                expected_checkpoint_sha256=current_sha,
                previous_executed_path=previous_executed_path,
                gp_cap=int(args.gp_cap),
                ess_target=float(args.ess_target),
                verifier_executor=executor,
                T=int(args.T),
                outdir=gather_dir,
            )
            gather_seconds = float(time.perf_counter() - gather_started)
            gather_marker = _read_json(
                os.path.join(gather_dir, "COMPLETE.json")
            )
            executed_path = os.path.join(gather_dir, "executed_round.pt")
            executed = OS.ExecutedRoundShard.load(executed_path)
            if executed.Dminus:
                raise RuntimeError("ordinary executed D unexpectedly has D-")
            positive_records = OS.positive_records(executed)
            _, neutral_records = MR._neutral_records(
                os.path.join(gather_dir, "neutral_round.pt")
            )
            guided_records = []
            if collect_gplus:
                _, guided_records = _guided_positive_records(
                    os.path.join(gather_dir, RA.GUIDED_POSITIVE_STORE)
                )
                if not guided_records:
                    raise RuntimeError("pilot round produced no certified G+")
            if include_dplus and not positive_records:
                raise RuntimeError("--include-dplus yes requires nonempty D+")

            fixed_before = {}
            if guided_records:
                fixed_before["Gplus"] = NS._fixed_loss(
                    policy, guided_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if positive_records:
                fixed_before["Dplus"] = NS._fixed_loss(
                    policy, positive_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )

            updates = []
            if include_dplus:
                updates.append(MR._population_update(
                    policy,
                    optimizer,
                    positive_records,
                    population="Dplus",
                    inner_steps=int(args.update_passes),
                    batch=int(args.batch),
                    device=args.device,
                    seed=int(args.train_seed) + round_i * 1_000_003,
                ))
                BX._save_checkpoint(
                    policy,
                    os.path.join(
                        checkpoints_dir,
                        f"round_{round_i:02d}_post_positive.pt",
                    ),
                    {
                        "study": STATUS,
                        "round": round_i,
                        "phase": "post_Dplus",
                        "pilot_config": config,
                    },
                )
            if guided_records:
                updates.append(MR._population_update(
                    policy,
                    optimizer,
                    guided_records,
                    population="Gplus",
                    inner_steps=int(args.update_passes),
                    batch=int(args.batch),
                    device=args.device,
                    seed=int(args.train_seed) + round_i * 7_000_003,
                ))

            fixed_after = {}
            if guided_records:
                fixed_after["Gplus"] = NS._fixed_loss(
                    policy, guided_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if positive_records:
                fixed_after["Dplus"] = NS._fixed_loss(
                    policy, positive_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )

            round_checkpoint = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}.pt",
            )
            BX._save_checkpoint(policy, round_checkpoint, {
                "study": STATUS,
                "round": round_i,
                "phase": "post_Gplus",
                "pilot_config": config,
            })
            record = {
                "status": ROUND_STATUS,
                "round": int(round_i),
                "scenarios": list(scenarios),
                "gather_dir": gather_dir,
                "gather_seconds": gather_seconds,
                "seconds": float(time.perf_counter() - started),
                "source_checkpoint": current_checkpoint,
                "source_checkpoint_sha256": current_sha,
                "round_checkpoint": round_checkpoint,
                "round_checkpoint_sha256": FA._sha256_file(round_checkpoint),
                "gather_counts": gather_marker["counts"],
                "gather_outcomes": gather_marker["outcomes"],
                "guided_collect": gather_marker.get("guided_collect"),
                "crunch_full_pool": gather_marker.get("crunch_full_pool"),
                "guided_positive_shard": gather_marker.get(
                    "guided_positive_shard"
                ),
                "executed_shard": gather_marker["executed_shard"],
                "neutral_shard": gather_marker["neutral_shard"],
                "populations": {
                    "Gplus": _population_summary(guided_records),
                    "Dplus": _population_summary(positive_records),
                    "D0": {
                        "windows": len(neutral_records),
                        "trained_on": False,
                    },
                },
                "Gplus_by_step_bin": (
                    _step_bins(
                        guided_records,
                        until_step=int(args.guided_until_step),
                    )
                    if collect_gplus else []
                ),
                "Gplus_reused_from_repair": int(sum(
                    bool(row["reused_from_repair"])
                    for _, row in guided_records
                )),
                "Gplus_selected_for_execution": int(sum(
                    bool(row["selected_for_execution"])
                    for _, row in guided_records
                )),
                "fixed_loss_before": fixed_before,
                "fixed_loss_after": fixed_after,
                "updates": updates,
            }
            FA._write_json(
                os.path.join(round_dir, "ROUND_COMPLETE.json"), record,
            )
            history.append(record)
            previous_executed_path = executed_path
            current_checkpoint = round_checkpoint

    final_checkpoint = current_checkpoint
    evaluation = None
    if int(args.eval_m) > 0:
        evaluation = _run_paired_eval(
            args,
            checkpoints=[round0_path, final_checkpoint],
            labels=["r0", f"r{int(args.rounds)}"],
            output_dir=os.path.join(output_root, "paired_eval"),
        )

    delivery = dict(
        status=STATUS,
        config=config,
        rounds=history,
        final_checkpoint=final_checkpoint,
        final_checkpoint_sha256=FA._sha256_file(final_checkpoint),
        round0_checkpoint=round0_path,
        round0_checkpoint_sha256=FA._sha256_file(round0_path),
        encoder_sha256_start=encoder_sha_start,
        encoder_sha256_end=BS.module_sha256(policy.enc_grid),
        evaluation=evaluation,
        caveats=[
            "single unreplicated pilot; not a confirmation",
            "G+ is collection-only and never executed in the gather",
            "D0 is collected and audited but never trained on",
        ],
    )
    if delivery["encoder_sha256_end"] != encoder_sha_start:
        raise RuntimeError("frozen visual encoder changed during the pilot")
    path = os.path.join(output_root, "PILOT_COMPLETE.json")
    FA._write_json(path, delivery)
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--scenario-ep0", type=int, default=DEFAULT_SCENARIO_EP0,
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+",
        default=tuple(map(float, SP.GAMMAS)),
    )
    parser.add_argument("--T", type=int, default=180)
    parser.add_argument(
        "--guided-until-step", type=int, default=DEFAULT_GUIDED_UNTIL_STEP,
        help=(
            "run the guided generator proactively at every live context with "
            "step < this value and harvest the certified positives as G+; "
            "the default covers the measured NVP crunch band"
        ),
    )
    parser.add_argument(
        "--guided-generator", choices=RA.GUIDED_GENERATORS,
        default="kazuki_full",
        help=(
            "kazuki_full runs the complete locked external controller; "
            "same_latent is the weak repair operator, kept for ablation"
        ),
    )
    parser.add_argument(
        "--guided-topk", type=int, default=RA.DEFAULT_GUIDED_TOPK,
        help="kazuki_full only: refined elite modes verified per context",
    )
    parser.add_argument(
        "--crunch-full-pool-until-step", type=int, default=0,
        help=(
            "independent teacher-free arm: verify all K=16 proposals instead "
            "of the GP's B=4 below this step; changes the admissible set and "
            "therefore the executed trajectory"
        ),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--update-passes", type=int, default=DEFAULT_UPDATE_PASSES,
        help="whole-population Adam steps per population per round",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--include-dplus", choices=("yes", "no"), default="yes",
        help="yes trains a separate pass block on executed D+ before G+",
    )
    parser.add_argument(
        "--selector", choices=("margin", "progress_gated_margin"),
        default="margin",
    )
    parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    parser.add_argument("--ell", type=float, default=RA.DEFAULT_ELL)
    parser.add_argument("--gp-cap", type=int, default=MR.DEFAULT_GP_CAP)
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--sample-seed", type=int, default=700_000)
    parser.add_argument("--audit-seed", type=int, default=2_026_073_0)
    parser.add_argument("--train-seed", type=int, default=2_026_073_1)
    parser.add_argument(
        "--eval-m", type=int, default=0,
        help="0 skips the paired offline evaluation entirely",
    )
    parser.add_argument("--eval-ep0", type=int, default=DEFAULT_EVAL_EP0)
    parser.add_argument("--eval-noise-seed", type=int, default=2_026_073_3)
    parser.add_argument(
        "--require-clean-worktree", action="store_true",
        help=(
            "fail unless the tracked worktree is clean; off by default "
            "because this pilot rides an uncommitted collector extension"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
