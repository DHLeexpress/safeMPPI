#!/usr/bin/env python3
"""Stateful temperature-1 expansion until one checkpoint clears the 3-gamma gate.

Gathering always uses T=1.0.  Measurements, checkpoint screening, and the paper
rollout records use T=0.5.  A gate passes only when the *same* checkpoint has at
least one successful matched-seed rollout for gamma 0.1, 0.5, and 1.0.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT.parents[1]
for path in (WORK, ROOT.parent, ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

from giant_obstacle_ood.stage4_frozen_ood import save_records  # noqa: E402
from giant_obstacle_ood.stage5_evaluate import evaluate_checkpoint  # noqa: E402
from giant_obstacle_ood.stage5_window_expand import CHECKPOINT, run_arm  # noqa: E402
from viz_style import GAMMAS  # noqa: E402


STAGE = HERE / "stage_results/05_window_expand"
RUN = STAGE / "runs/gatherT1_evalT0.5/full"
GATE = STAGE / "gatherT1_evalT0.5_gate"
TARGET_GAMMAS = (0.1, 0.5, 1.0)
GATHER_TEMPERATURE = 1.0
EVALUATION_TEMPERATURE = 0.5
PERSISTENT_ROUTE_BIT = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_iter(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("iter", 0))


def checkpoint_candidates() -> list[Path]:
    candidates = sorted(
        RUN.glob("ckpt_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[-1])
    )
    final = RUN / "final.pt"
    if final.exists() and not any(checkpoint_iter(path) == checkpoint_iter(final) for path in candidates):
        candidates.append(final)
    return candidates


def load_gate_index() -> dict:
    path = GATE / "gate_index.json"
    if not path.exists():
        return {"status": "RUNNING", "created_utc": now(), "evaluations": []}
    return json.loads(path.read_text())


def write_gate_index(index: dict) -> None:
    GATE.mkdir(parents=True, exist_ok=True)
    (GATE / "gate_index.json").write_text(json.dumps(index, indent=2) + "\n")


def gate_counts(summary: dict) -> dict[str, int]:
    return {
        f"{gamma:g}": int(summary["per_gamma"][str(float(gamma))]["successes"])
        for gamma in TARGET_GAMMAS
    }


def screen_new_checkpoints(device: torch.device, repetitions: int,
                           min_gate_iter: int) -> tuple[Path | None, dict]:
    index = load_gate_index()
    evaluated = {
        (int(row["iter"]), str(row["checkpoint_sha256"])) for row in index["evaluations"]
    }
    for checkpoint in checkpoint_candidates():
        iteration = checkpoint_iter(checkpoint)
        checksum = sha256(checkpoint)
        if (iteration, checksum) in evaluated:
            prior = next(
                row for row in index["evaluations"]
                if int(row["iter"]) == iteration and row["checkpoint_sha256"] == checksum
            )
            if prior["passed"]:
                return checkpoint, index
            continue
        records, summary = evaluate_checkpoint(
            checkpoint, temperature=EVALUATION_TEMPERATURE, repetitions=repetitions,
            device=device, method=f"Full gatherT1 it{iteration}", seed0=92500,
            persistent_route_bit=PERSISTENT_ROUTE_BIT,
        )
        directory = GATE / f"it{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        save_records(
            records, directory / f"rollouts_temp0.5_m{repetitions}.npz",
            checkpoint=np.asarray(str(checkpoint.resolve())),
            checkpoint_sha256=np.asarray(checksum),
            gather_temperature=np.asarray(GATHER_TEMPERATURE),
            evaluation_temperature=np.asarray(EVALUATION_TEMPERATURE),
            matched_seed0=np.asarray(92500),
            persistent_route_bit=np.asarray(PERSISTENT_ROUTE_BIT),
        )
        (directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        counts = gate_counts(summary)
        target_passed = all(count > 0 for count in counts.values())
        passed = bool(target_passed and iteration >= min_gate_iter)
        route_modes = Counter(
            record["route_mode"] for record in records if record["success"]
        )
        row = {
            "evaluated_utc": now(),
            "iter": iteration,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checksum,
            "evaluation_temperature": EVALUATION_TEMPERATURE,
            "M_per_gamma": repetitions,
            "matched_seed0": 92500,
            "persistent_route_bit": bool(PERSISTENT_ROUTE_BIT),
            "target_success_counts": counts,
            "target_passed": target_passed,
            "gate_eligible": bool(iteration >= min_gate_iter),
            "minimum_gate_iteration": int(min_gate_iter),
            "passed": passed,
            "overall_SR": float(summary["overall"]["a_SR"]),
            "overall_CR": float(summary["overall"]["b_CR"]),
            "successful_route_modes": dict(route_modes),
        }
        index["evaluations"].append(row)
        index["updated_utc"] = now()
        print(
            f"TARGET_GATE it={iteration} matched-M={repetitions} "
            + " ".join(f"g{gamma}={count}" for gamma, count in counts.items())
            + f" target_passed={target_passed} eligible={iteration >= min_gate_iter} "
            + f"passed={passed} overall_SR={summary['overall']['a_SR']:.3f} "
            + f"persistent_route_bit={PERSISTENT_ROUTE_BIT}",
            flush=True,
        )
        if passed:
            index["status"] = "PASS"
            index["selected"] = row
            write_gate_index(index)
            (GATE / "selected_checkpoint.json").write_text(json.dumps(row, indent=2) + "\n")
            return checkpoint, index
        write_gate_index(index)
        del records, summary
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return None, index


def health_audit(final_iter: int) -> dict:
    rows = [
        json.loads(line) for line in (RUN / "probe.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # A crash/retry can repeat a line; the committed final observation wins.
    by_iter = {int(row["iter"]): row for row in rows if int(row["iter"]) <= final_iter}
    rows = [by_iter[index] for index in sorted(by_iter)]
    rollbacks = [int(row["iter"]) for row in rows if row.get("rollback")]
    readiness_failures = [
        int(row["iter"]) for row in rows
        if not (row.get("gamma_ready") and row.get("classes_ready") and row.get("gamma_class_ready"))
    ]
    invalid_losses = [
        int(row["iter"]) for row in rows
        if row.get("loss") is None or not math.isfinite(float(row["loss"]))
    ]
    missing_updates = [
        int(row["iter"]) for row in rows
        if int(row.get("batch_e", 0)) + int(row.get("batch_f", 0)) + int(row.get("batch_d", 0)) == 0
    ]
    semantics_failures = [
        int(row["iter"]) for row in rows
        if row.get("gather_audit", {}).get("aggregation_semantics") != "window_native_v1"
    ]
    frontier_failures = [
        int(row["iter"]) for row in rows
        if int(row.get("n_easy", 0)) == 0 or int(row.get("n_frontier", 0)) == 0
    ]
    recipe = json.loads((RUN / "recipe.json").read_text())
    temperature_ok = bool(
        np.isclose(recipe["gather_temperature"], GATHER_TEMPERATURE)
        and np.isclose(recipe["evaluation_temperature"], EVALUATION_TEMPERATURE)
    )
    status = "PASS" if not (
        rollbacks or readiness_failures or invalid_losses or missing_updates
        or semantics_failures or frontier_failures or not temperature_ok
    ) else "ALERT"
    audit = {
        "status": status,
        "through_iter": final_iter,
        "probe_iterations": len(rows),
        "gather_temperature": recipe["gather_temperature"],
        "evaluation_temperature": recipe["evaluation_temperature"],
        "temperature_contract_ok": temperature_ok,
        "aggregation_semantics": recipe["aggregation_semantics"],
        "enabled_window_predicates": recipe["enabled_window_predicates"],
        "rollbacks": rollbacks,
        "readiness_failures": readiness_failures,
        "invalid_losses": invalid_losses,
        "missing_updates": missing_updates,
        "window_semantics_failures": semantics_failures,
        "empty_easy_or_frontier_iterations": frontier_failures,
        "accepted_windows_total": int(sum(
            row.get("gather_audit", {}).get("accepted_windows", 0) for row in rows
        )),
        "accepted_from_whole_invalid_total": int(sum(
            row.get("gather_audit", {}).get("accepted_from_whole_invalid", 0) for row in rows
        )),
        "demo_fraction_schedule": recipe["demo_frac_schedule"],
    }
    GATE.mkdir(parents=True, exist_ok=True)
    (GATE / f"health_it{final_iter:04d}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"HEALTH it={final_iter} status={status} rollbacks={len(rollbacks)} "
        f"readiness_failures={len(readiness_failures)} "
        f"empty_classes={len(frontier_failures)} accepted={audit['accepted_windows_total']}",
        flush=True,
    )
    return audit


def train_chunk(additional_iters: int, *, initial_checkpoint: Path,
                demo_switch_iter: int, beta: float, alpha: float,
                success_replay_quota: int, inner_steps: int, early_inner_steps: int,
                learning_rate: float, encoder_lr_mult: float,
                early_rollout_scale: float, demo_stage_weights: tuple[float, float, float, float],
                cfm_eta: float,
                resume_reset_optimizer: bool,
                lwf_eta: float, lwf_x0_scale: float,
                eval_temp_cfm_weight: float,
                eval_endpoint_eta: float, eval_endpoint_quota: int, eval_endpoint_scope: str,
                eval_endpoint_gammas: tuple[float, ...],
                eval_goal_eta: float, eval_goal_velocity_weight: float,
                eval_mode_noise_coupling: bool,
                train_gammas: tuple[float, ...],
                allow_recipe_drift: bool,
                resume_initial_state: bool) -> int:
    final = RUN / "final.pt"
    if final.exists():
        start = checkpoint_iter(final)
        run_arm(
            "full", RUN, gather_temperature=GATHER_TEMPERATURE,
            evaluation_temperature=EVALUATION_TEMPERATURE,
            iters=additional_iters, checkpoint_path=final, resume=True,
            demo_switch_iter=demo_switch_iter, beta=beta, alpha=alpha,
            success_replay_quota=success_replay_quota,
            inner_steps=inner_steps, early_inner_steps=early_inner_steps,
            learning_rate=learning_rate,
            encoder_lr_mult=encoder_lr_mult,
            early_rollout_scale=early_rollout_scale,
            demo_stage_weights=demo_stage_weights,
            cfm_eta=cfm_eta,
            resume_reset_optimizer=resume_reset_optimizer,
            lwf_eta=lwf_eta,
            lwf_x0_scale=lwf_x0_scale,
            eval_temp_cfm_weight=eval_temp_cfm_weight,
            eval_endpoint_eta=eval_endpoint_eta,
            eval_endpoint_quota=eval_endpoint_quota,
            eval_endpoint_scope=eval_endpoint_scope,
            eval_endpoint_gammas=eval_endpoint_gammas,
            eval_goal_eta=eval_goal_eta,
            eval_goal_velocity_weight=eval_goal_velocity_weight,
            eval_mode_noise_coupling=eval_mode_noise_coupling,
            train_gammas=train_gammas,
            allow_recipe_drift=allow_recipe_drift,
        )
    else:
        if RUN.exists() and any(RUN.iterdir()):
            raise RuntimeError(f"nonempty run has no resumable final checkpoint: {RUN}")
        start = 0
        run_arm(
            "full", RUN, gather_temperature=GATHER_TEMPERATURE,
            evaluation_temperature=EVALUATION_TEMPERATURE,
            iters=additional_iters, checkpoint_path=initial_checkpoint,
            resume=resume_initial_state,
            demo_switch_iter=demo_switch_iter, beta=beta, alpha=alpha,
            success_replay_quota=success_replay_quota,
            inner_steps=inner_steps, early_inner_steps=early_inner_steps,
            learning_rate=learning_rate,
            encoder_lr_mult=encoder_lr_mult,
            early_rollout_scale=early_rollout_scale,
            demo_stage_weights=demo_stage_weights,
            cfm_eta=cfm_eta,
            resume_reset_optimizer=resume_reset_optimizer,
            lwf_eta=lwf_eta,
            lwf_x0_scale=lwf_x0_scale,
            eval_temp_cfm_weight=eval_temp_cfm_weight,
            eval_endpoint_eta=eval_endpoint_eta,
            eval_endpoint_quota=eval_endpoint_quota,
            eval_endpoint_scope=eval_endpoint_scope,
            eval_endpoint_gammas=eval_endpoint_gammas,
            eval_goal_eta=eval_goal_eta,
            eval_goal_velocity_weight=eval_goal_velocity_weight,
            eval_mode_noise_coupling=eval_mode_noise_coupling,
            train_gammas=train_gammas,
            allow_recipe_drift=allow_recipe_drift,
        )
    end = checkpoint_iter(final)
    print(f"CHUNK committed={start}->{end}", flush=True)
    return end


def main() -> None:
    global RUN, GATE, PERSISTENT_ROUTE_BIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-iters", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--screen-m", type=int, default=6)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--gate-dir", type=Path, default=GATE)
    parser.add_argument("--initial-checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--demo-switch-iter", type=int, default=11)
    parser.add_argument("--beta", type=float, default=0.2,
                        choices=(0.2, 0.3, 0.4, 0.5, 0.7, 1.0))
    parser.add_argument("--alpha", type=float, default=0.0,
                        choices=(0.0, 0.001, 0.002, 0.005, 0.01))
    parser.add_argument("--success-replay-quota", type=int, default=0,
                        choices=(0, 2, 4, 6, 8, 10))
    parser.add_argument("--inner-steps", type=int, default=2,
                        choices=(2, 4, 6, 8, 12, 16, 32, 64, 100))
    parser.add_argument("--early-inner-steps", type=int, default=2,
                        choices=(2, 4, 6, 8, 12, 16, 32, 64, 100))
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--encoder-lr-mult", type=float, default=0.3)
    parser.add_argument("--early-rollout-scale", type=float, default=1.0)
    parser.add_argument("--demo-stage-weights", type=float, nargs=4,
                        default=(0.25, 0.25, 0.25, 0.25),
                        metavar=("INITIAL", "APPROACH", "BOUNDARY", "POST"))
    parser.add_argument("--eval-temp-cfm-weight", type=float, default=0.0)
    parser.add_argument("--cfm-eta", type=float, default=1.0)
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--lwf-eta", type=float, default=0.0)
    parser.add_argument("--lwf-x0-scale", type=float, default=1.0)
    parser.add_argument("--eval-endpoint-eta", type=float, default=0.0)
    parser.add_argument("--eval-endpoint-quota", type=int, default=0)
    parser.add_argument("--eval-endpoint-scope", choices=("all", "pre", "post", "goal"), default="all")
    parser.add_argument("--eval-endpoint-gammas", type=float, nargs="*", default=())
    parser.add_argument("--eval-goal-eta", type=float, default=0.0)
    parser.add_argument("--eval-goal-velocity-weight", type=float, default=0.2)
    parser.add_argument("--eval-mode-noise-coupling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-route-bit-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-gammas", type=float, nargs="+", default=tuple(GAMMAS))
    parser.add_argument("--allow-recipe-drift", action="store_true")
    parser.add_argument("--resume-initial-state", action="store_true",
                        help="restore complete state from --initial-checkpoint even in a new run directory")
    parser.add_argument("--min-gate-iter", type=int, default=0,
                        help="do not promote a checkpoint before this absolute iteration")
    args = parser.parse_args()
    if args.chunk_iters <= 0 or args.max_iters <= 0 or args.screen_m <= 0:
        parser.error("all bounds must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.encoder_lr_mult <= 0:
        parser.error("--encoder-lr-mult must be positive")
    if not (0.0 < args.early_rollout_scale <= 1.0):
        parser.error("--early-rollout-scale must be in (0,1]")
    if min(args.demo_stage_weights) < 0 or sum(args.demo_stage_weights) <= 0:
        parser.error("--demo-stage-weights must be nonnegative with positive total mass")
    if args.eval_temp_cfm_weight < 0:
        parser.error("--eval-temp-cfm-weight must be nonnegative")
    if args.cfm_eta < 0:
        parser.error("--cfm-eta must be nonnegative")
    if args.lwf_eta < 0:
        parser.error("--lwf-eta must be nonnegative")
    if args.lwf_x0_scale <= 0:
        parser.error("--lwf-x0-scale must be positive")
    if args.eval_endpoint_eta < 0:
        parser.error("--eval-endpoint-eta must be nonnegative")
    if not (0 <= args.eval_endpoint_quota <= 16):
        parser.error("--eval-endpoint-quota must be in [0,16]")
    if (args.eval_endpoint_eta > 0) != (args.eval_endpoint_quota > 0):
        parser.error("--eval-endpoint-eta and --eval-endpoint-quota must be enabled together")
    if any(float(gamma) not in GAMMAS for gamma in args.eval_endpoint_gammas):
        parser.error("--eval-endpoint-gammas must be drawn from the configured gamma sweep")
    if not args.train_gammas or any(float(gamma) not in GAMMAS for gamma in args.train_gammas):
        parser.error("--train-gammas must be a nonempty subset of the configured gamma sweep")
    if args.eval_goal_eta < 0 or args.eval_goal_velocity_weight < 0:
        parser.error("goal-loss weights must be nonnegative")
    RUN = args.run_dir.resolve()
    GATE = args.gate_dir.resolve()
    PERSISTENT_ROUTE_BIT = bool(args.persistent_route_bit_gate)
    initial_checkpoint = args.initial_checkpoint.resolve()
    if not initial_checkpoint.exists():
        parser.error(f"initial checkpoint does not exist: {initial_checkpoint}")
    GATE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"DEVICE {device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | "
        f"gatherT={GATHER_TEMPERATURE:g} evalT={EVALUATION_TEMPERATURE:g}",
        flush=True,
    )

    index = load_gate_index()
    index.update({
        "run_dir": str(RUN),
        "initial_checkpoint": str(initial_checkpoint),
        "gather_temperature": GATHER_TEMPERATURE,
        "evaluation_temperature": EVALUATION_TEMPERATURE,
        "demo_switch_iter": int(args.demo_switch_iter),
        "beta": float(args.beta),
        "alpha": float(args.alpha),
        "success_replay_quota": int(args.success_replay_quota),
        "inner_steps": int(args.inner_steps),
        "early_inner_steps": int(args.early_inner_steps),
        "learning_rate": float(args.learning_rate),
        "encoder_lr_mult": float(args.encoder_lr_mult),
        "early_rollout_scale": float(args.early_rollout_scale),
        "demo_stage_weights": [float(value) for value in args.demo_stage_weights],
        "cfm_eta": float(args.cfm_eta),
        "resume_reset_optimizer": bool(args.resume_reset_optimizer),
        "lwf_eta": float(args.lwf_eta),
        "lwf_x0_scale": float(args.lwf_x0_scale),
        "eval_temp_cfm_weight": float(args.eval_temp_cfm_weight),
        "eval_endpoint_eta": float(args.eval_endpoint_eta),
        "eval_endpoint_quota": int(args.eval_endpoint_quota),
        "eval_endpoint_scope": str(args.eval_endpoint_scope),
        "eval_endpoint_gammas": [float(gamma) for gamma in args.eval_endpoint_gammas],
        "eval_goal_eta": float(args.eval_goal_eta),
        "eval_goal_velocity_weight": float(args.eval_goal_velocity_weight),
        "eval_mode_noise_coupling": bool(args.eval_mode_noise_coupling),
        "persistent_route_bit_gate": bool(args.persistent_route_bit_gate),
        "train_gammas": [float(gamma) for gamma in args.train_gammas],
        "allow_recipe_drift": bool(args.allow_recipe_drift),
        "resume_initial_state": bool(args.resume_initial_state),
        "minimum_gate_iteration": int(args.min_gate_iter),
    })
    write_gate_index(index)
    selected, index = screen_new_checkpoints(device, args.screen_m, args.min_gate_iter)
    if selected is not None:
        print(f"ALREADY_PASS {selected}", flush=True)
        return
    current = checkpoint_iter(RUN / "final.pt") if (RUN / "final.pt").exists() else 0
    while current < args.max_iters:
        additional = min(args.chunk_iters, args.max_iters - current)
        current = train_chunk(
            additional, initial_checkpoint=initial_checkpoint,
            demo_switch_iter=args.demo_switch_iter, beta=args.beta, alpha=args.alpha,
            success_replay_quota=args.success_replay_quota,
            inner_steps=args.inner_steps, early_inner_steps=args.early_inner_steps,
            learning_rate=args.learning_rate,
            encoder_lr_mult=args.encoder_lr_mult,
            early_rollout_scale=args.early_rollout_scale,
            demo_stage_weights=tuple(args.demo_stage_weights),
            cfm_eta=args.cfm_eta,
            resume_reset_optimizer=args.resume_reset_optimizer,
            lwf_eta=args.lwf_eta,
            lwf_x0_scale=args.lwf_x0_scale,
            eval_temp_cfm_weight=args.eval_temp_cfm_weight,
            eval_endpoint_eta=args.eval_endpoint_eta,
            eval_endpoint_quota=args.eval_endpoint_quota,
            eval_endpoint_scope=args.eval_endpoint_scope,
            eval_endpoint_gammas=tuple(args.eval_endpoint_gammas),
            eval_goal_eta=args.eval_goal_eta,
            eval_goal_velocity_weight=args.eval_goal_velocity_weight,
            eval_mode_noise_coupling=args.eval_mode_noise_coupling,
            train_gammas=tuple(args.train_gammas),
            allow_recipe_drift=args.allow_recipe_drift,
            resume_initial_state=args.resume_initial_state,
        )
        health = health_audit(current)
        if health["status"] != "PASS":
            index = load_gate_index()
            index["status"] = "HEALTH_ALERT"
            index["health_alert_iter"] = current
            write_gate_index(index)
            raise RuntimeError(f"training health alert at iteration {current}; refusing blind continuation")
        selected, index = screen_new_checkpoints(device, args.screen_m, args.min_gate_iter)
        if selected is not None:
            print(f"PASS selected={selected} iter={checkpoint_iter(selected)}", flush=True)
            return
        if checkpoint_iter(RUN / "final.pt") != current:
            raise RuntimeError("final checkpoint iteration changed unexpectedly")

    index = load_gate_index()
    index["status"] = "MAX_ITER_WITHOUT_PASS"
    index["max_iters"] = args.max_iters
    index["target_gammas"] = list(TARGET_GAMMAS)
    write_gate_index(index)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
