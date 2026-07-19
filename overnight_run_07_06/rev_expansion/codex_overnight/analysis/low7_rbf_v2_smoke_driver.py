#!/usr/bin/env python3
"""One fail-closed giant-obstacle V2 qualification run plus raw-M10 evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "grid_expand_afe_rbf.py"
EVALUATOR = ROOT / "paper_results" / "low7_raw_m50_eval.py"
DIAGNOSTICS = ROOT / "analysis" / "afe_rbf_sweep_diagnostics.py"
VIDEO = ROOT / "video_afe2.py"
SCENE = "low7_radius1_canonical_v1"
EVAL_PROFILE = "v2_smoke_m10_every_round"
BASELINE_STUDY = "baseline"
LINEAGE_MASS_STUDY = "lineage_mass"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainer_command(args, run_dir: Path) -> list[str]:
    study_profile = args.study_profile
    if study_profile not in {BASELINE_STUDY, LINEAGE_MASS_STUDY}:
        raise ValueError(f"unknown V2 study profile: {study_profile}")
    protocol_profile = (
        "v2_lineage_mass_smoke"
        if study_profile == LINEAGE_MASS_STUDY else "v2_smoke"
    )
    execution_rule = (
        "nominal_hp_max_step_margin"
        if study_profile == LINEAGE_MASS_STUDY
        else "nominal_hp_max_step_margin_only"
    )
    replay_loss_weighting = (
        "gamma_episode_context_query_equal_mass"
        if study_profile == LINEAGE_MASS_STUDY else "query_uniform"
    )
    command = [
        args.python,
        str(TRAINER),
        "--protocol-profile", protocol_profile,
        "--ckpt", str(args.ckpt),
        "--expected-ckpt-sha256", args.expected_ckpt_sha256,
        "--scene-profile", SCENE,
        "--outdir", str(run_dir),
        "--rounds", "10",
        "--rollout-replicas", "8",
        "--K", "16",
        "--B", "4",
        "--T", "300",
        "--M-eval", "0",
        "--batch", "128",
        "--afe-steps", "0",
        "--afe-lr", "1e-5",
        "--gp-cap", "512",
        "--gp-lam", "1e-2",
        "--acquisition-mode", "sequential",
        "--adaptive-ess-target", "0.5",
        "--adaptive-beta-contexts-per-gamma", "64",
        "--adaptive-beta-equalize-gammas",
        "--replay-window", "2",
        "--replay-sampling", "round_gamma_replica_context",
        "--replay-update-mode", "one_epoch_without_replacement",
        "--replay-loss-weighting", replay_loss_weighting,
        "--gp-replay-window", "2",
        "--gp-replay-sampling", "round_gamma_replica_context",
        "--lengthscale-multiplier", "1.0",
        "--negative-alpha", "0",
        "--execution-rule", execution_rule,
        "--conditioning-schema", "low7_closest_boundary",
        "--freeze-visual-encoder",
        "--skip-training-probes",
        "--calibration-replicas", "8",
        "--calibration-control-steps", "4",
        "--sweep-compact-artifacts",
        "--compact-checkpoint-every", "1",
        "--route-metric-steps", "10",
        "--verifier-workers", str(args.verifier_workers),
        "--seed", "910",
    ]
    if study_profile == LINEAGE_MASS_STUDY:
        command.append("--nvp-audit-all-k")
    return command


def _run(command: list[str], log_path: Path) -> None:
    with log_path.open("x") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _video_record(path: Path, expected_frames: int) -> dict:
    payload = json.loads(subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames",
            "-of", "json", str(path),
        ],
        text=True,
    ))
    streams = payload.get("streams") or []
    if len(streams) != 1 or int(streams[0].get("nb_read_frames", -1)) != expected_frames:
        raise RuntimeError(f"V2 video is not the declared {expected_frames}-frame artifact")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "frames": expected_frames,
        "codec": streams[0].get("codec_name"),
        "width": int(streams[0]["width"]),
        "height": int(streams[0]["height"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--expected-ckpt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verifier-workers", type=int, default=64)
    parser.add_argument(
        "--study-profile",
        choices=(BASELINE_STUDY, LINEAGE_MASS_STUDY),
        required=True,
    )
    parser.add_argument(
        "--python",
        default="/home/dohyun/miniforge3/envs/cfm_mppi/bin/python",
    )
    args = parser.parse_args()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.expected_ckpt_sha256 = args.expected_ckpt_sha256.lower()
    if args.out.exists():
        raise FileExistsError(f"V2 smoke output root must be absent: {args.out}")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if sha256_file(args.ckpt) != args.expected_ckpt_sha256:
        raise RuntimeError("V2 smoke checkpoint SHA-256 mismatch")
    if args.verifier_workers < 1:
        raise ValueError("verifier workers must be positive")

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise RuntimeError("V2 smoke requires committed clean tracked source")
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("V2 smoke requires an empty index")

    args.out.mkdir(parents=True)
    run_dir = args.out / "run"
    evaluation_dir = args.out / "evaluation"
    presentation_dir = args.out / "presentation"
    diagnostic_path = args.out / "training_diagnostic.png"
    video_path = args.out / "training_expansion.mp4"
    started = time.time()
    train = trainer_command(args, run_dir)
    evaluate = [
        args.python,
        str(EVALUATOR),
        "--run-root", str(run_dir),
        "--scene-profile", SCENE,
        "--outdir", str(evaluation_dir),
        "--eval-profile", EVAL_PROFILE,
        "--verifier-workers", str(args.verifier_workers),
    ]
    validate = [
        args.python,
        str(EVALUATOR),
        "--outdir", str(evaluation_dir),
        "--validate-only",
    ]
    render = [
        args.python,
        str(EVALUATOR),
        "--outdir", str(evaluation_dir),
        "--render-only",
        "--presentation-outdir", str(presentation_dir),
    ]
    diagnostics = [
        args.python,
        str(DIAGNOSTICS),
        "--run", str(run_dir),
        "--out", str(diagnostic_path),
    ]
    video = [
        args.python,
        str(VIDEO),
        "--run", str(run_dir),
        "--out", str(video_path),
        "--dense-until", "10",
        "--every-after", "10",
    ]
    _run(train, args.out / "train.log")
    if _load_json(run_dir / "COMPLETE.json").get("status") != "COMPLETE":
        raise RuntimeError("V2 trainer did not deliver COMPLETE")
    _run(evaluate, args.out / "evaluation.log")
    _run(validate, args.out / "validation.log")
    _run(render, args.out / "presentation.log")
    report_records = []
    for suffix in ("png", "pdf"):
        source = presentation_dir / f"report.{suffix}"
        destination = args.out / f"report.{suffix}"
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"V2 true-evaluation report is missing: {source}")
        shutil.copy2(source, destination)
        report_records.append(
            {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    gallery_source = evaluation_dir / "raw_m10_r0_best_final_gallery.png"
    gallery_path = args.out / "gallery.png"
    if not gallery_source.is_file() or gallery_source.stat().st_size == 0:
        raise RuntimeError(f"V2 true-evaluation gallery is missing: {gallery_source}")
    shutil.copy2(gallery_source, gallery_path)
    gallery_record = {
        "path": str(gallery_path),
        "sha256": sha256_file(gallery_path),
        "bytes": gallery_path.stat().st_size,
    }
    _run(diagnostics, args.out / "diagnostic.log")
    _run(video, args.out / "video.log")
    completion = _load_json(evaluation_dir / "EVALUATION_COMPLETE.json")
    if completion.get("status") != (
        "AFE_RBF_RAW_M10_EVERY_ROUND_EVALUATION_DELIVERY_COMPLETE"
    ):
        raise RuntimeError("V2 raw-M10 evaluator did not deliver its declared status")
    if not diagnostic_path.is_file() or diagnostic_path.stat().st_size == 0:
        raise RuntimeError("V2 training diagnostic is missing")
    video_record = _video_record(video_path, expected_frames=10)

    manifest = {
        "status": (
            "LOW7_RBF_V2_LINEAGE_MASS_SMOKE_DELIVERY_COMPLETE"
            if args.study_profile == LINEAGE_MASS_STUDY
            else "LOW7_RBF_V2_SMOKE_DELIVERY_COMPLETE"
        ),
        "source_git_commit": source_commit,
        "scene_profile": SCENE,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": args.expected_ckpt_sha256,
        "elapsed_seconds": time.time() - started,
        "protocol": (
            "v2_lineage_mass_smoke"
            if args.study_profile == LINEAGE_MASS_STUDY else "v2_smoke"
        ),
        "study_profile": args.study_profile,
        "evaluation_profile": EVAL_PROFILE,
        "run": str(run_dir),
        "evaluation": str(evaluation_dir),
        "true_evaluation_reports": report_records,
        "true_evaluation_gallery": gallery_record,
        "presentation": str(presentation_dir),
        "training_diagnostic": str(diagnostic_path),
        "training_video": video_record,
        "commands": {
            "train": train,
            "evaluate": evaluate,
            "validate": validate,
            "render": render,
            "diagnostics": diagnostics,
            "video": video,
        },
    }
    with (args.out / "DELIVERY_COMPLETE.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        f"LOW7 RBF V2 {args.study_profile.upper()} SMOKE COMPLETE: {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
