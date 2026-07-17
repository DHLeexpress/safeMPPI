"""Authenticate one completed AFE-RBF run and its rendered report/video."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise FileNotFoundError(path)
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--expected-video-frames", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    recipe_path = os.path.join(args.run, "recipe.json")
    complete_path = os.path.join(args.run, "COMPLETE.json")
    with open(recipe_path) as stream:
        recipe = json.load(stream)
    with open(complete_path) as stream:
        complete = json.load(stream)
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("run is not complete")
    if recipe.get("algorithm") not in {
        "afe_rbf_previous_round_parallel_v1",
        "afe_rbf_batch_conditional_parallel_v2",
        "afe_rbf_sequential_operational_parallel_v3",
        "afe_rbf_adaptive_ess_parallel_v4",
        "afe_uniform_parallel_v1",
    }:
        raise RuntimeError("unexpected algorithm")
    records = [
        json.loads(line)
        for line in open(os.path.join(args.run, "probe.jsonl"))
        if line.strip()
    ]
    if [int(record["round"]) for record in records] != list(
        range(int(recipe["rounds"]) + 1)
    ):
        raise RuntimeError("probe does not contain exactly round 0..R")
    if recipe["algorithm"] == "afe_rbf_adaptive_ess_parallel_v4":
        target = float(recipe["adaptive_ess_target"])
        for round_i, record in enumerate(records[1:], start=1):
            calibration = record.get("adaptive_beta_calibration") or {}
            if float(calibration.get("target", -1.0)) != target:
                raise RuntimeError(f"round {round_i} adaptive beta target is inconsistent")
            if float(record["beta_next"]) != float(calibration["beta"]):
                raise RuntimeError(f"round {round_i} beta_next is inconsistent")
            if round_i > 1 and float(record["beta_used"]) != float(
                records[round_i - 1]["beta_next"]
            ):
                raise RuntimeError(f"round {round_i} did not use prior beta_next")
    if recipe["algorithm"] == "afe_uniform_parallel_v1":
        for round_i, record in enumerate(records[1:], start=1):
            if record.get("acquisition_mode") != "uniform":
                raise RuntimeError(f"round {round_i} is not uniform acquisition")
            if abs(float(record.get("ess_med", 0.0)) - 1.0) > 1.0e-12:
                raise RuntimeError(f"round {round_i} uniform acquisition ESS is not one")
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = os.path.join(args.run, relative)
        if not os.path.isfile(path) or sha256_file(path) != expected:
            raise RuntimeError(f"trainer artifact hash mismatch: {relative}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height", "-of", "json", args.video],
        check=True, capture_output=True, text=True,
    )
    video_metadata = json.loads(probe.stdout)
    streams = video_metadata.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("rendered video must contain exactly one video stream")
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise RuntimeError("rendered video has invalid dimensions")
    frame_count = int(stream.get("nb_frames", 0))
    if frame_count <= 0:
        raise RuntimeError("rendered video has no frames")
    if args.expected_video_frames is not None and frame_count != args.expected_video_frames:
        raise RuntimeError(
            f"rendered video has {frame_count} frames; expected {args.expected_video_frames}"
        )
    delivery = {
        "status": "AFE_RBF_DELIVERY_COMPLETE",
        "algorithm": recipe["algorithm"],
        "scene_profile": recipe["scene"]["profile"]["name"],
        "scene_sha256": recipe["scene"]["sha256"],
        "source_git_commit": recipe["source_git_commit"],
        "completed_round": complete["completed_round"],
        "run": artifact(complete_path),
        "recipe": artifact(recipe_path),
        "report": artifact(args.report),
        "video": {**artifact(args.video), "ffprobe": video_metadata},
    }
    with open(args.out, "w") as stream:
        json.dump(delivery, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"validated {args.out}")


if __name__ == "__main__":
    main()
