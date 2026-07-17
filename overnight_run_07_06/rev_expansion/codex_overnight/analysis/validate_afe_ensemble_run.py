"""Authenticate one completed AFE deep-ensemble run and rendered assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import torch


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
        "afe_deep_ensemble_parallel_v1",
        "afe_deep_ensemble_adaptive_ess_parallel_v2",
    }:
        raise RuntimeError("unexpected algorithm")
    adaptive = recipe["algorithm"] == "afe_deep_ensemble_adaptive_ess_parallel_v2"
    if recipe.get("kernel") is not None:
        raise RuntimeError("deep-ensemble arm must not declare a kernel")
    if recipe.get("beta") is None:
        raise RuntimeError("deep-ensemble beta calibration is missing")
    if complete.get("algorithm") != recipe["algorithm"]:
        raise RuntimeError("completion algorithm does not match recipe")
    if int(complete.get("completed_round", -1)) != int(recipe["rounds"]):
        raise RuntimeError("completion round does not match recipe")
    if complete.get("scene_sha256") != recipe["scene"]["sha256"]:
        raise RuntimeError("completion scene does not match recipe")
    if complete.get("checkpoint_sha256") != recipe["source_checkpoint_sha256"]:
        raise RuntimeError("completion checkpoint does not match recipe")
    if complete.get("source_git_commit") != recipe["source_git_commit"]:
        raise RuntimeError("completion source commit does not match recipe")
    calibration_path = os.path.join(args.run, "ensemble_calibration.json")
    with open(calibration_path) as stream:
        calibration = json.load(stream)
    if float(calibration.get("beta")) != float(recipe["beta"]):
        raise RuntimeError("calibration beta does not match recipe beta")
    probe_path = os.path.join(args.run, "probe.jsonl")
    records = [json.loads(line) for line in open(probe_path) if line.strip()]
    expected_rounds = list(range(int(recipe["rounds"]) + 1))
    if [int(record["round"]) for record in records] != expected_rounds:
        raise RuntimeError("probe does not contain exactly round 0..R")
    for record in records[1:]:
        round_i = int(record["round"])
        expected_mode = "uniform_bootstrap" if round_i == 1 else "ensemble_tilt"
        if record.get("acquisition_mode") != expected_mode:
            raise RuntimeError(f"round {round_i} has the wrong acquisition mode")
        if adaptive:
            if float(record.get("beta_next")) != float(record.get("beta")):
                raise RuntimeError(f"round {round_i} beta record is inconsistent")
            calibration = record.get("beta_calibration") or {}
            if float(calibration.get("target", -1.0)) != float(
                recipe["adaptive_ess_target"]
            ):
                raise RuntimeError(f"round {round_i} adaptive beta target is inconsistent")
            if round_i > 1 and float(record.get("beta_used")) != float(
                records[round_i - 1]["beta_next"]
            ):
                raise RuntimeError(f"round {round_i} did not use prior beta_next")
        elif float(record.get("beta")) != float(recipe["beta"]):
            raise RuntimeError(f"round {round_i} did not keep beta fixed")
        ensemble = record.get("ensemble", {})
        if int(ensemble.get("n", -1)) != int(record["n_D"]):
            raise RuntimeError(f"round {round_i} ensemble did not fit cumulative D")
        positive_fraction = float(ensemble.get("positive_fraction", -1.0))
        if not 0.0 <= positive_fraction <= 1.0:
            raise RuntimeError(f"round {round_i} has an invalid label fraction")
        if int(ensemble.get("label_unique_count", 0)) not in {1, 2}:
            raise RuntimeError(f"round {round_i} has invalid verifier label classes")
        gamma_total = sum(
            int(row["total"])
            for row in ensemble.get("per_gamma_labels", {}).values()
        )
        if gamma_total != int(record["n_D"]):
            raise RuntimeError(f"round {round_i} per-gamma labels do not sum to D")
    canonical = {
        "recipe.json",
        "ensemble_calibration.json",
        "probe.jsonl",
        "final.pt",
        "dstore.pt",
        *[f"ckpt_{index}.pt" for index in expected_rounds],
        *[f"ensemble_round{index}.pt" for index in expected_rounds],
        *[f"viz_db/round{index}.pt" for index in expected_rounds[1:]],
    }
    inventory = complete.get("artifact_sha256", {})
    if set(inventory) != canonical:
        raise RuntimeError("completion artifact inventory is not canonical")
    for relative, expected in inventory.items():
        path = os.path.join(args.run, relative)
        if not os.path.isfile(path) or sha256_file(path) != expected:
            raise RuntimeError(f"trainer artifact hash mismatch: {relative}")
    for round_i in expected_rounds:
        payload = torch.load(
            os.path.join(args.run, f"ensemble_round{round_i}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if int(payload.get("round", -1)) != round_i:
            raise RuntimeError(f"ensemble checkpoint {round_i} has the wrong round")
        expected_beta = (
            None if round_i == 0 else (
                float(records[round_i]["beta_next"])
                if adaptive else float(recipe["beta"])
            )
        )
        if payload.get("beta") != expected_beta:
            raise RuntimeError(f"ensemble checkpoint {round_i} has the wrong beta")
        if payload.get("source_git_commit") != recipe["source_git_commit"]:
            raise RuntimeError(f"ensemble checkpoint {round_i} has the wrong source commit")
        estimator_state = payload.get("estimator", {})
        model_states = estimator_state.get("model_states")
        if round_i == 0 and model_states is not None:
            raise RuntimeError("round-0 ensemble checkpoint must be unfit")
        if round_i > 0 and len(model_states or []) != 5:
            raise RuntimeError(f"ensemble checkpoint {round_i} must contain five members")
        if round_i > 0 and int(estimator_state["fit_diagnostics"]["n"]) != int(
            records[round_i]["n_D"]
        ):
            raise RuntimeError(f"ensemble checkpoint {round_i} fit rows do not match D")
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
    if int(stream.get("nb_frames", 0)) <= 0:
        raise RuntimeError("rendered video has no frames")
    if (
        args.expected_video_frames is not None
        and int(stream.get("nb_frames", 0)) != args.expected_video_frames
    ):
        raise RuntimeError(
            f"rendered video has {stream.get('nb_frames')} frames; "
            f"expected {args.expected_video_frames}"
        )
    delivery = {
        "status": "AFE_ENSEMBLE_DELIVERY_COMPLETE",
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
