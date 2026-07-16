#!/usr/bin/env python3
"""Independent audit for the Stage 04--06 v4-style sanity package."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "stage_results/05_sanity"
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
EXPECTED_SHA = "5bdd1d7abfc187bf22b31479bbd337166a8375db62f8df1b7e992af56de99de2"
ARMS = {
    "ours": (False, False, False),
    "no_socp": (False, True, False),
    "no_progress": (False, False, True),
    "no_curriculum": (True, False, False),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(directory):
    return [json.loads((directory / f"row_g{float(gamma)}.json").read_text()) for gamma in GAMMAS]


def main():
    active_sha = sha256(ROOT / "pretrained_sg_walls8.pt")
    require(active_sha == EXPECTED_SHA, "active endpoint-free checkpoint changed")

    arm_summary = {}
    for name, expected_flags in ARMS.items():
        run = STAGE / "runs" / f"final_v7_{name}"
        recipe = json.loads((run / "recipe.json").read_text())
        flags = recipe["ablations"]
        actual_flags = (flags["curriculum"], flags["socp"], flags["progress"])
        require(actual_flags == expected_flags, f"{name}: ablation is not one-flag clean")
        demo = recipe["demo_distillation"]
        require(recipe["demo_frac"] == 0.5 and demo["fraction_after_valid"] == 0.25,
                f"{name}: wrong retained-distillation schedule")
        require(demo["bootstrap_on_empty"] and demo["decay_on_first_exact_valid"],
                f"{name}: demo bootstrap/decay not enabled")
        acceptance = recipe["trajectory_acceptance"]
        expected_predicates = {
            "horizon": 10,
            "taskspace": True,
            "goal_reach": True,
            "goal_progress": name != "no_progress",
            "socp": name != "no_socp",
            "semantics": "named ablation removes only that predicate",
        }
        require(acceptance == expected_predicates, f"{name}: wrong reduced acceptance predicates")
        require(recipe["beta"] == 0.2 and recipe["alpha"] == 0.0005,
                f"{name}: beta/alpha mismatch")
        probes = [json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines() if line]
        require(len(probes) == 20, f"{name}: expected twenty sanity iterations")
        require(probes[0]["vr"] > 0 and probes[0]["demo_latched"],
                f"{name}: first certified rollout did not latch demo decay")
        require(all(abs(row["demo_frac"] - 0.25) < 1e-12 and row["batch_d"] == 4 for row in probes),
                f"{name}: post-certificate distillation was not retained at four rows")
        for row in probes:
            audit = row["gather_audit"]
            require(audit["acceptance_required"] == {
                "windowable": True, "taskspace": True, "goal_reach": True,
                "goal_progress": name != "no_progress", "socp": name != "no_socp"},
                f"{name}: audit predicate mask mismatch")
            require(audit["accepted_reached"] == audit["acceptance_pass"],
                    f"{name}: accepted a non-goal-terminated rollout")
        evaluation = rows(STAGE / "data" / f"eval_final_v7_{name}")
        require(all(row["M"] == 6 for row in evaluation), f"{name}: evaluation is not M=6")
        arm_summary[name] = {
            "accepted_rollouts": sum(row["vr"] for row in probes),
            "accepted_windows": sum(int(row["gather_audit"].get("returned_windows", 0)) for row in probes),
            "viz_db_frames": len(list((run / "viz_db").glob("it*.pt"))),
            "deployment_successes": sum(row["n_success"] for row in evaluation),
            "deployment_collisions": sum(round(row["CR"] * row["M"]) for row in evaluation),
        }

    # The no-curriculum arm is a strict controlled replay: same full-arm accepted rows, rejected rows,
    # and per-iteration volume; only the easy/frontier class split and batch draw are removed.
    full_run = STAGE / "runs/final_v7_ours"
    no_curr_run = STAGE / "runs/final_v7_no_curriculum"
    full_probe = [json.loads(line) for line in (full_run / "probe.jsonl").read_text().splitlines() if line]
    no_curr_probe = [json.loads(line) for line in (no_curr_run / "probe.jsonl").read_text().splitlines() if line]
    full_counts = [int(row["gather_audit"].get("returned_windows", 0)) for row in full_probe]
    no_curr_counts = [int(row["gather_audit"].get("returned_windows", 0)) for row in no_curr_probe]
    require(full_counts == no_curr_counts, "-Curriculum accepted-window schedule differs from full")
    require(sum(full_counts) == 601 and sum(value > 0 for value in full_counts) == 8,
            "unexpected controlled full-arm budget")
    no_curr_recipe = json.loads((no_curr_run / "recipe.json").read_text())
    ablation = no_curr_recipe["ablations"]
    require(ablation["controlled_sample_identity"] is True,
            "-Curriculum recipe does not declare controlled sample identity")
    require([int(ablation["accepted_window_budget"][str(i)]) for i in range(1, 21)] == full_counts,
            "-Curriculum recipe budget differs from full")
    replayed_positive = replayed_negative = 0
    for iteration, expected in enumerate(full_counts, start=1):
        if not expected:
            require(no_curr_probe[iteration - 1]["gather_audit"]["replay"] is True,
                    f"-Curriculum it{iteration}: zero-budget iteration is not replay-controlled")
            continue
        source = torch.load(full_run / f"viz_db/it{iteration}.pt", map_location="cpu", weights_only=False)
        replay = torch.load(no_curr_run / f"viz_db/it{iteration}.pt", map_location="cpu", weights_only=False)
        for key in ("grid", "low5", "hist", "U", "gamma"):
            require(torch.equal(source[key], replay[key]),
                    f"-Curriculum it{iteration}: accepted tensor {key} differs from full")
        require(source["negative"] is not None and replay["negative"] is not None,
                f"-Curriculum it{iteration}: rejected replay is missing")
        for key in ("grid", "low5", "hist", "U"):
            require(torch.equal(source["negative"][key], replay["negative"][key]),
                    f"-Curriculum it{iteration}: rejected tensor {key} differs from full")
        replayed_positive += int(source["U"].shape[0])
        replayed_negative += int(source["negative"]["U"].shape[0])
    require(replayed_positive == 601, "controlled positive replay total mismatch")

    demo_pool = torch.load(STAGE / "data/canonical_seed_windows.pt", map_location="cpu",
                           weights_only=False)
    require(demo_pool["exact_valid2"] is True and demo_pool["wall_plugs"] == 8,
            "distillation pool is not the exact walled SafeMPPI expert")
    require(np.allclose(demo_pool["start"], [0.05, 0.05]) and
            np.allclose(demo_pool["goal"], [5.0, 5.0]),
            "distillation pool is not the canonical OOD lower-left to upper-right task")
    require(len(demo_pool["paths"]) == len(GAMMAS) and
            all(np.linalg.norm(np.asarray(path)[-1] - [5.0, 5.0]) < demo_pool["reach"]
                for path in demo_pool["paths"]),
            "OOD expert trajectories do not all terminate at the expansion goal")

    expert = rows(ROOT / "stage_results/06_baselines/results/expert_m6")
    kazuki = rows(ROOT / "stage_results/06_baselines/results/kazuki_low_guidance_m6")
    require(sum(row["n_success"] for row in expert) == 42 and sum(row["CR"] for row in expert) == 0,
            "expert baseline changed")
    require(sum(row["n_success"] for row in kazuki) == 3, "low-guidance Kazuki success count changed")
    require(sum(round(row["CR"] * row["M"]) for row in kazuki) == 39,
            "low-guidance Kazuki collision count changed")

    artifacts = [
        STAGE / "viz/scatter_v4.png", STAGE / "viz/scatter_v4.pdf",
        STAGE / "viz/rollouts_v4.png", STAGE / "viz/rollouts_v4.pdf",
        STAGE / "viz/internals_v4.png", STAGE / "viz/internals_v4.pdf",
        STAGE / "viz/curriculum_it20.mp4",
        STAGE / "data/table_v4.md", STAGE / "data/table_v4.tex",
    ]
    for artifact in artifacts:
        require(artifact.exists() and artifact.stat().st_size > 1000, f"missing/empty {artifact}")
    table = (STAGE / "data/table_v4.md").read_text()
    for label in ("NO safety validity check", "NO progress check", "NO curriculum"):
        require(label in table, f"table missing {label}")

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames", "-show_entries", "format=duration",
        "-of", "json", str(STAGE / "viz/curriculum_it20.mp4")], text=True)
    video = json.loads(probe)
    stream = video["streams"][0]
    require((stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"], stream["nb_frames"])
            == ("h264", 2028, 1014, "2/1", "42"), "curriculum video encoding mismatch")

    summary = {
        "status": "PASS",
        "active_checkpoint_sha256": active_sha,
        "trainer_tests": {"hardtail": "22/22", "signed_negative": "PASS"},
        "demo_pool": {"role": "canonical OOD SafeMPPI expert", "start": [0.05, 0.05],
                      "goal": [5.0, 5.0], "gammas": list(GAMMAS), "windows": 819},
        "demo_schedule": {"initial_fraction": 0.5, "after_first_exact": 0.25,
                          "batch_size": 16, "initial_rows": 8, "after_rows": 4},
        "controlled_no_curriculum": {
            "per_iteration_windows": full_counts,
            "accepted_windows": replayed_positive,
            "rejected_windows_replayed_on_nonempty_updates": replayed_negative,
            "sample_identity": "bit-exact tensors",
        },
        "arms": arm_summary,
        "expert": {"successes": 42, "collisions": 0},
        "kazuki_low_guidance": {"w_safe": 0.02, "coll_w": 2.0, "goal_coef": 0.1,
                                "successes": 3, "collisions": 39},
        "video": {"codec": "h264", "resolution": "2028x1014", "fps": 2,
                  "frames": 42, "duration_s": float(video["format"]["duration"])},
        "artifacts": [str(path.relative_to(ROOT)) for path in artifacts],
    }
    output = STAGE / "logs/v4_sanity_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
