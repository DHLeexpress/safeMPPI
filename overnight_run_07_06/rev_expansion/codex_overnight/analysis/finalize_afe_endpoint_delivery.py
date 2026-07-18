"""Combine authenticated trainer and endpoint-evaluation deliveries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "DELIVERY_COMPLETE.json"
    if output.exists():
        raise FileExistsError(output)
    trainer_path = root / "TRAINER_DELIVERY_COMPLETE.json"
    evaluation_path = root / "evaluation/EVALUATION_COMPLETE.json"
    trainer = json.loads(trainer_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if trainer.get("status") != "AFE_ENSEMBLE_DELIVERY_COMPLETE":
        raise RuntimeError("trainer delivery is incomplete")
    if evaluation.get("status") != "AFE_M20_EVALUATION_DELIVERY_COMPLETE":
        raise RuntimeError("endpoint evaluation delivery is incomplete")
    if trainer.get("source_git_commit") != evaluation.get("trainer_source_commit"):
        raise RuntimeError("trainer/evaluation source commits disagree")
    if trainer.get("scene_sha256") != evaluation.get("scene_sha256"):
        raise RuntimeError("trainer/evaluation scene hashes disagree")
    evaluation_root = evaluation_path.parent
    for relative, expected in evaluation.get("artifact_sha256", {}).items():
        path = evaluation_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"endpoint evaluation artifact mismatch: {relative}")
    payload = {
        "status": "LOW7_AFE100_DELIVERY_COMPLETE",
        "algorithm": trainer["algorithm"],
        "scene_profile": trainer["scene_profile"],
        "scene_sha256": trainer["scene_sha256"],
        "source_git_commit": trainer["source_git_commit"],
        "completed_round": trainer["completed_round"],
        "trainer_delivery": artifact(trainer_path),
        "evaluation_delivery": artifact(evaluation_path),
        "report_png": artifact(root / "report.png"),
        "report_pdf": artifact(root / "report.pdf"),
        "expansion_video": artifact(root / "video.mp4"),
        "endpoint_gallery": artifact(
            root / "evaluation/afe_m20_fixed_index_gallery.png"
        ),
        "evaluation_curves": artifact(
            root / "evaluation/afe_m20_probe_curves.png"
        ),
        "metrics": artifact(root / "evaluation/metrics.jsonl"),
        "gpu_provenance": artifact(root / "gpu_provenance.json"),
        "checkpoint_and_scene_provenance": artifact(root / "preflight.json"),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"validated {output}")


if __name__ == "__main__":
    main()
