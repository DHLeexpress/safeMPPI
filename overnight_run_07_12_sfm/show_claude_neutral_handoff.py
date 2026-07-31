"""Print the frozen Claude handoff from existing Helios artifacts only.

This command never launches a rollout or recomputes a metric.  It authenticates
the shared visualization, checkpoint, and fresh disjoint-M50 delivery before
printing the two requested reference rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


VIDEO = Path(
    "/data3/research1/sfm_neutral_claude_handoff_assets/"
    "neutral_continuation.mp4"
)
VIDEO_SHA256 = "c9437021d5a2c25f2c735521f00ded0727b819f4ab2beca3e97c4beb5370b45f"
DELIVERY = Path(
    "/data3/research1/sfm_neutral_gamma_temp_0e441d6/"
    "DELIVERY_COMPLETE.json"
)
DELIVERY_STATUS = "SFM_NEUTRAL_GAMMA_TEMPERATURE_M50_COMPLETE"
METRICS_SOURCE_COMMIT = "0e441d68644e89017a4c312ad59e7b520ca80208"
CHECKPOINT = Path(
    "/home/dohyun/projects/sfm_hp10_b1_runs/103476d/pretrained_hp10.pt"
)
CHECKPOINT_SHA256 = "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}"
            )


def main() -> None:
    require_file(VIDEO, VIDEO_SHA256)
    require_file(CHECKPOINT, CHECKPOINT_SHA256)
    require_file(DELIVERY)
    payload = json.loads(DELIVERY.read_text())
    if payload.get("status") != DELIVERY_STATUS:
        raise RuntimeError("cached M50 delivery is incomplete")
    if payload.get("source_commit") != METRICS_SOURCE_COMMIT:
        raise RuntimeError("cached M50 delivery source changed")
    bank = payload["fresh_confirmation_bank"]
    if (bank["M_per_gamma"], bank["ep0"]) != (50, 485000):
        raise RuntimeError("fresh disjoint M50 bank changed")
    by_method = {row["method"]: row for row in payload["final_records"]}
    if set(("pretrained", "kazuki_locked")) - set(by_method):
        raise RuntimeError("cached reference methods are missing")

    print(f"NEUTRAL_CONTINUATION_MP4={VIDEO}")
    print(f"NEUTRAL_CONTINUATION_MP4_SHA256={VIDEO_SHA256}")
    print(f"PRETRAINED_CHECKPOINT={CHECKPOINT}")
    print(f"PRETRAINED_CHECKPOINT_SHA256={CHECKPOINT_SHA256}")
    print(
        "CACHED_REFERENCE_BANK="
        f"fresh disjoint M={bank['M_per_gamma']}/gamma, ep0={bank['ep0']}, "
        f"noise_seed={bank['noise_seed']}"
    )
    print("method\tSR\tCR\ttimeout\tValidity\tclearance_m\ttime_to_goal_s")
    for method in ("pretrained", "kazuki_locked"):
        row = by_method[method]
        values = row["pooled"]
        print(
            f"{method}\t{values['SR']:.6f}\t{values['CR']:.6f}\t"
            f"{values['timeout']:.6f}\t{values['Validity']:.6f}\t"
            f"{values['clearance']:.6f}\t{values['time_to_goal']:.6f}"
        )
        schedule = row.get("temperature_by_gamma")
        if schedule is not None:
            print(
                f"{method}_temperature_by_gamma="
                f"{dict(zip(('0.1','0.2','0.3','0.4','0.5','0.7','1.0'), schedule))}"
            )
    print(f"CACHED_REFERENCE_DELIVERY={DELIVERY}")
    print("No rollout or metric was recomputed.")


if __name__ == "__main__":
    main()
