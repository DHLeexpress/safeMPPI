"""Fail-closed physical-GPU provenance capture for canonical launchers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-index", type=int, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    fields = "index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu"
    line = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(args.physical_index),
            f"--query-gpu={fields}", "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in line.split(",")]
    if len(values) != 7:
        raise RuntimeError(f"unexpected nvidia-smi row: {line}")
    index, uuid, name, driver, used, total, utilization = values
    if index != str(args.physical_index) or uuid != args.expected_uuid:
        raise RuntimeError(
            f"physical GPU identity mismatch: index={index}, uuid={uuid}"
        )
    compute_pids = [
        row.strip()
        for row in subprocess.check_output(
            [
                "nvidia-smi", "-i", str(args.physical_index),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        if row.strip()
    ]
    if compute_pids:
        raise RuntimeError(
            f"physical GPU {args.physical_index} is not idle: pids={compute_pids}"
        )
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_index": int(index),
        "uuid": uuid,
        "name": name,
        "driver_version": driver,
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "utilization_percent": int(utilization),
        "compute_pids": compute_pids,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(args.physical_index),
        "process_device": "cuda:0",
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
