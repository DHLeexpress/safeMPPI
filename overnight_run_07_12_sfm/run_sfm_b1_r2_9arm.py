#!/usr/bin/env python3
"""Fail-closed launcher for the two-round SFM B1 alpha/replay sweep.

The launcher deliberately knows only the narrow CLI contract of
``sfm_b1_r2_alpha_replay.py``:

    --checkpoint PATH --outdir ABSENT_DIR
    --alpha FLOAT --replay-epochs INT
    --verifier-workers INT --seed INT --device cuda:0

Each arm must atomically write ``COMPLETE.json`` with status
``R2_ALPHA_REPLAY_COMPLETE`` and authenticated ``round_00.pt`` through
``round_02.pt`` sidecars.  This launcher does not import training code and
does not evaluate checkpoints.  Once all arms validate, it writes a compact
``CHECKPOINT_INDEX.json`` for a separate common-bank evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAINER = HERE / "sfm_b1_r2_alpha_replay.py"
ALPHAS = (0.0, 0.01, 0.1)
REPLAY_EPOCHS = (1, 10, 100)
ROUNDS = 2
ARM_STATUS = "R2_ALPHA_REPLAY_COMPLETE"
MAX_VERIFIER_WORKERS = 8
ELL = 0.24210826720721101
CAP = 256
SCENE_PROFILE = "double_density_velocity_ood"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | os.PathLike[str], payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


@dataclass(frozen=True)
class Arm:
    alpha: float
    replay_epochs: int

    @property
    def name(self) -> str:
        # Match ExperimentConfig.arm_name without importing the training module.
        alpha = str(float(self.alpha)).replace(".", "p")
        return f"margin_alpha{alpha}_epochs{self.replay_epochs:03d}"


def arm_grid() -> tuple[Arm, ...]:
    return tuple(
        Arm(alpha, epochs) for alpha in ALPHAS for epochs in REPLAY_EPOCHS
    )


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int
    pci_bus_id: str


def _nvidia_lines(arguments: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"nvidia-smi query failed: {error}") from error
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot() -> tuple[list[GPU], list[dict], str]:
    rows = _nvidia_lines([
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if len(values) != 7:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
        gpus.append(GPU(
            index=values[0], uuid=values[1], name=values[2],
            memory_total_mib=int(values[3]), memory_used_mib=int(values[4]),
            utilization_percent=int(values[5]), pci_bus_id=values[6],
        ))
    processes = []
    for row in _nvidia_lines([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
    ]):
        values = [value.strip() for value in row.split(",")]
        if len(values) == 4:
            processes.append(dict(
                gpu_uuid=values[0], pid=int(values[1]), process_name=values[2],
                used_memory_mib=int(values[3]),
            ))
    try:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        topology = ""
    return gpus, processes, topology


def select_idle_gpus(gpus: list[GPU], processes: list[dict], requested: str,
                     *, max_memory_mib: int, max_utilization: int) -> list[GPU]:
    if requested == "auto":
        candidates = list(gpus)
    else:
        indices = [value.strip() for value in requested.split(",") if value.strip()]
        if len(indices) != len(set(indices)) or not indices:
            raise ValueError("--gpu-indices must be 'auto' or unique comma-separated indices")
        by_index = {gpu.index: gpu for gpu in gpus}
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise RuntimeError(f"requested GPU indices are unavailable: {missing}")
        candidates = [by_index[index] for index in indices]
    active = {row["gpu_uuid"] for row in processes}
    busy = [
        gpu for gpu in candidates
        if (gpu.uuid in active or gpu.memory_used_mib > int(max_memory_mib)
            or gpu.utilization_percent > int(max_utilization))
    ]
    if requested != "auto" and busy:
        detail = [
            dict(index=gpu.index, uuid=gpu.uuid, memory_used_mib=gpu.memory_used_mib,
                 utilization_percent=gpu.utilization_percent,
                 compute_process=gpu.uuid in active)
            for gpu in busy
        ]
        raise RuntimeError(f"explicitly requested GPUs are not idle: {detail}")
    selected = [gpu for gpu in candidates if gpu not in busy]
    if not selected:
        raise RuntimeError("no idle GPU satisfies the launch contract")
    return selected


def assign_arms(arms: list[Arm], gpus: list[GPU],
                *, max_arms_per_gpu: int = 3) -> dict[str, list[Arm]]:
    """Create a deterministic balanced allocation.

    With four GPUs and the declared grid this places all three one-step arms
    together and one 10/100 pair on each remaining GPU, yielding 3/2/2/2.
    For other GPU counts, a least-loaded greedy allocation is used.
    """
    if not gpus:
        raise ValueError("at least one GPU is required")
    if int(max_arms_per_gpu) < 1:
        raise ValueError("max_arms_per_gpu must be positive")
    if len(arms) > len(gpus) * int(max_arms_per_gpu):
        raise RuntimeError(
            f"{len(arms)} arms exceed {len(gpus)} GPUs x "
            f"{int(max_arms_per_gpu)} arms/GPU"
        )
    ordered_gpus = sorted(gpus, key=lambda gpu: int(gpu.index))
    allocation = {gpu.uuid: [] for gpu in ordered_gpus}
    if len(arms) == 9 and len(ordered_gpus) == 4 and set(arms) == set(arm_grid()):
        by_steps = {
            steps: sorted(
            [arm for arm in arms if arm.replay_epochs == steps],
                key=lambda arm: arm.alpha,
            )
            for steps in REPLAY_EPOCHS
        }
        allocation[ordered_gpus[0].uuid].extend(by_steps[1])
        for gpu, ten, hundred in zip(
                ordered_gpus[1:], by_steps[10], by_steps[100]):
            allocation[gpu.uuid].extend((ten, hundred))
        return allocation
    # The fallback cost is dominated by gathering, so arm count is the primary
    # balance term; replay work is only a deterministic tie-break.
    for arm in sorted(
            arms, key=lambda value: (-value.replay_epochs, value.alpha)):
        eligible = [
            gpu for gpu in ordered_gpus
            if len(allocation[gpu.uuid]) < int(max_arms_per_gpu)
        ]
        gpu = min(
            eligible,
            key=lambda value: (
                len(allocation[value.uuid]),
                sum(item.replay_epochs for item in allocation[value.uuid]),
                int(value.index),
            ),
        )
        allocation[gpu.uuid].append(arm)
    return allocation


def source_provenance() -> dict:
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=ROOT, text=True, env=environment,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"git {' '.join(arguments)} failed") from error

    status = git("status", "--porcelain")
    if status:
        raise RuntimeError("source worktree must be clean before launch")
    branch = git("branch", "--show-current")
    if not branch:
        raise RuntimeError("a named pushed branch is required")
    head = git("rev-parse", "HEAD")
    try:
        remote = subprocess.check_output(
            ["git", "ls-remote", "--heads", "origin", branch],
            cwd=ROOT, text=True, env=environment,
        ).split()
    except subprocess.CalledProcessError as error:
        raise RuntimeError("cannot authenticate origin branch") from error
    if not remote or remote[0] != head:
        raise RuntimeError("source HEAD is not the pushed origin branch head")
    return dict(branch=branch, commit=head, remote_commit=remote[0])


def _cpu_affinity() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def allocate_cpu_pools(arms: list[Arm], workers: int) -> dict[str, list[int]]:
    cpus = _cpu_affinity()
    needed = len(arms) * int(workers)
    if len(cpus) < needed:
        raise RuntimeError(
            f"{len(arms)} arms x {workers} verifier workers require {needed} "
            f"available CPUs, only {len(cpus)} are in the launcher affinity"
        )
    return {
        arm.name: cpus[index * int(workers):(index + 1) * int(workers)]
        for index, arm in enumerate(arms)
    }


def _expected_arm_contract(arm: Arm, *, checkpoint_sha256: str, scene_profile: str,
                           ell: float, cap: int, seed: int,
                           verifier_workers: int, rounds: int = ROUNDS) -> dict:
    return dict(
        alpha=float(arm.alpha), replay_epochs=int(arm.replay_epochs),
        rounds=int(rounds), checkpoint_sha256=str(checkpoint_sha256),
        scene_profile=str(scene_profile), ell=float(ell), cap=int(cap),
        seed=int(seed), verifier_workers=int(verifier_workers), lr=1.0e-4,
    )


def validate_complete_arm(arm_dir: str | os.PathLike[str], arm: Arm, *,
                          checkpoint_sha256: str, scene_profile: str,
                          ell: float, cap: int, seed: int,
                          verifier_workers: int) -> dict | None:
    arm_dir = Path(arm_dir)
    marker = arm_dir / "COMPLETE.json"
    if not marker.exists():
        if arm_dir.exists() and any(arm_dir.iterdir()):
            raise RuntimeError(f"incomplete nonempty arm directory: {arm_dir}")
        return None
    with marker.open() as stream:
        payload = json.load(stream)
    if payload.get("status") != ARM_STATUS:
        raise RuntimeError(f"invalid arm completion status: {marker}")
    if payload.get("experiment") != arm.name:
        raise RuntimeError(f"arm name mismatch in {marker}")
    expected = _expected_arm_contract(
        arm, checkpoint_sha256=checkpoint_sha256,
        scene_profile=scene_profile, ell=ell, cap=cap, seed=seed,
        verifier_workers=verifier_workers,
    )
    recipe = payload.get("recipe", {})
    constants = payload.get("constants", {})
    contract = dict(
        alpha=recipe.get("alpha"), replay_epochs=recipe.get("replay_epochs"),
        rounds=recipe.get("rounds"),
        checkpoint_sha256=payload.get("source_checkpoint_sha256"),
        scene_profile=recipe.get("scene_profile"),
        ell=constants.get("ell"), cap=constants.get("cap"),
        seed=recipe.get("seed"), verifier_workers=recipe.get("verifier_workers"),
        lr=recipe.get("lr"),
    )
    if contract != expected:
        raise RuntimeError(
            f"arm completion contract mismatch for {arm.name}: "
            f"{contract!r} != {expected!r}"
        )
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != ROUNDS:
        raise RuntimeError(f"{marker} must contain two round-history records")
    history_by_round = {int(row.get("round", -1)): row for row in history}
    validated = []
    for round_i in range(ROUNDS + 1):
        path = (arm_dir / f"round_{round_i:02d}.pt").resolve()
        expected_name = f"round_{round_i:02d}.pt"
        if path.name != expected_name or not path.is_file():
            raise RuntimeError(f"missing expected checkpoint: {path}")
        observed = sha256_file(path)
        sidecar = Path(str(path) + ".COMPLETE.json")
        if not sidecar.is_file():
            raise RuntimeError(f"missing checkpoint completion sidecar: {sidecar}")
        with sidecar.open() as stream:
            sidecar_payload = json.load(stream)
        if (sidecar_payload.get("status") != "COMPLETE"
                or sidecar_payload.get("sha256") != observed):
            raise RuntimeError(f"checkpoint sidecar mismatch: {sidecar}")
        if round_i > 0 and history_by_round.get(round_i, {}).get(
                "checkpoint_sha256") != observed:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        validated.append(dict(
            round=round_i, path=str(path), sha256=observed,
            complete_sidecar=str(sidecar.resolve()),
            complete_sidecar_sha256=sha256_file(sidecar),
        ))
    return dict(marker=str(marker.resolve()), marker_sha256=sha256_file(marker),
                checkpoints=validated, payload=payload)


def _trainer_command(args, arm: Arm, arm_dir: Path) -> list[str]:
    return [
        sys.executable, str(TRAINER),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--outdir", str(arm_dir.resolve()),
        "--alpha", str(arm.alpha),
        "--replay-epochs", str(arm.replay_epochs),
        "--verifier-workers", str(args.verifier_workers),
        "--seed", str(args.seed),
        "--device", "cuda:0",
    ]


def _child_environment(gpu: GPU) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu.uuid,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TORCH_NUM_THREADS="1",
        PYTHONPATH=str(HERE) + os.pathsep + environment.get("PYTHONPATH", ""),
    )
    return environment


def _launch_pending(jobs: list[dict], log_dir: Path) -> list[str]:
    taskset = shutil.which("taskset")
    running = []
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for job in jobs:
            log_path = log_dir / f"{job['arm'].name}.log"
            stream = log_path.open("w")
            command = list(job["command"])
            if taskset:
                command = [
                    taskset, "-c", ",".join(map(str, job["cpu_pool"])), *command,
                ]
            process = subprocess.Popen(
                command, cwd=ROOT, env=_child_environment(job["gpu"]),
                stdout=stream, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,
            )
            running.append(dict(
                process=process, stream=stream, log_path=str(log_path.resolve()),
                arm=job["arm"],
            ))
        while running:
            failure = None
            for item in running:
                code = item["process"].poll()
                if code not in (None, 0):
                    failure = (item["arm"].name, code, item["log_path"])
                    break
            if failure is not None:
                for item in running:
                    if item["process"].poll() is None:
                        os.killpg(item["process"].pid, signal.SIGTERM)
                deadline = time.monotonic() + 10.0
                for item in running:
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        item["process"].wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(item["process"].pid, signal.SIGKILL)
                        item["process"].wait()
                raise RuntimeError(
                    f"arm {failure[0]} failed with code {failure[1]}; "
                    f"all peers were stopped; log={failure[2]}"
                )
            finished = [item for item in running if item["process"].poll() == 0]
            for item in finished:
                item["stream"].close()
                running.remove(item)
            if running:
                time.sleep(0.25)
    except BaseException:
        for item in running:
            if item["process"].poll() is None:
                os.killpg(item["process"].pid, signal.SIGTERM)
            item["stream"].close()
        raise
    finally:
        for item in running:
            if not item["stream"].closed:
                item["stream"].close()
    return [job["log_path"] for job in jobs]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--scene-profile", default=SCENE_PROFILE, choices=(SCENE_PROFILE,))
    parser.add_argument("--ell", type=float, default=ELL)
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--verifier-workers", type=int, default=8)
    parser.add_argument("--gpu-indices", default="auto")
    parser.add_argument("--max-arms-per-gpu", type=int, default=3)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-utilization-percent", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args) -> dict:
    if not (1 <= int(args.verifier_workers) <= MAX_VERIFIER_WORKERS):
        raise ValueError(
            f"--verifier-workers must be in [1,{MAX_VERIFIER_WORKERS}]"
        )
    if float(args.ell) != ELL or int(args.cap) != CAP:
        raise ValueError(f"trainer fixes ell={ELL} and cap={CAP}")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint_sha = sha256_file(checkpoint)
    if observed_checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: {observed_checkpoint_sha} != "
            f"{args.expected_checkpoint_sha256}"
        )
    if not TRAINER.is_file():
        raise FileNotFoundError(
            f"training module is not present at the frozen source path: {TRAINER}"
        )
    source = source_provenance()
    arms = list(arm_grid())
    outdir = Path(args.outdir).resolve()
    declaration_contract = dict(
        version=1, source_commit=source["commit"],
        trainer_sha256=sha256_file(TRAINER),
        checkpoint=str(checkpoint), checkpoint_sha256=observed_checkpoint_sha,
        scene_profile=args.scene_profile, rounds=ROUNDS,
        alphas=list(ALPHAS), replay_epochs=list(REPLAY_EPOCHS),
        ell=float(args.ell), cap=int(args.cap), seed=int(args.seed),
        verifier_workers=int(args.verifier_workers),
    )
    declaration = dict(
        status="SFM_B1_R2_9ARM_DECLARED",
        contract=declaration_contract,
        contract_sha256=_sha256_json(declaration_contract),
    )
    declaration_path = outdir / "RUN_DECLARATION.json"
    if declaration_path.exists():
        with declaration_path.open() as stream:
            existing = json.load(stream)
        if existing != declaration:
            raise RuntimeError(f"existing run declaration differs: {declaration_path}")
    elif outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"nonempty output root lacks a matching declaration: {outdir}")

    completed, pending_arms = {}, []
    for arm in arms:
        arm_dir = outdir / "arms" / arm.name
        complete = validate_complete_arm(
            arm_dir, arm, checkpoint_sha256=observed_checkpoint_sha,
            scene_profile=args.scene_profile, ell=args.ell, cap=args.cap,
            seed=args.seed, verifier_workers=args.verifier_workers,
        )
        if complete is not None:
            completed[arm.name] = complete
            continue
        pending_arms.append(arm)
    gpus, compute_processes, topology = gpu_snapshot()
    if pending_arms:
        selected_gpus = select_idle_gpus(
            gpus, compute_processes, args.gpu_indices,
            max_memory_mib=args.idle_memory_mib,
            max_utilization=args.idle_utilization_percent,
        )
        allocation = assign_arms(
            pending_arms, selected_gpus, max_arms_per_gpu=args.max_arms_per_gpu,
        )
        by_uuid = {gpu.uuid: gpu for gpu in selected_gpus}
        cpu_pools = allocate_cpu_pools(pending_arms, args.verifier_workers)
        arm_gpu = {
            arm: by_uuid[uuid]
            for uuid, values in allocation.items() for arm in values
        }
    else:
        selected_gpus, allocation, cpu_pools, arm_gpu = [], {}, {}, {}
    pending = [
        dict(
            arm=arm, gpu=arm_gpu[arm], arm_dir=outdir / "arms" / arm.name,
            cpu_pool=cpu_pools[arm.name],
            command=_trainer_command(args, arm, outdir / "arms" / arm.name),
        )
        for arm in pending_arms
    ]
    plan = dict(
        status="SFM_B1_R2_9ARM_DRY_RUN" if args.dry_run else "SFM_B1_R2_9ARM_PLAN",
        generated_at=_utc_now(), source=source, declaration=declaration,
        all_gpus=[asdict(gpu) for gpu in gpus],
        selected_gpus=[asdict(gpu) for gpu in selected_gpus],
        compute_processes=compute_processes, topology=topology,
        allocation={
            gpu.index: [arm.name for arm in allocation[gpu.uuid]]
            for gpu in selected_gpus
        },
        completed_arms=sorted(completed),
        pending=[dict(
            arm=item["arm"].name, alpha=item["arm"].alpha,
            replay_epochs=item["arm"].replay_epochs,
            gpu_index=item["gpu"].index,
            gpu_uuid=item["gpu"].uuid, cpu_pool=item["cpu_pool"],
            command=item["command"],
        ) for item in pending],
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan

    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(declaration_path, declaration)
    _write_json(outdir / "GPU_PROVENANCE.json", plan)
    started = time.perf_counter()
    jobs = []
    for item in pending:
        item["log_path"] = str(
            (outdir / "logs" / f"{item['arm'].name}.log").resolve()
        )
        jobs.append(item)
    logs = _launch_pending(jobs, outdir / "logs") if jobs else []

    index_rows = []
    for arm in arms:
        complete = validate_complete_arm(
            outdir / "arms" / arm.name, arm,
            checkpoint_sha256=observed_checkpoint_sha,
            scene_profile=args.scene_profile, ell=args.ell, cap=args.cap,
            seed=args.seed, verifier_workers=args.verifier_workers,
        )
        if complete is None:
            raise RuntimeError(f"arm returned without COMPLETE.json: {arm.name}")
        index_rows.append(dict(
            arm=arm.name, alpha=arm.alpha,
            replay_epochs=arm.replay_epochs,
            complete_marker=complete["marker"],
            complete_marker_sha256=complete["marker_sha256"],
            checkpoints=complete["checkpoints"],
        ))
    checkpoint_index = dict(
        status="SFM_B1_R2_CHECKPOINT_INDEX_COMPLETE",
        created_at=_utc_now(), source=source,
        run_contract=declaration_contract,
        common_evaluation_requirement=(
            "Evaluate the unique r0 checkpoint once and every arm r1/r2 checkpoint "
            "on one predeclared raw temp=1 M50/gamma common-noise bank; do not "
            "force the M50 r0 estimate to equal the archival M100 statistic."
        ),
        arms=index_rows,
    )
    index_path = outdir / "CHECKPOINT_INDEX.json"
    _write_json(index_path, checkpoint_index)
    complete = dict(
        status="SFM_B1_R2_9ARM_TRAINING_COMPLETE",
        finished_at=_utc_now(), wall_seconds=time.perf_counter() - started,
        source=source, declaration_sha256=sha256_file(declaration_path),
        gpu_provenance_sha256=sha256_file(outdir / "GPU_PROVENANCE.json"),
        checkpoint_index=str(index_path), checkpoint_index_sha256=sha256_file(index_path),
        resumed_arms=sorted(completed), launched_arms=[item["arm"].name for item in jobs],
        logs=logs,
    )
    _write_json(outdir / "TRAINING_COMPLETE.json", complete)
    print(json.dumps(complete, indent=2, allow_nan=False))
    return complete


def main(argv=None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
