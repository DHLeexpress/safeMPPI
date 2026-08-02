"""From-scratch HP100 SFM pretraining with ID-only checkpoint promotion.

The loader keeps each freshly collected ``[N,32,100]`` Hp file memory-mapped.
It stores only ten integer source indices per row and gathers the corresponding
newest-to-oldest history in ``Dataset.__getitem__``.  It therefore never
materializes a second ``[N,10,32,100]`` dataset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_dynamics as DYN
import sfm_hp100_history as HPH
import sfm_protocol as SP


SUCCESSFUL_LINEAGES_PER_GAMMA = 500
SCHEMA_VERSION = "sfm_hp100_id_demonstrations_v1"
HISTORY_LENGTH = 10
DEFAULT_EPOCHS = 120
DEFAULT_BATCH = 256
DEFAULT_LR = 3.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_WARMUP = 5
DEFAULT_CHECKPOINT_EVERY = 10


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gamma_path(dataset, gamma) -> Path:
    return Path(dataset).resolve() / f"sfm_hp100_windows_g{float(gamma)}.pt"


def _manifest(dataset) -> tuple[dict, Path, str]:
    path = Path(dataset).resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing authenticated HP100 manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "HP100_ID_DATASET_COMPLETE":
        raise ValueError("HP100 dataset manifest is not complete")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"HP100 schema changed: {payload.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return payload, path, sha256_file(path)


def _manifest_files(manifest: dict) -> dict[float, dict]:
    rows = {}
    for row in manifest.get("files", []):
        gamma = float(row["gamma"])
        if gamma in rows:
            raise ValueError(f"duplicate manifest file row for gamma {gamma}")
        rows[gamma] = row
    return rows


def _history_indices(episodes: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Return local newest-to-oldest row indices without copying Hp frames."""
    episodes = torch.as_tensor(episodes, dtype=torch.int64).reshape(-1)
    steps = torch.as_tensor(steps, dtype=torch.int64).reshape(-1)
    if len(episodes) != len(steps):
        raise ValueError("episode/step lengths differ")
    result = torch.empty((len(episodes), HISTORY_LENGTH), dtype=torch.int64)
    for episode in torch.unique(episodes, sorted=True).tolist():
        rows = torch.nonzero(episodes == int(episode), as_tuple=False).flatten()
        order = torch.argsort(steps[rows], stable=True)
        rows = rows[order]
        episode_steps = steps[rows]
        expected = torch.arange(
            int(episode_steps[0]), int(episode_steps[0]) + len(rows), dtype=torch.int64
        )
        if not torch.equal(episode_steps, expected):
            raise ValueError(f"episode {episode} is duplicated, unordered, or non-contiguous")
        for position, row in enumerate(rows.tolist()):
            history_positions = [max(0, position - lag) for lag in range(HISTORY_LENGTH)]
            result[row] = rows[history_positions]
    return result


def _validate_source(payload: dict, path: Path, gamma: float) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"wrong HP100 schema in {path}")
    if not payload.get("success_only", False):
        raise ValueError(f"pretraining source is not successful-only: {path}")
    if float(payload.get("gamma")) != float(gamma):
        raise ValueError(f"gamma mismatch in {path}")
    required = {
        "hp": (32, 100), "low5": (5,), "hist": (16, 2), "U": (10, 2),
    }
    size = None
    for key, trailing in required.items():
        if key not in payload:
            raise ValueError(f"missing {key} in {path}")
        tensor = payload[key]
        if tuple(tensor.shape[1:]) != trailing:
            raise ValueError(f"{key} has shape {tuple(tensor.shape)} in {path}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must be float32 in {path}, got {tensor.dtype}")
        size = len(tensor) if size is None else size
        if len(tensor) != size:
            raise ValueError(f"window tensor lengths differ in {path}")
    for key in ("episode", "step"):
        if key not in payload or len(payload[key]) != size:
            raise ValueError(f"missing or mis-sized {key} in {path}")
    unique = torch.unique(payload["episode"].to(torch.int64), sorted=True)
    declared = int(payload.get("n_traj", -1))
    if declared != SUCCESSFUL_LINEAGES_PER_GAMMA or len(unique) != declared:
        raise ValueError(
            f"gamma {gamma} requires exactly {SUCCESSFUL_LINEAGES_PER_GAMMA} "
            f"successful lineages, found declared={declared}, unique={len(unique)}"
        )
    if payload.get("dynamics") != DYN.contract():
        raise ValueError(f"dataset dynamics differ from the HP100 training contract: {path}")


class HP100WindowDataset(Dataset):
    """A split view over memory-mapped current Hp frames."""

    def __init__(self, sources: list[dict], gamma_rows, source_rows):
        self.sources = sources
        self.gamma_rows = torch.as_tensor(gamma_rows, dtype=torch.int64).contiguous()
        self.source_rows = torch.as_tensor(source_rows, dtype=torch.int64).contiguous()
        if len(self.gamma_rows) != len(self.source_rows):
            raise ValueError("gamma/source row maps differ in length")
        self.episodes = torch.empty(len(self.source_rows), dtype=torch.int64)
        for gamma_index in torch.unique(self.gamma_rows, sorted=True).tolist():
            mask = self.gamma_rows == int(gamma_index)
            rows = self.source_rows[mask]
            self.episodes[mask] = sources[int(gamma_index)]["episode"][rows].to(torch.int64)

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, index):
        gamma_index = int(self.gamma_rows[index])
        row = int(self.source_rows[index])
        source = self.sources[gamma_index]
        history_rows = source["_history_indices"][row]
        hp100 = source["hp"].index_select(0, history_rows)
        return (
            hp100,
            source["low5"][row],
            source["hist"][row],
            source["U"][row],
            source["episode"][row].to(torch.int64),
            torch.tensor(gamma_index, dtype=torch.int64),
        )


def load_split(dataset, gammas=SP.GAMMAS, val_frac=0.1, seed=20260720):
    """Load seven ID files and split whole successful trajectories 90/10."""
    if not math.isclose(float(val_frac), 0.1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("canonical HP100 pretraining requires a 90/10 trajectory split")
    manifest, manifest_path, manifest_sha = _manifest(dataset)
    file_rows = _manifest_files(manifest)
    sources, train_gamma, train_rows, val_gamma, val_rows = [], [], [], [], []
    split_meta = {}
    for gamma_index, gamma in enumerate(map(float, gammas)):
        path = _gamma_path(dataset, gamma)
        row = file_rows.get(gamma)
        if row is None:
            raise ValueError(f"manifest has no file for gamma {gamma}")
        if path.name != row.get("file"):
            raise ValueError(f"manifest filename mismatch for gamma {gamma}")
        actual_hash = sha256_file(path)
        if actual_hash != row.get("sha256"):
            raise RuntimeError(f"dataset digest mismatch: {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        _validate_source(payload, path, gamma)
        payload["_history_indices"] = _history_indices(payload["episode"], payload["step"])
        sources.append(payload)

        unique = torch.unique(payload["episode"].to(torch.int64), sorted=True)
        generator = torch.Generator().manual_seed(
            int(seed) + 1009 * gamma_index + int(round(1000 * gamma))
        )
        shuffled = unique[torch.randperm(len(unique), generator=generator)]
        n_val = int(round(SUCCESSFUL_LINEAGES_PER_GAMMA * float(val_frac)))
        val_episodes, train_episodes = shuffled[:n_val], shuffled[n_val:]
        episodes = payload["episode"].to(torch.int64)
        is_val = torch.isin(episodes, val_episodes)
        local_val = torch.nonzero(is_val, as_tuple=False).flatten()
        local_train = torch.nonzero(~is_val, as_tuple=False).flatten()
        train_gamma.append(torch.full_like(local_train, gamma_index))
        train_rows.append(local_train)
        val_gamma.append(torch.full_like(local_val, gamma_index))
        val_rows.append(local_val)
        split_meta[str(gamma)] = {
            "file": str(path), "sha256": actual_hash,
            "train_episodes": sorted(map(int, train_episodes.tolist())),
            "val_episodes": sorted(map(int, val_episodes.tolist())),
            "train_lineages": len(train_episodes), "val_lineages": len(val_episodes),
            "train_windows": len(local_train), "val_windows": len(local_val),
        }
    train = HP100WindowDataset(sources, torch.cat(train_gamma), torch.cat(train_rows))
    val = HP100WindowDataset(sources, torch.cat(val_gamma), torch.cat(val_rows))
    metadata = {
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
        "files": split_meta, "split_seed": int(seed), "val_fraction": float(val_frac),
        "required_successful_lineages_per_gamma": SUCCESSFUL_LINEAGES_PER_GAMMA,
    }
    return train, val, metadata


def hierarchical_sampler_weights(episodes, gamma_indices):
    """Uniform objective mass gamma -> successful trajectory -> window."""
    episodes = torch.as_tensor(episodes, dtype=torch.int64)
    gamma_indices = torch.as_tensor(gamma_indices, dtype=torch.int64)
    if len(episodes) != len(gamma_indices):
        raise ValueError("episode/gamma lengths differ")
    weights = torch.zeros(len(episodes), dtype=torch.float64)
    gammas = torch.unique(gamma_indices, sorted=True)
    for gamma in gammas.tolist():
        gamma_mask = gamma_indices == int(gamma)
        trajectories = torch.unique(episodes[gamma_mask], sorted=True)
        for trajectory in trajectories.tolist():
            mask = gamma_mask & (episodes == int(trajectory))
            weights[mask] = 1.0 / (
                len(gammas) * len(trajectories) * int(mask.sum())
            )
    if not torch.isclose(
        weights.sum(), torch.tensor(1.0, dtype=weights.dtype), atol=1.0e-10
    ):
        raise RuntimeError("pretraining hierarchical mass does not sum to one")
    return weights


@contextmanager
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def deterministic_validation(policy, dataset, gammas, device, batch=512, seed=41017):
    """Fixed CFM bases/times make every epoch directly comparable."""
    policy.eval()
    totals = np.zeros(len(gammas), dtype=np.float64)
    counts = np.zeros(len(gammas), dtype=np.int64)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch), shuffle=False, num_workers=0)
    with preserve_rng():
        for hp100, low, hist, controls, _episode, gamma_index in loader:
            batch_size = len(controls)
            x1_cpu = (controls.float() / float(policy.u_max)).reshape(batch_size, policy.d)
            x0_cpu = torch.randn(x1_cpu.shape, generator=generator, dtype=torch.float32)
            tau_cpu = torch.rand((batch_size,), generator=generator).clamp(1.0e-4, 1.0)
            hp100 = hp100.to(device, non_blocking=True)
            low = low.to(device, non_blocking=True)
            hist = hist.to(device, non_blocking=True)
            x1 = x1_cpu.to(device, non_blocking=True)
            x0 = x0_cpu.to(device, non_blocking=True)
            tau = tau_cpu.to(device, non_blocking=True)
            x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
            target = x1 - x0
            prediction = policy(x_tau, tau, policy.ctx_from(hp100, low, hist))
            losses = ((prediction - target) ** 2).mean(dim=1).cpu()
            for index in torch.unique(gamma_index, sorted=True).tolist():
                mask = gamma_index == int(index)
                totals[int(index)] += float(losses[mask].sum())
                counts[int(index)] += int(mask.sum())
    if np.any(counts == 0):
        raise RuntimeError("validation split has missing gamma support")
    per_gamma = {
        str(float(gamma)): float(totals[index] / counts[index])
        for index, gamma in enumerate(gammas)
    }
    return float(np.mean(list(per_gamma.values()))), per_gamma


def atomic_save(payload, path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_hashes() -> dict:
    modules = {
        "trainer": inspect.getmodule(_source_hashes),
        "architecture": GPS,
        "dynamics": DYN,
        "history": HPH,
    }
    result = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve()
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {"root": str(root), "head": head, "clean": not bool(status), "status": status}


def checkpoint_payload(policy, epoch, history, args, dataset_meta, provenance):
    return {
        "state_dict": {
            name: value.detach().cpu() for name, value in policy.state_dict().items()
        },
        "config": policy.config(),
        "epoch": int(epoch),
        "history": history,
        "training_args": vars(args),
        "dataset": dataset_meta,
        "provenance": provenance,
        "initialization": "from_scratch",
        "partial_transplant": False,
    }


def _resolve_id_raw_gate():
    """Load the ID-only evaluator without importing any OOD evaluation code."""
    try:
        module = importlib.import_module("sfm_hp100_eval")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HP100 promotion requires sfm_hp100_eval.id_raw_gate; "
            "refusing to train or promote without the fixed ID temp=1 gate"
        ) from error
    gate = getattr(module, "id_raw_gate", None)
    if not callable(gate):
        raise RuntimeError("sfm_hp100_eval.id_raw_gate is missing or not callable")
    return gate


def _validate_gate(result: dict, gammas) -> None:
    if not isinstance(result, dict):
        raise TypeError("sfm_hp100_eval.id_raw_gate must return a dictionary")
    if float(result.get("temperature", float("nan"))) != 1.0:
        raise RuntimeError("HP100 promotion gate must use raw sampling temperature 1")
    if result.get("distribution") != "ID":
        raise RuntimeError("HP100 checkpoint promotion may inspect only ID scenarios")
    required = {str(float(gamma)) for gamma in gammas}
    if set(result.get("per_gamma", {})) != required:
        raise RuntimeError("HP100 ID gate did not return every declared gamma")
    for gamma, row in result["per_gamma"].items():
        if not {"SR", "CR"}.issubset(row):
            raise RuntimeError(f"HP100 ID gate lacks SR/CR for gamma {gamma}")


def _gate_score(result: dict, validation_cfm: float, gammas) -> tuple:
    rows = [result["per_gamma"][str(float(gamma))] for gamma in gammas]
    return (
        max(float(row["CR"]) for row in rows),
        float(np.mean([row["CR"] for row in rows])),
        -min(float(row["SR"]) for row in rows),
        -float(np.mean([row["SR"] for row in rows])),
        float(validation_cfm),
    )


def _lr_multiplier(epoch_index: int, epochs: int, warmup: int) -> float:
    if epoch_index < warmup:
        return float(epoch_index + 1) / max(1, int(warmup))
    # LambdaLR evaluates index zero before the first optimizer step.  Reaching
    # one only at index ``epochs`` avoids training epoch 120 at exactly zero LR.
    progress = min(1.0, (epoch_index - warmup) / max(1, epochs - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--policy-out", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--val-batch", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--gate-m", type=int, default=10)
    parser.add_argument("--gate-top", type=int, default=3)
    parser.add_argument("--gate-episode-start", type=int, default=SP.PRETRAIN_GATE_EP0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--smoke", action="store_true",
        help="one-epoch frozen integration smoke; never a promotable scientific checkpoint",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.samples_per_epoch = args.samples_per_epoch or 512
        args.ckpt_every = 1
        args.gate_m = 1
        args.gate_top = 1
    if not args.smoke and (args.epochs != DEFAULT_EPOCHS or args.batch != DEFAULT_BATCH):
        raise ValueError("canonical HP100 pretraining requires 120 epochs and batch 256")
    if args.lr != DEFAULT_LR or args.weight_decay != DEFAULT_WEIGHT_DECAY:
        raise ValueError("canonical HP100 pretraining requires AdamW lr=3e-4, wd=1e-4")
    if not args.smoke and (
        args.warmup != DEFAULT_WARMUP or args.ckpt_every != DEFAULT_CHECKPOINT_EVERY
    ):
        raise ValueError("canonical HP100 pretraining requires warmup=5 and ckpt-every=10")

    # Fail before a long GPU run if the fixed ID promotion evaluator is absent.
    id_raw_gate = _resolve_id_raw_gate()
    git = _git_provenance()
    if git["head"] != args.expected_source_commit:
        raise RuntimeError(
            f"source commit {git['head']} != expected {args.expected_source_commit}"
        )
    if not git["clean"]:
        raise RuntimeError("HP100 pretraining requires a clean frozen worktree")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train, val, dataset_meta = load_split(
        args.dataset, SP.GAMMAS, val_frac=0.1, seed=args.seed
    )
    weights = hierarchical_sampler_weights(train.episodes, train.gamma_rows)
    samples_per_epoch = int(args.samples_per_epoch) or len(train)
    policy = GPS.build_sfm_hp100_policy(device=args.device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _lr_multiplier(epoch, args.epochs, args.warmup),
    )
    provenance = {
        "dataset_manifest": dataset_meta["manifest"],
        "dataset_manifest_sha256": dataset_meta["manifest_sha256"],
        "source_hashes": _source_hashes(),
        "source_git": git,
        "dynamics": DYN.contract(),
        "architecture": policy.config(),
        "optimizer": {
            "name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup, "schedule": "warmup_then_cosine",
        },
        "sampler_mass": "gamma -> successful trajectory -> window",
        "promotion": "top ID validation checkpoints -> fixed raw temp=1 ID gate; OOD forbidden",
    }

    history, candidates = [], []
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        sampler = WeightedRandomSampler(
            weights, samples_per_epoch, replacement=True, generator=generator
        )
        loader = DataLoader(
            train, batch_size=args.batch, sampler=sampler, drop_last=True,
            num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
        )
        policy.train()
        lr_used = float(optimizer.param_groups[0]["lr"])
        losses = []
        for hp100, low, hist, controls, _episode, _gamma in loader:
            hp100 = hp100.to(args.device, non_blocking=True)
            low = low.to(args.device, non_blocking=True)
            hist = hist.to(args.device, non_blocking=True)
            controls = controls.to(args.device, non_blocking=True)
            loss = policy.cfm_loss(controls, policy.ctx_from(hp100, low, hist))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        macro, per_gamma = deterministic_validation(
            policy, val, SP.GAMMAS, args.device, args.val_batch
        )
        row = {
            "epoch": epoch + 1,
            "train_cfm": float(np.mean(losses)),
            "val_macro_cfm": macro,
            "val_per_gamma": per_gamma,
            "lr": lr_used,
        }
        history.append(row)
        if (epoch + 1) % args.ckpt_every == 0:
            path = outdir / f"ckpt_{epoch + 1}.pt"
            atomic_save(
                checkpoint_payload(policy, epoch + 1, history, args, dataset_meta, provenance),
                path,
            )
            candidates.append((macro, str(path)))
        print(json.dumps(row, sort_keys=True), flush=True)

    gates = []
    for validation_cfm, path in sorted(candidates)[:max(1, int(args.gate_top))]:
        candidate, _ = GPS.load_sfm_hp100_policy(path, device=args.device)
        result = id_raw_gate(
            candidate,
            M=int(args.gate_m),
            ep0=int(args.gate_episode_start),
            device=args.device,
        )
        # The imported hook is, by construction, the matched-ID-only entry
        # point.  Record that constraint explicitly in the promoted payload.
        result = {"distribution": "ID", **result}
        _validate_gate(result, SP.GAMMAS)
        gates.append({
            **result,
            "checkpoint": str(Path(path).resolve()),
            "checkpoint_sha256": sha256_file(path),
            "val_macro_cfm": float(validation_cfm),
        })
    selected = min(
        gates,
        key=lambda row: _gate_score(row, row["val_macro_cfm"], SP.GAMMAS),
    )
    promoted, _ = GPS.load_sfm_hp100_policy(selected["checkpoint"], device="cpu")
    Path(args.policy_out).resolve().parent.mkdir(parents=True, exist_ok=True)
    GPS.save_sfm_hp100_policy(
        promoted,
        args.policy_out,
        extra={
            "selected_by": "trajectory-disjoint ID validation + fixed raw temp=1 ID gate",
            "selected_gate": selected,
            "all_id_gates": gates,
            "dataset": dataset_meta,
            "provenance": provenance,
            "pretrained_from_scratch": True,
            "partial_transplant": False,
        },
    )
    report = {
        "status": (
            "HP100_PRETRAIN_SMOKE_COMPLETE" if args.smoke
            else "HP100_PRETRAIN_COMPLETE"
        ),
        "policy": str(Path(args.policy_out).resolve()),
        "policy_sha256": sha256_file(args.policy_out),
        "selected": selected,
        "gates": gates,
        "dataset": dataset_meta,
        "provenance": provenance,
    }
    report_path = outdir / "pretraining_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, report_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
