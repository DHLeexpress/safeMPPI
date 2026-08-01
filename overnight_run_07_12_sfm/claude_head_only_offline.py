"""Track D1: head-only offline CFM updates on already-acquired windows.

Hypothesis under test (project lead): a full-network CFM update with a few
hundred certified windows cannot move a 331k-parameter policy, but updating
*only* ``policy.head`` (``nn.Linear(256 -> 20)``, exactly 5,140 parameters)
with the visual encoder, low encoder, GRU, residual trunk and time embedding
frozen may extract usable signal from the same windows.  ``demo_frac`` mixes
pinned in-distribution pretraining windows into every microbatch as a
stabiliser.

Every arm starts from the same frozen pretrained checkpoint.  The acquired
windows all come from gathers collected *under that exact checkpoint*, so no
arm ever trains on data produced by an already-updated policy.

Populations
-----------
D+   executed, verifier-certified (y=1) windows            (ExecutedRoundShard)
G+   guided candidates certified (y=1) but never executed  (guided_positive)
D0   executed windows whose audited label is y=0 forever.  Training on D0 is a
     deliberate *toxicity probe*: it imitates windows the verifier refused,
     exactly the way a naive "train on everything you rolled" loop would.

Weighting keeps the canonical hierarchy (equal mass per gamma, then equal per
(round, scenario) cell, then equal per context, then equal per window), with
the per-microbatch weights renormalised to mean 1 so every optimiser step sees
a proper weighted mean rather than a shrinking share of a whole-buffer sum.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import claude_guided_positive_pilot as GPP
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as MR
import sfm_b1_neutral_teacher_sanity as NS
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_protocol as SP

STATUS = "CLAUDE_HEAD_ONLY_OFFLINE_COMPLETE"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = (
    "/home/dohyun/projects/sfm_hp10_b1_runs/103476d/pretrained_hp10.pt"
)
FASTLAB = "/data3/research1/claude_sfm_neutral_6c99a05/fastlab"
DEFAULT_GATHERS = (
    os.path.join(
        FASTLAB, "guided_positive/kfull_u50_topk2_P4/rounds/round_01/gather"
    ),
    os.path.join(
        FASTLAB, "guided_positive/kfull_x3_P4_v2/rounds/round_01/gather"
    ),
)
HEAD_PARAMETERS = 5_140
PROBE_SEED = 20_260_741
ID_PROBE_SEED = 20_260_742
DEMO_SEED = 20_260_743

# round labels keep (round, query_id) identities globally unique once the two
# gathers and the three populations are pooled into one training buffer.
ROUND_LABEL = {
    ("Dplus", 0): 11, ("Dplus", 1): 12,
    ("Gplus", 0): 21, ("Gplus", 1): 22,
    ("D0", 0): 31, ("D0", 1): 32,
}

ARMS = {
    "head_Dplus": dict(pools=("Dplus",), demo_frac=0.0),
    "head_Gplus": dict(pools=("Gplus",), demo_frac=0.0),
    "head_DG": dict(pools=("Dplus", "Gplus"), demo_frac=0.0),
    "head_DG_demo50": dict(pools=("Dplus", "Gplus"), demo_frac=0.5),
    "head_Dplus_demo50": dict(pools=("Dplus",), demo_frac=0.5),
    "head_D0": dict(pools=("D0",), demo_frac=0.0),
}
ARM_NOTES = {
    "head_D0": (
        "TOXICITY PROBE: D0 windows keep audit truth y=0 forever; this arm "
        "imitates them the way a naive collect-everything loop would."
    ),
}


# ---------------------------------------------------------------- data loading
def _relabel(holder, round_label):
    holder.round_i = int(round_label)
    for context in holder.contexts:
        context["round"] = int(round_label)
    return holder


def _load_population(gathers, population):
    """Return (records, per-gather diagnostics) for one population."""
    records = []
    sources = []
    for index, gather in enumerate(gathers):
        if population == "Dplus":
            path = os.path.join(gather, "executed_round.pt")
            shard = OS.ExecutedRoundShard.load(path)
            if shard.Dminus:
                raise RuntimeError("executed shard unexpectedly carries D-")
            holder = _relabel(shard, ROUND_LABEL[(population, index)])
            rows = OS.positive_records(holder)
        elif population == "Gplus":
            path = os.path.join(gather, RA.GUIDED_POSITIVE_STORE)
            holder, rows = GPP._guided_positive_records(path)
            _relabel(holder, ROUND_LABEL[(population, index)])
        elif population == "D0":
            path = os.path.join(gather, "neutral_round.pt")
            holder, rows = MR._neutral_records(path)
            _relabel(holder, ROUND_LABEL[(population, index)])
        else:
            raise ValueError(population)
        if not rows:
            raise RuntimeError(f"{population} empty at {path}")
        records.extend(rows)
        sources.append(dict(
            population=population,
            gather=os.path.abspath(gather),
            file=os.path.abspath(path),
            sha256=FA._sha256_file(path),
            round_label=int(ROUND_LABEL[(population, index)]),
            **GPP._population_summary(rows),
        ))
    return records, sources


def _gamma_of(holder, row):
    if "gamma" in row:
        return float(row["gamma"])
    return float(holder.contexts[int(row["context_id"])]["gamma"])


def _pool_summary(records):
    per_gamma = Counter(f"{_gamma_of(h, r):g}" for h, r in records)
    contexts = {
        (int(h.round_i), int(r["context_id"])) for h, r in records
    }
    return dict(
        windows=len(records),
        contexts=len(contexts),
        per_gamma={key: int(per_gamma[key]) for key in sorted(per_gamma)},
    )


# ------------------------------------------------------------------- ID demos
def _demo_bundle(checkpoint):
    preflight = MR._id_anchor_preflight(MR.DEFAULT_ID_ANCHOR_DATASET, 7)
    bundle = MR._id_anchor_banks(preflight, checkpoint)
    banks = bundle["banks"]
    # global gamma-balanced categorical over every retained ID train window
    offsets = np.cumsum([0] + [int(bank["windows"]) for bank in banks])
    probability = np.concatenate([
        np.asarray(bank["mass"], np.float64) / float(len(banks))
        for bank in banks
    ])
    total = float(probability.sum())
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError("ID demo distribution does not sum to one")
    return dict(
        banks=banks,
        offsets=offsets,
        probability=probability / total,
        manifest_sha256=bundle["manifest_sha256"],
        file_sha256=bundle["file_sha256"],
        split=bundle["split"],
        dataset=bundle["dataset"],
        windows=int(offsets[-1]),
    )


def _demo_draw(bundle, size, generator):
    flat = generator.choice(
        len(bundle["probability"]), size=int(size), replace=True,
        p=bundle["probability"],
    )
    offsets = bundle["offsets"]
    bank_index = np.searchsorted(offsets, flat, side="right") - 1
    row_index = flat - offsets[bank_index]
    return list(zip(bank_index.tolist(), row_index.tolist()))


def _demo_tensors(bundle, selection, device):
    banks = bundle["banks"]
    stack = lambda key: torch.stack(  # noqa: E731
        [banks[b][key][i] for b, i in selection]
    ).to(device)
    return stack("hp10"), stack("low5"), stack("hist"), stack("U")


def _demo_loss(policy, bundle, selection, device, seed):
    hp10, low5, hist, controls = _demo_tensors(bundle, selection, device)
    torch.manual_seed(int(seed))
    return policy.cfm_loss(controls, policy.ctx_from(hp10, low5, hist))


@torch.no_grad()
def _id_probe_loss(policy, bundle, selection, device, *, batch=128):
    """Deterministic mean CFM loss on the fixed ID-train probe."""
    was_training = policy.training
    policy.eval()
    total = 0.0
    for start in range(0, len(selection), int(batch)):
        chunk = selection[start:start + int(batch)]
        loss = _demo_loss(
            policy, bundle, chunk, device, ID_PROBE_SEED + start,
        )
        total += float(loss) * (len(chunk) / float(len(selection)))
    if was_training:
        policy.train()
    return float(total)


# ------------------------------------------------------------------- training
def _chunk_loss(policy, chunk, mass, device, seed):
    grid, low, hist, controls = BS._tensor_batch(chunk, device)
    context = policy.ctx_from(grid, low, hist)
    weights = torch.as_tensor(
        [float(mass[(id(holder), int(row["query_id"]))]) for holder, row in chunk],
        dtype=controls.dtype, device=device,
    )
    weights = weights * (len(chunk) / weights.sum())
    torch.manual_seed(int(seed))
    loss = policy.cfm_loss(controls, context, weights=weights)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite head-only CFM loss")
    return loss


def _freeze_head_only(policy):
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.head.parameters():
        parameter.requires_grad_(True)
    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        (name, int(parameter.numel()))
        for name, parameter in policy.named_parameters()
        if not parameter.requires_grad
    ]
    trainable_total = sum(value for _, value in trainable)
    frozen_total = sum(value for _, value in frozen)
    names = sorted(name for name, _ in trainable)
    if names != ["head.bias", "head.weight"]:
        raise RuntimeError(f"unexpected trainable parameters: {names}")
    if trainable_total != HEAD_PARAMETERS:
        raise RuntimeError(
            f"head-only trainable count {trainable_total} != {HEAD_PARAMETERS}"
        )
    report = dict(
        trainable_parameters=trainable_total,
        frozen_parameters=frozen_total,
        total_parameters=trainable_total + frozen_total,
        trainable_tensors=[dict(name=n, numel=v) for n, v in trainable],
        frozen_tensor_names=[n for n, _ in frozen],
        head_module=str(policy.head),
    )
    print(
        f"[freeze] trainable {trainable_total} ({names}) | frozen "
        f"{frozen_total} over {len(frozen)} tensors | total "
        f"{trainable_total + frozen_total}"
    )
    return report


def _module_digests(policy):
    digests = {}
    for name, parameter in policy.named_parameters():
        digests[name] = hashlib.sha256(
            parameter.detach().to("cpu", torch.float32).numpy().tobytes()
        ).hexdigest()
    return digests


def run(args):
    started = time.perf_counter()
    arm = str(args.arm)
    spec = ARMS[arm]
    demo_frac = float(spec["demo_frac"])
    device = str(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = os.path.abspath(args.checkpoint)
    source_sha = FA._sha256_file(checkpoint)
    if source_sha != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source pretrained checkpoint SHA changed")

    gathers = [os.path.abspath(path) for path in args.gathers]
    pools = {}
    sources = []
    for population in ("Dplus", "Gplus", "D0"):
        if population in spec["pools"]:
            records, meta = _load_population(gathers, population)
            pools[population] = records
            sources.extend(meta)
    records = [row for population in spec["pools"] for row in pools[population]]
    mass, accounting = BS.hierarchy_mass(records)
    identities = [(int(h.round_i), int(r["query_id"])) for h, r in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("pooled records collide on (round, query_id)")

    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    freeze = _freeze_head_only(policy)
    digests_before = _module_digests(policy)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=float(args.lr)
    )

    bundle = None
    demo_meta = None
    if demo_frac > 0.0 or args.id_probe_windows > 0:
        bundle = _demo_bundle(checkpoint)
        demo_meta = dict(
            dataset=bundle["dataset"],
            manifest_sha256=bundle["manifest_sha256"],
            file_sha256=bundle["file_sha256"],
            split=bundle["split"],
            train_windows=int(bundle["windows"]),
        )
    probe_selection = []
    if args.id_probe_windows > 0:
        probe_selection = _demo_draw(
            bundle, int(args.id_probe_windows),
            np.random.default_rng(ID_PROBE_SEED),
        )

    batch = int(args.batch)
    n_demo = int(round(batch * demo_frac))
    n_arm = batch - n_demo
    if n_arm < 1:
        raise ValueError("demo_frac leaves no room for arm windows")

    policy.eval()
    pool_losses = [NS._fixed_loss(
        policy, records, batch=batch, device=device, seed=PROBE_SEED,
    )]
    id_probe = (
        [_id_probe_loss(policy, bundle, probe_selection, device)]
        if probe_selection else []
    )
    print(
        f"[{arm}] pool={len(records)} windows, batch={batch} "
        f"(arm {n_arm} + demo {n_demo}), pool loss {pool_losses[0]:.5f}"
        + (f", ID probe {id_probe[0]:.5f}" if id_probe else "")
    )

    epoch_train_losses = []
    epoch_demo_losses = []
    demo_generator = np.random.default_rng(int(args.seed) + DEMO_SEED)
    for epoch in range(int(args.epochs)):
        order = np.random.default_rng(
            int(args.seed) + 1_000_003 * (epoch + 1)
        ).permutation(len(records))
        policy.train()
        batch_losses = []
        demo_losses = []
        for step, start in enumerate(range(0, len(order), n_arm)):
            chunk = [records[int(i)] for i in order[start:start + n_arm]]
            if len(chunk) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            seed = int(args.seed) + 7919 * step + 1_000_003 * (epoch + 1)
            arm_loss = _chunk_loss(policy, chunk, mass, device, seed)
            loss = arm_loss
            if n_demo:
                selection = _demo_draw(bundle, n_demo, demo_generator)
                demo_loss = _demo_loss(
                    policy, bundle, selection, device, seed + 104_729,
                )
                loss = (1.0 - demo_frac) * arm_loss + demo_frac * demo_loss
                demo_losses.append(float(demo_loss.detach()))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(arm_loss.detach()))
        policy.eval()
        epoch_train_losses.append(float(np.mean(batch_losses)))
        if demo_losses:
            epoch_demo_losses.append(float(np.mean(demo_losses)))
        pool_losses.append(NS._fixed_loss(
            policy, records, batch=batch, device=device, seed=PROBE_SEED,
        ))
        if probe_selection:
            id_probe.append(
                _id_probe_loss(policy, bundle, probe_selection, device)
            )
        print(
            f"[{arm}] epoch {epoch + 1:2d} train {epoch_train_losses[-1]:.5f} "
            f"pool {pool_losses[-1]:.5f}"
            + (f" ID {id_probe[-1]:.5f}" if probe_selection else "")
            + (f" demo {epoch_demo_losses[-1]:.5f}" if demo_losses else "")
        )

    digests_after = _module_digests(policy)
    changed = sorted(
        name for name in digests_before
        if digests_before[name] != digests_after[name]
    )
    if changed != ["head.bias", "head.weight"]:
        raise RuntimeError(f"parameters outside the head changed: {changed}")

    out_checkpoint = os.path.join(output_dir, f"{arm}.pt")
    BX._save_checkpoint(policy, out_checkpoint, {
        "study": STATUS,
        "arm": arm,
        "round": 1,
        "phase": "head_only",
        "source_checkpoint": checkpoint,
        "source_sha256": source_sha,
    })

    payload = dict(
        status=STATUS,
        arm=arm,
        note=ARM_NOTES.get(arm, ""),
        source=FA._source(),
        config=dict(
            checkpoint=checkpoint,
            checkpoint_sha256=source_sha,
            gathers=gathers,
            pools=list(spec["pools"]),
            demo_frac=demo_frac,
            batch=batch,
            arm_windows_per_batch=n_arm,
            demo_windows_per_batch=n_demo,
            epochs=int(args.epochs),
            lr=float(args.lr),
            seed=int(args.seed),
            device=device,
            optimizer="Adam",
            id_probe_windows=int(args.id_probe_windows),
            weighting=(
                "canonical hierarchy (gamma -> cell -> context -> window), "
                "renormalised to mean 1 within each microbatch"
            ),
            dropout_mode="policy.train() during updates (canonical)",
        ),
        freeze=freeze,
        sources=sources,
        pool=_pool_summary(records),
        pool_by_population={
            key: _pool_summary(value) for key, value in pools.items()
        },
        mass=MR._compact_mass(accounting),
        id_demo=demo_meta,
        losses=dict(
            train_pool_mean_per_epoch=epoch_train_losses,
            demo_mean_per_epoch=epoch_demo_losses,
            fixed_pool_loss_by_epoch=pool_losses,
            id_train_probe_loss_by_epoch=id_probe,
        ),
        parameter_digests=dict(
            changed=changed,
            head_weight_before=digests_before["head.weight"],
            head_weight_after=digests_after["head.weight"],
            unchanged_tensors=len(digests_before) - len(changed),
        ),
        outputs=dict(
            checkpoint=out_checkpoint,
            checkpoint_sha256=FA._sha256_file(out_checkpoint),
        ),
        seconds=float(time.perf_counter() - started),
    )
    path = os.path.join(output_dir, f"{arm}_train.json")
    FA._write_json(path, payload)
    print(path)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gathers", nargs="+", default=list(DEFAULT_GATHERS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_744)
    parser.add_argument("--id-probe-windows", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
