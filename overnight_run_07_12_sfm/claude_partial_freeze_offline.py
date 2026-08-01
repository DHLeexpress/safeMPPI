"""Track E2: freeze-DEPTH sweep with aggressive multi-step offline training.

Track D1 (``claude_head_only_offline.py``) established that a 5,140-parameter
head-only update *fits* the acquired windows without moving closed-loop CR/SR,
and that a full-network update with the same few hundred windows is inert.
Three axes were never separated in that grid, and this track separates them:

depth
    Four nested trainable scopes between "only the flow read-out" and the
    canonical expansion trainability (everything except the frozen visual
    encoder ``enc_grid``)::

        S1  head.*                                                     5,140
        S2  head.* + trunk.blocks.1.*                                137,236
        S3  S2 + trunk.blocks.0.* + trunk.inp.*                      311,572
        S4  everything except enc_grid.*  (canonical)                317,060

dose
    Canonical expansion took ONE accumulated Adam step per round.  Every arm
    here takes an ``optimizer.step()`` on EVERY microbatch for 20 epochs over
    its pool (~25 microbatches/epoch => ~500 steps), i.e. two to three orders
    of magnitude more optimiser work on the same frozen windows.

dedup
    G+ is multi-modal per context: 929 certified guided windows over 491
    ``(scenario, gamma, step)`` contexts, and averaging conflicting modes
    through a linear head was actively harmful in Track D (CR .286 -> .471).
    ``Gplus_top1`` keeps exactly one window per context -- the locked Kazuki
    controller's own rank-0 refined elite, which is the plan it would have
    executed (``candidate_id = 2K + rank``; see the ranking contract in
    ``sfm_b1_kazuki_repair.locked_full_kazuki_plans``).  The ``mode`` field
    carried by the records is a left/right *behaviour* label, not a rank, so
    it cannot order candidates and the candidate id is used instead.

demo
    ``demo_frac`` mixes pinned in-distribution pretraining windows (ID-anchor
    train split, pinned manifest + per-file digests) into every microbatch.

Every arm starts from the same frozen pretrained checkpoint, and every window
was collected under that exact checkpoint, so no arm trains on data produced
by an already-updated policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
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

STATUS = "CLAUDE_PARTIAL_FREEZE_OFFLINE_COMPLETE"
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
PROBE_SEED = 20_260_741
ID_PROBE_SEED = 20_260_742
DEMO_SEED = 20_260_743
DEFAULT_SEED = 20_260_745

# round labels keep (round, query_id) identities globally unique once the two
# gathers and the populations are pooled into one training buffer.
ROUND_LABEL = {
    ("Dplus", 0): 11, ("Dplus", 1): 12,
    ("Gplus", 0): 21, ("Gplus", 1): 22,
}

# ---------------------------------------------------------------- freeze scopes
# Each scope freezes EVERY parameter and then unfreezes exactly the tensors
# whose names start with one of these prefixes.  S4 is expressed through the
# canonical helper so it is trainability-identical to a real expansion round.
SCOPES = {
    "S1": dict(
        prefixes=("head.",),
        expect=5_140,
        label="head",
    ),
    "S2": dict(
        prefixes=("head.", "trunk.blocks.1."),
        expect=137_236,
        label="+block1",
    ),
    "S3": dict(
        prefixes=("head.", "trunk.blocks.1.", "trunk.blocks.0.", "trunk.inp."),
        expect=311_572,
        label="+trunk",
    ),
    "S4": dict(
        prefixes=None,  # canonical: everything except enc_grid.*
        expect=317_060,
        label="full-canonical",
    ),
}
SCOPE_LR = {"S1": 1.0e-3, "S2": 3.0e-4, "S3": 1.0e-4, "S4": 1.0e-4}

ARMS = {
    "E2a": dict(scope="S1", pools=("Dplus", "Gplus_top1"), demo_frac=0.5),
    "E2b": dict(scope="S2", pools=("Dplus", "Gplus_top1"), demo_frac=0.5),
    "E2c": dict(scope="S3", pools=("Dplus", "Gplus_top1"), demo_frac=0.5),
    "E2d": dict(scope="S4", pools=("Dplus", "Gplus_top1"), demo_frac=0.5),
    "E2e": dict(scope="S2", pools=("Dplus", "Gplus"), demo_frac=0.5),
    "E2f": dict(scope="S2", pools=("Dplus", "Gplus_top1"), demo_frac=0.0),
}
ARM_NOTES = {
    "E2d": "aggressive full-net (canonical trainability) with demo50 + dedup",
    "E2e": "mode-dedup ablation: identical to E2b but every G+ mode kept",
    "E2f": "demo ablation: identical to E2b but no ID-demo mixing",
}
ARM_ORDER = ["E2a", "E2b", "E2c", "E2d", "E2e", "E2f"]


# ---------------------------------------------------------------- data loading
def _relabel(holder, round_label):
    holder.round_i = int(round_label)
    for context in holder.contexts:
        context["round"] = int(round_label)
    return holder


def _guided_records_with_candidates(path):
    """``GPP._guided_positive_records`` plus the raw candidate id / mode.

    The pilot's replay rows deliberately drop ``candidate_id`` and ``mode``
    (they are not needed for replay), but the mode-dedup needs the locked
    controller's ranking.  ``guided_id`` is the payload record index, so the
    two are re-joined positionally after the pilot's own authentication has
    already passed.
    """
    holder, records = GPP._guided_positive_records(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sources = payload["records"]
    if len(sources) != len(records):
        raise RuntimeError("guided payload/record count mismatch")
    for index, (_, row) in enumerate(records):
        source = sources[index]
        if int(source["guided_id"]) != index or int(row["query_id"]) != index:
            raise RuntimeError("guided_id is not the payload record index")
        row["candidate_id"] = int(source["candidate_id"])
        row["mode"] = str(source["mode"])
    return holder, records


def _dedup_top1(records):
    """Keep one window per ``(round, context_id)``: lowest ``candidate_id``.

    ``candidate_id = 2K + rank`` where ``rank`` indexes the locked Kazuki
    controller's own cost ordering over its refined elite modes, so rank 0 --
    the lowest candidate id -- is the plan the controller would have executed.
    Ties (never observed) fall back to the lowest ``query_id``.
    """
    best = {}
    modes = defaultdict(set)
    per_context = Counter()
    for holder, row in records:
        key = (int(holder.round_i), int(row["context_id"]))
        modes[key].add(str(row["mode"]))
        per_context[key] += 1
        rank = (int(row["candidate_id"]), int(row["query_id"]))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, (holder, row))
    kept = [
        pair for _, pair in sorted(
            (
                (int(holder.round_i), int(row["query_id"])), (holder, row)
            )
            for _, (holder, row) in best.values()
        )
    ]
    diagnostics = dict(
        windows_before=len(records),
        windows_after=len(kept),
        contexts=len(best),
        windows_per_context_histogram={
            str(key): int(value)
            for key, value in sorted(Counter(per_context.values()).items())
        },
        contexts_with_multiple_windows=int(sum(
            1 for value in per_context.values() if value > 1
        )),
        contexts_with_conflicting_modes=int(sum(
            1 for value in modes.values() if len(value) > 1
        )),
        candidate_id_before=dict(sorted(Counter(
            int(row["candidate_id"]) for _, row in records
        ).items())),
        candidate_id_after=dict(sorted(Counter(
            int(row["candidate_id"]) for _, row in kept
        ).items())),
        mode_before=dict(sorted(Counter(
            str(row["mode"]) for _, row in records
        ).items())),
        mode_after=dict(sorted(Counter(
            str(row["mode"]) for _, row in kept
        ).items())),
        mode_semantics=(
            "left/right behaviour label emitted by the locked controller, not "
            "a rank; dedup therefore orders by candidate_id = 2K + elite "
            "rank, whose rank 0 is the plan the controller would execute"
        ),
    )
    return kept, diagnostics


def _load_population(gathers, population):
    """Return (records, per-gather diagnostics, dedup) for one population."""
    records = []
    sources = []
    dedup = None
    for index, gather in enumerate(gathers):
        if population == "Dplus":
            path = os.path.join(gather, "executed_round.pt")
            shard = OS.ExecutedRoundShard.load(path)
            if shard.Dminus:
                raise RuntimeError("executed shard unexpectedly carries D-")
            holder = _relabel(shard, ROUND_LABEL[(population, index)])
            rows = OS.positive_records(holder)
        elif population in ("Gplus", "Gplus_top1"):
            path = os.path.join(gather, RA.GUIDED_POSITIVE_STORE)
            holder, rows = _guided_records_with_candidates(path)
            _relabel(holder, ROUND_LABEL[("Gplus", index)])
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
            round_label=int(ROUND_LABEL[
                (("Gplus" if population.startswith("Gplus") else population),
                 index)
            ]),
            **GPP._population_summary(rows),
        ))
    if population == "Gplus_top1":
        records, dedup = _dedup_top1(records)
    return records, sources, dedup


def _gamma_of(holder, row):
    if "gamma" in row:
        return float(row["gamma"])
    return float(holder.contexts[int(row["context_id"])]["gamma"])


def _pool_summary(records):
    per_gamma = Counter(f"{_gamma_of(h, r):g}" for h, r in records)
    contexts = {(int(h.round_i), int(r["context_id"])) for h, r in records}
    return dict(
        windows=len(records),
        contexts=len(contexts),
        per_gamma={key: int(per_gamma[key]) for key in sorted(per_gamma)},
    )


# ------------------------------------------------------------------- ID demos
def _demo_bundle(checkpoint):
    """Pinned ID-anchor TRAIN banks (identical machinery to Track D)."""
    preflight = MR._id_anchor_preflight(MR.DEFAULT_ID_ANCHOR_DATASET, 7)
    bundle = MR._id_anchor_banks(preflight, checkpoint)
    banks = bundle["banks"]
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
        loss = _demo_loss(policy, bundle, chunk, device, ID_PROBE_SEED + start)
        total += float(loss) * (len(chunk) / float(len(selection)))
    if was_training:
        policy.train()
    return float(total)


# ------------------------------------------------------------------- training
def _chunk_loss(policy, chunk, mass, device, seed):
    grid, low, hist, controls = BS._tensor_batch(chunk, device)
    context = policy.ctx_from(grid, low, hist)
    weights = torch.as_tensor(
        [float(mass[(id(holder), int(row["query_id"]))])
         for holder, row in chunk],
        dtype=controls.dtype, device=device,
    )
    weights = weights * (len(chunk) / weights.sum())
    torch.manual_seed(int(seed))
    loss = policy.cfm_loss(controls, context, weights=weights)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite CFM loss")
    return loss


def _configure_scope(policy, scope):
    """Freeze everything, then unfreeze exactly the scope's tensors."""
    spec = SCOPES[str(scope)]
    if spec["prefixes"] is None:
        BS.configure_expansion_trainability(policy)
    else:
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        for name, parameter in policy.named_parameters():
            if name.startswith(tuple(spec["prefixes"])):
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
    if trainable_total != int(spec["expect"]):
        raise RuntimeError(
            f"scope {scope} trains {trainable_total} parameters, "
            f"not the pre-registered {spec['expect']}"
        )
    if scope == "S1" and trainable_total != 5_140:
        raise RuntimeError("S1 is not the 5,140-parameter head")
    if scope == "S4":
        if trainable_total != 317_060:
            raise RuntimeError("S4 is not the canonical 317,060")
        if sorted(name for name, _ in frozen) != sorted(
            f"enc_grid.{name}" for name, _ in policy.enc_grid.named_parameters()
        ):
            raise RuntimeError("S4 froze something other than enc_grid")
    report = dict(
        scope=str(scope),
        scope_label=str(spec["label"]),
        trainable_parameters=trainable_total,
        frozen_parameters=frozen_total,
        total_parameters=trainable_total + frozen_total,
        trainable_tensors=[dict(name=n, numel=v) for n, v in trainable],
        trainable_tensor_names=[n for n, _ in trainable],
        frozen_tensor_names=[n for n, _ in frozen],
    )
    print(
        f"[scope {scope} {spec['label']}] trainable {trainable_total} over "
        f"{len(trainable)} tensors | frozen {frozen_total} over "
        f"{len(frozen)} tensors | total {trainable_total + frozen_total}",
        flush=True,
    )
    return report


def _module_digests(policy):
    return {
        name: hashlib.sha256(
            parameter.detach().to("cpu", torch.float32).numpy().tobytes()
        ).hexdigest()
        for name, parameter in policy.named_parameters()
    }


def run_train(args):
    started = time.perf_counter()
    arm = str(args.arm)
    spec = ARMS[arm]
    scope = str(spec["scope"])
    demo_frac = float(
        spec["demo_frac"] if args.demo_frac is None else args.demo_frac
    )
    lr = float(SCOPE_LR[scope] if args.lr is None else args.lr)
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
    dedup = None
    for population in spec["pools"]:
        records, meta, population_dedup = _load_population(gathers, population)
        pools[population] = records
        sources.extend(meta)
        if population_dedup is not None:
            dedup = population_dedup
            print(
                f"[dedup] G+ {population_dedup['windows_before']} windows -> "
                f"{population_dedup['windows_after']} over "
                f"{population_dedup['contexts']} contexts "
                f"(candidate ids kept {population_dedup['candidate_id_after']})",
                flush=True,
            )
    records = [row for population in spec["pools"] for row in pools[population]]
    mass, accounting = BS.hierarchy_mass(records)
    identities = [(int(h.round_i), int(r["query_id"])) for h, r in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("pooled records collide on (round, query_id)")

    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    freeze = _configure_scope(policy, scope)
    digests_before = _module_digests(policy)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
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
    n_demo = int(math.floor(batch * demo_frac))
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
    steps_per_epoch = len([
        start for start in range(0, len(records), n_arm)
        if len(records) - start >= 2
    ])
    print(
        f"[{arm}] scope={scope} lr={lr:g} pool={len(records)} windows "
        f"batch={batch} (arm {n_arm} + demo {n_demo}) "
        f"steps/epoch={steps_per_epoch} epochs={args.epochs} "
        f"total_steps={steps_per_epoch * int(args.epochs)} "
        f"pool loss {pool_losses[0]:.5f}"
        + (f" ID probe {id_probe[0]:.5f}" if id_probe else ""),
        flush=True,
    )

    epoch_train_losses = []
    epoch_demo_losses = []
    demo_generator = np.random.default_rng(int(args.seed) + DEMO_SEED)
    total_steps = 0
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
            # aggressive dose: one Adam step per microbatch, not one per round
            optimizer.step()
            total_steps += 1
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
            + (f" demo {epoch_demo_losses[-1]:.5f}" if demo_losses else ""),
            flush=True,
        )

    digests_after = _module_digests(policy)
    changed = sorted(
        name for name in digests_before
        if digests_before[name] != digests_after[name]
    )
    trainable_names = set(freeze["trainable_tensor_names"])
    leaked = sorted(set(changed) - trainable_names)
    if leaked:
        raise RuntimeError(f"parameters outside the scope changed: {leaked}")

    out_checkpoint = os.path.join(output_dir, f"{arm}.pt")
    BX._save_checkpoint(policy, out_checkpoint, {
        "study": STATUS,
        "arm": arm,
        "round": 1,
        "phase": "partial_freeze",
        "scope": scope,
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
            scope=scope,
            scope_label=SCOPES[scope]["label"],
            pools=list(spec["pools"]),
            demo_frac=demo_frac,
            batch=batch,
            arm_windows_per_batch=n_arm,
            demo_windows_per_batch=n_demo,
            epochs=int(args.epochs),
            steps_per_epoch=steps_per_epoch,
            optimizer_steps=int(total_steps),
            lr=lr,
            seed=int(args.seed),
            device=device,
            optimizer="Adam",
            stepping="one optimizer.step() per microbatch (aggressive dose)",
            id_probe_windows=int(args.id_probe_windows),
            weighting=(
                "canonical hierarchy (gamma -> cell -> context -> window), "
                "renormalised to mean 1 within each microbatch"
            ),
            dropout_mode="policy.train() during updates (canonical)",
        ),
        freeze=freeze,
        sources=sources,
        dedup=dedup,
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
            changed_count=len(changed),
            trainable_tensor_count=len(trainable_names),
            unchanged_trainable=sorted(trainable_names - set(changed)),
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
    print(path, flush=True)
    return payload


# --------------------------------------------------------------------- report
METRICS = ("SR", "CR", "Validity", "successful_clearance",
           "successful_time_to_goal")
DELTA_METRICS = ("SR", "CR", "Validity", "successful_clearance",
                 "successful_time_to_goal")


def _cell_metrics(cell):
    def _scalar(key):
        value = cell[key]
        if isinstance(value, dict):
            value = value.get("mean")
        return float("nan") if value is None else float(value)
    return dict(
        n=int(cell["n"]),
        SR=_scalar("SR"),
        CR=_scalar("CR"),
        timeout=_scalar("timeout"),
        Validity=_scalar("Validity"),
        successful_clearance=_scalar("successful_clearance"),
        successful_time_to_goal=_scalar("successful_time_to_goal"),
    )


def _read_eval(path):
    with open(path) as stream:
        payload = json.load(stream)
    out = {}
    for record in payload["records"]:
        summary = record["cell"]["summary"]
        out[str(record["label"])] = dict(
            checkpoint=record["cell"]["checkpoint"],
            checkpoint_sha256=record["cell"]["checkpoint_sha256"],
            pooled=_cell_metrics(summary["pooled"]),
            per_gamma={
                key: _cell_metrics(value)
                for key, value in summary["per_gamma"].items()
            },
            rows={
                (round(float(row["gamma"]), 8), int(row["episode"])): row
                for row in record["cell"]["rows"]
            },
        )
    return out


def _exact_mcnemar(b, c):
    """Two-sided exact McNemar p-value on the discordant pairs (b, c)."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
    return float(min(1.0, 2.0 * tail))


def _paired(r0_rows, r1_rows, *, seed, draws=4000):
    """CRN-paired comparison: the same (gamma, episode) latent in both cells."""
    keys = sorted(set(r0_rows) & set(r1_rows))
    if len(keys) != len(r0_rows) or len(keys) != len(r1_rows):
        raise RuntimeError("paired evaluation cells do not share episodes")
    success_b = sum(
        1 for k in keys
        if r0_rows[k]["success"] and not r1_rows[k]["success"]
    )
    success_c = sum(
        1 for k in keys
        if not r0_rows[k]["success"] and r1_rows[k]["success"]
    )
    collision_b = sum(
        1 for k in keys
        if not r0_rows[k]["collision"] and r1_rows[k]["collision"]
    )
    collision_c = sum(
        1 for k in keys
        if r0_rows[k]["collision"] and not r1_rows[k]["collision"]
    )
    validity = np.array(
        [float(r1_rows[k]["validity"]) - float(r0_rows[k]["validity"])
         for k in keys]
    )
    generator = np.random.default_rng(int(seed))
    index = generator.integers(0, len(keys), size=(int(draws), len(keys)))
    boot = validity[index].mean(axis=1)
    return dict(
        episodes=len(keys),
        SR_lost=int(success_b),
        SR_gained=int(success_c),
        SR_mcnemar_p=_exact_mcnemar(success_b, success_c),
        CR_gained=int(collision_b),
        CR_avoided=int(collision_c),
        CR_mcnemar_p=_exact_mcnemar(collision_b, collision_c),
        Validity_paired_mean=float(validity.mean()),
        Validity_paired_bootstrap95=[
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ],
    )


def run_report(args):
    root = os.path.abspath(args.root)
    figures = os.path.join(root, "figs")
    os.makedirs(figures, exist_ok=True)
    arms = {}
    r0 = None
    for arm in ARM_ORDER:
        metrics = os.path.join(
            root, arm, "eval", "raw_m20_offline_metrics.json"
        )
        if not os.path.isfile(metrics):
            print(f"[report] missing {metrics}", flush=True)
            continue
        cells = _read_eval(metrics)
        if r0 is None:
            r0 = cells["r0"]
        elif r0["checkpoint_sha256"] != cells["r0"]["checkpoint_sha256"]:
            raise RuntimeError("arms disagree on the r0 reference cell")
        train = os.path.join(root, arm, f"{arm}_train.json")
        with open(train) as stream:
            train_payload = json.load(stream)
        arms[arm] = dict(
            train=train_payload,
            r1=cells["r1"],
            paired=_paired(
                cells["r0"]["rows"], cells["r1"]["rows"],
                seed=PROBE_SEED + int(
                    hashlib.sha256(arm.encode()).hexdigest()[:8], 16
                ) % 100_000,
            ),
            delta={
                key: cells["r1"]["pooled"][key] - r0["pooled"][key]
                for key in DELTA_METRICS
            },
            delta_per_gamma={
                gamma: {
                    key: cells["r1"]["per_gamma"][gamma][key]
                    - r0["per_gamma"][gamma][key]
                    for key in DELTA_METRICS
                }
                for gamma in cells["r1"]["per_gamma"]
            },
        )
    if not arms:
        raise RuntimeError("no arm evaluations found")

    summary = dict(
        status="CLAUDE_PARTIAL_FREEZE_REPORT_COMPLETE",
        source=FA._source(),
        r0=dict(
            checkpoint=r0["checkpoint"],
            checkpoint_sha256=r0["checkpoint_sha256"],
            pooled=r0["pooled"],
            per_gamma=r0["per_gamma"],
        ),
        arms={
            arm: dict(
                scope=value["train"]["config"]["scope"],
                scope_label=value["train"]["config"]["scope_label"],
                pools=value["train"]["config"]["pools"],
                demo_frac=value["train"]["config"]["demo_frac"],
                lr=value["train"]["config"]["lr"],
                trainable_parameters=value["train"]["freeze"][
                    "trainable_parameters"],
                optimizer_steps=value["train"]["config"]["optimizer_steps"],
                pool_windows=value["train"]["pool"]["windows"],
                checkpoint=value["train"]["outputs"]["checkpoint"],
                pooled=value["r1"]["pooled"],
                per_gamma=value["r1"]["per_gamma"],
                delta=value["delta"],
                delta_per_gamma=value["delta_per_gamma"],
                paired=value["paired"],
                losses=value["train"]["losses"],
            )
            for arm, value in arms.items()
        },
    )
    path = os.path.join(root, "trackE_summary.json")
    FA._write_json(path, summary)
    print(path, flush=True)

    _figure(summary, figures, args.figure_copies)
    return summary


def _figure(summary, figures, copies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [arm for arm in ARM_ORDER if arm in summary["arms"]]
    metrics = [
        ("SR", "SR"),
        ("CR", "CR"),
        ("Validity", "Validity"),
        ("successful_clearance", "clearance (m)"),
        ("successful_time_to_goal", "time to goal (s)"),
    ]
    colors = ["#2f6fb0", "#3f8f5f", "#b06f2f", "#8f4f8f", "#3f8f8f", "#a03f3f"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17.5, 4.3))
    width = 0.8
    for axis, (key, label) in zip(axes, metrics):
        values = [summary["arms"][arm]["delta"][key] for arm in names]
        positions = np.arange(len(names))
        axis.bar(positions, values, width=width,
                 color=[colors[i % len(colors)] for i in range(len(names))],
                 edgecolor="black", linewidth=0.6)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(positions)
        labels = [
            f"{arm}\n{summary['arms'][arm]['scope']}"
            for arm in names
        ]
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_title(f"Δ {label} vs r0", fontsize=10)
        span = max(abs(min(values)), abs(max(values)), 1e-6)
        axis.set_ylim(-1.35 * span, 1.35 * span)
        for position, value in zip(positions, values):
            axis.annotate(
                f"{value:+.3f}",
                (position, value),
                textcoords="offset points",
                xytext=(0, 4 if value >= 0 else -12),
                ha="center", fontsize=7,
            )
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    reference = summary["r0"]["pooled"]
    fig.suptitle(
        "Track E2 freeze-depth sweep: paired raw temp-1 M20 deltas vs the "
        "pretrained policy r0 "
        f"(SR {reference['SR']:.3f} / CR {reference['CR']:.3f} / "
        f"Val {reference['Validity']:.3f}); "
        "S1 head 5,140 | S2 +block1 137,236 | S3 +trunk 311,572 | "
        "S4 canonical 317,060",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    written = []
    for directory in [figures, *copies]:
        os.makedirs(directory, exist_ok=True)
        for extension in ("png", "pdf"):
            target = os.path.join(
                directory, f"trackE_partial_freeze_deltas.{extension}"
            )
            fig.savefig(target, dpi=170)
            written.append(target)
    plt.close(fig)
    for target in written:
        print(target, flush=True)
    return written


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--arm", required=True, choices=ARM_ORDER)
    train.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train.add_argument("--gathers", nargs="+", default=list(DEFAULT_GATHERS))
    train.add_argument("--output-dir", required=True)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--demo-frac", type=float, default=None)
    train.add_argument("--batch", type=int, default=128)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--id-probe-windows", type=int, default=1024)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=run_train)

    report = sub.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--figure-copies", nargs="*", default=[])
    report.set_defaults(func=run_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
