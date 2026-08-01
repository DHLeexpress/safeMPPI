"""Track F1: multi-positive "smooth CFM" -- the repair for the single-Dirac blind spot.

Every offline arm to date (Track D head-only, Track E freeze-depth x dose x
dedup x demo) trained conditional flow matching against **one** executed control
window per context.  That is a Dirac target: at a context where the exact
verifier certified four different H=10 windows, canonical CFM asks the flow to
put all of its probability on the single one that happened to be executed, and
the other certified modes are treated as if they were negatives.  Two axes were
never separated from that choice, and this track separates them:

target set (axis A)
    Per ``(scenario, gamma, step)`` context the certified positives form a SET,
    not a point.  ``BASE`` is the set of the uncertainty-tilted B=4 acquisition
    queries that the exact verifier certified (``candidate_id < 16``, ``y == 1``,
    ``full_h``) -- the policy's own proposals, including the executed one.
    ``TEACHER`` adds the G+ windows harvested from the locked external Kazuki
    controller at the same context.  Per *exposure* of a context the CFM target
    is drawn uniformly from its set (redrawn every epoch, seeded), so the
    objective is a stochastic-target estimator of the CFM loss against the
    empirical *mixture* over certified modes rather than against one atom.
    This is deliberately NOT the Track D ``G+`` pooling that averaged conflicting
    modes inside one microbatch: the flow never sees two conflicting targets for
    one context in the same gradient evaluation, so nothing is mode-averaged.

exposure smoothing (axis B)
    Canonical CFM draws ONE ``(x0, tau)`` pair per record per pass.  ``K``
    draws per exposure, averaged inside the loss, leave the expected gradient
    unchanged and divide the CFM sampling variance by ``K``.  ``K = 1`` is the
    canonical noise level.

hierarchy mass
    Mass is equal PER CONTEXT (gamma -> (round, scenario) -> context), so a
    context with six certified windows carries exactly the same total mass as a
    context with one.  Set size therefore changes *what* the target is, never
    *how loud* the context is -- without this, "more positives" would silently
    become "more dose", which Track E already showed is harmful.

Everything else is held at the Track E2b operating point (scope S2 = head +
trunk.blocks.1, 137,236 parameters, lr 3e-4, batch 128, 20 epochs, one Adam
step per microbatch, demo_frac 0.5 ID-anchor mixing) so the only moving parts
are the two axes above.  Demo windows are single-target and always K=1.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import claude_guided_positive_pilot as GPP
import claude_partial_freeze_offline as PF
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as MR  # noqa: F401  (imported by PF helpers)
import sfm_b1_offline_store as OS
import sfm_b1_store as BS

STATUS = "CLAUDE_MULTIPOS_OFFLINE_COMPLETE"
REPORT_STATUS = "CLAUDE_MULTIPOS_REPORT_COMPLETE"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = PF.DEFAULT_CHECKPOINT
FASTLAB = PF.FASTLAB
DEFAULT_ROOT = os.path.join(FASTLAB, "multipos")
EVAL_CACHE = os.path.join(FASTLAB, "head_only", "eval_cache")
R0_CHECKPOINT = os.path.join(
    FASTLAB, "guided_positive/kfull_x3_P4_v2/checkpoints/round_00.pt"
)
TRACK_E_SUMMARY = os.path.join(FASTLAB, "partial_freeze", "trackE_summary.json")

# The five pretrained-policy gathers.  Every one is authenticated against the
# frozen pretrained SHA through its own COMPLETE.json before a byte is used.
DEFAULT_GATHERS = (
    os.path.join(
        FASTLAB, "guided_positive/kfull_u50_topk2_P4/rounds/round_01/gather"
    ),
    os.path.join(
        FASTLAB, "guided_positive/kfull_x3_P4_v2/rounds/round_01/gather"
    ),
    os.path.join(FASTLAB, "early_crunch/harvest_260014/rounds/round_01/gather"),
    os.path.join(FASTLAB, "early_crunch/harvest_260016/rounds/round_01/gather"),
    os.path.join(FASTLAB, "early_crunch/harvest_260018/rounds/round_01/gather"),
)
# Unique per-gather round labels keep contexts from different gathers distinct
# even when they share (scenario, gamma, step).
ROUND_LABELS = (31, 32, 33, 34, 35)

# candidate_id = acquisition pool index in [0, K) for the B=4 base queries,
# K + parent for same-latent repair queries, 2K + offset for guided proposals.
# K = 16 in every gather (guided_collect.candidate_id_base == 2K == 32).
POOL_K = 16

PROBE_SEED = 20_260_751
ID_PROBE_SEED = PF.ID_PROBE_SEED  # identical fixed 1024-window ID-train probe
DEMO_SEED = 20_260_753
TARGET_SEED = 20_260_754
DEFAULT_SEED = 20_260_755
EVAL_EP0 = 270_000
EVAL_NOISE_SEED = 20_260_733
EVAL_M = 20

TARGET_SETS = ("base", "base_teacher", "executed")

ARMS = {
    "F1a": dict(scope="S2", targets="base", k_draws=1, lr=3.0e-4,
                note="SET-BASE: target set = the certified B=4 base positives"),
    "F1b": dict(scope="S2", targets="base_teacher", k_draws=1, lr=3.0e-4,
                note="SET-BASE+G: base positives union the G+ teacher windows"),
    "F1c": dict(scope="S2", targets="executed", k_draws=1, lr=3.0e-4,
                note="DIRAC control: executed window only, same new pipeline"),
    "F1d": dict(scope="S2", targets="base", k_draws=8, lr=3.0e-4,
                note="SET-BASE + K=8 exposure smoothing"),
    "F1e": dict(scope="S2", targets="executed", k_draws=8, lr=3.0e-4,
                note="DIRAC + K=8: exposure smoothing alone"),
    "F1f": dict(scope="S4", targets="base_teacher", k_draws=1, lr=1.0e-4,
                note="SET-BASE+G at S4 full-canonical trainability"),
}
ARM_ORDER = ["F1a", "F1b", "F1c", "F1d", "F1e", "F1f"]


# ------------------------------------------------------------------- dataset
def _context_key(row):
    return (
        int(row["scenario_id"]), round(float(row["gamma"]), 8), int(row["step"])
    )


def _same_features(left, right):
    return all(
        np.array_equal(
            np.asarray(left[key], np.float32), np.asarray(right[key], np.float32)
        )
        for key in ("hp10", "low5", "hist")
    )


def _authenticate(gather):
    """Every store is checked against the gather's own COMPLETE manifest."""
    with open(os.path.join(gather, "COMPLETE.json")) as stream:
        manifest = json.load(stream)
    if str(manifest["checkpoint_sha256"]) != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"{gather} was not gathered under the frozen policy")
    if int(manifest["guided_collect"]["candidate_id_base"]) != 2 * POOL_K:
        raise RuntimeError(f"{gather} does not use the K={POOL_K} pool layout")
    files = dict(
        sidecar=os.path.join(gather, "query_sidecar.pt"),
        executed=os.path.join(gather, "executed_round.pt"),
        guided=os.path.join(gather, RA.GUIDED_POSITIVE_STORE),
    )
    digests = {key: FA._sha256_file(path) for key, path in files.items()}
    expected = dict(
        sidecar=str(manifest["query_sidecar"]["sha256"]),
        executed=str(manifest["executed_shard"]["sha256"]),
        guided=str(manifest["guided_positive_shard"]["sha256"]),
    )
    for key, digest in digests.items():
        if digest != expected[key]:
            raise RuntimeError(f"{files[key]} digest changed since collection")
    return files, digests, manifest


def _load_gather(gather, round_label):
    """One gather -> context-keyed multi-positive records + diagnostics."""
    files, digests, manifest = _authenticate(gather)
    sidecar = BS.RoundShard.load(files["sidecar"])
    executed = OS.ExecutedRoundShard.load(files["executed"])
    _, guided = GPP._guided_positive_records(files["guided"])

    contexts = {}
    for row in sidecar.contexts:
        key = _context_key(row)
        if key in contexts:
            raise RuntimeError("duplicate sidecar context key")
        contexts[key] = dict(
            round=int(round_label),
            gather=os.path.abspath(gather),
            scenario_id=int(row["scenario_id"]),
            gamma=float(row["gamma"]),
            step=int(row["step"]),
            hp10=np.asarray(row["hp10"], np.float32),
            low5=np.asarray(row["low5"], np.float32),
            hist=np.asarray(row["hist"], np.float32),
            base=[], base_candidate_ids=[], teacher=[],
            executed=None, executed_candidate_id=None, executed_in_base=False,
        )

    # BASE: the exact-verifier-certified members of the uncertainty-tilted B=4
    # acquisition queries.  Repair (K..2K) and guided (2K..) ids are excluded.
    base_sources = Counter()
    for row in sidecar.queries:
        if int(row["y"]) != 1 or not bool(row["full_h"]):
            continue
        if int(row["candidate_id"]) >= POOL_K:
            continue
        key = _context_key(sidecar.contexts[int(row["context_id"])])
        record = contexts[key]
        record["base"].append(np.asarray(row["controls"], np.float32))
        record["base_candidate_ids"].append(int(row["candidate_id"]))
        base_sources[str(row.get("query_source"))] += 1
    for record in contexts.values():
        if len(record["base"]) > 1:
            order = np.argsort(record["base_candidate_ids"])
            record["base"] = [record["base"][i] for i in order]
            record["base_candidate_ids"] = [
                record["base_candidate_ids"][i] for i in order
            ]

    # EXECUTED: the canonical one-window-per-context offline D+ store.  Its
    # controls are cross-checked against the sidecar row flagged executed.
    sidecar_executed = {}
    for row in sidecar.queries:
        if not bool(row["executed"]):
            continue
        sidecar_executed[_context_key(sidecar.contexts[int(row["context_id"])])] = row
    executed_matched = 0
    executed_in_base = 0
    for window in executed.Dplus:
        context = executed.contexts[int(window["context_id"])]
        key = _context_key(context)
        record = contexts.get(key)
        if record is None:
            raise RuntimeError("executed context missing from the sidecar")
        if not _same_features(context, record):
            raise RuntimeError("executed/sidecar context features disagree")
        controls = np.asarray(window["controls"], np.float32)
        mirror = sidecar_executed.get(key)
        if mirror is None or not np.array_equal(
            controls, np.asarray(mirror["controls"], np.float32)
        ):
            raise RuntimeError("executed window is not the flagged sidecar row")
        executed_matched += 1
        record["executed"] = controls
        record["executed_candidate_id"] = int(mirror["candidate_id"])
        record["executed_in_base"] = any(
            np.array_equal(controls, value) for value in record["base"]
        )
        executed_in_base += int(record["executed_in_base"])

    # TEACHER: G+ windows keyed to the same context.
    teacher_contexts = set()
    for holder, row in guided:
        key = (int(row["scenario_id"]), round(float(row["gamma"]), 8),
               int(row["step"]))
        record = contexts.get(key)
        if record is None:
            raise RuntimeError("G+ context missing from the sidecar")
        if not _same_features(holder.contexts[int(row["context_id"])], record):
            raise RuntimeError("G+/sidecar context features disagree")
        record["teacher"].append(np.asarray(row["controls"], np.float32))
        teacher_contexts.add(key)

    diagnostics = dict(
        gather=os.path.abspath(gather),
        round_label=int(round_label),
        sha256=digests,
        sidecar_contexts=len(sidecar.contexts),
        sidecar_queries=len(sidecar.queries),
        base_windows=sum(len(v["base"]) for v in contexts.values()),
        base_query_sources={k: int(v) for k, v in sorted(base_sources.items())},
        base_contexts=sum(1 for v in contexts.values() if v["base"]),
        executed_windows=executed_matched,
        executed_in_base=executed_in_base,
        executed_outside_base=executed_matched - executed_in_base,
        teacher_windows=sum(len(v["teacher"]) for v in contexts.values()),
        teacher_contexts=len(teacher_contexts),
        teacher_only_contexts=sum(
            1 for v in contexts.values() if v["teacher"] and not v["base"]
        ),
        guided_generator=str(manifest["guided_collect"]["generator"]),
        guided_topk=int(manifest["guided_collect"]["topk"]),
    )
    return contexts, diagnostics


def build_dataset(gathers, round_labels=ROUND_LABELS):
    dataset = []
    per_gather = []
    for gather, label in zip(gathers, round_labels):
        contexts, diagnostics = _load_gather(gather, label)
        per_gather.append(diagnostics)
        for key in sorted(contexts):
            record = contexts[key]
            record["key"] = (int(label), *key)
            dataset.append(record)
    if len({record["key"] for record in dataset}) != len(dataset):
        raise RuntimeError("context keys collide across gathers")
    return dataset, per_gather


def targets_of(record, target_set):
    if target_set == "base":
        return record["base"]
    if target_set == "base_teacher":
        return [*record["base"], *record["teacher"]]
    if target_set == "executed":
        return [] if record["executed"] is None else [record["executed"]]
    raise ValueError(target_set)


def _set_histogram(sizes):
    return {str(key): int(value) for key, value in sorted(Counter(sizes).items())}


def dataset_stats(dataset, per_gather):
    base_sizes = [len(r["base"]) for r in dataset if r["base"]]
    union_sizes = [
        len(r["base"]) + len(r["teacher"]) for r in dataset
        if r["base"] or r["teacher"]
    ]
    teacher_sizes = [len(r["teacher"]) for r in dataset if r["teacher"]]
    executed = [r for r in dataset if r["executed"] is not None]
    per_gamma = defaultdict(lambda: dict(contexts=0, base=0, teacher=0))
    for record in dataset:
        if not (record["base"] or record["teacher"]):
            continue
        cell = per_gamma[f"{record['gamma']:g}"]
        cell["contexts"] += 1
        cell["base"] += len(record["base"])
        cell["teacher"] += len(record["teacher"])
    return dict(
        gathers=per_gather,
        sidecar_contexts=len(dataset),
        contexts_with_base=len(base_sizes),
        contexts_with_targets=len(union_sizes),
        contexts_with_executed=len(executed),
        base_windows=int(sum(base_sizes)),
        teacher_windows=int(sum(teacher_sizes)),
        certified_windows_total=int(sum(union_sizes)),
        base_set_size_histogram=_set_histogram(base_sizes),
        base_set_size_mean=float(np.mean(base_sizes)) if base_sizes else 0.0,
        union_set_size_histogram=_set_histogram(union_sizes),
        union_set_size_mean=float(np.mean(union_sizes)) if union_sizes else 0.0,
        teacher_set_size_histogram=_set_histogram(teacher_sizes),
        teacher_only_contexts=int(sum(
            1 for r in dataset if r["teacher"] and not r["base"]
        )),
        executed_in_base=int(sum(1 for r in executed if r["executed_in_base"])),
        executed_outside_base=int(sum(
            1 for r in executed if not r["executed_in_base"]
        )),
        contexts_with_base_but_no_executed=int(sum(
            1 for r in dataset if r["base"] and r["executed"] is None
        )),
        contexts_with_executed_but_no_base=int(sum(
            1 for r in dataset if r["executed"] is not None and not r["base"]
        )),
        multi_positive_share_base=float(
            np.mean([size > 1 for size in base_sizes]) if base_sizes else 0.0
        ),
        per_gamma={key: per_gamma[key] for key in sorted(per_gamma)},
    )


# ---------------------------------------------------------------------- mass
def context_mass(contexts):
    """Equal mass per CONTEXT: gamma -> (round, scenario) -> context."""
    grouped = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(contexts):
        grouped[round(float(record["gamma"]), 8)][
            (int(record["round"]), int(record["scenario_id"]))
        ].append(index)
    mass = np.zeros(len(contexts), np.float64)
    n_gamma = len(grouped)
    for cells in grouped.values():
        for indices in cells.values():
            value = 1.0 / (n_gamma * len(cells) * len(indices))
            for index in indices:
                mass[index] = value
    total = float(mass.sum())
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError(f"context mass sums to {total}, not one")
    gamma_mass = defaultdict(float)
    for index, record in enumerate(contexts):
        gamma_mass[f"{record['gamma']:g}"] += float(mass[index])
    accounting = dict(
        total=total,
        gammas=n_gamma,
        cells=int(sum(len(cells) for cells in grouped.values())),
        contexts=len(contexts),
        per_gamma={key: gamma_mass[key] for key in sorted(gamma_mass)},
        semantics=(
            "equal mass per context; set size changes the target, never the "
            "context's weight"
        ),
    )
    return mass, accounting


# ------------------------------------------------------------------ batching
def _context_tensors(contexts, indices, device):
    grid = torch.as_tensor(
        np.stack([contexts[i]["hp10"] for i in indices]), device=device
    ).float()
    low = torch.as_tensor(
        np.stack([contexts[i]["low5"] for i in indices]), device=device
    ).float()
    hist = torch.as_tensor(
        np.stack([contexts[i]["hist"] for i in indices]), device=device
    ).float()
    return grid, low, hist


def _arm_loss(policy, contexts, target_lists, picks, indices, mass, device,
              seed, k_draws):
    grid, low, hist = _context_tensors(contexts, indices, device)
    controls = torch.as_tensor(
        np.stack([target_lists[i][int(picks[i])] for i in indices]),
        device=device,
    ).float()
    weights = torch.as_tensor(
        mass[list(indices)], dtype=controls.dtype, device=device
    )
    weights = weights * (len(indices) / weights.sum())
    context = policy.ctx_from(grid, low, hist)
    if int(k_draws) > 1:
        # K independent (x0, tau) draws per exposure, averaged inside the loss:
        # same expected gradient, CFM sampling variance divided by K.  The
        # context embedding is computed once and tiled, so only the noise moves.
        context = context.repeat(int(k_draws), 1)
        controls = controls.repeat(int(k_draws), 1, 1)
        weights = weights.repeat(int(k_draws))
    torch.manual_seed(int(seed))
    loss = policy.cfm_loss(controls, context, weights=weights)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite CFM loss")
    return loss


@torch.no_grad()
def _window_probe(policy, contexts, pairs, device, *, seed, batch=256):
    """Deterministic unweighted CFM loss over an explicit (context, window) list."""
    if not pairs:
        return float("nan")
    was_training = policy.training
    policy.eval()
    total = 0.0
    for start in range(0, len(pairs), int(batch)):
        chunk = pairs[start:start + int(batch)]
        indices = [index for index, _ in chunk]
        grid, low, hist = _context_tensors(contexts, indices, device)
        controls = torch.as_tensor(
            np.stack([window for _, window in chunk]), device=device
        ).float()
        torch.manual_seed(int(seed) + start)
        loss = policy.cfm_loss(controls, policy.ctx_from(grid, low, hist))
        total += float(loss) * (len(chunk) / float(len(pairs)))
    if was_training:
        policy.train()
    return float(total)


# ------------------------------------------------------------------ training
def run_train(args):
    started = time.perf_counter()
    arm = str(args.arm)
    spec = ARMS[arm]
    scope = str(spec["scope"])
    target_set = str(spec["targets"])
    k_draws = int(spec["k_draws"] if args.k_draws is None else args.k_draws)
    lr = float(spec["lr"] if args.lr is None else args.lr)
    demo_frac = float(args.demo_frac)
    device = str(args.device)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = os.path.abspath(args.checkpoint)
    source_sha = FA._sha256_file(checkpoint)
    if source_sha != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source pretrained checkpoint SHA changed")

    gathers = [os.path.abspath(path) for path in args.gathers]
    dataset, per_gather = build_dataset(gathers)
    stats = dataset_stats(dataset, per_gather)

    all_targets = [targets_of(record, target_set) for record in dataset]
    active = [index for index, values in enumerate(all_targets) if values]
    contexts = [dataset[index] for index in active]
    target_lists = [all_targets[index] for index in active]
    set_sizes = np.array([len(values) for values in target_lists], np.int64)
    mass, accounting = context_mass(contexts)

    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    freeze = PF._configure_scope(policy, scope)
    digests_before = PF._module_digests(policy)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
    )

    bundle = PF._demo_bundle(checkpoint)
    demo_meta = dict(
        dataset=bundle["dataset"],
        manifest_sha256=bundle["manifest_sha256"],
        file_sha256=bundle["file_sha256"],
        split=bundle["split"],
        train_windows=int(bundle["windows"]),
    )
    probe_selection = PF._demo_draw(
        bundle, int(args.id_probe_windows), np.random.default_rng(ID_PROBE_SEED)
    )

    # Fixed diagnostic probes, identical for every arm regardless of its own
    # target set, so mode coverage and Dirac fit are directly comparable.
    pool_pairs = [
        (index, window)
        for index, values in enumerate(target_lists) for window in values
    ]
    executed_pairs = [
        (index, record["executed"]) for index, record in enumerate(contexts)
        if record["executed"] is not None
    ]
    unexecuted_pairs = [
        (index, window)
        for index, record in enumerate(contexts)
        for window in record["base"]
        if record["executed"] is None
        or not np.array_equal(window, record["executed"])
    ]
    teacher_pairs = [
        (index, window) for index, record in enumerate(contexts)
        for window in record["teacher"]
    ]

    batch = int(args.batch)
    n_demo = int(math.floor(batch * demo_frac))
    n_arm = batch - n_demo
    if n_arm < 1:
        raise ValueError("demo_frac leaves no room for arm contexts")

    def _probes():
        return dict(
            target_set=_window_probe(
                policy, contexts, pool_pairs, device, seed=PROBE_SEED),
            executed=_window_probe(
                policy, contexts, executed_pairs, device,
                seed=PROBE_SEED + 1_001),
            unexecuted_base=_window_probe(
                policy, contexts, unexecuted_pairs, device,
                seed=PROBE_SEED + 2_002),
            teacher=_window_probe(
                policy, contexts, teacher_pairs, device,
                seed=PROBE_SEED + 3_003),
        )

    policy.eval()
    probe_before = _probes()
    pool_losses = [probe_before["target_set"]]
    id_probe = [PF._id_probe_loss(policy, bundle, probe_selection, device)]
    steps_per_epoch = len([
        start for start in range(0, len(contexts), n_arm)
        if len(contexts) - start >= 2
    ])
    print(
        f"[{arm}] scope={scope} targets={target_set} K={k_draws} lr={lr:g} "
        f"contexts={len(contexts)} windows={len(pool_pairs)} "
        f"mean|set|={float(set_sizes.mean()):.3f} batch={batch} "
        f"(arm {n_arm} contexts + demo {n_demo} windows) "
        f"steps/epoch={steps_per_epoch} epochs={args.epochs} "
        f"pool {pool_losses[0]:.5f} ID {id_probe[0]:.5f}",
        flush=True,
    )

    epoch_train_losses = []
    epoch_demo_losses = []
    demo_generator = np.random.default_rng(int(args.seed) + DEMO_SEED)
    total_steps = 0
    draw_counts = np.zeros(len(contexts), np.int64)
    for epoch in range(int(args.epochs)):
        # One target draw per exposure, uniform over the context's set,
        # redrawn every epoch from a stream that is seeded and independent of
        # the microbatch permutation.
        picks = np.random.default_rng(
            int(args.seed) + TARGET_SEED + 1_000_003 * (epoch + 1)
        ).integers(0, np.maximum(set_sizes, 1))
        draw_counts += (picks > 0).astype(np.int64)
        order = np.random.default_rng(
            int(args.seed) + 1_000_003 * (epoch + 1)
        ).permutation(len(contexts))
        policy.train()
        batch_losses = []
        demo_losses = []
        for step, start in enumerate(range(0, len(order), n_arm)):
            indices = [int(i) for i in order[start:start + n_arm]]
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            seed = int(args.seed) + 7919 * step + 1_000_003 * (epoch + 1)
            arm_loss = _arm_loss(
                policy, contexts, target_lists, picks, indices, mass, device,
                seed, k_draws,
            )
            loss = arm_loss
            if n_demo:
                selection = PF._demo_draw(bundle, n_demo, demo_generator)
                demo_loss = PF._demo_loss(
                    policy, bundle, selection, device, seed + 104_729,
                )
                loss = (1.0 - demo_frac) * arm_loss + demo_frac * demo_loss
                demo_losses.append(float(demo_loss.detach()))
            loss.backward()
            optimizer.step()
            total_steps += 1
            batch_losses.append(float(arm_loss.detach()))
        policy.eval()
        epoch_train_losses.append(float(np.mean(batch_losses)))
        if demo_losses:
            epoch_demo_losses.append(float(np.mean(demo_losses)))
        pool_losses.append(
            _window_probe(policy, contexts, pool_pairs, device, seed=PROBE_SEED)
        )
        id_probe.append(
            PF._id_probe_loss(policy, bundle, probe_selection, device)
        )
        print(
            f"[{arm}] epoch {epoch + 1:2d} train {epoch_train_losses[-1]:.5f} "
            f"pool {pool_losses[-1]:.5f} ID {id_probe[-1]:.5f}"
            + (f" demo {epoch_demo_losses[-1]:.5f}" if demo_losses else ""),
            flush=True,
        )
    probe_after = _probes()

    digests_after = PF._module_digests(policy)
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
        "phase": "multipos",
        "scope": scope,
        "target_set": target_set,
        "k_draws": int(k_draws),
        "source_checkpoint": checkpoint,
        "source_sha256": source_sha,
    })

    payload = dict(
        status=STATUS,
        arm=arm,
        note=str(spec["note"]),
        source=FA._source(),
        config=dict(
            checkpoint=checkpoint,
            checkpoint_sha256=source_sha,
            gathers=gathers,
            scope=scope,
            scope_label=PF.SCOPES[scope]["label"],
            target_set=target_set,
            k_draws=int(k_draws),
            demo_frac=demo_frac,
            batch=batch,
            arm_contexts_per_batch=n_arm,
            demo_windows_per_batch=n_demo,
            epochs=int(args.epochs),
            steps_per_epoch=steps_per_epoch,
            optimizer_steps=int(total_steps),
            lr=lr,
            seed=int(args.seed),
            device=device,
            optimizer="Adam",
            stepping="one optimizer.step() per microbatch (Track E dose)",
            id_probe_windows=int(args.id_probe_windows),
            objective=(
                "per exposure the CFM target is drawn uniformly from the "
                "context's certified set (redrawn each epoch, seeded); K "
                "independent (x0,tau) draws are averaged inside the loss; "
                "hierarchy mass is equal per CONTEXT"
            ),
            dropout_mode="policy.train() during updates (canonical)",
        ),
        freeze=freeze,
        dataset=stats,
        pool=dict(
            contexts=len(contexts),
            windows=len(pool_pairs),
            set_size_histogram=_set_histogram(set_sizes.tolist()),
            set_size_mean=float(set_sizes.mean()),
            multi_target_contexts=int((set_sizes > 1).sum()),
            executed_probe_windows=len(executed_pairs),
            unexecuted_base_probe_windows=len(unexecuted_pairs),
            teacher_probe_windows=len(teacher_pairs),
            epochs_with_non_first_target=int(draw_counts.sum()),
        ),
        mass=accounting,
        id_demo=demo_meta,
        losses=dict(
            train_pool_mean_per_epoch=epoch_train_losses,
            demo_mean_per_epoch=epoch_demo_losses,
            fixed_pool_loss_by_epoch=pool_losses,
            id_train_probe_loss_by_epoch=id_probe,
            probes_before=probe_before,
            probes_after=probe_after,
        ),
        parameter_digests=dict(
            changed=changed,
            changed_count=len(changed),
            trainable_tensor_count=len(trainable_names),
            unchanged_trainable=sorted(trainable_names - set(changed)),
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


def run_stats(args):
    dataset, per_gather = build_dataset(
        [os.path.abspath(path) for path in args.gathers]
    )
    stats = dataset_stats(dataset, per_gather)
    stats["status"] = "CLAUDE_MULTIPOS_DATASET_COMPLETE"
    stats["source"] = FA._source()
    for name in TARGET_SETS:
        sizes = [len(targets_of(record, name)) for record in dataset]
        sizes = [size for size in sizes if size]
        stats.setdefault("target_sets", {})[name] = dict(
            contexts=len(sizes), windows=int(sum(sizes)),
            mean_set_size=float(np.mean(sizes)) if sizes else 0.0,
        )
    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    path = os.path.join(os.path.abspath(args.output_dir), "dataset_stats.json")
    FA._write_json(path, stats)
    print(json.dumps({
        key: value for key, value in stats.items()
        if key not in ("gathers", "source")
    }, indent=1))
    print(path, flush=True)
    return stats


# ---------------------------------------------------------------------- eval
def run_eval(args):
    arm = str(args.arm)
    root = os.path.abspath(args.root)
    checkpoint = os.path.join(root, arm, f"{arm}.pt")
    output_dir = os.path.join(root, arm, "eval")
    os.makedirs(output_dir, exist_ok=True)
    command = [
        sys.executable, os.path.join(HERE, "sfm_b1_offline_eval.py"),
        "--checkpoints", os.path.abspath(args.r0), checkpoint,
        "--labels", "r0", "r1",
        "--ep0", str(EVAL_EP0),
        "--noise-seed", str(EVAL_NOISE_SEED),
        "--m-per-gamma", str(EVAL_M),
        "--device", str(args.device),
        "--workers", str(int(args.workers)),
        "--cache-dir", os.path.abspath(args.cache_dir),
        "--output-dir", output_dir,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=HERE, text=True, capture_output=True, check=False
    )
    with open(os.path.join(output_dir, "eval.log"), "w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.write(completed.stdout)
        stream.write("\n--- stderr ---\n")
        stream.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"eval failed for {arm}; see {output_dir}/eval.log")
    print(f"[{arm}] eval done in {time.perf_counter() - started:.1f}s",
          flush=True)
    return 0


# -------------------------------------------------------------------- report
DELTA_METRICS = PF.DELTA_METRICS


def _track_e_reference():
    if not os.path.isfile(TRACK_E_SUMMARY):
        return None
    with open(TRACK_E_SUMMARY) as stream:
        payload = json.load(stream)
    arm = payload["arms"].get("E2b")
    if arm is None:
        return None
    return dict(
        arm="E2b", pooled=arm["pooled"], delta=arm["delta"],
        paired=arm["paired"], scope=arm["scope"],
        note="Track E best non-winner: S2, D+ & G+top1 windows, demo50",
    )


def run_report(args):
    root = os.path.abspath(args.root)
    figures = os.path.join(root, "figs")
    os.makedirs(figures, exist_ok=True)
    arms = {}
    r0 = None
    for arm in ARM_ORDER:
        metrics = os.path.join(root, arm, "eval", "raw_m20_offline_metrics.json")
        if not os.path.isfile(metrics):
            print(f"[report] missing {metrics}", flush=True)
            continue
        cells = PF._read_eval(metrics)
        if r0 is None:
            r0 = cells["r0"]
        elif r0["checkpoint_sha256"] != cells["r0"]["checkpoint_sha256"]:
            raise RuntimeError("arms disagree on the r0 reference cell")
        with open(os.path.join(root, arm, f"{arm}_train.json")) as stream:
            train = json.load(stream)
        arms[arm] = dict(
            train=train,
            r1=cells["r1"],
            paired=PF._paired(
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

    reference = _track_e_reference()
    summary = dict(
        status=REPORT_STATUS,
        source=FA._source(),
        r0=dict(
            checkpoint=r0["checkpoint"],
            checkpoint_sha256=r0["checkpoint_sha256"],
            pooled=r0["pooled"],
            per_gamma=r0["per_gamma"],
        ),
        reference_E2b=reference,
        dataset=next(iter(arms.values()))["train"]["dataset"],
        arms={
            arm: dict(
                scope=value["train"]["config"]["scope"],
                target_set=value["train"]["config"]["target_set"],
                k_draws=value["train"]["config"]["k_draws"],
                lr=value["train"]["config"]["lr"],
                note=value["train"]["note"],
                trainable_parameters=value["train"]["freeze"][
                    "trainable_parameters"],
                optimizer_steps=value["train"]["config"]["optimizer_steps"],
                pool=value["train"]["pool"],
                checkpoint=value["train"]["outputs"]["checkpoint"],
                checkpoint_sha256=value["train"]["outputs"][
                    "checkpoint_sha256"],
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
    path = os.path.join(root, "trackF_summary.json")
    FA._write_json(path, summary)
    print(path, flush=True)

    machine = {
        arm: dict(
            checkpoint_path=value["checkpoint"],
            sha256=value["checkpoint_sha256"],
            SR=value["pooled"]["SR"],
            CR=value["pooled"]["CR"],
            Val=value["pooled"]["Validity"],
            clr=value["pooled"]["successful_clearance"],
            ttg=value["pooled"]["successful_time_to_goal"],
            pSR=value["paired"]["SR_mcnemar_p"],
            pCR=value["paired"]["CR_mcnemar_p"],
        )
        for arm, value in summary["arms"].items()
    }
    machine["r0"] = dict(
        checkpoint_path=summary["r0"]["checkpoint"],
        sha256=summary["r0"]["checkpoint_sha256"],
        SR=summary["r0"]["pooled"]["SR"],
        CR=summary["r0"]["pooled"]["CR"],
        Val=summary["r0"]["pooled"]["Validity"],
        clr=summary["r0"]["pooled"]["successful_clearance"],
        ttg=summary["r0"]["pooled"]["successful_time_to_goal"],
        pSR=None, pCR=None,
    )
    machine_path = os.path.join(root, "trackF_arms.json")
    FA._write_json(machine_path, machine)
    print(machine_path, flush=True)

    _figure(summary, figures, args.figure_copies)
    return summary


def _figure(summary, figures, copies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [arm for arm in ARM_ORDER if arm in summary["arms"]]
    reference = summary.get("reference_E2b")
    metrics = [
        ("SR", "SR"),
        ("CR", "CR"),
        ("Validity", "Validity"),
        ("successful_clearance", "clearance (m)"),
        ("successful_time_to_goal", "time to goal (s)"),
    ]
    palette = {
        "base": "#2f6fb0", "base_teacher": "#3f8f5f", "executed": "#b06f2f",
    }
    short = {"base": "base", "base_teacher": "base+G", "executed": "Dirac"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.5, 4.8))
    for axis, (key, label) in zip(axes, metrics):
        values = []
        colors = []
        labels = []
        errors = []
        stars = []
        if reference is not None:
            values.append(reference["delta"][key])
            colors.append("#9a9a9a")
            labels.append("E2b\nTrack E\nD+&G+")
            errors.append(
                reference["paired"]["Validity_paired_bootstrap95"]
                if key == "Validity" else None
            )
            stars.append(False)
        for arm in names:
            entry = summary["arms"][arm]
            values.append(entry["delta"][key])
            colors.append(palette[entry["target_set"]])
            scope = "" if entry["scope"] == "S2" else f" {entry['scope']}"
            labels.append(
                f"{arm}\n{short[entry['target_set']]}\nK{entry['k_draws']}{scope}"
            )
            interval = entry["paired"]["Validity_paired_bootstrap95"]
            errors.append(interval if key == "Validity" else None)
            stars.append(
                key == "Validity" and (interval[0] > 0.0 or interval[1] < 0.0)
            )
        positions = np.arange(len(values))
        axis.bar(positions, values, width=0.78, color=colors,
                 edgecolor="black", linewidth=0.6)
        if key == "Validity":
            for position, value, interval in zip(positions, values, errors):
                if interval is None:
                    continue
                axis.plot(
                    [position, position], interval, color="black",
                    linewidth=1.1, solid_capstyle="butt", zorder=3,
                )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=7.0)
        axis.set_title(f"$\\Delta$ {label} vs r0", fontsize=10)
        low = min([min(v, *(e or [v])) for v, e in zip(values, errors)])
        high = max([max(v, *(e or [v])) for v, e in zip(values, errors)])
        span = max(abs(low), abs(high), 1e-6)
        axis.set_ylim(-1.5 * span, 1.5 * span)
        for position, value, star in zip(positions, values, stars):
            axis.annotate(
                f"{value:+.3f}" + (" *" if star else ""), (position, value),
                textcoords="offset points",
                xytext=(0, 5 if value >= 0 else -13), ha="center", fontsize=7.0,
                fontweight="bold" if star else "normal",
            )
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    pooled = summary["r0"]["pooled"]
    dataset = summary["dataset"]
    fig.suptitle(
        "Track F1 multi-positive CFM -- paired raw temp-1 M20 deltas vs the "
        f"pretrained policy r0 (SR {pooled['SR']:.3f} / CR {pooled['CR']:.3f} "
        f"/ Val {pooled['Validity']:.3f}, 140 paired episodes)\n"
        f"per-context certified target SETS: {dataset['contexts_with_base']} "
        f"contexts, {dataset['base_windows']} base + "
        f"{dataset['teacher_windows']} teacher windows, mean |set| "
        f"{dataset['base_set_size_mean']:.2f}; K = (x0,tau) draws averaged per "
        "exposure; scope S2 unless marked\n"
        "blue = base set, green = base+teacher set, orange = single executed "
        "Dirac, grey = Track E best non-winner E2b.  Validity bars carry the "
        "paired bootstrap 95% CI; * = CI excludes zero",
        fontsize=8.5, y=0.995, va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    written = []
    for directory in [figures, *copies]:
        os.makedirs(directory, exist_ok=True)
        for extension in ("png", "pdf"):
            target = os.path.join(
                directory, f"trackF_multipos_deltas.{extension}"
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

    stats = sub.add_parser("stats")
    stats.add_argument("--gathers", nargs="+", default=list(DEFAULT_GATHERS))
    stats.add_argument("--output-dir", default=DEFAULT_ROOT)
    stats.set_defaults(func=run_stats)

    train = sub.add_parser("train")
    train.add_argument("--arm", required=True, choices=ARM_ORDER)
    train.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    train.add_argument("--gathers", nargs="+", default=list(DEFAULT_GATHERS))
    train.add_argument("--output-dir", required=True)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--k-draws", type=int, default=None)
    train.add_argument("--demo-frac", type=float, default=0.5)
    train.add_argument("--batch", type=int, default=128)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--id-probe-windows", type=int, default=1024)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=run_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--arm", required=True, choices=ARM_ORDER)
    evaluate.add_argument("--root", default=DEFAULT_ROOT)
    evaluate.add_argument("--r0", default=R0_CHECKPOINT)
    evaluate.add_argument("--cache-dir", default=EVAL_CACHE)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--workers", type=int, default=24)
    evaluate.set_defaults(func=run_eval)

    report = sub.add_parser("report")
    report.add_argument("--root", default=DEFAULT_ROOT)
    report.add_argument("--figure-copies", nargs="*", default=[])
    report.set_defaults(func=run_report)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
