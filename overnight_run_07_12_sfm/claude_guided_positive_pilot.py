"""Guided-positive early-collection pilot (fastlab, isolated from the study).

Motivation.  Guided-repair rescues concentrate at early episode steps, where
the OOD SFM crowd is fast and the learned policy is weakest; late-episode
executed positives are easy and nearly straight.  The canonical loop only
invokes the locked-Kazuki guidance when a repair trigger fires, so the
exact-verifier-certified early-step dodges that the guidance *can* produce are
never observed, let alone trained on.

This pilot changes exactly one thing.  During gathering it asks
``sfm_b1_kazuki_repair_audit.collect`` to run a guided generator at every live
context with ``step < --guided-until-step``, independently of any trigger, and
to keep the windows the exact verifier certifies positive as a new
collection-only population ``G+`` (``guided_positive_round.pt``).  Execution,
selection, repair triggering, the D0 neutral store, and the GP are untouched:
``G+`` rows are never executed and never enter the GP.

Two generators are available.  ``same_latent`` is the collector's existing
weak repair operator (kept only as an ablation: it certifies ~1% of the time
in the early band, so it harvests almost nothing there).  ``kazuki_full``, the
default, runs the *complete* locked external Kazuki controller at the context
-- 200 generated samples, 10 elites, 200 MPPI perturbations per elite, locked
goal .5 / safe .3 -- and verifies its top ``--guided-topk`` refined modes.

``--crunch-full-pool-until-step`` is a separate, independently launchable arm
with no external teacher at all: below that step every one of the K=16
proposals is verified instead of only the B=4 the GP acquires, which attacks
the finite-B NVP budget artifact directly.  It changes the admissible set and
therefore the executed trajectory, so it is a different arm, not a control.

The pilot then runs ``--update-passes`` whole-buffer Adam steps over ``G+``
(and, by default, a separate equally sized pass block over the ordinary
executed ``D+``), each pass being one optimizer step over the whole population
with the canonical ``gamma -> (round, scenario) -> context -> window``
hierarchy mass.  ``D0`` is collected and audited as usual but is deliberately
*not* trained on, so the pilot isolates the ``G+`` signal.

Nothing here is a confirmation.  It is a single-round, unreplicated pilot whose
only job is to measure the early-step ``G+`` yield and whether a massive update
on it moves the paired offline metrics at all.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as MR
import sfm_b1_neutral_teacher_sanity as NS
import sfm_b1_offline_store as OS
import sfm_b1_store as BS
import sfm_protocol as SP


STATUS = "CLAUDE_GUIDED_POSITIVE_PILOT_COMPLETE"
ROUND_STATUS = "CLAUDE_GUIDED_POSITIVE_PILOT_ROUND_COMPLETE"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENARIO_EP0 = MR.DEFAULT_SCENARIO_EP0
DEFAULT_EVAL_EP0 = MR.DEFAULT_EVAL_EP0
DEFAULT_GUIDED_UNTIL_STEP = 50
DEFAULT_UPDATE_PASSES = 4
DEFAULT_LR = MR.DEFAULT_LR
DEFAULT_BATCH = 128
STEP_BIN = 10
# policy.head is the single nn.Linear(width=256 -> d=2H=20) flow read-out.
HEAD_PARAMETERS = 256 * 20 + 20
# Population-specific fold used by run() for the acquired-replay seed; the demo
# draw reuses it so D+ and G+ never see the same ID microbatch in a round.
DPLUS_SEED_FOLD = 1_000_003
GPLUS_SEED_FOLD = 7_000_003


def _configure_train_scope(policy, scope):
    """Apply ``--train-scope`` on top of the canonical expansion trainability.

    ``full`` is exactly today's behaviour: every parameter except the frozen
    visual encoder ``enc_grid`` trains.  ``head`` additionally freezes the GRU,
    ``enc_low`` and the residual trunk and re-enables exactly ``policy.head.*``
    -- the final ``nn.Linear(256, 2H)`` velocity read-out -- so an update can
    only re-aim the flow field, never move the representation that produced it.
    """
    frozen = BS.configure_expansion_trainability(policy)
    scope = str(scope)
    if scope not in ("full", "head"):
        raise ValueError("--train-scope must be full or head")
    if scope == "head":
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        for parameter in policy.head.parameters():
            parameter.requires_grad_(True)
        expected = {f"head.{name}" for name, _ in policy.head.named_parameters()}
        trainable_names = {
            name for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        if trainable_names != expected:
            raise RuntimeError(
                "head scope leaked outside policy.head: "
                f"{sorted(trainable_names ^ expected)}"
            )
        frozen = sorted(
            name for name, parameter in policy.named_parameters()
            if not parameter.requires_grad
        )
    trainable = sorted(
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    trainable_count = int(sum(
        parameter.numel() for parameter in policy.parameters()
        if parameter.requires_grad
    ))
    frozen_count = int(sum(
        parameter.numel() for parameter in policy.parameters()
        if not parameter.requires_grad
    ))
    if scope == "head" and trainable_count != HEAD_PARAMETERS:
        raise RuntimeError(
            f"head scope trains {trainable_count} parameters, "
            f"not the expected {HEAD_PARAMETERS}"
        )
    summary = dict(
        train_scope=scope,
        trainable_parameters=trainable,
        trainable_parameter_count=trainable_count,
        frozen_parameters=frozen,
        frozen_parameter_count=frozen_count,
    )
    print(
        f"[train-scope] {scope}: trainable {trainable_count} "
        f"({len(trainable)} tensors: {trainable}) | frozen {frozen_count} "
        f"({len(frozen)} tensors)",
        flush=True,
    )
    return summary


def _demo_plan(batch, demo_frac):
    """Microbatch composition ceil((1-F)*B) acquired + floor(F*B) demo.

    The two counts always sum to ``B`` because
    ``ceil((1-F)B) == B - floor(FB)`` for every real ``F``.
    """
    batch = int(batch)
    demo_frac = float(demo_frac)
    if not 0.0 <= demo_frac < 1.0:
        raise ValueError("--demo-frac must lie in [0, 1)")
    acquired = int(math.ceil((1.0 - demo_frac) * batch))
    demo = int(math.floor(demo_frac * batch))
    if acquired + demo != batch:
        raise RuntimeError("demo plan does not tile the microbatch")
    if demo_frac > 0.0 and (acquired < 1 or demo < 1):
        raise ValueError("--demo-frac leaves an empty half of the microbatch")
    return acquired, demo


def _demo_bundle(dataset, checkpoint):
    """Authenticate and load the frozen ID pretraining TRAIN banks once.

    Delegates to the study's own ``MR._id_anchor_preflight`` /
    ``MR._id_anchor_banks``, so the pinned manifest digest, the seven pinned
    per-file digests, the ``success_only`` flag and the promoted checkpoint's
    ``split_meta`` train-episode restriction are all re-verified here.
    """
    preflight = MR._id_anchor_preflight(dataset, len(SP.GAMMAS))
    bundle = MR._id_anchor_banks(preflight, checkpoint)
    if len(bundle["banks"]) != len(SP.GAMMAS):
        raise RuntimeError("ID-demo bundle does not cover the seven gammas")
    return bundle


def _demo_provenance(bundle):
    return dict(
        dataset=str(bundle["dataset"]),
        manifest_sha256=str(bundle["manifest_sha256"]),
        file_sha256=dict(bundle["file_sha256"]),
        split=str(bundle["split"]),
        split_source=str(bundle["split_source"]),
        per_gamma_support=[
            {
                "gamma": float(bank["gamma"]),
                "trajectories": int(bank["trajectories"]),
                "windows": int(bank["windows"]),
                "source_windows": int(bank["source_windows"]),
                "sha256": str(bank["sha256"]),
            }
            for bank in bundle["banks"]
        ],
    )


def _demo_draw(bundle, count, seed):
    """Gamma-balanced draw of ``count`` ID train windows.

    Gamma slots are round-robin over a seeded permutation of the seven gammas,
    so per-gamma counts differ by at most one and which gamma receives the
    surplus rotates with the seed.  Inside a gamma the window is drawn with
    replacement from the frozen hierarchical ``MR._id_anchor_mass`` (uniform
    over train trajectories, then uniform over that trajectory's windows) --
    the same law ``MR._id_anchor_update`` uses.
    """
    banks = bundle["banks"]
    count = int(count)
    if count < 1:
        raise ValueError("ID-demo draw requires a positive count")
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(banks))
    slots = [int(order[index % len(banks)]) for index in range(count)]
    drawn = {}
    for bank_index, number in sorted(Counter(slots).items()):
        bank = banks[bank_index]
        drawn[bank_index] = [
            int(row) for row in generator.choice(
                len(bank["mass"]), size=int(number), replace=True,
                p=bank["mass"],
            )
        ]
    cursor = Counter()
    selection = []
    for bank_index in slots:
        selection.append((bank_index, drawn[bank_index][cursor[bank_index]]))
        cursor[bank_index] += 1
    return selection


def _demo_tensors(bundle, selection, device):
    banks = bundle["banks"]
    stack = lambda key: torch.stack(  # noqa: E731
        [banks[bank][key][row] for bank, row in selection]
    ).to(device=device, dtype=torch.float32)
    return stack("hp10"), stack("low5"), stack("hist"), stack("U")


@torch.no_grad()
def _demo_fixed_loss(policy, bundle, *, windows, device, seed):
    """Deterministic unweighted CFM loss on a fixed ID-demo probe batch."""
    selection = _demo_draw(bundle, int(windows), int(seed))
    grid, low, hist, controls = _demo_tensors(bundle, selection, device)
    was_training = policy.training
    policy.eval()
    torch.manual_seed(int(seed))
    loss = policy.cfm_loss(controls, policy.ctx_from(grid, low, hist))
    if was_training:
        policy.train()
    return float(loss)


def _mixed_population_update(
    policy,
    optimizer,
    records,
    *,
    population,
    inner_steps,
    batch,
    device,
    seed,
    demo_frac,
    demo_bundle,
):
    """Sibling of ``MR._population_update`` with ID-demo microbatch mixing.

    Semantics deliberately mirrored from ``MR._population_update`` /
    ``NS._objective``: one Adam step per pass, the acquired population ordered
    by ``BS.hierarchical_order`` and therefore exposed exactly once per pass,
    per-sample acquired weights ``n_acquired * hierarchy_mass``, non-finite
    loss rejection, and the frozen-encoder SHA assertion.

    The one change: each microbatch carries ``ceil((1-F)*batch)`` acquired rows
    *and* ``floor(F*batch)`` ID-demo rows through a single ``ctx_from`` +
    ``cfm_loss`` call, the demo rows weighted 1.0.  Because ``cfm_loss``
    averages over the composed batch, the microbatch objective is exactly
    ``(1-F) * sum_i mass_i * per_i  +  F * mean(demo per)`` -- i.e. canonical
    demo-fraction mixing, reducing to the unmixed objective at ``F = 0``.

    Demo seed: ``demo_seed = (seed * 9176) + 97 * batch_index + 13`` with
    ``seed = train_seed + round * fold + pass * 1_000_003`` (fold 1_000_003 for
    D+, 7_000_003 for G+), so the draw folds train_seed, round, population,
    pass index and batch index.
    """
    if not records:
        raise ValueError(f"{population} replay requires nonempty support")
    acquired_n, demo_n = _demo_plan(batch, demo_frac)
    if demo_n < 1:
        raise ValueError("mixed replay requires a positive demo fraction")
    mass, accounting = BS.hierarchy_mass(records)
    encoder_before = BS.module_sha256(policy.enc_grid)
    expected = {MR._identity(holder, row) for holder, row in records}
    losses = []
    encoder_gradient_norms = []
    trainable_gradient_norms = []
    exposure_hashes = []
    demo_windows = []
    demo_gamma = Counter()
    policy.train()
    for inner in range(int(inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        pass_seed = int(seed) + inner * 1_000_003
        ordered = BS.hierarchical_order(records, pass_seed)
        visited = []
        total_loss = 0.0
        drawn = 0
        for index, start in enumerate(range(0, len(ordered), acquired_n)):
            values = ordered[start:start + acquired_n]
            grid, low, hist, controls = BS._tensor_batch(values, device)
            weights = torch.as_tensor(
                [
                    len(values) * mass[(id(holder), int(row["query_id"]))]
                    for holder, row in values
                ],
                dtype=controls.dtype,
                device=device,
            )
            # A ragged final chunk keeps the F ratio rather than the count.
            share = max(1, int(round(
                len(values) * float(demo_n) / float(acquired_n)
            )))
            selection = _demo_draw(
                demo_bundle, share, pass_seed * 9176 + 97 * index + 13,
            )
            demo_grid, demo_low, demo_hist, demo_controls = _demo_tensors(
                demo_bundle, selection, device,
            )
            context = policy.ctx_from(
                torch.cat([grid, demo_grid]),
                torch.cat([low, demo_low]),
                torch.cat([hist, demo_hist]),
            )
            composed = torch.cat([controls, demo_controls])
            composed_weights = torch.cat([
                weights,
                torch.ones(
                    len(selection), dtype=controls.dtype, device=device,
                ),
            ])
            torch.manual_seed(pass_seed + start)
            loss = policy.cfm_loss(composed, context, weights=composed_weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite mixed CFM loss in {population}"
                )
            loss.backward()
            total_loss += float(loss.detach())
            drawn += len(selection)
            if inner == 0:
                for bank_index, _ in selection:
                    demo_gamma[
                        str(demo_bundle["banks"][bank_index]["gamma"])
                    ] += 1
            visited.extend(
                (int(holder.round_i), int(row["query_id"]))
                for holder, row in values
            )
        if len(visited) != len(expected) or set(visited) != expected:
            raise RuntimeError(
                f"{population} pass duplicated or omitted a sample"
            )
        squared = torch.zeros((), dtype=torch.float64)
        for parameter in policy.enc_grid.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().to(
                    dtype=torch.float64,
                ).square().sum().cpu()
        encoder_gradient_norms.append(float(squared.sqrt()))
        squared = torch.zeros((), dtype=torch.float64)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    squared += parameter.grad.detach().to(
                        dtype=torch.float64,
                    ).square().sum().cpu()
        gradient_norm = float(squared.sqrt())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite {population} trainable gradient norm"
            )
        trainable_gradient_norms.append(gradient_norm)
        optimizer.step()
        losses.append(float(total_loss))
        exposure_hashes.append(MR._sha256_jsonable(visited))
        demo_windows.append(int(drawn))
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    encoder_after = BS.module_sha256(policy.enc_grid)
    if (
        not any(
            parameter.requires_grad
            for parameter in policy.enc_grid.parameters()
        )
        and encoder_after != encoder_before
    ):
        raise RuntimeError("visual encoder changed during replay")
    return {
        "population": str(population),
        "records": len(records),
        "inner_steps": int(inner_steps),
        "optimizer_steps": int(inner_steps),
        "sample_exposures": len(records) * int(inner_steps),
        "exact_once_per_inner_step": True,
        "exposure_identity_sha256": exposure_hashes,
        "losses": losses,
        "encoder_gradient_norms": encoder_gradient_norms,
        "trainable_gradient_norms": trainable_gradient_norms,
        "mass": MR._compact_mass(accounting),
        "encoder_sha_before": encoder_before,
        "encoder_sha_after": encoder_after,
        "demo_frac": float(demo_frac),
        "demo_acquired_per_microbatch": int(acquired_n),
        "demo_windows_per_microbatch": int(demo_n),
        "demo_windows_per_pass": demo_windows,
        "demo_per_gamma_first_pass": {
            key: int(value) for key, value in sorted(demo_gamma.items())
        },
        "demo_seed_formula": (
            "(train_seed + round*fold + pass*1000003)*9176 + 97*batch + 13; "
            "fold = 1000003 (Dplus) / 7000003 (Gplus)"
        ),
        "microbatch_noise_seed_formula": (
            "train_seed + round*fold + pass*1000003 + acquired_offset"
        ),
    }


class _GuidedHolder:
    """Minimal shard-like holder so G+ reuses the canonical replay helpers."""

    def __init__(self, round_i):
        self.round_i = int(round_i)
        self.contexts = []
        self.windows = []


def _guided_positive_records(path):
    """Authenticate one ``guided_positive_round.pt`` and expose replay rows.

    Windows harvested at the same ``(scenario, gamma, step)`` share one context
    entry, so the hierarchy mass keeps equal weight per *context* rather than
    silently up-weighting steps where more than one guided candidate was
    certified.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("status") != RA.GUIDED_POSITIVE_STATUS:
        raise ValueError("not an authenticated guided-positive payload")
    holder = _GuidedHolder(payload["round"])
    context_ids = {}
    records = []
    for expected, source in enumerate(payload["records"]):
        result = source.get("verifier_result", {})
        if (
            int(source["guided_id"]) != expected
            or source["population"] != "Gplus"
            or source["semantic_label"] != "guided_positive"
            or source["generator"] not in RA.GUIDED_GENERATORS
            or source["source"] != RA.GUIDED_SOURCES[source["generator"]]
            or int(source["verifier_y"]) != 1
            or not result.get("resolved")
            or int(result.get("y", -1)) != 1
            or not result.get("full_h")
            or int(result.get("terminal_step", -1)) != SP.H
            or not source["train_eligible"]
            or source["replay_default"]
            or source["gp_eligible"]
        ):
            raise RuntimeError("G+ semantics changed")
        key = (
            int(source["scenario_id"]),
            round(float(source["gamma"]), 8),
            int(source["step"]),
        )
        if key not in context_ids:
            context_ids[key] = len(holder.contexts)
            holder.contexts.append({
                "context_id": context_ids[key],
                "round": int(payload["round"]),
                "scenario_id": int(source["scenario_id"]),
                "gamma": float(source["gamma"]),
                "step": int(source["step"]),
                "state": np.asarray(source["state"], np.float32),
                "hp10": np.asarray(source["hp10"], np.float32),
                "low5": np.asarray(source["low5"], np.float32),
                "hist": np.asarray(source["hist"], np.float32),
                "ped_xy": np.asarray(source["ped_xy"], np.float32),
                "ped_vel": np.asarray(source["ped_vel"], np.float32),
            })
        row = {
            "window_id": expected,
            "query_id": expected,
            "context_id": context_ids[key],
            "controls": np.asarray(source["controls"], np.float32),
            "x0": (
                None if source["x0"] is None
                else np.asarray(source["x0"], np.float32)
            ),
            "y": 1,
            "semantic_label": "guided_positive",
            "generator": str(source["generator"]),
            "step": int(source["step"]),
            "gamma": float(source["gamma"]),
            "scenario_id": int(source["scenario_id"]),
            "reused_from_repair": bool(source["reused_from_repair"]),
            "selected_for_execution": bool(source["selected_for_execution"]),
        }
        if tuple(row["controls"].shape) != (SP.H, 2):
            raise RuntimeError("invalid G+ controls shape")
        if row["x0"] is not None and tuple(row["x0"].shape) != (2 * SP.H,):
            raise RuntimeError("invalid G+ x0 shape")
        holder.windows.append(row)
        records.append((holder, row))
    if len(records) != int(payload["summary"]["Gplus"]):
        raise RuntimeError("G+ payload count mismatch")
    return holder, records


def _step_bins(records, *, until_step, bin_size=STEP_BIN):
    """G+ yield by early-step bin, with per-gamma and context breakdowns."""
    bins = defaultdict(lambda: dict(
        windows=0, contexts=set(), reused_from_repair=0, per_gamma=Counter(),
    ))
    for _, row in records:
        index = int(row["step"]) // int(bin_size)
        start = index * int(bin_size)
        cell = bins[start]
        cell["windows"] += 1
        cell["contexts"].add(
            (int(row["scenario_id"]), round(float(row["gamma"]), 8),
             int(row["step"]))
        )
        cell["reused_from_repair"] += int(bool(row["reused_from_repair"]))
        cell["per_gamma"][str(row["gamma"])] += 1
    ordered = []
    for start in range(0, int(until_step), int(bin_size)):
        cell = bins.get(start)
        ordered.append({
            "step_start": int(start),
            "step_end": int(min(start + int(bin_size), int(until_step))),
            "windows": 0 if cell is None else int(cell["windows"]),
            "contexts": 0 if cell is None else len(cell["contexts"]),
            "reused_from_repair": (
                0 if cell is None else int(cell["reused_from_repair"])
            ),
            "per_gamma": (
                {} if cell is None
                else {key: int(value) for key, value in cell["per_gamma"].items()}
            ),
        })
    return ordered


def _population_summary(records):
    # Rows are (holder, row) pairs: G+ rows carry gamma themselves, while
    # ExecutedRoundShard windows resolve gamma via the shard's context list.
    def _gamma(holder, row):
        if "gamma" in row:
            return row["gamma"]
        return holder.contexts[int(row["context_id"])]["gamma"]

    per_gamma = Counter(str(_gamma(holder, row)) for holder, row in records)
    return dict(
        windows=len(records),
        per_gamma={key: int(value) for key, value in per_gamma.items()},
    )


def _read_json(path):
    with open(path) as stream:
        return json.load(stream)


def _run_paired_eval(args, *, checkpoints, labels, output_dir):
    command = [
        sys.executable,
        os.path.join(HERE, "sfm_b1_offline_eval.py"),
        "--checkpoints", *checkpoints,
        "--labels", *labels,
        "--scene-profile", args.scene_profile,
        "--ep0", str(int(args.eval_ep0)),
        "--noise-seed", str(int(args.eval_noise_seed)),
        "--m-per-gamma", str(int(args.eval_m)),
        "--device", str(args.device),
        "--workers", str(int(args.workers)),
        "--output-dir", output_dir,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=HERE, text=True, capture_output=True, check=False,
    )
    log_path = os.path.join(output_dir, "offline_eval.log")
    os.makedirs(output_dir, exist_ok=True)
    with open(log_path, "w") as stream:
        stream.write(completed.stdout)
        stream.write("\n--- stderr ---\n")
        stream.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"paired offline eval failed; see {log_path}")
    return dict(
        command=command,
        returncode=int(completed.returncode),
        log=log_path,
        seconds=float(time.perf_counter() - started),
        stdout_tail=completed.stdout.strip().splitlines()[-8:],
    )


def run(args):
    gammas = tuple(map(float, args.gammas))
    if (
        not gammas
        or len(set(gammas)) != len(gammas)
        or any(value not in tuple(map(float, SP.GAMMAS)) for value in gammas)
    ):
        raise ValueError(f"--gammas must be a distinct subset of {SP.GAMMAS}")
    if int(args.T) != 180:
        raise ValueError("the collector protocol is pinned to T=180")
    if not 0 <= int(args.guided_until_step) <= int(args.T):
        raise ValueError("--guided-until-step must lie in [0, T]")
    if not 0 <= int(args.crunch_full_pool_until_step) <= int(args.T):
        raise ValueError("--crunch-full-pool-until-step must lie in [0, T]")
    collect_gplus = int(args.guided_until_step) > 0
    include_dplus = str(args.include_dplus) == "yes"
    if not collect_gplus and int(args.crunch_full_pool_until_step) == 0:
        raise ValueError("enable --guided-until-step, the crunch arm, or both")
    if not collect_gplus and not include_dplus:
        raise ValueError(
            "the crunch-only arm has no G+ population, so D+ must be trained"
        )
    if int(args.update_passes) < 1:
        raise ValueError("--update-passes must be positive")
    if int(args.rounds) < 1:
        raise ValueError("--rounds must be positive")
    if int(args.batch) != DEFAULT_BATCH:
        raise ValueError("canonical microbatch size is 128")
    demo_frac = float(args.demo_frac)
    acquired_per_microbatch, demo_per_microbatch = _demo_plan(
        int(args.batch), demo_frac,
    )
    mix_demos = demo_frac > 0.0
    if args.head_lr is not None and str(args.train_scope) != "head":
        raise ValueError("--head-lr only applies to --train-scope head")

    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root):
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    checkpoint = os.path.abspath(args.checkpoint)
    source_sha = FA._sha256_file(checkpoint)
    if source_sha != RA.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("source pretrained checkpoint SHA changed")
    source = FA._source()
    if args.require_clean_worktree and not source["tracked_worktree_clean"]:
        raise RuntimeError("pilot was asked for a clean frozen worktree")

    # Loaded once, before any output exists, so a bad/unpinned ID dataset
    # fails the pilot before it burns a gather.
    demo_bundle = (
        _demo_bundle(args.demo_dataset, checkpoint) if mix_demos else None
    )

    os.makedirs(output_root)
    checkpoints_dir = os.path.join(output_root, "checkpoints")
    rounds_dir = os.path.join(output_root, "rounds")
    os.makedirs(checkpoints_dir)
    os.makedirs(rounds_dir)

    policy, _ = GPS.load_sfm_policy(checkpoint, device=args.device)
    scope = _configure_train_scope(policy, args.train_scope)
    frozen = scope["frozen_parameters"]
    effective_lr = float(args.lr)
    if str(args.train_scope) == "head" and args.head_lr is not None:
        effective_lr = float(args.head_lr)
    encoder_sha_start = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.Adam(
        [
            parameter for parameter in policy.parameters()
            if parameter.requires_grad
        ],
        lr=effective_lr,
    )
    config = dict(
        status=STATUS,
        source=source,
        checkpoint=checkpoint,
        checkpoint_sha256=source_sha,
        scene_profile=str(args.scene_profile),
        selector=str(args.selector),
        gammas=list(gammas),
        scenario_ep0=int(args.scenario_ep0),
        rounds=int(args.rounds),
        T=int(args.T),
        guided_until_step=int(args.guided_until_step),
        guided_generator=str(args.guided_generator),
        guided_topk=int(args.guided_topk),
        crunch_full_pool_until_step=int(args.crunch_full_pool_until_step),
        update_passes=int(args.update_passes),
        lr=float(args.lr),
        head_lr=(None if args.head_lr is None else float(args.head_lr)),
        effective_lr=float(effective_lr),
        train_scope=str(args.train_scope),
        trainable_parameters=scope["trainable_parameters"],
        trainable_parameter_count=int(scope["trainable_parameter_count"]),
        frozen_parameter_count=int(scope["frozen_parameter_count"]),
        demo_frac=float(demo_frac),
        demo_acquired_per_microbatch=int(acquired_per_microbatch),
        demo_windows_per_microbatch=int(demo_per_microbatch),
        demo_source=(
            None if demo_bundle is None else _demo_provenance(demo_bundle)
        ),
        batch=int(args.batch),
        include_dplus=bool(include_dplus),
        train_D0=False,
        ell=float(args.ell),
        gp_cap=int(args.gp_cap),
        ess_target=float(args.ess_target),
        device=str(args.device),
        workers=int(args.workers),
        sample_seed=int(args.sample_seed),
        audit_seed=int(args.audit_seed),
        train_seed=int(args.train_seed),
        eval_m=int(args.eval_m),
        eval_ep0=int(args.eval_ep0),
        eval_noise_seed=int(args.eval_noise_seed),
        frozen_parameters=frozen,
        encoder_sha256_start=encoder_sha_start,
    )
    FA._write_json(os.path.join(output_root, "pilot_config.json"), config)

    trunk_sha_start = BS.module_sha256(policy.trunk)

    round0_path = os.path.join(checkpoints_dir, "round_00.pt")
    BX._save_checkpoint(policy, round0_path, {
        "study": STATUS,
        "round": 0,
        "phase": "pretrained",
        "source_checkpoint": checkpoint,
        "source_sha256": source_sha,
        "pilot_config": config,
    })

    history = []
    current_checkpoint = checkpoint
    previous_executed_path = None
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers), mp_context=spawn,
    ) as executor:
        for round_i in range(1, int(args.rounds) + 1):
            started = time.perf_counter()
            round_dir = os.path.join(rounds_dir, f"round_{round_i:02d}")
            os.makedirs(round_dir)
            gather_dir = os.path.join(round_dir, "gather")
            scenarios = tuple(range(
                int(args.scenario_ep0) + (round_i - 1) * 2,
                int(args.scenario_ep0) + round_i * 2,
            ))
            current_sha = FA._sha256_file(current_checkpoint)
            gather_started = time.perf_counter()
            RA.collect(
                current_checkpoint,
                scenarios=scenarios,
                gammas=gammas,
                scene_profile=str(args.scene_profile),
                selector=str(args.selector),
                device=args.device,
                verifier_workers=int(args.workers),
                sample_seed=int(args.sample_seed),
                audit_seed=int(args.audit_seed),
                ell=float(args.ell),
                neutral_continuation=True,
                guided_collect_until_step=int(args.guided_until_step),
                guided_generator=str(args.guided_generator),
                guided_topk=int(args.guided_topk),
                crunch_full_pool_until_step=int(
                    args.crunch_full_pool_until_step
                ),
                round_i=round_i,
                expected_checkpoint_sha256=current_sha,
                previous_executed_path=previous_executed_path,
                gp_cap=int(args.gp_cap),
                ess_target=float(args.ess_target),
                verifier_executor=executor,
                T=int(args.T),
                outdir=gather_dir,
            )
            gather_seconds = float(time.perf_counter() - gather_started)
            gather_marker = _read_json(
                os.path.join(gather_dir, "COMPLETE.json")
            )
            executed_path = os.path.join(gather_dir, "executed_round.pt")
            executed = OS.ExecutedRoundShard.load(executed_path)
            if executed.Dminus:
                raise RuntimeError("ordinary executed D unexpectedly has D-")
            positive_records = OS.positive_records(executed)
            _, neutral_records = MR._neutral_records(
                os.path.join(gather_dir, "neutral_round.pt")
            )
            guided_records = []
            if collect_gplus:
                _, guided_records = _guided_positive_records(
                    os.path.join(gather_dir, RA.GUIDED_POSITIVE_STORE)
                )
                if not guided_records:
                    raise RuntimeError("pilot round produced no certified G+")
            if include_dplus and not positive_records:
                raise RuntimeError("--include-dplus yes requires nonempty D+")

            fixed_before = {}
            if guided_records:
                fixed_before["Gplus"] = NS._fixed_loss(
                    policy, guided_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if positive_records:
                fixed_before["Dplus"] = NS._fixed_loss(
                    policy, positive_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if demo_bundle is not None:
                fixed_before["ID_demo"] = _demo_fixed_loss(
                    policy, demo_bundle, windows=int(args.batch),
                    device=args.device, seed=int(args.train_seed),
                )

            def _update(records, *, population, fold):
                # F == 0 keeps the canonical study updater byte-for-byte.
                if not mix_demos:
                    return MR._population_update(
                        policy,
                        optimizer,
                        records,
                        population=population,
                        inner_steps=int(args.update_passes),
                        batch=int(args.batch),
                        device=args.device,
                        seed=int(args.train_seed) + round_i * fold,
                    )
                return _mixed_population_update(
                    policy,
                    optimizer,
                    records,
                    population=population,
                    inner_steps=int(args.update_passes),
                    batch=int(args.batch),
                    device=args.device,
                    seed=int(args.train_seed) + round_i * fold,
                    demo_frac=demo_frac,
                    demo_bundle=demo_bundle,
                )

            updates = []
            if include_dplus:
                updates.append(_update(
                    positive_records,
                    population="Dplus",
                    fold=DPLUS_SEED_FOLD,
                ))
                BX._save_checkpoint(
                    policy,
                    os.path.join(
                        checkpoints_dir,
                        f"round_{round_i:02d}_post_positive.pt",
                    ),
                    {
                        "study": STATUS,
                        "round": round_i,
                        "phase": "post_Dplus",
                        "pilot_config": config,
                    },
                )
            if guided_records:
                updates.append(_update(
                    guided_records,
                    population="Gplus",
                    fold=GPLUS_SEED_FOLD,
                ))

            fixed_after = {}
            if guided_records:
                fixed_after["Gplus"] = NS._fixed_loss(
                    policy, guided_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if positive_records:
                fixed_after["Dplus"] = NS._fixed_loss(
                    policy, positive_records, batch=int(args.batch),
                    device=args.device, seed=int(args.train_seed) + round_i,
                )
            if demo_bundle is not None:
                fixed_after["ID_demo"] = _demo_fixed_loss(
                    policy, demo_bundle, windows=int(args.batch),
                    device=args.device, seed=int(args.train_seed),
                )

            round_checkpoint = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}.pt",
            )
            BX._save_checkpoint(policy, round_checkpoint, {
                "study": STATUS,
                "round": round_i,
                "phase": "post_Gplus",
                "pilot_config": config,
            })
            record = {
                "status": ROUND_STATUS,
                "round": int(round_i),
                "scenarios": list(scenarios),
                "gather_dir": gather_dir,
                "gather_seconds": gather_seconds,
                "seconds": float(time.perf_counter() - started),
                "source_checkpoint": current_checkpoint,
                "source_checkpoint_sha256": current_sha,
                "round_checkpoint": round_checkpoint,
                "round_checkpoint_sha256": FA._sha256_file(round_checkpoint),
                "gather_counts": gather_marker["counts"],
                "gather_outcomes": gather_marker["outcomes"],
                "guided_collect": gather_marker.get("guided_collect"),
                "crunch_full_pool": gather_marker.get("crunch_full_pool"),
                "guided_positive_shard": gather_marker.get(
                    "guided_positive_shard"
                ),
                "executed_shard": gather_marker["executed_shard"],
                "neutral_shard": gather_marker["neutral_shard"],
                "populations": {
                    "Gplus": _population_summary(guided_records),
                    "Dplus": _population_summary(positive_records),
                    "D0": {
                        "windows": len(neutral_records),
                        "trained_on": False,
                    },
                },
                "Gplus_by_step_bin": (
                    _step_bins(
                        guided_records,
                        until_step=int(args.guided_until_step),
                    )
                    if collect_gplus else []
                ),
                "Gplus_reused_from_repair": int(sum(
                    bool(row["reused_from_repair"])
                    for _, row in guided_records
                )),
                "Gplus_selected_for_execution": int(sum(
                    bool(row["selected_for_execution"])
                    for _, row in guided_records
                )),
                "fixed_loss_before": fixed_before,
                "fixed_loss_after": fixed_after,
                "updates": updates,
            }
            FA._write_json(
                os.path.join(round_dir, "ROUND_COMPLETE.json"), record,
            )
            history.append(record)
            previous_executed_path = executed_path
            current_checkpoint = round_checkpoint

    final_checkpoint = current_checkpoint
    evaluation = None
    if int(args.eval_m) > 0:
        evaluation = _run_paired_eval(
            args,
            checkpoints=[round0_path, final_checkpoint],
            labels=["r0", f"r{int(args.rounds)}"],
            output_dir=os.path.join(output_root, "paired_eval"),
        )

    delivery = dict(
        status=STATUS,
        config=config,
        rounds=history,
        final_checkpoint=final_checkpoint,
        final_checkpoint_sha256=FA._sha256_file(final_checkpoint),
        round0_checkpoint=round0_path,
        round0_checkpoint_sha256=FA._sha256_file(round0_path),
        encoder_sha256_start=encoder_sha_start,
        encoder_sha256_end=BS.module_sha256(policy.enc_grid),
        trunk_sha256_start=trunk_sha_start,
        trunk_sha256_end=BS.module_sha256(policy.trunk),
        train_scope=scope,
        evaluation=evaluation,
        caveats=[
            "single unreplicated pilot; not a confirmation",
            "G+ is collection-only and never executed in the gather",
            "D0 is collected and audited but never trained on",
        ] + ([
            "train scope is head-only: policy.head is the sole trainable "
            "module, so the representation is frozen by construction",
        ] if str(args.train_scope) == "head" else []) + ([
            f"every microbatch mixes {demo_per_microbatch} frozen ID "
            f"pretraining-train demo windows with "
            f"{acquired_per_microbatch} acquired rows",
        ] if mix_demos else []),
    )
    if delivery["encoder_sha256_end"] != encoder_sha_start:
        raise RuntimeError("frozen visual encoder changed during the pilot")
    if (
        str(args.train_scope) == "head"
        and delivery["trunk_sha256_end"] != trunk_sha_start
    ):
        raise RuntimeError("trunk changed under head-only training scope")
    path = os.path.join(output_root, "PILOT_COMPLETE.json")
    FA._write_json(path, delivery)
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--scenario-ep0", type=int, default=DEFAULT_SCENARIO_EP0,
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+",
        default=tuple(map(float, SP.GAMMAS)),
    )
    parser.add_argument("--T", type=int, default=180)
    parser.add_argument(
        "--guided-until-step", type=int, default=DEFAULT_GUIDED_UNTIL_STEP,
        help=(
            "run the guided generator proactively at every live context with "
            "step < this value and harvest the certified positives as G+; "
            "the default covers the measured NVP crunch band"
        ),
    )
    parser.add_argument(
        "--guided-generator", choices=RA.GUIDED_GENERATORS,
        default="kazuki_full",
        help=(
            "kazuki_full runs the complete locked external controller; "
            "same_latent is the weak repair operator, kept for ablation"
        ),
    )
    parser.add_argument(
        "--guided-topk", type=int, default=RA.DEFAULT_GUIDED_TOPK,
        help="kazuki_full only: refined elite modes verified per context",
    )
    parser.add_argument(
        "--crunch-full-pool-until-step", type=int, default=0,
        help=(
            "independent teacher-free arm: verify all K=16 proposals instead "
            "of the GP's B=4 below this step; changes the admissible set and "
            "therefore the executed trajectory"
        ),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--update-passes", type=int, default=DEFAULT_UPDATE_PASSES,
        help="whole-population Adam steps per population per round",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--train-scope", choices=("full", "head"), default="full",
        help=(
            "full (default, today's behaviour) trains everything except the "
            "frozen visual encoder; head freezes all of that too and trains "
            "exactly policy.head, the nn.Linear(256, 2H) flow read-out "
            f"({HEAD_PARAMETERS} parameters)"
        ),
    )
    parser.add_argument(
        "--head-lr", type=float, default=None,
        help=(
            "learning rate for --train-scope head; default None reuses --lr. "
            "Rejected under --train-scope full"
        ),
    )
    parser.add_argument(
        "--demo-frac", type=float, default=0.0,
        help=(
            "0 (default) is today's behaviour. F>0 composes every update "
            "microbatch of BOTH trained populations from ceil((1-F)*batch) "
            "acquired records plus floor(F*batch) gamma-balanced ID windows "
            "drawn from the frozen pretraining TRAIN split under the same "
            "CFM loss; the acquired rows keep their hierarchical mass and "
            "are still exposed exactly once per pass"
        ),
    )
    parser.add_argument(
        "--demo-dataset", default=MR.DEFAULT_ID_ANCHOR_DATASET,
        help="frozen ID window banks used by --demo-frac (digest-pinned)",
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--include-dplus", choices=("yes", "no"), default="yes",
        help="yes trains a separate pass block on executed D+ before G+",
    )
    parser.add_argument(
        "--selector", choices=("margin", "progress_gated_margin"),
        default="margin",
    )
    parser.add_argument(
        "--scene-profile", default="double_density_velocity_ood",
    )
    parser.add_argument("--ell", type=float, default=RA.DEFAULT_ELL)
    parser.add_argument("--gp-cap", type=int, default=MR.DEFAULT_GP_CAP)
    parser.add_argument("--ess-target", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--sample-seed", type=int, default=700_000)
    parser.add_argument("--audit-seed", type=int, default=2_026_073_0)
    parser.add_argument("--train-seed", type=int, default=2_026_073_1)
    parser.add_argument(
        "--eval-m", type=int, default=0,
        help="0 skips the paired offline evaluation entirely",
    )
    parser.add_argument("--eval-ep0", type=int, default=DEFAULT_EVAL_EP0)
    parser.add_argument("--eval-noise-seed", type=int, default=2_026_073_3)
    parser.add_argument(
        "--require-clean-worktree", action="store_true",
        help=(
            "fail unless the tracked worktree is clean; off by default "
            "because this pilot rides an uncommitted collector extension"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
