"""Track G: closed-loop confirmation of the M50-confirmed F1e offline recipe.

Track F measured F1e -- CFM on **executed-Dirac** windows only, K=8 (x0, tau)
draws per exposure, scope S2, lr 3e-4, batch 128, 20 epochs, demo_frac 0.5,
equal mass per context, refit from the frozen pretrained checkpoint -- against a
*fixed* five-gather pool, and the M50 replication confirmed a CI-clean
``Validity +.046`` with SR/CR favourable-but-ns and ``ttg +0.90`` as the
confirmed cost.  That is an OFFLINE statement: the pool was gathered once, by
the pretrained policy, and never moved.

This driver asks the closed-loop question instead.  Per arm, for rounds
``r = 1..R``:

1. **collect** two fresh scenarios with the arm's *current* policy through the
   frozen ``sfm_b1_kazuki_repair_audit.collect`` gatherer (T=180, the pinned
   double-shift OOD profile, exact-verifier labels, no teacher harvest);
2. **pool** the cumulative list of ``executed_round.pt`` shards (warm-start
   shards, if any, plus every round so far);
3. **refit** the F1e recipe *from the pretrained checkpoint* on that pool --
   never an incremental update of the previous round's weights, so every round
   is an honest re-application of the confirmed offline recipe to a larger,
   partly self-collected pool;
4. at pre-registered milestone rounds, run the frozen paired M20 offline
   evaluation against the same ``r0`` reference cell every Track D/E/F arm used.

The training call is ``claude_multipos_offline.run_train`` itself with
``arm="F1e"``: the recipe is not re-implemented here, it is imported.  The one
substitution is the dataset builder, because the canonical
``claude_multipos_offline.build_dataset`` authenticates each *gather directory*
against the frozen pretrained SHA through its ``COMPLETE.json`` -- correct for a
pool that a frozen policy produced, impossible for a closed loop whose rounds
are gathered by evolving checkpoints.  ``build_executed_pool`` below reads the
same ``executed_round.pt`` shards directly, keeps the per-shard round labels
distinct, sorts each shard by ``(scenario, gamma, step)`` exactly as the
canonical builder does, and emits records in the identical schema, so
``run_train`` sees a byte-compatible dataset for ``target_set="executed"``.
``verify-warm`` exists to prove that claim: refitting on the five warm-start
shards alone must reproduce the Track F ``F1e.pt``.

Two deliberate deviations from the canonical multi-round protocol, both
conservative, both recorded in ``CAMPAIGN_COMPLETE.json``:

* every ``collect`` runs with ``round_i=1`` and ``previous_executed_path=None``.
  In the canonical chain, round ``i > 1`` seeds the GP acquisition buffer from
  round ``i-1``'s executed support; here each gather is standalone, so the GP
  acquires from an EMPTY support every round, exactly like the reference round
  1.  Acquisition is therefore never sharpened by accumulated support -- the
  conservative choice, and the one that keeps every round's gather protocol
  identical to the protocol under which the warm-start shards were produced.
* the refit is from the pretrained checkpoint every round (see above), so the
  only thing that compounds across rounds is the DATA, never the weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401
import claude_multipos_offline as MP
import claude_partial_freeze_offline as PF
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_offline_store as OS
import sfm_scene as SS

STATUS = "CLAUDE_CLOSED_LOOP_REFIT_COMPLETE"
ROUND_STATUS = "CLAUDE_CLOSED_LOOP_ROUND_COMPLETE"
PROGRESS_STATUS = "CLAUDE_CLOSED_LOOP_IN_PROGRESS"
VERIFY_STATUS = "CLAUDE_CLOSED_LOOP_VERIFY_WARM_COMPLETE"

HERE = os.path.dirname(os.path.abspath(__file__))
FASTLAB = PF.FASTLAB

# The confirmed recipe, referenced by name so it cannot drift from Track F.
ARM = "F1e"
RECIPE = MP.ARMS[ARM]

PRETRAINED = PF.DEFAULT_CHECKPOINT
PRETRAINED_SHA = RA.EXPECTED_CHECKPOINT_SHA256
F1E_CHECKPOINT = os.path.join(FASTLAB, "multipos", ARM, f"{ARM}.pt")

# Warm start = the five pretrained-policy gathers Track F trained on.  Their
# round labels are the Track F labels, so a warm-only refit is bit-comparable.
WARM_START_SHARDS = tuple(
    os.path.join(gather, "executed_round.pt") for gather in MP.DEFAULT_GATHERS
)
WARM_START_LABELS = tuple(MP.ROUND_LABELS)
ROUND_LABEL_BASE = 40  # round r shard carries label ROUND_LABEL_BASE + r

R0_CHECKPOINT = MP.R0_CHECKPOINT
EVAL_CACHE = MP.EVAL_CACHE
EVAL_SCRIPT = os.path.join(HERE, "sfm_b1_offline_eval.py")
EVAL_EP0 = MP.EVAL_EP0            # 270_000
EVAL_NOISE_SEED = MP.EVAL_NOISE_SEED  # 20_260_733
EVAL_M = MP.EVAL_M                # 20 episodes per gamma
EVAL_TEMPERATURE = 1.0

DEFAULT_SCENARIO_EP0 = 260_020
DEFAULT_ROUNDS = 8
DEFAULT_EVAL_ROUNDS = "2,5,8"
DEFAULT_T = 180
DEFAULT_SAMPLE_SEED = RA.DEFAULT_SAMPLE_SEED   # 700_000
DEFAULT_AUDIT_SEED = RA.DEFAULT_AUDIT_SEED     # 20_260_730
DEFAULT_TRAIN_SEED = MP.DEFAULT_SEED           # 20_260_755
DEFAULT_ELL = RA.DEFAULT_ELL


# ------------------------------------------------------------------- dataset
def _shard_records(path, round_label):
    """One ``executed_round.pt`` -> canonical multipos context records.

    The executed store holds at most one exact full-H window per context, so
    the record's target set is the single Dirac ``executed`` window and both
    ``base`` and ``teacher`` are empty -- which is precisely what the F1e
    ``target_set="executed"`` recipe consumes.
    """
    path = os.path.abspath(path)
    shard = OS.ExecutedRoundShard.load(path)
    if shard.Dminus:
        raise RuntimeError(f"{path} contains verifier-negative executed windows")
    records = []
    for window in shard.Dplus:
        context = shard.contexts[int(window["context_id"])]
        records.append(dict(
            round=int(round_label),
            gather=os.path.dirname(path),
            shard=path,
            scenario_id=int(context["scenario_id"]),
            gamma=float(context["gamma"]),
            step=int(context["step"]),
            hp10=np.asarray(context["hp10"], np.float32),
            low5=np.asarray(context["low5"], np.float32),
            hist=np.asarray(context["hist"], np.float32),
            base=[],
            base_candidate_ids=[],
            teacher=[],
            executed=np.asarray(window["controls"], np.float32),
            executed_candidate_id=(
                None if window["candidate_id"] is None
                else int(window["candidate_id"])
            ),
            executed_in_base=False,
        ))
    # The canonical builder walks each gather's contexts in sorted key order;
    # match it so the pooled dataset ordering (and therefore the seeded
    # microbatch permutation) is identical for the shared shards.
    records.sort(
        key=lambda r: (r["scenario_id"], round(r["gamma"], 8), r["step"])
    )
    for record in records:
        record["key"] = (
            int(round_label), int(record["scenario_id"]),
            round(float(record["gamma"]), 8), int(record["step"]),
        )
    diagnostics = dict(
        shard=path,
        gather=os.path.dirname(path),
        round_label=int(round_label),
        shard_round=int(shard.round_i),
        sha256=FA._sha256_file(path),
        shard_contexts=len(shard.contexts),
        executed_windows=len(shard.D),
        executed_positive=len(shard.Dplus),
        executed_negative=len(shard.Dminus),
        base_windows=0,
        teacher_windows=0,
    )
    return records, diagnostics


def build_executed_pool(shards, round_labels):
    """``claude_multipos_offline.build_dataset`` stand-in for the closed loop."""
    shards = [os.path.abspath(path) for path in shards]
    round_labels = [int(label) for label in round_labels]
    if len(shards) != len(round_labels):
        raise ValueError("one round label per shard is required")
    if len(set(round_labels)) != len(round_labels):
        raise ValueError("round labels must be distinct")
    dataset = []
    per_shard = []
    for path, label in zip(shards, round_labels):
        records, diagnostics = _shard_records(path, label)
        per_shard.append(diagnostics)
        dataset.extend(records)
    if len({record["key"] for record in dataset}) != len(dataset):
        raise RuntimeError("context keys collide across pooled shards")
    return dataset, per_shard


def refit(shards, round_labels, output_dir, *, device, seed,
          checkpoint=PRETRAINED, epochs=20, batch=128, demo_frac=0.5):
    """Run the frozen F1e trainer on a pooled executed-Dirac shard list."""
    captured = {}

    def _builder(gathers, labels=None):
        dataset, per_shard = build_executed_pool(
            gathers, round_labels if labels is None else labels
        )
        captured["dataset"] = dataset
        return dataset, per_shard

    original = MP.build_dataset
    MP.build_dataset = _builder
    try:
        payload = MP.run_train(argparse.Namespace(
            arm=ARM,
            checkpoint=checkpoint,
            gathers=[os.path.abspath(path) for path in shards],
            output_dir=output_dir,
            lr=None,          # arm default: 3e-4
            k_draws=None,     # arm default: K = 8
            demo_frac=float(demo_frac),
            batch=int(batch),
            epochs=int(epochs),
            seed=int(seed),
            id_probe_windows=1024,
            device=str(device),
        ))
    finally:
        MP.build_dataset = original
    freeze = payload["freeze"]
    if int(freeze["trainable_parameters"]) != PF.SCOPES["S2"]["expect"]:
        raise RuntimeError("refit did not train the pre-registered S2 scope")
    if payload["config"]["target_set"] != "executed":
        raise RuntimeError("refit target set is not the executed Dirac")
    if int(payload["config"]["k_draws"]) != 8:
        raise RuntimeError("refit is not the K=8 recipe")
    if payload["config"]["checkpoint_sha256"] != PRETRAINED_SHA:
        raise RuntimeError("refit did not start from the frozen pretrained base")
    dataset = captured["dataset"]
    per_gamma = Counter(f"{record['gamma']:g}" for record in dataset)
    per_shard = Counter(str(record["round"]) for record in dataset)
    payload["closed_loop"] = dict(
        per_gamma_contexts={key: int(per_gamma[key]) for key in sorted(per_gamma)},
        per_shard_contexts={key: int(per_shard[key]) for key in sorted(per_shard)},
    )
    return payload


# ----------------------------------------------------------------- milestones
def _run_milestone_eval(round_checkpoint, round_i, output_dir, *, eval_gpu,
                        workers, r0=R0_CHECKPOINT, cache_dir=EVAL_CACHE,
                        m_per_gamma=EVAL_M):
    """Frozen paired M20 offline evaluation: r0 vs this round's checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    label = f"r{int(round_i)}"
    command = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoints", os.path.abspath(r0), os.path.abspath(round_checkpoint),
        "--labels", "r0", label,
        "--ep0", str(EVAL_EP0),
        "--noise-seed", str(EVAL_NOISE_SEED),
        "--m-per-gamma", str(int(m_per_gamma)),
        "--temperature", str(EVAL_TEMPERATURE),
        "--device", "cuda",
        "--workers", str(int(workers)),
        "--cache-dir", os.path.abspath(cache_dir),
        "--output-dir", os.path.abspath(output_dir),
    ]
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(int(eval_gpu))
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=HERE, env=environment, text=True, capture_output=True,
        check=False,
    )
    log_path = os.path.join(output_dir, "eval.log")
    with open(log_path, "w") as stream:
        stream.write(" ".join(command) + "\n")
        stream.write(f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}\n")
        stream.write(completed.stdout)
        stream.write("\n--- stderr ---\n")
        stream.write(completed.stderr)
    seconds = float(time.perf_counter() - started)
    record = dict(
        round=int(round_i),
        label=label,
        command=command,
        eval_gpu=int(eval_gpu),
        workers=int(workers),
        m_per_gamma=int(m_per_gamma),
        output_dir=os.path.abspath(output_dir),
        log=log_path,
        seconds=seconds,
        returncode=int(completed.returncode),
    )
    metrics = os.path.join(output_dir, "raw_m20_offline_metrics.json")
    if completed.returncode != 0 or not os.path.isfile(metrics):
        # A milestone is a measurement, not the experiment: a failed eval is
        # recorded loudly and the training chain continues.
        record["status"] = "failed"
        print(f"[eval r{round_i}] FAILED rc={completed.returncode}; see "
              f"{log_path}", flush=True)
        return record
    cells = PF._read_eval(metrics)
    r0_cell = cells["r0"]
    arm_cell = cells[label]
    record.update(
        status="ok",
        metrics_json=metrics,
        r0=dict(
            checkpoint=r0_cell["checkpoint"],
            checkpoint_sha256=r0_cell["checkpoint_sha256"],
            pooled=r0_cell["pooled"],
            per_gamma=r0_cell["per_gamma"],
        ),
        cell=dict(
            checkpoint=arm_cell["checkpoint"],
            checkpoint_sha256=arm_cell["checkpoint_sha256"],
            pooled=arm_cell["pooled"],
            per_gamma=arm_cell["per_gamma"],
        ),
        delta={
            key: arm_cell["pooled"][key] - r0_cell["pooled"][key]
            for key in PF.DELTA_METRICS
        },
        delta_per_gamma={
            gamma: {
                key: arm_cell["per_gamma"][gamma][key]
                - r0_cell["per_gamma"][gamma][key]
                for key in PF.DELTA_METRICS
            }
            for gamma in arm_cell["per_gamma"]
        },
        paired=PF._paired(
            r0_cell["rows"], arm_cell["rows"],
            seed=MP.PROBE_SEED + 977 * int(round_i),
        ),
    )
    pooled = arm_cell["pooled"]
    print(
        f"[eval r{round_i}] {seconds:.1f}s  SR {pooled['SR']:.3f} "
        f"({record['delta']['SR']:+.3f})  CR {pooled['CR']:.3f} "
        f"({record['delta']['CR']:+.3f})  Val {pooled['Validity']:.3f} "
        f"({record['delta']['Validity']:+.3f})  ttg "
        f"{pooled['successful_time_to_goal']:.2f} "
        f"({record['delta']['successful_time_to_goal']:+.2f})  "
        f"Val95 {record['paired']['Validity_paired_bootstrap95']}",
        flush=True,
    )
    return record


# ------------------------------------------------------------------- campaign
def _parse_eval_rounds(text):
    if text is None:
        return []
    return sorted({
        int(token) for token in str(text).replace(",", " ").split() if token
    })


def run_campaign(args):
    started = time.perf_counter()
    name = str(args.name)
    output_root = os.path.abspath(args.output_root)
    rounds = int(args.rounds)
    eval_rounds = _parse_eval_rounds(args.eval_rounds)
    gammas = tuple(map(float, args.gammas))
    workers = int(args.workers)

    complete_path = os.path.join(output_root, "CAMPAIGN_COMPLETE.json")
    if os.path.isfile(complete_path):
        raise FileExistsError(f"campaign already complete: {complete_path}")
    rounds_dir = os.path.join(output_root, "rounds")
    checkpoints_dir = os.path.join(output_root, "checkpoints")
    evals_dir = os.path.join(output_root, "evals")
    for directory in (output_root, rounds_dir, checkpoints_dir, evals_dir):
        os.makedirs(directory, exist_ok=True)

    initial_checkpoint = os.path.abspath(
        args.initial_checkpoint or PRETRAINED
    )
    initial_sha = FA._sha256_file(initial_checkpoint)
    warm = str(args.warm_start) == "fastlab5"
    pool = []
    if warm:
        for path, label in zip(WARM_START_SHARDS, WARM_START_LABELS):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            pool.append(dict(
                path=os.path.abspath(path), label=int(label),
                source="warm_start", sha256=FA._sha256_file(path),
            ))

    config = dict(
        status=PROGRESS_STATUS,
        name=name,
        source=FA._source(),
        recipe=dict(
            arm=ARM, note=str(RECIPE["note"]), **{
                key: RECIPE[key] for key in ("scope", "targets", "k_draws", "lr")
            },
            batch=int(args.batch), epochs=int(args.epochs),
            demo_frac=float(args.demo_frac),
            trainer="claude_multipos_offline.run_train (imported, unmodified)",
            refit_base=PRETRAINED, refit_base_sha256=PRETRAINED_SHA,
            refit_semantics=(
                "every round retrains FROM the frozen pretrained checkpoint on "
                "the cumulative pool; weights never compound, only data"
            ),
        ),
        collect=dict(
            selector=str(args.selector),
            gammas=list(gammas),
            scenario_ep0=int(args.scenario_ep0),
            scenarios_per_round=2,
            T=int(args.T),
            scene_profile="double_density_velocity_ood",
            neutral_continuation=True,
            guided_collect_until_step=0,
            sample_seed=int(args.sample_seed),
            audit_seed=int(args.audit_seed),
            ell=float(args.ell),
            gp_cap=int(args.gp_cap),
            ess_target=float(args.ess_target),
            verifier_workers=workers,
            round_i=1,
            previous_executed_path=None,
            deviation=(
                "DEVIATION FROM THE CANONICAL GP-SUPPORT CHAIN: every round "
                "collects with round_i=1 and previous_executed_path=None, so "
                "the GP acquires from an EMPTY support each round exactly like "
                "the reference round 1.  The canonical multi-round chain seeds "
                "round i's GP from round i-1's executed support; declining that "
                "is the conservative choice (acquisition is never sharpened by "
                "accumulated support) and keeps every gather protocol identical "
                "to the one that produced the warm-start shards."
            ),
        ),
        warm_start=dict(
            mode=str(args.warm_start),
            shards=[entry["path"] for entry in pool],
            labels=[entry["label"] for entry in pool],
            sha256=[entry["sha256"] for entry in pool],
        ),
        initial_checkpoint=initial_checkpoint,
        initial_checkpoint_sha256=initial_sha,
        rounds=rounds,
        eval_rounds=eval_rounds,
        eval=dict(
            script=EVAL_SCRIPT, r0=os.path.abspath(R0_CHECKPOINT),
            ep0=EVAL_EP0, noise_seed=EVAL_NOISE_SEED,
            m_per_gamma=int(args.eval_m), temperature=EVAL_TEMPERATURE,
            cache_dir=os.path.abspath(args.eval_cache),
            gpu=int(args.eval_gpu), workers=int(args.eval_workers),
        ),
        device=str(args.device),
        workers=workers,
        train_seed=int(args.train_seed),
        output_root=output_root,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        started_unix=time.time(),
    )
    FA._write_json(os.path.join(output_root, "campaign_config.json"), config)
    print(
        f"[{name}] selector={args.selector} warm={args.warm_start} "
        f"rounds={rounds} eval_rounds={eval_rounds} workers={workers} "
        f"eval_gpu={args.eval_gpu} pool0={len(pool)} shards\n"
        f"[{name}] initial checkpoint {initial_checkpoint} "
        f"({initial_sha[:12]})",
        flush=True,
    )

    history = []
    milestones = []
    current_checkpoint = initial_checkpoint
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as executor:
        for round_i in range(1, rounds + 1):
            round_started = time.perf_counter()
            round_dir = os.path.join(rounds_dir, f"round_{round_i:02d}")
            os.makedirs(round_dir, exist_ok=True)
            gather_dir = os.path.join(round_dir, "gather")
            scenarios = tuple(range(
                int(args.scenario_ep0) + (round_i - 1) * 2,
                int(args.scenario_ep0) + round_i * 2,
            ))
            current_sha = FA._sha256_file(current_checkpoint)

            # -- 1. collect with the arm's CURRENT policy ------------------
            gather_started = time.perf_counter()
            RA.collect(
                current_checkpoint,
                scenarios=scenarios,
                gammas=gammas,
                scene_profile="double_density_velocity_ood",
                selector=str(args.selector),
                device=str(args.device),
                verifier_workers=workers,
                sample_seed=int(args.sample_seed),
                audit_seed=int(args.audit_seed),
                ell=float(args.ell),
                neutral_continuation=True,
                guided_collect_until_step=0,
                crunch_full_pool_until_step=0,
                round_i=1,
                expected_checkpoint_sha256=current_sha,
                previous_executed_path=None,
                gp_cap=int(args.gp_cap),
                ess_target=float(args.ess_target),
                verifier_executor=executor,
                T=int(args.T),
                outdir=gather_dir,
            )
            gather_seconds = float(time.perf_counter() - gather_started)
            with open(os.path.join(gather_dir, "COMPLETE.json")) as stream:
                marker = json.load(stream)
            executed_path = os.path.join(gather_dir, "executed_round.pt")

            # -- 2. pool --------------------------------------------------
            pool.append(dict(
                path=os.path.abspath(executed_path),
                label=ROUND_LABEL_BASE + round_i,
                source=f"round_{round_i:02d}",
                sha256=FA._sha256_file(executed_path),
            ))

            # -- 3. refit the confirmed recipe from the pretrained base ----
            refit_started = time.perf_counter()
            refit_dir = os.path.join(round_dir, "refit")
            train = refit(
                [entry["path"] for entry in pool],
                [entry["label"] for entry in pool],
                refit_dir,
                device=str(args.device),
                seed=int(args.train_seed),
                epochs=int(args.epochs),
                batch=int(args.batch),
                demo_frac=float(args.demo_frac),
            )
            refit_seconds = float(time.perf_counter() - refit_started)
            round_checkpoint = os.path.join(
                checkpoints_dir, f"round_{round_i:02d}.pt"
            )
            shutil.copy2(train["outputs"]["checkpoint"], round_checkpoint)
            round_sha = FA._sha256_file(round_checkpoint)
            if round_sha != train["outputs"]["checkpoint_sha256"]:
                raise RuntimeError("round checkpoint copy is not byte-identical")
            FA._write_json(round_checkpoint + ".COMPLETE.json", dict(
                status="COMPLETE", path=round_checkpoint, sha256=round_sha,
                round=int(round_i), arm=ARM, source_checkpoint=PRETRAINED,
            ))

            losses = train["losses"]
            record = dict(
                status=ROUND_STATUS,
                round=int(round_i),
                scenarios=list(scenarios),
                gather_dir=gather_dir,
                gather_seconds=gather_seconds,
                gather_counts=marker["counts"],
                gather_outcomes=marker["outcomes"],
                gather_acquisition=marker.get("acquisition"),
                collect_checkpoint=current_checkpoint,
                collect_checkpoint_sha256=current_sha,
                pool=[
                    dict(path=e["path"], label=e["label"], source=e["source"],
                         sha256=e["sha256"]) for e in pool
                ],
                pool_shards=len(pool),
                pool_contexts=int(train["pool"]["contexts"]),
                pool_windows=int(train["pool"]["windows"]),
                pool_contexts_by_shard={
                    str(entry["round_label"]): int(entry["executed_positive"])
                    for entry in train["dataset"]["gathers"]
                },
                per_gamma_contexts=train["closed_loop"]["per_gamma_contexts"],
                per_shard_contexts=train["closed_loop"]["per_shard_contexts"],
                refit_dir=refit_dir,
                refit_seconds=refit_seconds,
                refit_train_json=os.path.join(refit_dir, f"{ARM}_train.json"),
                trainable_parameters=int(
                    train["freeze"]["trainable_parameters"]
                ),
                optimizer_steps=int(train["config"]["optimizer_steps"]),
                loss=dict(
                    train_first=losses["train_pool_mean_per_epoch"][0],
                    train_last=losses["train_pool_mean_per_epoch"][-1],
                    fixed_pool_first=losses["fixed_pool_loss_by_epoch"][0],
                    fixed_pool_last=losses["fixed_pool_loss_by_epoch"][-1],
                    demo_last=(
                        losses["demo_mean_per_epoch"][-1]
                        if losses["demo_mean_per_epoch"] else None
                    ),
                ),
                id_probe=dict(
                    first=losses["id_train_probe_loss_by_epoch"][0],
                    last=losses["id_train_probe_loss_by_epoch"][-1],
                    by_epoch=losses["id_train_probe_loss_by_epoch"],
                ),
                round_checkpoint=round_checkpoint,
                round_checkpoint_sha256=round_sha,
                seconds=float(time.perf_counter() - round_started),
            )
            FA._write_json(
                os.path.join(round_dir, "ROUND_COMPLETE.json"), record
            )
            history.append(record)
            current_checkpoint = round_checkpoint
            print(
                f"[{name}] round {round_i}/{rounds} scenarios={list(scenarios)} "
                f"gather {gather_seconds:.1f}s (+{pool[-1]['label']}: "
                f"{record['pool_contexts_by_shard'].get(str(pool[-1]['label']))}"
                f" ctx) pool {record['pool_shards']} shards / "
                f"{record['pool_contexts']} contexts | refit "
                f"{refit_seconds:.1f}s train "
                f"{record['loss']['train_first']:.5f}->"
                f"{record['loss']['train_last']:.5f} ID "
                f"{record['id_probe']['first']:.5f}->"
                f"{record['id_probe']['last']:.5f} | {round_sha[:12]} | "
                f"{record['seconds']:.1f}s",
                flush=True,
            )

            # -- 4. milestone evaluation (sequential inside the arm) -------
            if round_i in eval_rounds:
                milestone = _run_milestone_eval(
                    round_checkpoint, round_i,
                    os.path.join(evals_dir, f"round_{round_i:02d}"),
                    eval_gpu=int(args.eval_gpu),
                    workers=int(args.eval_workers),
                    r0=os.path.abspath(args.r0),
                    cache_dir=os.path.abspath(args.eval_cache),
                    m_per_gamma=int(args.eval_m),
                )
                milestones.append(milestone)
                record["milestone_eval"] = milestone
                FA._write_json(
                    os.path.join(round_dir, "ROUND_COMPLETE.json"), record
                )

            FA._write_json(
                os.path.join(output_root, "campaign_progress.json"),
                dict(status=PROGRESS_STATUS, name=name, config=config,
                     rounds_done=len(history), rounds_total=rounds,
                     latest=record, milestones=milestones,
                     seconds=float(time.perf_counter() - started)),
            )

    # -- 5. campaign summary ------------------------------------------------
    config["status"] = STATUS
    summary = dict(
        status=STATUS,
        name=name,
        source=FA._source(),
        config=config,
        rounds=history,
        milestones=milestones,
        milestone_evals={
            f"r{entry['round']}": dict(
                status=entry["status"],
                output_dir=entry["output_dir"],
                metrics_json=entry.get("metrics_json"),
                pooled=entry.get("cell", {}).get("pooled"),
                r0_pooled=entry.get("r0", {}).get("pooled"),
                delta=entry.get("delta"),
                paired=entry.get("paired"),
                seconds=entry["seconds"],
            )
            for entry in milestones
        },
        final_checkpoint=current_checkpoint,
        final_checkpoint_sha256=FA._sha256_file(current_checkpoint),
        final_pool=[
            dict(path=e["path"], label=e["label"], source=e["source"],
                 sha256=e["sha256"]) for e in pool
        ],
        final_pool_shards=len(pool),
        final_pool_contexts=(
            int(history[-1]["pool_contexts"]) if history else 0
        ),
        seconds=float(time.perf_counter() - started),
    )
    FA._write_json(complete_path, summary)
    print(f"[{name}] CAMPAIGN COMPLETE in {summary['seconds'] / 60.0:.1f} min "
          f"-> {complete_path}", flush=True)
    return summary


# --------------------------------------------------------------- verify-warm
def verify_warm(args):
    """Refit on the five warm-start shards alone and compare to Track F F1e."""
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    started = time.perf_counter()
    payload = refit(
        list(WARM_START_SHARDS), list(WARM_START_LABELS), output_dir,
        device=str(args.device), seed=int(args.train_seed),
    )
    reference = None
    if os.path.isfile(F1E_CHECKPOINT):
        reference = FA._sha256_file(F1E_CHECKPOINT)
    produced = payload["outputs"]["checkpoint_sha256"]
    result = dict(
        status=VERIFY_STATUS,
        source=FA._source(),
        output_dir=output_dir,
        pool_contexts=int(payload["pool"]["contexts"]),
        pool_windows=int(payload["pool"]["windows"]),
        trainable_parameters=int(payload["freeze"]["trainable_parameters"]),
        reference_checkpoint=F1E_CHECKPOINT,
        reference_sha256=reference,
        produced_sha256=produced,
        sha256_match=(reference is not None and reference == produced),
        train_loss_last=payload["losses"]["train_pool_mean_per_epoch"][-1],
        id_probe_last=payload["losses"]["id_train_probe_loss_by_epoch"][-1],
        seconds=float(time.perf_counter() - started),
    )
    FA._write_json(os.path.join(output_dir, "VERIFY_WARM.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "source"},
                     indent=1), flush=True)
    return result


# ------------------------------------------------------------------- parsing
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    def _common(target):
        target.add_argument("--device", default="cuda")
        target.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
        target.add_argument("--batch", type=int, default=128)
        target.add_argument("--epochs", type=int, default=20)
        target.add_argument("--demo-frac", type=float, default=0.5)

    campaign = sub.add_parser("campaign")
    campaign.add_argument("--name", required=True)
    campaign.add_argument(
        "--selector", default="margin",
        choices=("margin", "progress_gated_margin"),
    )
    campaign.add_argument(
        "--warm-start", default="none", choices=("none", "fastlab5"),
    )
    campaign.add_argument("--initial-checkpoint", default=PRETRAINED)
    campaign.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    campaign.add_argument(
        "--scenario-ep0", type=int, default=DEFAULT_SCENARIO_EP0,
    )
    campaign.add_argument("--output-root", required=True)
    campaign.add_argument("--workers", type=int, default=30)
    campaign.add_argument("--eval-gpu", type=int, default=3)
    campaign.add_argument("--eval-rounds", default=DEFAULT_EVAL_ROUNDS)
    campaign.add_argument("--eval-workers", type=int, default=20)
    campaign.add_argument("--eval-m", type=int, default=EVAL_M)
    campaign.add_argument("--eval-cache", default=EVAL_CACHE)
    campaign.add_argument("--r0", default=R0_CHECKPOINT)
    campaign.add_argument(
        "--gammas", type=float, nargs="+", default=tuple(map(float, SS.GAMMAS)),
    )
    campaign.add_argument("--T", type=int, default=DEFAULT_T)
    campaign.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    campaign.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED)
    campaign.add_argument("--ell", type=float, default=DEFAULT_ELL)
    campaign.add_argument("--gp-cap", type=int, default=512)
    campaign.add_argument("--ess-target", type=float, default=0.5)
    _common(campaign)
    campaign.set_defaults(func=run_campaign)

    verify = sub.add_parser("verify-warm")
    verify.add_argument("--output-dir", required=True)
    _common(verify)
    verify.set_defaults(func=verify_warm)
    return parser


SUBCOMMANDS = ("campaign", "verify-warm")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `campaign` is the default: the driver's flat flag form
    # (`--name ... --output-root ...`) is what the launchers use.
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["campaign", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.error("a subcommand is required")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
