"""Collection-only guided-positive gather for one winner checkpoint.

This driver exists so a checkpoint produced elsewhere (the fastlab winner) can
be dropped into exactly the collection protocol
``claude_guided_positive_pilot.py`` used for its round 1, and nothing else.  It
runs ``sfm_b1_kazuki_repair_audit.collect`` once, with ``round_i=1``, no
previous GP support, neutral continuation on, and the proactive locked-Kazuki
teacher enabled below ``--guided-until-step``.  There is **no training, no
checkpoint write, and no evaluation**: the output is the gather directory the
D+/D0/G+ acquisition-support renderer consumes
(``claude_gplus_acquisition_viz.py``).

Every protocol constant that the pilot pins is pinned here identically and
recorded in ``gather_config.json``: the two scenarios ``ep0`` and ``ep0 + 1``,
the seven-gamma sweep, T=180, margin selection, the double-shift OOD scene, the
GP length scale and cap, ESS target 0.5, sample seed 700000 and audit seed
20260730, and the spawn ``ProcessPoolExecutor`` that owns the exact verifier
workers.  The checkpoint SHA is checked here *and* passed to ``collect`` as
``expected_checkpoint_sha256``, so a wrong file fails before any compute.

Usage::

    python claude_winner_gather.py \
        --checkpoint .../round_10.pt --checkpoint-sha256 <sha> \
        --outdir .../fastlab/viz/winner_gather --workers 40 --device cuda
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
import time

import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_full_episode_audit as FA
import sfm_b1_kazuki_repair_audit as RA
import sfm_b1_neutral_multiround as MR
import sfm_protocol as SP


STATUS = "CLAUDE_WINNER_GATHER_COMPLETE"
# Pinned to claude_guided_positive_pilot.py round 1; see module docstring.
SCENE_PROFILE = "double_density_velocity_ood"
SELECTOR = "margin"
T = 180
ROUND_I = 1
ESS_TARGET = 0.5
SAMPLE_SEED = 700_000
AUDIT_SEED = 20_260_730
DEFAULT_SCENARIO_EP0 = 260_000
DEFAULT_GUIDED_UNTIL_STEP = 50
GUIDED_GENERATOR = "kazuki_full"


def _summary(gather_dir):
    """Shard counts from the collector's own authenticated markers."""
    with open(os.path.join(gather_dir, "COMPLETE.json")) as stream:
        marker = json.load(stream)
    if marker["status"] != RA.STATUS:
        raise RuntimeError("gather marker is not an authenticated audit")
    guided = marker.get("guided_positive_shard") or {}
    return dict(
        status=marker["status"],
        round=int(marker["round"]),
        checkpoint_sha256=marker["checkpoint_sha256"],
        Dplus=int(marker["executed_shard"]["Dplus"]),
        Dminus=int(marker["executed_shard"]["Dminus"]),
        executed_contexts=int(marker["executed_shard"]["contexts"]),
        D0=int(marker["neutral_shard"]["D0"]),
        Gplus=int(guided.get("Gplus", 0)),
        Gplus_contexts=int(guided.get("contexts", 0)),
        guided_verifier_queries=int(
            marker["counts"].get("guided_proactive_verifier_queries", 0)
        ),
        guided_verifier_positive=int(
            marker["counts"].get("guided_proactive_verifier_positive", 0)
        ),
        neutral_executions=int(marker["counts"].get("neutral_executions", 0)),
        executed_windows=int(marker["counts"].get("executed_windows", 0)),
        counts=marker["counts"],
        outcomes=marker.get("outcomes"),
    )


def run(args):
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(f"refusing to reuse output root: {outdir}")
    expected_sha = str(args.checkpoint_sha256).strip().lower()
    if len(expected_sha) != 64:
        raise ValueError("--checkpoint-sha256 must be a 64-hex-digit digest")
    observed_sha = FA._sha256_file(checkpoint)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected {expected_sha}, "
            f"observed {observed_sha}"
        )
    guided_until_step = int(args.guided_until_step)
    if not 1 <= guided_until_step <= T:
        raise ValueError("--guided-until-step must lie in [1, T]")
    guided_topk = int(args.guided_topk)
    if not 1 <= guided_topk <= 16:
        raise ValueError("--guided-topk must lie in [1, K]")
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be positive")
    gammas = tuple(map(float, SP.GAMMAS))
    scenarios = (
        int(args.scenario_ep0) + (ROUND_I - 1) * 2,
        int(args.scenario_ep0) + (ROUND_I - 1) * 2 + 1,
    )

    # Load once on CPU: proves the file is a strict SFM Hp10 checkpoint and
    # records its weight digest before any verifier worker is spawned.  The
    # collector loads it again itself on --device.
    policy, _ = GPS.load_sfm_policy(checkpoint, device="cpu")
    policy_sha = BX.policy_sha256(policy)
    del policy

    os.makedirs(outdir)
    gather_dir = os.path.join(outdir, "gather")
    config = dict(
        status=STATUS,
        mode="collection_only",
        source=FA._source(),
        checkpoint=checkpoint,
        checkpoint_sha256=observed_sha,
        policy_sha256=policy_sha,
        gather_dir=gather_dir,
        scenarios=list(scenarios),
        scenario_ep0=int(args.scenario_ep0),
        gammas=list(gammas),
        scene_profile=SCENE_PROFILE,
        selector=SELECTOR,
        T=T,
        round_i=ROUND_I,
        previous_executed_path=None,
        neutral_continuation=True,
        guided_until_step=guided_until_step,
        guided_generator=GUIDED_GENERATOR,
        guided_topk=guided_topk,
        crunch_full_pool_until_step=0,
        ell=float(RA.DEFAULT_ELL),
        gp_cap=int(MR.DEFAULT_GP_CAP),
        ess_target=float(ESS_TARGET),
        sample_seed=SAMPLE_SEED,
        audit_seed=AUDIT_SEED,
        device=str(args.device),
        workers=workers,
        trains=False,
        evaluates=False,
        mirrors="claude_guided_positive_pilot.py round 1 collection",
    )
    FA._write_json(os.path.join(outdir, "gather_config.json"), config)
    print(
        f"[winner-gather] checkpoint {checkpoint}\n"
        f"[winner-gather] sha {observed_sha}\n"
        f"[winner-gather] scenarios {scenarios} gammas {gammas}\n"
        f"[winner-gather] guided kazuki_full t<{guided_until_step} "
        f"topk {guided_topk} | workers {workers} device {args.device}",
        flush=True,
    )

    started = time.perf_counter()
    spawn = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=spawn,
    ) as executor:
        RA.collect(
            checkpoint,
            scenarios=scenarios,
            gammas=gammas,
            scene_profile=SCENE_PROFILE,
            selector=SELECTOR,
            device=args.device,
            verifier_workers=workers,
            sample_seed=SAMPLE_SEED,
            audit_seed=AUDIT_SEED,
            ell=float(RA.DEFAULT_ELL),
            neutral_continuation=True,
            guided_collect_until_step=guided_until_step,
            guided_generator=GUIDED_GENERATOR,
            guided_topk=guided_topk,
            crunch_full_pool_until_step=0,
            round_i=ROUND_I,
            expected_checkpoint_sha256=observed_sha,
            previous_executed_path=None,
            gp_cap=int(MR.DEFAULT_GP_CAP),
            ess_target=float(ESS_TARGET),
            verifier_executor=executor,
            T=T,
            outdir=gather_dir,
        )
    seconds = float(time.perf_counter() - started)

    summary = _summary(gather_dir)
    if summary["checkpoint_sha256"] != observed_sha:
        raise RuntimeError("collector recorded a different checkpoint SHA")
    if guided_until_step > 0 and summary["Gplus"] < 1:
        raise RuntimeError("guided collection produced no certified G+")
    payload = dict(
        status=STATUS,
        config=config,
        gather_seconds=seconds,
        summary=summary,
    )
    FA._write_json(os.path.join(outdir, "winner_gather_summary.json"), payload)
    print(
        f"[winner-gather] done in {seconds / 60.0:.1f} min | "
        f"D+ {summary['Dplus']} D0 {summary['D0']} G+ {summary['Gplus']} "
        f"(G+ contexts {summary['Gplus_contexts']}, "
        f"guided queries {summary['guided_verifier_queries']})\n"
        f"[winner-gather] gather {gather_dir}",
        flush=True,
    )
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-sha256", required=True,
        help="expected digest; checked here and inside collect()",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="new directory; the gather lands in <outdir>/gather",
    )
    parser.add_argument(
        "--scenario-ep0", type=int, default=DEFAULT_SCENARIO_EP0,
        help="round-1 scenarios are ep0 and ep0+1",
    )
    parser.add_argument(
        "--guided-until-step", type=int, default=DEFAULT_GUIDED_UNTIL_STEP,
    )
    parser.add_argument("--guided-topk", type=int, default=2)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
