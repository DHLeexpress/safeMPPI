"""One non-resumable optimizer step from a saved exact-certified visualization pool.

This is a controlled batch-size/gradient-variance probe.  It does not gather, loosen
acceptance, or manufacture data; every row comes from a prior ready gather snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT.parent.parent)]

import grid_expand_hardtail as HT  # noqa: E402
import grid_hp_expt as HP  # noqa: E402
import grid_scene as GS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--viz-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hard-quota", type=int, default=48)
    ap.add_argument("--guard-quota", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--boundary-adapter", action="store_true",
                    help="freeze the base policy and train only compact-support boundary residuals")
    ap.add_argument("--adapter-side", choices=["both", "origin", "goal"], default="both")
    ap.add_argument("--adapter-hidden", type=int, default=0)
    ap.add_argument("--origin-gate", type=float, nargs=4, default=None,
                    metavar=("XMAX", "YMAX", "XWIDTH", "YWIDTH"))
    ap.add_argument("--goal-gate", type=float, nargs=4, default=None,
                    metavar=("XMIN", "YMIN", "XWIDTH", "YWIDTH"))
    ap.add_argument("--guard-x0", choices=["", "inbounds", "near-hard"], default="")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hard-side", choices=["both", "origin-ordinary", "origin-start", "goal", "goal-brake"], default="both")
    ap.add_argument("--hard-x0-cand", type=int, default=32)
    ap.add_argument("--hard-x0-select", choices=["worst", "random-oob"], default="worst")
    ap.add_argument("--guard-side", choices=["interior", "origin-start", "origin-ordinary"], default="interior")
    ap.add_argument("--fixed-origin-x0-seed", type=int, default=-1)
    ap.add_argument("--hard-x0-allow-majority", action="store_true")
    ap.add_argument("--hard-gamma-augment", action="store_true",
                    help="relabel physically verified hard rows across all gamma conditions")
    ap.add_argument("--hard-focus-gamma", type=float, default=None)
    ap.add_argument("--endpoint-eta", type=float, default=0.0)
    ap.add_argument("--cfm-eta", type=float, default=1.0)
    ap.add_argument("--guard-teacher-endpoint", action="store_true")
    ap.add_argument("--guard-teacher-ckpt", default=None,
                    help="fixed preservation teacher; trust anchor still uses the branch checkpoint")
    ap.add_argument("--escape-replay", default=None)
    ap.add_argument("--escape-quota", type=int, default=0)
    ap.add_argument("--escape-eta", type=float, default=1.0)
    ap.add_argument("--strip-goal-band", type=float, nargs=8, default=None, metavar="V",
                    help="override cfg.recovery_goal_band (x0 x1 y0 y1 vx0 vx1 vy0 vy1) so the "
                         "hard-row goal-strip flag targets a narrower context band")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, ck = HP.load_hp(args.ckpt, device="cpu")
    if args.boundary_adapter:
        policy.enable_boundary_adapter(args.adapter_hidden)
        if args.origin_gate is not None:
            policy.boundary_origin_gate = tuple(args.origin_gate)
        if args.goal_gate is not None:
            policy.boundary_goal_gate = tuple(args.goal_gate)
    policy = policy.to(device)
    anchor_teacher, _ = HP.load_hp(args.ckpt, device="cpu")
    teacher, _ = HP.load_hp(args.guard_teacher_ckpt or args.ckpt, device="cpu")
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if args.boundary_adapter:
        for p in policy.parameters():
            p.requires_grad_(False)
        field = []
        if args.adapter_side in ("both", "origin"):
            field += list(policy.adapter_origin.parameters())
        if args.adapter_side in ("both", "goal"):
            field += list(policy.adapter_goal.parameters())
        for p in field:
            p.requires_grad_(True)
    else:
        for p in policy.encoder_modules():
            p.requires_grad_(False)
        field = list(policy.trunk.parameters()) + list(policy.head.parameters())
    opt = torch.optim.Adam([{"params": field, "lr": args.lr}])

    db = torch.load(args.viz_db, map_location="cpu", weights_only=False)
    required = ("grid", "low5", "hist", "U", "gamma", "rid", "widx", "mode", "label")
    missing = [k for k in required if k not in db]
    if missing:
        raise RuntimeError(f"saved pool missing fields: {missing}")
    if not all(bool(db.get(k)) for k in ("gamma_ready", "classes_ready", "gamma_class_ready", "mode_ready")):
        raise RuntimeError("saved pool did not pass the original readiness gates")
    fresh = {k: db[k] for k in ("grid", "low5", "hist", "U", "gamma")}
    fresh.update({k: np.asarray(db[k]) for k in ("rid", "widx", "mode")})
    labels = np.asarray(db["label"]).astype(str)

    cfg = HT.CurConfig()
    cfg.batch_cap = args.batch
    cfg.demo_frac = 0.125
    cfg.lwf_eta = 0.05
    cfg.lr = args.lr
    cfg.nfe_explore = 8
    cfg.field_grad_clip = 1.0
    cfg.max_functional_step = 0.025
    cfg.max_anchor_drift = 0.016
    cfg.hard_quota = args.hard_quota
    cfg.guard_quota = args.guard_quota
    cfg.guard_x0 = args.guard_x0
    cfg.hard_x0 = "oob"
    cfg.hard_x0_cand = args.hard_x0_cand
    cfg.hard_x0_select = args.hard_x0_select
    cfg.fixed_origin_x0_seed = args.fixed_origin_x0_seed
    cfg.hard_x0_allow_majority = args.hard_x0_allow_majority
    cfg.endpoint_eta = args.endpoint_eta
    cfg.cfm_eta = args.cfm_eta
    cfg.guard_teacher_endpoint = args.guard_teacher_endpoint
    cfg.escape_quota = args.escape_quota
    cfg.escape_eta = args.escape_eta
    cfg.recovery_origin_band = (0.0, 1.0, -0.05, 0.18, 0.0, 0.45, -0.28, 0.05)
    cfg.recovery_goal_band = (tuple(args.strip_goal_band) if args.strip_goal_band is not None
                              else (4.3, 5.0, 4.6, 5.06, -0.30, 0.30, -0.05, 0.35))
    env = GS.make_grid()
    gamma_aug_certified = {}
    if args.hard_gamma_augment and args.hard_side == "goal-brake":
        flags0 = HT._strip_flags(fresh["low5"].numpy(), cfg)
        states0 = np.stack([HT.GX2.state_from_low5(x) for x in fresh["low5"].numpy()])
        sel = np.where((flags0 == "goal") & (states0[:, 3] > 0.2) &
                       (fresh["U"][:, 0, 1].numpy() < -0.2))[0]
        if not len(sel):
            raise RuntimeError("no goal-brake rows to gamma-augment")
        # The verifier certificate is gamma-dependent.  Never assume that an
        # unchanged physical target can be relabelled safely: certify every
        # duplicated row at its destination gamma before adding it.
        for g in cfg.gammas:
            bad = []
            for i in sel:
                ok, _margin, _residual = HT.GM2.window_socp_stats(
                    states0[i], fresh["U"][i].numpy(), env, float(g))
                if not ok:
                    bad.append(int(i))
            gamma_aug_certified[str(float(g))] = int(len(sel) - len(bad))
            if bad:
                raise RuntimeError(
                    f"gamma augmentation is not exactly certified at gamma={g}: rows={bad}")
        tensor_add = {k: [] for k in ("grid", "low5", "hist", "U", "gamma")}
        array_add = {k: [] for k in ("rid", "widx", "mode")}
        for j, g in enumerate(cfg.gammas):
            tensor_add["grid"].append(fresh["grid"][sel].clone())
            low = fresh["low5"][sel].clone(); low[:, 4] = float(g); tensor_add["low5"].append(low)
            tensor_add["hist"].append(fresh["hist"][sel].clone())
            tensor_add["U"].append(fresh["U"][sel].clone())
            tensor_add["gamma"].append(torch.full((len(sel),), float(g)))
            array_add["rid"].append(fresh["rid"][sel] + (j + 1) * 100000)
            array_add["widx"].append(fresh["widx"][sel])
            array_add["mode"].append(fresh["mode"][sel])
        for k, v in tensor_add.items(): fresh[k] = torch.cat([fresh[k]] + v)
        for k, v in array_add.items(): fresh[k] = np.concatenate([fresh[k]] + v)
        labels = np.concatenate([labels, np.repeat("easy", len(sel) * len(cfg.gammas))])
    easy = np.where(labels == "easy")[0]
    frontier = np.where(labels == "frontier")[0]
    fresh["strip"] = HT._strip_flags(fresh["low5"].numpy(), cfg)
    if args.hard_side != "both":
        modes = np.asarray(fresh["mode"]).astype(str)
        if args.hard_side in ("origin-ordinary", "origin-start"):
            keep = (fresh["strip"] == "origin") & ~np.char.startswith(modes, "recovery_")
            if args.hard_side == "origin-start":
                keep &= np.asarray(fresh["widx"]) == 0
        else:
            keep = fresh["strip"] == "goal"
            if args.hard_side == "goal-brake":
                states = np.stack([HT.GX2.state_from_low5(x) for x in fresh["low5"].numpy()])
                keep &= (states[:, 3] > 0.2) & (fresh["U"][:, 0, 1].numpy() < -0.2)
        fresh["strip"][~keep] = ""
    if args.hard_focus_gamma is not None:
        fresh["strip"][~np.isclose(np.asarray(fresh["gamma"]), args.hard_focus_gamma)] = ""
    if args.guard_side in ("origin-start", "origin-ordinary"):
        modes = np.asarray(fresh["mode"]).astype(str)
        ordinary = ~np.char.startswith(modes, "recovery_")
        if args.guard_side == "origin-start":
            ordinary &= np.asarray(fresh["widx"]) == 0
        ordinary &= HT._strip_flags(fresh["low5"].numpy(), cfg) == "origin"
        fresh["guard"] = np.where(ordinary, "origin", "").astype("U8")
    demo = HT._load_demo(cfg)
    escape_replay = (torch.load(args.escape_replay, map_location="cpu", weights_only=False)
                     if args.escape_replay else None)
    anchor_teacher = anchor_teacher.to(device).eval()
    if args.adapter_side == "goal" and escape_replay is not None:
        n = min(224, len(escape_replay["x0"]))
        ai = torch.linspace(0, len(escape_replay["x0"]) - 1, n).long()
        gen = torch.Generator().manual_seed(20260711)
        ax = torch.randn(n, policy.d, generator=gen).to(device)
        at = torch.full((n,), 0.5, device=device)
        ag = escape_replay["grid"][ai].to(device); al = escape_replay["low5"][ai].to(device)
        ah = escape_replay["hist"][ai].to(device)
        with torch.no_grad():
            ar = anchor_teacher.forward(
                ax, at, anchor_teacher._expand_ctx(anchor_teacher.ctx_from(ag, al, ah), n)).detach()
        anchor = {"grid": ag, "low5": al, "hist": ah, "x": ax, "tau": at, "ref": ar}
    else:
        anchor = HT._make_origin_trust_anchor(anchor_teacher, env, list(cfg.gammas), device)
    stats = HT.update_flow_fresh(
        policy, opt, fresh, easy, frontier, tuple(db["mix"]), args.steps, cfg,
        field, [], device, demo=demo, teacher=teacher, pile=None,
        trust_anchor=anchor, env=env, escape_replay=escape_replay,
    )
    if stats is None:
        raise RuntimeError(f"update rejected: {stats}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "config": policy.config(),
        "iter": float(ck.get("iter", 0)) + 1,
        "srcr": {},
        "recipe": {
            "algorithm": "one_step_exact_certified_pool_variance_probe",
            "source_checkpoint": args.ckpt,
            "source_viz_db": args.viz_db,
            "seed": args.seed,
            "batch": args.batch,
            "hard_quota": args.hard_quota,
            "guard_quota": args.guard_quota,
            "boundary_adapter": bool(args.boundary_adapter),
            "adapter_side": args.adapter_side,
            "adapter_hidden": args.adapter_hidden,
            "origin_gate": list(policy.boundary_origin_gate) if args.boundary_adapter else None,
            "goal_gate": list(policy.boundary_goal_gate) if args.boundary_adapter else None,
            "guard_x0": args.guard_x0,
            "steps": args.steps,
            "hard_side": args.hard_side,
            "hard_x0_cand": args.hard_x0_cand,
            "hard_x0_select": args.hard_x0_select,
            "guard_side": args.guard_side,
            "fixed_origin_x0_seed": args.fixed_origin_x0_seed,
            "hard_x0_allow_majority": bool(args.hard_x0_allow_majority),
            "hard_gamma_augment": bool(args.hard_gamma_augment),
            "gamma_aug_certified": gamma_aug_certified,
            "hard_focus_gamma": args.hard_focus_gamma,
            "endpoint_eta": args.endpoint_eta,
            "cfm_eta": args.cfm_eta,
            "guard_teacher_endpoint": bool(args.guard_teacher_endpoint),
            "guard_teacher_ckpt": args.guard_teacher_ckpt or args.ckpt,
            "escape_replay": args.escape_replay,
            "escape_quota": args.escape_quota,
            "escape_eta": args.escape_eta,
            "valid2_unchanged": True,
            "deployment_only": True,
            "stats": stats,
        },
        "train_state": None,
        "resumable": False,
    }
    torch.save(payload, out)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(payload["recipe"], f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
