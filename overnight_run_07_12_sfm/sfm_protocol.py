"""Frozen protocol constants and disjoint seed banks for SFM Hp10/B1."""
from __future__ import annotations

import sfm_scene as SS

GAMMAS = SS.GAMMAS
K = 16
B = 4
T = 180
H = 10
W = 2
BATCH = 128
LR = 1.0e-5
ESS_TARGET = 0.5
ROUNDS = 20
SCENARIOS_PER_ROUND = 8
N_PED = 20

# Demonstrations are below 8,000. Every named bank is mutually disjoint.
PRETRAIN_GATE_EP0 = 12_000
EXPANSION_EP0 = 20_000
SCREEN_EP0 = 50_000
CONFIRM_EP0 = 80_000
KAZUKI_CONFIRM_EP0 = 90_000
SMOKE_EP0 = 110_000


def expansion_scenarios(round_i, *, smoke=False):
    if int(round_i) < 1:
        raise ValueError("round indices start at one")
    base = SMOKE_EP0 if smoke else EXPANSION_EP0
    start = base + (int(round_i) - 1) * SCENARIOS_PER_ROUND
    return tuple(range(start, start + SCENARIOS_PER_ROUND))


def raw_bank(ep0, m_per_gamma):
    return {str(g): tuple(range(int(ep0), int(ep0) + int(m_per_gamma))) for g in GAMMAS}
