"""Shared sys.path bootstrap for the overnight_run_2026-06-30 smoke test.

Import this first from every module in this folder so the bare-name imports used by
`overnight_run_today/src/*` (``from dynamics import ...``) and the package import
``cfm_mppi.safegpc_adapter.safemppi`` both resolve.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))                       # cfm_mppi repo root
RUN_TODAY_SRC = os.path.join(ROOT, "overnight_run_today", "src")       # dynamics/flow_policy/safeflow/...
RUN_28 = os.path.join(ROOT, "overnight_run_2026-06-28")               # best_area_mode4.json, di_gap idioms

BEST_CONFIG = os.path.join(RUN_28, "best_area_mode4.json")

for _p in (HERE, RUN_TODAY_SRC, ROOT, RUN_28):
    if _p not in sys.path:
        sys.path.insert(0, _p)
