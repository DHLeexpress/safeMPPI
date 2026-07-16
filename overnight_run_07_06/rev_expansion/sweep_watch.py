"""sweep_watch.py — the token-saving "switch". Polls the rev_sweep arms' history.json and EXITS (which
re-invokes the operator) only when a metric goes BAD or all arms finish. Silent otherwise.

Usage: python sweep_watch.py --dir results/rev_sweep --iters 1000 --poll 60 --arms base roll6 roll14 ...
Exit 1 + "BAD: ..." on first trip; exit 0 + summary when every arm has final.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def load_hist(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def bad_reason(arm, hist, done, iters):
    """Return a string reason if this arm looks bad, else None."""
    if not hist:
        return None
    meas = [h for h in hist if h.get("iter", 0) > 0]
    if not meas:
        return None
    last = meas[-1]
    peak = max(h["SR"] for h in hist)
    it = last["iter"]
    # (a) collapse: finished early (collapse-term fired) OR SR far below peak for >=2 late measures
    if done and it < iters - 1:
        return f"collapsed/early-stop at it{it} (SR {last['SR']:.2f}, peak {peak:.2f})"
    late = [h for h in meas if h["iter"] >= 300]
    if len(late) >= 2 and all(h["SR"] < 0.45 * peak for h in late[-2:]) and peak > 0.1:
        return f"SR collapse: last2 < 0.45*peak ({peak:.2f}) at it{it}"
    # (b) class death for >=2 consecutive measures after it200
    cd = [h for h in meas if h["iter"] >= 200]
    if len(cd) >= 2 and all(h.get("n_easy", 1) == 0 for h in cd[-2:]):
        return f"easy class died (n_easy=0) at it{it}"
    if len(cd) >= 2 and all(h.get("n_frontier", 1) == 0 for h in cd[-2:]):
        return f"frontier class died (n_frontier=0) at it{it}"
    # (c) stall: at it>=500 SR still very low
    if it >= 500 and last["SR"] < 0.30:
        return f"stall: SR {last['SR']:.2f} < 0.30 at it{it}"
    # (c2) no valid data: n_valid ~0 for >=2 measures after it200 (degenerate gathering)
    nd = [h for h in meas if h["iter"] >= 200]
    if len(nd) >= 2 and all(h.get("n_valid", h.get("n_pos", 1)) < 8 for h in nd[-2:]):
        return f"no valid samples (n_valid<8) at it{it}"
    # (d) CR blowup: high and rising after it400
    if it >= 400 and len(meas) >= 2 and last["CR"] > 0.5 and last["CR"] > meas[-2]["CR"]:
        return f"CR blowup: CR {last['CR']:.2f} rising at it{it}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--grace-min", type=float, default=60.0, help="trip if no history.json after this")
    ap.add_argument("--stale-min", type=float, default=60.0, help="trip if history.json unchanged this long (crash)")
    args = ap.parse_args()
    t0 = time.time()
    while True:
        done_arms, statuses = [], []
        for a in args.arms:
            adir = os.path.join(args.dir, a)
            hpath = os.path.join(adir, "history.json")
            hist = load_hist(hpath)
            done = os.path.exists(os.path.join(adir, "final.pt"))
            if done:
                done_arms.append(a)
            now = time.time()
            if not done:                                   # crash/stall detection (no final.pt)
                if not os.path.exists(hpath):
                    if (now - t0) / 60.0 > args.grace_min:
                        print(f"BAD: arm {a} — no history.json after {args.grace_min:.0f} min (import/early crash?)", flush=True)
                        sys.exit(1)
                elif (now - os.path.getmtime(hpath)) / 60.0 > args.stale_min:
                    print(f"BAD: arm {a} — history.json stale {args.stale_min:.0f} min, no final.pt (crashed/hung)", flush=True)
                    sys.exit(1)
            r = bad_reason(a, hist, done, args.iters)
            last = (hist[-1] if hist else {})
            statuses.append((a, last.get("iter", -1), last.get("SR"), last.get("CR"),
                             last.get("n_easy"), last.get("n_frontier"), done))
            if r:
                print(f"BAD: arm {a} — {r}", flush=True)
                print("STATUS:", flush=True)
                for s in statuses:
                    print(f"  {s[0]:10s} it{s[1]} SR {s[2]} CR {s[3]} e/f {s[4]}/{s[5]} done={s[6]}", flush=True)
                sys.exit(1)
        if len(done_arms) == len(args.arms):
            print("ALL DONE:", flush=True)
            for s in statuses:
                print(f"  {s[0]:10s} it{s[1]} SR {s[2]} CR {s[3]} e/f {s[4]}/{s[5]}", flush=True)
            sys.exit(0)
        if (time.time() - t0) / 3600.0 > args.max_hours:
            print(f"WATCH TIMEOUT after {args.max_hours}h; done {len(done_arms)}/{len(args.arms)}", flush=True)
            sys.exit(2)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
