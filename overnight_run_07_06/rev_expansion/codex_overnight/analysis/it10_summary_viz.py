"""it10 summary (2026-07-14): the emergent-gamma sweep after 10 iters. Two panels:
(A) per-gamma CLEARANCE (policy vs expert) — the main a-d gap; is 'safer-than-expert' happening?
(B) per-gamma VALID2 rate pretrained vs best arm — are gamma 0.1/0.2 joining (leaving 0%)?
Reads results/p2/eval_*/scorecard.json + hardcoded per_gamma_valid numbers passed in.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); P2 = os.path.dirname(HERE)
G = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# per-gamma valid2 rate (from the it10 report)
VALID = {
    "pretrained": [0, 0, 24, 60, 76, 80, 92],
    "b02": [0, 0, 28, 64, 64, 84, 92],
    "b03": [0, 0, 28, 52, 68, 76, 80],
    "b04": [0, 0, 32, 68, 72, 84, 88],
}


def load(tag):
    p = os.path.join(P2, "results/p2", f"eval_{tag}", "scorecard.json")
    return json.load(open(p))["per_gamma"]


def main():
    pre = load("pretrained_it0"); b03 = load("fsw_b03_it10")
    clr_pre = [pre[str(g)]["clr"] for g in G]
    clr_b03 = [b03[str(g)]["clr"] for g in G]
    exp = [pre[str(g)]["exp_clr"] for g in G]

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6))
    # A: clearance
    a.plot(G, exp, "-o", c="k", lw=2.4, label="expert (target to beat)")
    a.plot(G, clr_pre, "--s", c="#888888", lw=1.8, label="pretrained (it0)")
    a.plot(G, clr_b03, "-^", c="#0072B2", lw=2.2, label="ours it10 (β0.3)")
    a.fill_between(G, clr_b03, exp, where=[c < e for c, e in zip(clr_b03, exp)],
                   color="#cc3311", alpha=0.12)
    a.set_xlabel("γ (safety level)"); a.set_ylabel("clearance (m), successful eps")
    a.set_title("(A) clearance vs expert — 'safer' (c✓) needs to CROSS the black line\n"
                "it10 is FLAT at pretrained level, below expert at every γ", fontsize=12)
    a.legend(fontsize=10); a.grid(alpha=.3)
    # B: valid2 rate
    b.plot(G, VALID["pretrained"], "--s", c="#888888", lw=1.8, label="pretrained")
    b.plot(G, VALID["b03"], "-^", c="#0072B2", lw=2.2, label="ours it10 (β0.3)")
    b.axhspan(-2, 2, color="#cc3311", alpha=0.10)
    b.annotate("γ0.1/0.2 stuck at 0%\n(no certified windows → no fresh training,\nshared weights didn't lift them in 10 it)",
               (0.15, 3), (0.35, 30), fontsize=10, color="#aa3311",
               arrowprops=dict(arrowstyle="->", color="#aa3311"))
    b.set_xlabel("γ (safety level)"); b.set_ylabel("valid2 rate (%)")
    b.set_title("(B) per-γ valid2 — do low γ 'join'?\nγ0.3–1.0 healthy; γ0.1/0.2 still 0%", fontsize=12)
    b.legend(fontsize=10); b.grid(alpha=.3); b.set_ylim(-4, 100)
    fig.suptitle("Emergent-γ expansion at it10: SR/CR improve modestly, but clearance is flat and "
                 "γ0.1/0.2 haven't joined", fontsize=13)
    fig.tight_layout()
    out = os.path.join(P2, "grand_final_reports_rev", "it10_summary.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
