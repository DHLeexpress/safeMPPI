# table_v6 — pure AFE-minimal Safe Flow Expansion (2026-07-16)

Protocol: M=40/γ × 7 γ, T=350, reach 0.15, start (0.3,0.3), goal (4.7,4.7), 8-plug walls; all
numbers pooled from `row_g*.json` (single consistent source). Gather-time columns from each arm's
`probe.jsonl` (800 shielded episodes / 100 rounds). a-d = per-γ axes won vs expert (28 = 7γ × 4).

| method | SR | CR | clearance | time [s] | covΣ | a-d | gather deaths | V̂_adv (end) |
|---|---|---|---|---|---|---|---|---|
| **Ours — pure AFE, exec∝π (pure_pi_s910)** | **0.954** | **0.046** | 0.253 | 11.26 | **34** | 7/28 | **0 / 800** | 0.438 (flat) |
| pure AFE, exec=argmax-prog (s910/911/912) | 0.90–0.94 | 0.06–0.10 | 0.245–0.248 | 10.3–10.6 | 26–30 | 5–11/28 | 0 / 800 | 0.43–0.44 (flat) |
| pure AFE, λ=0.01 reference | 0.936 | 0.064 | 0.249 | 10.46 | 29 | 11/28 | 0 / 800 | 0.44 (flat) |
| −Verifier (brother) | 0.954 | 0.046 | 0.249 | 10.73 | 34 | 8/28 | **68 / 800 (8.5%)** | 0.435 |
| −Fallback (brother) | 0.946 | 0.054 | 0.251 | 11.09 | 33 | 6/28 | **39 / 800 (4.9%)** | 0.435 |
| −Prox (brother) | **1.000** | **0.000** | **0.262** | 10.91 | **18** | **24/28** | 0 / 800 | **0.393 (−4.5 pts)** |
| Pretrained (base) | 0.918 | 0.082 | 0.257 | 10.82 | **52** | 9/28 | — | 0.438 |
| Curriculum+anchor recipe (prev. Ours) | 0.943 | 0.057 | 0.255 | 11.08 | 51 | 8/28 | (uncertified gather) | — |
| CFM-MPPI* (goal-tuned) | 0.983 | 0.017 | 0.241 | 5.44 | 8 | — | — | — |
| CFM-MPPI* (safety-calibrated) | 0.350 | 0.000 | 0.364 | 12.04 | 3 | — | — | — |
| Expert (SafeMPPI) | 1.000 | 0.000 | 0.259 | 10.80 | 56 | — | — | — |

## How to read it (the paradigm, gate by gate)
- **Verifier (full window SOCP BEFORE execution)** buys the certified training set and gather-time
  safety: without it, 8.5% of gathering episodes end in collision/OOB and uncertified plans enter
  D⁺ (dithering share ×4). Its effect on final eval CR is small here because the base policy is
  already mostly safe — the guarantee, not the eval delta, is the point.
- **Certified SafeMPPI fallback** is what makes shielded gathering collision-free (0 vs 39 deaths).
  Its location tells the expansion story: mid-route fallbacks are learned away (12→3), the
  goal-corner ones persist (31→40) because no certifiable plan exists there to learn from.
- **Proximal bound** is the validity/diversity preserver, NOT an SR maximizer: removing it gives the
  best closed-loop numbers (SR 1.0, CR 0, a-d 24/28) while collapsing routes (covΣ 18 ≈ 2.6/γ) and
  ERODING the untilted audit validity (V̂_adv 0.438→0.393) — it trades off-distribution validity
  mass for on-task exploitation. This is the measured content of "bounded continued pretraining".
- **Execution rule ∝π** (not argmax-progress) is the pure method's diversity lever: covΣ 34 vs
  26–30, and the best SR/CR of the shielded pure arms. No arm reaches the pretrained's own
  diversity (52); the curriculum+anchor recipe preserved it (51) at the cost of every stabilizer
  the pure method removed.
- **CFM-MPPI*** shows the safety-reliability trap: goal-tuned = fast but lowest clearance and 8
  routes; safety-calibrated = highest clearance but SR 0.35 (circles to timeout). No single tuning
  gives reliability + safety + diversity.

Numbers provenance: `results/p2/eval_afe_*`, `eval_base_pretrained_g47`, `eval_faithful_div_it100`,
`expert_g47`, `kazuki_g47(_trap)`; gather-time from `results/afe/*/probe.jsonl`.
