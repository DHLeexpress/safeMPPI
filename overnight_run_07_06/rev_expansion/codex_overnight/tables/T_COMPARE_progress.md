# T_COMPARE — expansion progress vs baselines (SR / CR / clearance / time / coverage)

Goal (final claim, M>=100): every gamma SR 100% / CR 0% / clearance > P1 / time < P1 / coverage >= 14 and > P1.

| Method | gamma | SR | CR | Clearance (m) | Time (s) | Coverage | M |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter0 (pretrained) | 0.1 | 44% | 0% | 0.321 | 16.89 | 8 | 25 |
| iter0 (pretrained) | 0.2 | 48% | 4% | 0.314 | 13.66 | 4 | 25 |
| iter0 (pretrained) | 0.3 | 32% | 12% | 0.316 | 11.93 | 3 | 25 |
| iter0 (pretrained) | 0.4 | 40% | 4% | 0.312 | 11.85 | 2 | 25 |
| iter0 (pretrained) | 0.5 | 32% | 4% | 0.319 | 11.71 | 2 | 25 |
| iter0 (pretrained) | 0.7 | 32% | 0% | 0.322 | 12.26 | 2 | 25 |
| iter0 (pretrained) | 1.0 | 24% | 0% | 0.335 | 12.98 | 2 | 25 |
| t104 (selected, pre-repair) | 0.1 | 92% | 0% | 0.305 | 18.20 | 6 | 25 |
| t104 (selected, pre-repair) | 0.2 | 96% | 0% | 0.294 | 13.80 | 4 | 25 |
| t104 (selected, pre-repair) | 0.3 | 96% | 0% | 0.295 | 12.10 | 3 | 25 |
| t104 (selected, pre-repair) | 0.4 | 92% | 0% | 0.297 | 11.78 | 3 | 25 |
| t104 (selected, pre-repair) | 0.5 | 92% | 0% | 0.298 | 11.73 | 3 | 25 |
| t104 (selected, pre-repair) | 0.7 | 92% | 0% | 0.296 | 12.32 | 2 | 25 |
| t104 (selected, pre-repair) | 1.0 | 96% | 0% | 0.297 | 12.65 | 3 | 25 |
| s792 (11/11 flips, M25) | 0.1 | 100% | 0% | 0.306 | 17.80 | 5 | 25 |
| s792 (11/11 flips, M25) | 0.2 | 100% | 0% | 0.292 | 13.70 | 3 | 25 |
| s792 (11/11 flips, M25) | 0.3 | 100% | 0% | 0.295 | 11.89 | 3 | 25 |
| s792 (11/11 flips, M25) | 0.4 | 100% | 0% | 0.294 | 11.54 | 3 | 25 |
| s792 (11/11 flips, M25) | 0.5 | 100% | 0% | 0.295 | 11.60 | 3 | 25 |
| s792 (11/11 flips, M25) | 0.7 | 100% | 0% | 0.294 | 12.16 | 3 | 25 |
| s792 (11/11 flips, M25) | 1.0 | 92% | 0% | 0.297 | 12.35 | 2 | 25 |
| s792 (M100 audit) | 0.1 | 100% | 0% | 0.307 | 17.86 | 10 | 100 |
| s792 (M100 audit) | 0.2 | 99% | 0% | 0.293 | 13.55 | 5 | 100 |
| s792 (M100 audit) | 0.3 | 97% | 2% | 0.296 | 11.92 | 4 | 100 |
| s792 (M100 audit) | 0.4 | 97% | 3% | 0.296 | 11.56 | 4 | 100 |
| s792 (M100 audit) | 0.5 | 97% | 2% | 0.296 | 11.63 | 5 | 100 |
| s792 (M100 audit) | 0.7 | 97% | 1% | 0.297 | 12.12 | 4 | 100 |
| s792 (M100 audit) | 1.0 | 97% | 0% | 0.297 | 12.31 | 3 | 100 |
| P1 SafeMPPI expert | all | 100% | 0% | .281-.333 | 10.54-15.13 | 6-11 | 100 |
| T3 Kazuki tuned 5-coef mix | all | 100% | 0% | .372-.375 | 8.96-10.47 | 5-8 | 200 |

## Kazuki w_safe vulnerability sweep (single coefficient, ALL 200 MPPI samples, M=25, gamma_ctx=.5)

| w_safe | SR | CR | outcome |
|---:|---:|---:|---|
| 0.05 | 0% | 0% | all timeouts |
| 0.3 | 0% | 0% | all timeouts |
| 0.9 | 0% | 0% | all timeouts |
| 2.0 | 0% | 0% | all timeouts |
| 5.0 | 0% | 0% | all timeouts |
| tuned 5-coef mix (T3) | 100% | 0% | requires the exact hand-tuned ensemble |

Reading: no SINGLE safety coefficient completes the task at all; only the hand-tuned mixed ensemble
does. Our method exposes gamma as a conditioning input with a verifier certificate instead of a
fragile cost weight. (Arm-2 hardtail rows to be appended per checkpoint by run_gate.sh.)
