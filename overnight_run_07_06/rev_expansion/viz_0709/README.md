# viz_0709 — micro-experiment round (all arms finished)
Reading order:
1. 01_data_566trajs_per_gamma — the ACTUAL druni dataset: 566 uniform starts x 7 gamma SafeMPPI expert trajs (per-gamma minis below).
2. 02_early_iters_easy_vs_sigma_base_vs_combo — THE disease/cure evidence: windows colored by sigma, green ring = stamped easy.
   base rings sigma=1.0 origin windows from it1; combo: "easy 0 - ALL frontier" until sigma < 0.25 (first easy it5, sigma 0.24).
3. 03_micro20_first20iters_5fixes — 20-iter probes of the 5 single fixes vs control (escape frac / heading std / composition).
4. 04_micro100_SR50_cov_internals_6arms — 100-iter, per-iter M=50 probe: SR50/coverage/CR/gradRMS. WINNER m_combo:
   SR50 last10 0.78, std .04, floor 0.72 after it30, 7-gamma SR 0.82.
5. 05_pile_axis_vs_freshonly — P-series (pile revival) FAILS: all oscillate worse despite smooth grads + diverse batches;
   ablations confirm warm-up helps (P5>>P5_nw) and replace=False helps (P5>>P5_rT, cov->1).
6/7. style refs (uniform starts, gamma colorbar).
Arms glossary: m_base=locked recipe; m_bup=beta 0.3->1.0; m_strat=batch round-robin across rollouts;
m_sigabs=absolute sigma<0.25 easy gate + demo backfill; m_skipf=first-2 windows never easy; m_combo=all four.
P3/P5/P7=fresh_frac .3/.5/.7 + warmup10 + FIFO3k LRU pile; P5_nw=no warmup; P5_rT=with-replacement.
Raw per-iter data: results/micro100/<arm>/probe.jsonl, results/micro_pile/<arm>/probe.jsonl; snapshots in <arm>/viz_db/.
