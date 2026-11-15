# Claims this project will not make

1. **No cipher is broken, secure or "unbreakable".** Every result applies only to reduced-round SIMON32/64 in the single-key setting, at the evaluated round counts, under this data distribution (random-bit negatives, as in AutoND), with this model (DBitNet), this training recipe and this compute budget.
2. **A chance-level accuracy (≈ 50%) does not mean a design is secure.** It describes one model on one distribution. NIST SP 800-22 Rev. 1a states that statistical testing cannot certify cryptographic suitability or replace cryptanalysis.
3. **A distinguisher is not a key-recovery attack.** No data or time complexity for key recovery is claimed.
4. **The intervals carry no formal guarantee.** `bias_bounded` intervals are empirical. Their fluctuation term relies on a normal approximation, and they come with no formal coverage guarantee under adaptive stopping. They are validated by repeated trials (Phase 4). The McDiarmid bound is formally valid per look (union-bounded), but it is reported only as a sensitivity analysis.
5. **Novelty is limited to what [related_work_matrix.md](related_work_matrix.md) lists as ours.**
   - **Ours:** the spectral (low-degree Walsh) view of AutoND's score; the certified accuracy bracket; the unbiased estimator with its χ² test; the empirical evaluations; and the pre-registered negative allocation result.
   - **Not ours:** successive elimination, bandit top-K identification, Parseval/capacity and differential-linear correlations are established techniques. The claim is their combination and evaluation in this pipeline.
   - **Still provisional:** the claim stays provisional until the DDC-degree paper has been read in full.
6. **Multiple input differences are not new.** Feeding several differences to a distinguisher is prior work (polytopic distinguishers, Su et al. 2020; Baksi et al. 2021; Wang–Wang–Sun 2024; Yuan–Wang 2025).
7. **Rotated differences are not treated as equivalent.** Rotational variants of a difference are not assumed to induce the same distribution. Their near-equal *scores* are an empirical observation; the round constants and key schedule mean exact equivalence would need a proof.
8. **Unverified speed-up figures are not adopted.** No "30% faster search" (PCA + K-means) or similar figure is used unless it is reproduced under the same budget here.
9. **AutoND_New's 51.93% (8-round Speck32/64, two differences) stays tied to its own setup.** It is not compared with these results.
10. **Ciphers present in AutoND's code but not validated here are not claimed as supported.** That includes SIMECK in `ciphers/`, and GIFT is absent entirely.
11. **Not claimed: that the 2-bit masks explain the extra search round in E3.** The degree-1 ablation shows how much of that gain comes from the calibrated χ² stopping rule instead (see results.md, E3). The clean evidence for the degree-2 contribution is E2, at t = 10⁵.
12. **Not claimed: full-scale neural accuracies from E1.** E1 uses reduced-scale training: 2·10⁵ samples per round versus the paper's 10⁷, 3 epochs, a PyTorch port, one seed (plus 4 repeats). Its results are about *ranking* candidates at that scale. They must be replicated at full scale (GPU) before any claim about absolute accuracy or best-known distinguishers.
13. **Not claimed: that adaptive allocation beats uniform allocation.** The pre-registered criterion failed on SIMON32 (bias) and on SPECK32 (bias and spectral). Only automatic *stopping* is claimed.
14. **Not claimed: anything about ciphers other than SIMON32/64 and SPECK32/64**, in the single-key setting.
15. **Not claimed: novelty over the DDC-degree paper (IEEE 2025).** Its full text was not accessible (IEEE and ResearchGate return 403), so the comparison is outstanding.
