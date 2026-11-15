# Pre-registration: spectral (low-degree Walsh) scoring vs AutoND bias scoring

**Frozen 14 Nov 2026, before any neural training for E1.** Proxy scores and the candidate selection had already been computed (`results/novel/e1_speck3264/selection.json`). They use no neural information.

## Questions
- **Q1 (proxy validity):** Which cheap score better predicts the held-out accuracy of a trained DBitNet across candidate input differences?
- **Q2 (certified floor):** Is the certified accuracy lower bound `½ + (max|ĉ| − ε)/4` respected by trained networks? A violation means the network is under-trained relative to a one-mask rule, which is a diagnostic in its own right.
- **Q3 (selection):** Does the top candidate by the spectral score reach at least the neural accuracy of the top candidate by the AutoND score?

## Testbed (E1)
- **Cipher and candidates:** SPECK32/64, single-key. The 24 candidates were selected by the rule in `novel_e1_score_candidates.py`, plus the reference differences 0x400000 and 0x102000.
- **Proxy scores:** t = 5·10⁴ per round. The AutoND weighted bias score `Σ_{r≤R} r·b̃_r` uses R = 5, AutoND's own highest biased round for this pool. The spectral scores are `Ĉ_{≤2}` and `max|ĉ|` at rounds 6 and 7.
- **Neural recipe, reduced scale and identical for every candidate:** the PyTorch port of DBitNet (validated against AutoND's released weights). Training is staged over rounds 5 → 6 → 7 with 3 epochs per round and 2·10⁵ fresh training samples per round. Each round has a selection set of 5·10⁴, used only to keep the best-validation-loss weights. BatchNorm statistics are recalibrated, and neural seed 1 is used.
- **Test:** 2·10⁵ fresh samples at rounds 6 and 7 from frozen stream `test_seed = 770001`, evaluated once per candidate.
- **Training-noise estimate:** 4 candidates are re-trained with seed 2.

## Analysis (fixed in advance)
- **Primary:** Spearman ρ between the round-7 test accuracy and (a) the AutoND weighted bias score and (b) the spectral `Ĉ_{≤2}` at round 7. Report the 95% bootstrap CI over candidates (10,000 resamples) and the bootstrap CI of ρ_b − ρ_a.
- **Secondary:** the same at round 6; `max|ĉ|` as a proxy; AutoND bias at single rounds 6 and 7; certified-bound violations; the Q3 comparison.
- **Decision rule:** "the spectral score predicts better" only if the 95% CI of ρ_b − ρ_a at round 7 excludes 0. Otherwise the result is reported as inconclusive or negative.

## Not claimed
- These are reduced-scale networks: 2·10⁵ samples versus the paper's 10⁷. Absolute accuracies are lower than the paper's.
- The claim covers ranking validity on SPECK32 at this scale only, until it is replicated on further ciphers and at full scale.

---

# E5: allocation policies with both statistics (frozen 15 Nov 2026, before running)

- **Pool and K:** the frozen SPECK32 pool from E1 (`results/novel/e1_speck3264/pool.json`, 97 candidates), K = 8, 3 seeds. The fixed budget is t_fixed = 10⁴ per round.
- **Statistics:**
  - *bias*: AutoND's weighted score Σ_{r≤R} r·b̃_r with R = 5, AutoND's highest biased round for this pool.
  - *spectral*: the unbiased C_{≤2} at round R+1 = 6. This is the same deepest-round score used for selection in E1 and E3.
- **Policies:** A fixed; B adaptive (hard cap = cost(A), t₀ = 250, doubling, α = 0.01, tolerance 0.05·S_K); C uniform at B's actual spend; C2 successive halving under the same cap.
- **Reference:** each statistic at t = 10⁵ on independent streams.
- **Outcomes:** regret and top-K overlap against the *same statistic's* reference, plus encryptions.
- **Decision rule:** the same as in `preregistration.md` §4. B "wins on allocation" only if C at matched spend is worse than B by more than one pooled SD.
