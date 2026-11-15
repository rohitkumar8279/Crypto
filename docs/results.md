# Results log

*Every number here is read from `results/<phase>/<config>/`, and every one of those folders contains a `manifest.json`. CPU results come from the development laptop listed in [setup_notes.md](setup_notes.md). The neural results (Phases 2, 3b and 5b) are **pending a GPU run**.*

## Configuration `simon32_demo` (development / profiling)

The demo uses AutoND's repository demonstration settings: pool search with 5 generations, 10³ samples and T = 0.05. Scoring is compared at `t_fixed = 10³` with K = 8 and 3 seeds. The reference ranking uses t = 2·10⁴.

### Phase 1: verification (all passed)
- The published SIMON32/64 test vector matches in both AutoND and our independent reference. The two implementations agree on 300 random inputs for every round count 1–32, with 0 mismatches.
- Labels after 1 round: 100% of real pairs show the expected right-half difference, versus 0% of random pairs. Label balance is 0.497.
- Our scorer matches AutoND's `evaluate_multiple_differences` exactly: maximum absolute difference 0.0.
- The bias score on random data is 0.01263 at t = 10³ and 0.00400 at t = 10⁴. Theory (√(1/(2πt))) gives 0.01262 and 0.00399; the paper reports ≈ 0.004.
- The pool has 116 candidates, the highest biased round is 7, the run took 9.5 s, and 0x400 is in the pool.

### Phase 3a: score variability

| t (per round) | encryptions | median SD of S | Spearman between reps | top-8 overlap between reps | top-8 overlap with reference |
|---|---|---|---|---|---|
| 100 | 81,900 | 0.070 | 0.949 | 0.61 | 0.63 |
| 300 | 245,700 | 0.047 | 0.953 | 0.71 | 0.63 |
| 1,000 | 819,000 | 0.023 | 0.976 | 0.60 | 0.58 |
| 3,000 | 2,457,000 | 0.014 | 0.978 | 0.61 | 0.65 |

**Finding.** Top-K overlap does **not** improve with more samples. The reference ranking has 13 single-bit differences, the rotational variants, with weighted scores between 6.192 and 6.213, a spread of 0.3%. The next group sits at ≈ 4.93. With K = 8, the cut falls inside the tied group, so *which* 8 get chosen is noise at any affordable t. This is consistent with the paper's observation of 16 near-identical SIMON32 differences.

**Consequence for the design.** The primary quality metric became regret against the reference score, with top-K overlap kept as secondary. A tie tolerance was added for the adaptive scorer. Both changes are logged in [preregistration.md](preregistration.md) and were made before any main-configuration run.

### Phase 4: scorer validation
- **Synthetic coverage at a nominal 90%,** over 200 trials for each cell, t ∈ {250, 1000, 4000}, and 0/4/16/32 biased bits out of 32:
  - `bias_bounded` reached **0.945–1.000**.
  - `mcdiarmid` reached 1.000 (very wide).
  - `basic` bootstrap reached only 0.035–0.895, and `percentile` 0.000–0.940. Both fail whenever unbiased bits are present.
- **On the real cipher pool** (t = 400, 99% intervals, 20 repetitions, against the high-t reference): `bias_bounded` covered 100% with a mean width of 0.94; `mcdiarmid` covered 100% with width 3.58; `basic` covered 54% and `percentile` 1%.
- **Allocation check** (80 synthetic candidates, K = 10):
  - The Spearman correlation between samples spent and closeness to the K-th score is **0.95**.
  - The shortlist was exactly the true top-10.
  - Spending was 194,400 of the 324,000 cap, and the budget was respected.

### Phase 5a: scoring comparison (3 seeds)

| Policy | Budget fraction | Encryptions (mean) | Regret (mean ± SD) | ε-close precision |
|---|---|---|---|---|
| A fixed | 1 | 819,000 | 0.00072 ± 0.00019 | 1.0 |
| A fixed | 1/8 | 102,375 | 0.00051 ± 0.00025 | 1.0 |
| **B adaptive** | 1 (cap) | **180,133** (stopped itself, "resolved") | 0.00097 ± 0.00019 | 1.0 |
| C uniform at B's spend | — | 179,907 | 0.00063 ± 0.00028 | 1.0 |
| C2 successive halving | 1 (cap) | 210,000 | 0.00060 ± 0.00010 | 1.0 |

On seed 1, B ran 3 stages (t = 100 → 200 → 400). It eliminated 98 candidates and froze 18 as practical ties, then stopped.

**Interpretation (demo; neutral result).** All policies reach regret below 0.1% and perfect ε-close precision at every budget tested, including fixed scoring at 1/8 of the budget. B stopped on its own at **22% of A's cost** with equal quality. However, uniform allocation at the same spend (C) is equally good. On this easy pool, which has one dominant tied group far above the rest, the saving comes from **knowing when to stop**, not from adaptive allocation. That does **not** support the allocation part of H2/H3 under the pre-registered criterion. It shows only that the fixed t = 10³ is more than this setting needs. Differences in regret of order 10⁻⁴ are within reference-score noise.

## Configuration `simon32_main` (pre-registered; completed 14 Nov 2026)

These are the paper's search settings: 50 generations, 10⁴ samples, T = 0.01. The pool has **114 candidates** and the highest biased round is **8**; the search took 905 s on CPU. The scoring comparison used K = 16, t_fixed = 10⁴, 5 seeds, and a reference ranking at t = 2·10⁵.

### Phase 3a
- The pool contains all **16** rotational single-bit differences. Their reference scores lie between 6.3135 and 6.3244, a 0.17% spread. The next group starts at 5.035.
- With K = 16 the boundary sits in this gap, so top-16 overlap is **1.0 at every t from 250 to 10⁴**, both between repetitions and against the reference.
- The 0.011 spread among the 16 rotations is about 4.6 reference SDs (SD ≈ 0.0024 at t = 2·10⁵). That is slightly more than the expected range of 16 exchangeable draws (≈ 3.5 SD), so it hints at small real differences from the key schedule beyond round 4 (see Lemma 7 in [method_spectral.md](method_spectral.md)). This is not tested formally.

### Phase 4 (main pool)
- `bias_bounded` coverage was 100% on the real cipher pool, with mean width 0.75.
- `basic` bootstrap covered 45% and `percentile` 0%.
- The allocation check gave Spearman 0.96, with budget respected and the shortlist exact.

### Phase 5a: pre-registered comparison (5 seeds)

| Policy | Encryptions (mean) | Fraction of A | Regret | ε-close precision | top-16 overlap |
|---|---|---|---|---|---|
| A fixed, t = 10⁴ | 9,200,000 | 100% | 0 | 1.0 | 1.0 |
| A fixed, t = 625 (f = 1/16) | 575,000 | 6.3% | 0 | 1.0 | 1.0 |
| **B adaptive** (self-stopped, every seed) | **375,200** | **4.1%** | 0 | 1.0 | 1.0 |
| C uniform at B's spend | 374,808 | 4.1% | 0 | 1.0 | 1.0 |
| C2 successive halving | 466,000 | 5.1% | 0 | 1.0 | 1.0 |

### Verdict against the pre-registered criteria
- **Criterion 1 is not met.** B reaches comparable quality at ≤ 50% of cost(A), but C at B's spend is *not* worse than B. The required condition fails.
- **Criterion 2 is not met.** Every policy ties at every budget fraction.
- **Therefore H2/H3, read as claims about adaptive *allocation*, are not supported on SIMON32/64.** What the data do support is narrower: confidence-based *stopping* found the AutoND-quality shortlist at **4.1% of the paper's scoring budget** without any tuning of t, in every seed. With the paper's settings, the scoring stage is over-provisioned by roughly 25× for this cipher.
- **Interpretation.** For SIMON32 the top of the bias-score ranking is a well-separated group, so careful sample allocation gains nothing. The open question is instead whether the bias score ranks *what the network will learn*. That motivated the spectral-scoring work ([method_spectral.md](method_spectral.md), [novel_preregistration.md](novel_preregistration.md)).

## Novel contribution: spectral (low-degree Walsh) scoring

The method and proofs are in [method_spectral.md](method_spectral.md) and the pre-registration in [novel_preregistration.md](novel_preregistration.md).

### Port validation (prerequisite for the CPU neural experiments)
- **Faithfulness check:** the PyTorch DBitNet port ([cgsearch/torch_dbitnet.py](../cgsearch/torch_dbitnet.py)) was loaded with AutoND's released `simon3264_single-key_0x400_round11_polished.h5`. It gives **0.5181 ± 0.0004** on 5 × 10⁶ fresh samples, with TPR 0.477 and TNR 0.559. AutoND's README documents 0.5182 ± 0.0004 with TPR ≈ 0.478 and TNR ≈ 0.559.
- **Parameter count:** 296,065 trainable plus 2,048 BatchNorm statistics, giving 298,113. The paper reports 298.11k.
- **Reduced-scale adaptation:** BatchNorm statistics are recalibrated before evaluation, because at 2·10⁵ samples there are too few steps per epoch for Keras-momentum running averages to converge.

### E2: detection depth at equal samples

Setup: t = 10⁵ per round, 3 repetitions, Bonferroni α = 0.01 over 13 rounds. Results are in `results/novel/e2_detection_depth/`.

| Difference | AutoND bias score (simulated null) | degree-1 χ² | **degree-≤2 spectral χ²** | DBitNet reach (paper) |
|---|---|---|---|---|
| SIMON32 0x400 | 9 (3/3 reps) | 9 (3/3) | **10 in 2/3 reps, 9 in 1/3** | 11 |
| SIMON32 0x8000 | 9 (3/3) | 9 (3/3) | **10 in 2/3 reps, 9 in 1/3** | — |
| SPECK32 0x400000 | 6 (3/3) | 6 (3/3) | **7 (3/3)**, log p ∈ [−180, −162] | 8 |
| SPECK32 0x102000 | 6 (3/3) | 6 (3/3) | **7 (3/3)**, log p ∈ [−284, −211] | — |

At round 7 on SPECK32, AutoND's bias score has log p between −5.5 and −0.2 across all repetitions and both differences, so it is not significant (the threshold is log(0.01/13) = −7.2). These depths were re-derived from `per_round.csv` during the 15 Nov audit.

**E2 is the clean test of the degree-2 contribution.** Degree-1 χ² and degree-≤2 χ² use the same calibrated test at the same t, and differ only in whether the 2-bit masks are included.

- **SPECK32 strongest mask:** at rounds 4–7 the strongest mask is `bit12 ⊕ bit28` (bit j of each word). That is the keyless one-round partial-decryption feature of SPECK, found here without any cipher knowledge.
- **Certified floors are respected** by the paper's networks on SPECK32 0x400000: 0.718 / 0.631 / 0.518 at rounds 5 / 6 / 7, against network accuracies of 0.929 / 0.788 / 0.616.

### E3: drop-in search (3 seeds, 10 generations, t = 10³, AutoND's unmodified `evo`)

| Cipher | Method | Highest round | Search encryptions | Wall time | Rank of known best |
|---|---|---|---|---|---|
| SPECK32 | AutoND bias, T = 0.05 | 5 (all seeds) | 24.1 M | 38 s | 1 |
| SPECK32 | **spectral, χ² stop** | **6** (all seeds) | 26.3 M (+9%) | 65 s | 1 |
| SIMON32 | AutoND bias, T = 0.05 | 7 (all seeds) | 31.0 M | 44 s | 1 (0x400 orbit) |
| SIMON32 | **spectral, χ² stop** | **8** (all seeds) | 37.7 M (+22%) | 90 s | 1 (0x400 orbit) |

- **Stopping rule:** the spectral search stops on a calibrated test (α = 0.01, a fresh-sample re-test of the selected best) instead of a threshold that depends on t. It covers one more round at a modest extra cost.
- **Orbits:** on SIMON, the top-16 of both rankings fall into 1–2 rotation orbits. Orbit-aware shortlisting would cut the neural budget by about 16× for the same information.

### E5: allocation policies under both statistics (SPECK32 pool, K = 8, 3 seeds, pre-registered)

Results are in `results/novel/e5_compare/`. Each statistic's quality is measured against its own reference at t = 10⁵.

| Statistic | Policy | Encryptions (mean) | Fraction of A | Regret (mean ± SD) | top-8 overlap |
|---|---|---|---|---|---|
| bias | A fixed | 4,900,000 | 100% | 0 | 1.00 |
| bias | B adaptive | 398,333 | 8.1% | 0.0013 ± 0.0022 | 0.96 |
| bias | C uniform at B's spend | 398,207 | 8.1% | **0.0005 ± 0.0008** | 0.96 |
| bias | C2 successive halving | 320,000 | 6.5% | 0.0015 ± 0.0015 | 0.92 |
| spectral | A fixed | 980,000 | 100% | 0 | 1.00 |
| spectral | B adaptive | 269,333 | 27.5% | 0.0035 ± 0.0061 | 0.96 |
| spectral | C uniform at B's spend | 269,271 | 27.5% | **0.0030 ± 0.0052** | 0.96 |
| spectral | C2 successive halving | 64,000 | 6.5% | 0.0149 ± 0.0076 | 0.83 |

- **Verdict against the pre-registered criterion: not supported, for either statistic.** Uniform allocation at B's matched spend is as good as, or better than, adaptive allocation. This replicates the SIMON32 main result on a second cipher and a second statistic.
- **What B does deliver** is automatic stopping at 8–28% of the fixed budget, with regret below 0.4%.
- **Successive halving**, which has no intervals, is cheapest but clearly worst under the spectral statistic: its regret is about 5× B's.
- **The two statistics disagree:** their reference top-8 shortlists overlap by only **0.75**. So the choice of score changes *which* candidates are shortlisted. E1 tests which choice the networks agree with.

### E1: predictive validity against trained DBitNet
*Running (SPECK32, 24 candidates).* Analysis follows the pre-registration.

## Neural phases (original TensorFlow pipeline)

*Pending a GPU run (`scripts/setup_gpu_linux.sh`, then `scripts/run_gpu_phases.sh`). TensorFlow cannot load on the development laptop; see [setup_notes.md](setup_notes.md).*
