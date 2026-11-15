# Related-work matrix (Phase 0)

*Compiled from the research plan and the AutoND paper. The novelty statement must be re-checked against IACR ePrint, ToSC, FSE, CRYPTO and journal work immediately before the final report (Phase 6). That search has **not** been done yet.*

| Work | Setting | Search for input differences | Neural model | Relation to this project |
|---|---|---|---|---|
| Gohr, CRYPTO 2019 | Speck32/64, single-key | Neural difference search (few-shot + greedy, Alg. 2 in the AutoND appendix) | ResNet | The origin of neural distinguishers. Its search is expensive per candidate, which is what motivated AutoND's bias score. |
| Benamira et al., EUROCRYPT 2021 | Speck32 | n/a | analyses Gohr's network | Truncated-differential explanation; the basis of AutoND's bias-score conjecture. |
| Bao et al., ASIACRYPT 2022 | SIMON32 | Manual / differential | SENet + staged training | The specialised SIMON baseline (11 rounds, 0.517). |
| Gohr, Leander, Neumann, AICrypt 2023 | many | manual | automated hyperparameter tuning | Generic baseline that AutoND compares against. |
| **AutoND**, ToSC 2023(3) | 13 primitives | **Evolutionary search + fixed-sample bias score** | DBitNet | **The baseline. We change only how scoring samples are allocated.** |
| Su et al., Inscrypt 2020 (polytopic) | SIMON32 | fixed | multi-difference | Prior multi-difference work; we do not claim that idea. |
| Baksi et al., DATE 2021 | GIMLI, ASCON, … | fixed set | MLP | Multi-difference (E = D). |
| Zhang, Wang, Wang, ePrint 2022/183 | Speck/Simon | fixed | Inception | Architecture line; not in our scope. |
| Wang, Wang, Sun, IET IS 2024 | Speck, related-key | two differences | ResNet | Two-difference distinguishers. |
| AutoND_New (GitHub) | Speck32/64, 8 rounds | two differences (0x400000, 0x502000) | DBitNet | Reports 0.5193; keep it tied to its own setup. |
| Yuan & Wang, ePrint 2025/697 / Comput. J. | SIMON/SIMECK, related-key | multi-differential | — | Related-key; out of our scope. |
| Sci. Reports 2025 (multiple ciphertext pairs) | Speck/Simon | fixed | multi-pair | Different data design; metrics not comparable. |
| PeerJ CS 2025 (Speck32-like rotations) | Speck variants | — | — | Changing rotation parameters has already been studied; out of scope. |
| Cryptography 8(4) 2024 (Secure LCB) | LCB variant | — | — | A cipher-modification case study; out of scope. |
| InputDiffExplorePCA (GitHub) | — | PCA + K-means clustering | — | Its speed-up claim is unverified; a candidate for later comparison only. |
| Gohr, Leander, Neumann, ePrint 2022/1521 | Speck/Simon/… | — | ResNet | Neural accuracy correlates with the distance of the output-difference distribution from uniform (≈ DDT-distinguisher accuracy). **Our Prop. 2–3 make this computable at low degree, with bounds.** |
| Wang & Wang, PeerJ CS 10:e2566 (2024) | SIMON/SIMECK, related-key | **round-weighted bias score** Σ r·b̃_r over HW-1/2 differences | SE-ResNet, 8 pairs | The weighting is also AutoND's; it is still a per-bit score. Not novel for us. |
| "DDC degree", IEEE (2025) | Feistel/SPN/ARX | entropy-inspired cluster metric of the difference distribution, ~1 s per difference | — | A competing proxy that reports beating the bias score. **Full text not accessed (403); must be compared before submission.** |
| Lv, Jin, Cui, J. Supercomputing 80 (2024) | Speck32 | n/a | differential-linear imbalance distinguishers | Shows NDs ≈ DL imbalances: supports using DL correlations as a proxy. They approximate a *given* ND; they do not select input differences. |
| Peyrin, Wang, Zhang, Zheng, ePrint 2026/1858 | Speck | fixed difference | Conv1DFully | Neural masks vs classical DL masks; no selection score. |
| CLAASP PR #660 (Oct 2026) | — | AutoND port in CLAASP | — | Fixes the AutoND port's defects (scored rounds, highest-round rule, mutation bounds, seeding). Engineering, not a new score. |
| Hwang et al., ePrint 2026/1136 | Speck32/64 | fixed | Gohr-style | Fourier parities of *trained* networks explained by modular-addition local constraints. Post-hoc interpretation; **no selection score, no bounds** (checked 15 Nov 2026, abstract). |
| Yuan et al. 2026 [YWZW26] (cited in 2026/1858) | Speck | fixed | trained ND | Fourier framework for the Boolean decision rules of trained networks. Interpretation, not selection. Full text not checked. |
| He, Hu, Niu, Preneel, Wang, ePrint 2026/2013 | ARX, Xoodoo, Ascon | classical DL search (CP models) | none | Walsh-transform unification of DL, rotational-DL and internal-DL correlations. Classical, **no neural component** (checked, abstract). |
| LibFWHT, ePrint 2026/1997 | KATAN-32, RC5-16, … | exact Walsh spectra | none | Exact DL correlations over the full codebook. Tooling, classical. |
| Ebrahimi et al., SAC 2023 (rotational-XOR ND) | AND-RX ciphers | **AutoND-style bias score** + evolutionary search | ND | Another user of AutoND's per-bit score. Our Prop. 1 applies to it too. |
| **Statistical background** | — | Successive elimination; LUCB / top-K arm identification (Even-Dar et al. 2006; Kalyanakrishnan et al. 2012); successive halving (Karnin et al. 2013) | — | **The adaptive scorer applies these established ideas. The claim is the equal-budget evaluation inside AutoND, not the statistical technique.** |

## Narrowed novelty statement (updated 14 Nov 2026 after the literature search)

**Novel to our knowledge.** The 2026 ePrints were re-checked on 15 Nov 2026: Fourier/Walsh work there either *interprets* trained networks or is purely classical. **The DDC-degree paper is still unread in full; that is the outstanding risk.**
1. **A unified spectral view.** AutoND's bias score is exactly a degree-1 L1 Walsh quantity. Gohr et al.'s distance criterion is the full-spectrum (Parseval) capacity. Differential-linear correlations are its degree-2 terms (Props. 1–2).
2. **A certified accuracy bracket** for difference-based distinguishers: ½ + ¼·max|c| ≤ acc\* ≤ ½ + ¼·√C, with a finite-sample (Hoeffding) certificate (Props. 3 and 6).
3. **An unbiased low-degree capacity estimator and a χ² test of known null distribution** to replace AutoND's empirical threshold. AutoND §5.4 itself lists this as future work (Props. 4–5).
4. **Empirical:**
   - E2: degree-2 masks add one round of detection depth at equal samples (t = 10⁵): 3/3 reps on SPECK32, 2/3 on SIMON32.
   - E1: predictive validity against trained DBitNet.
   - E3: drop-in search, with an ablation that separates calibrated stopping from the degree-2 masks.
   - E4/E5: confidence-guided scoring on the unbiased statistic.
5. **A negative, pre-registered result:** confidence-guided *allocation* gives no gain over uniform allocation at matched spend. This holds on SIMON32 (bias) and SPECK32 (bias and spectral). The gain is in stopping: 4–28% of the fixed scoring budget.

**Not novel (cited as prior work):**
- round-weighted bias scores;
- multi-difference distinguishers;
- differential-linear explanations of neural distinguishers;
- successive elimination and bandit top-K identification;
- capacity in multidimensional linear cryptanalysis (Hermelin, Cho and Nyberg). Prop. 2 applies that classical notion; it does not invent it.
