# Method: low-degree Walsh-spectrum scoring of input differences

**Implementation:** [cgsearch/spectral.py](../cgsearch/spectral.py), with tests in [tests/test_spectral.py](../tests/test_spectral.py).

## Setting and notation

- E_K is a block cipher with n-bit blocks, reduced to r rounds. δ is an input difference.
- D = E_K(P) ⊕ E_K(P ⊕ δ) ∈ F₂ⁿ, with (P, K) uniform. Its distribution is P_D, and U is the uniform distribution on F₂ⁿ.
- For a mask u ∈ F₂ⁿ, the correlation (Walsh coefficient) is c(u) = E[(−1)^{u·D}] = Σ_x P_D(x)(−1)^{u·x}, with c(0) = 1.
- **Real-vs-random experiment** (AutoND, `main.make_train_data`): label 1 is the ciphertext pair (C₀, C₀ ⊕ D); label 0 is 2n uniform bits; the priors are equal.

## Proposition 1. AutoND's bias score is a degree-1 L1 Walsh quantity

With p_j = Pr[D_j = 1] we have c(e_j) = 1 − 2p_j, so |½ − p_j| = |c(e_j)|/2 and

  b(δ) = (1/n) Σ_j |½ − p_j| = (1/2n) Σ_{j} |c(e_j)|.

The sample version is identical with ĉ in place of c. This holds exactly, and `test_autond_bias_is_degree1_walsh` checks it against AutoND's own code.

**Consequence.** The bias score is blind to every correlation of weight ≥ 2. For example, a pair of output bits can each be unbiased while their XOR is strongly biased (`test_unbiased_capacity_recovers_known_value`).

## Proposition 2. Parseval capacity is the full-spectrum version of Gohr et al.'s criterion

  C(δ) := Σ_{u≠0} c(u)² = 2ⁿ Σ_x P_D(x)² − 1 = 2ⁿ ‖P_D − U‖₂².

Gohr, Leander and Neumann (ePrint 2022/1521) observe that neural accuracy tracks the distance of P_D from uniform. C is the L2 form of that distance, and it decomposes over masks.

## Proposition 3. Certified accuracy bracket

Let acc*_Δ be the best accuracy of any distinguisher that sees only D (equal priors). Then

  ½ + ¼·max_{u≠0} |c(u)|  ≤  acc*_Δ  ≤  ½ + ¼·√C(δ).

*Proof.* The optimal test between P_D and U with equal priors has accuracy ½ + ½·TV(P_D, U).

- **Lower bound:** take A = {x : u·x = 0}. Then P_D(A) = (1 + c(u))/2 and U(A) = ½, so TV ≥ |c(u)|/2.
- **Upper bound:** by Cauchy–Schwarz, TV = ½ Σ_x |P_D(x) − U(x)| ≤ ½ (2ⁿ Σ_x (P_D − U)²)^{1/2} = ½ √C. ∎

**The lower bound is achieved.** The one-mask rule "predict *real* iff u·D = [c(u) < 0]" has accuracy exactly ½ + |c(u)|/4 (`test_one_mask_rule_achieves_lower_bound`).

**It also bounds pair distinguishers.** D is a function of the ciphertext pair, so the lower bound also holds for the optimal pair distinguisher, the one DBitNet approximates. The upper bound holds only for difference-only distinguishers: networks can exploit more than the difference (Gohr 2019, real-difference experiment).

## Truncation to low degree

C_{≤d} = Σ_{1 ≤ wt(u) ≤ d} c(u)².

- **d = 1** is a squared variant of AutoND's score.
- **d = 2** adds the n(n−1)/2 two-bit differential-linear masks (496 for n = 32).

All degree-≤2 correlations come from one Gram matrix: with S ∈ {±1}^{t×n}, ĉ(e_i ⊕ e_j) = (SᵀS)_{ij}/t. The cost is O(t·n²) per candidate, against O(t·n) for AutoND. Encryption remains the dominant cost.

## Proposition 4. Unbiased estimation (no noise floor)

With t i.i.d. samples and ĉ(u) = (1/t) Σ_i (−1)^{u·D_i}:

  E[ĉ²] = c² + (1 − c²)/t,  so  ĉ²_unb := (t·ĉ² − 1)/(t − 1)  is unbiased for c².

In contrast, AutoND's estimator satisfies E[b̃_t] ≥ b by Jensen's inequality, with a floor of about √(1/(2πt)) per unbiased bit: 0.0126 at t = 10³ and 0.0040 at t = 10⁴, which we reproduced in Phase 1. So a score's value depends on t, and the threshold T_b has to be retuned whenever t changes. The same |·| kink makes bootstrap intervals for b̃ badly miscalibrated (Phase 4).

## Proposition 5. A test statistic of known distribution

Under H₀ (D uniform), for distinct non-zero masks u ≠ v we have E[(−1)^{u·D}(−1)^{v·D}] = c(u ⊕ v) = 0. So the variables {√t·ĉ(u)}_{u∈M} are asymptotically independent standard normals, and

  T_M = t · Σ_{u∈M} ĉ(u)²  →  χ²_{|M|}.

This gives the "rigorous hypothesis testing procedure … with a test statistic of known distribution" that AutoND §5.4 proposes as future work. The empirical calibration is checked in `test_chi2_null_calibration`.

## Proposition 6. Finite-sample certificate

Each term (−1)^{u·D_i} lies in [−1, 1]. By Hoeffding's inequality and a union bound over |M| masks, with probability ≥ 1 − α we have |ĉ(u) − c(u)| ≤ ε = √(2 ln(2|M|/α)/t) for all u ∈ M. Hence

  acc* ≥ ½ + (max_{u∈M} |ĉ(u)| − ε)/4.

This is reported as `acc_lower_bound_certified`.

## Lemma 7. Rotational equivalence (SIMON-like ciphers, idealised keys)

Let ρ_k rotate both n/2-bit words by k. SIMON's round function f(x) = (S¹x ∧ S⁸x) ⊕ S²x satisfies f(ρx) = ρf(x).

**Claim.** With independent, uniform round keys, D(ρδ) and ρ·D(δ) have the same distribution, so c_{ρδ}(ρu) = c_δ(u) and every score above is invariant under rotation.

*Proof sketch.* Let k_i be independent and uniform, and P uniform. The map (P, k_1…k_r) ↦ (ρP, ρk_1…ρk_r) preserves the input distribution. Because f commutes with ρ and XOR commutes with ρ, it maps the trajectory for δ to the trajectory for ρδ. ∎

**Caveat.** SIMON's real key schedule makes round keys from round 5 onward dependent on the master key, through constants that are not rotation-invariant. So for r > 4 rounds the lemma is a model assumption. This is why we treat rotations as an empirically tested equivalence (`orbit_check` in E3), not a proven one.

## Proposition 8. Standard error for adaptive (confidence-guided) scoring

For Ĉ = Σ_{u∈M} ĉ²_unb with M = all masks of weight ≤ 2, the delta method gives a per-sample influence

  z_i = 2 Σ_{u∈M} ĉ(u)·(−1)^{u·D_i} = 2·( ĉ₁ · s_i + ½ · s_iᵀ C₂° s_i ),

where s_i ∈ {±1}ⁿ and C₂° is the matrix of pair correlations with zero diagonal. So Var(Ĉ) ≈ Var(z)/t. When every c(u) ≈ 0 the delta method degenerates. We therefore add the exact null variance of the unbiased squares, about 2|M|/t²:

  SE(Ĉ) = √( Var(z)/t + 2|M|/t² ).

This is `cap12_and_se`. The adaptive scorer with `statistic="spectral"` uses the intervals Ĉ ± z₁₋α/₂·SE. They are symmetric with no floor term, because the estimator is unbiased. Coverage at a nominal 90% is ≥ 84% in every tested regime (null, weak, strong); see `test_cap12_se_coverage`. These intervals are empirical under adaptive stopping, exactly as the `bias_bounded` ones are.

## Search stopping rule (E3)

AutoND stops its round loop when the best bias score falls below a fixed threshold T (0.05 at t = 10³ in the released code). We instead:

1. Take the round's best candidate from the evolutionary search.
2. Re-measure it on **fresh** samples, independent of the selection.
3. Compute the χ² log p-value (degree ≤ 2 with 528 degrees of freedom, or degree 1 with n).
4. Continue only if p < α = 0.01.

A single pre-selected candidate is tested per round, so no multiplicity correction over the thousands of candidates the search evaluated is needed; testing on the selection samples would suffer from the winner's curse. False-positive control on a pure-noise "cipher" is enforced in `test_chi2_stopping_controls_false_positives`.

**Attribution ablation.** `spectral_optimize(..., degree=1)` uses the same calibrated stopping rule with only the per-bit statistic. Comparing degree = 1 against degree = 2 separates the gain from *calibrated stopping* from the gain from *the 2-bit masks*. See the E3 results for which effect explains the extra round.

## Computational cost

- **Per candidate and round:** O(t·n) to encrypt (shared with AutoND), plus O(t·n²) for the Gram matrix. With n = 32 the Gram step is a small matrix multiply.
- **Measured:** the spectral search took about 1.7–2× AutoND's wall time at t = 10³, with encryptions differing by +9% to +22% because the search reaches deeper rounds.
- **Larger blocks:** degree 2 has n(n−1)/2 masks. For n = 128 that is 8,128 masks and the Gram matrix costs 128² per sample, which is still cheap next to encryption. Degree 3 (O(t·n³)) is not implemented.
