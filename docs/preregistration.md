# Pre-registration: confidence-guided scoring vs fixed-budget AutoND scoring

**Status:** frozen for the scoring comparison (Phase 5a) on 14 Nov 2026, *before* the main-configuration runs. The neural comparison (Phase 5b) is frozen by the same document. Any later change must be logged in the change log at the end, with its date and reason. Results produced before a change are reported, not discarded.

**Scope:** SIMON32/64, single-key, reduced rounds, AutoND commit `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` (unmodified), configuration [`experiments/configs/simon32_main.json`](../experiments/configs/simon32_main.json).

## 1. Estimand and cost model

- **Candidate score:** AutoND's weighted bias score `S(δ) = Σ_r r · b̃_r(δ)` over rounds `1..R`. `R` is the highest round at which the frozen pool search still found a bias above the threshold. This is the formula in AutoND's `optimizer.optimize`.
- **Cost:** block encryptions. Scoring `m` candidates on `b` samples at one round costs `b·(m+1)`: the shared reference ciphertexts C0 plus one C1 per candidate. Round-function evaluations are also logged. Every policy uses the same sampler and is charged the same way.
- **Candidate pool:** frozen before scoring. It is produced by AutoND's unmodified `evo`, run with seeded inputs using the paper's settings (50 generations, 10⁴ samples, T = 0.01). The cost of the pool search is shared by all policies, so it is excluded from the comparison.

## 2. Policies

| ID | Policy | Budget |
|---|---|---|
| A | Fixed: `t_fixed = 10⁴` samples per candidate per round (AutoND paper value) | defines `cost(A)` |
| B | Adaptive: stages at t = 250, 500, …, 10⁴ (doubling); common random numbers; bias-bounded 99% intervals; elimination / acceptance / tie-tolerance rules below | hard cap = f·cost(A) |
| C | Uniform allocation with B's *actual* spend | = spend of B |
| C2 | Successive halving on B's checkpoints, without intervals | hard cap = f·cost(A) |

Budget fractions: f ∈ {1/16, 1/8, 1/4, 1/2, 1}. Seeds: 1–5 (independent RNG streams per policy, seed and fraction).

### Adaptive rules (B)
- **Interval:** `bias_bounded`. The upper bound is `S + z·SE` and the lower bound is `S − z·SE − Σ_r r·0.5/√t_r`, where SE is the standard deviation over 200 Poisson-bootstrap replicates and z = z₀.₉₉₅. These intervals are **empirical**. Their only approximation is that the fluctuation term is treated as normal; the floor term is a rigorous bound. Validation is in Phase 4. They come with **no formal coverage guarantee under adaptive stopping**. The McDiarmid bound, which is distribution-free and union-bounded over looks and candidates, is run as a sensitivity analysis only.
- **Eliminate** a candidate when at least K other non-eliminated candidates have a lower bound above its upper bound.
- **Accept** a candidate when fewer than K others have an upper bound above its lower bound.
- **Tie tolerance:** stop sampling a candidate when its noise width (interval width minus the shared floor term) is at most 0.05 × S_K, where S_K is the K-th best point estimate. *Rationale, set after the demo profiling:* the 13–16 rotational single-bit differences score within 0.3% of each other, and no budget separates them (Phase 3: top-K overlap ≈ 0.6 at every t). AutoND itself treats candidates within 10% as equivalent, so 5% is stricter than the baseline's own criterion.
- **Shortlist:** the top K = 16 non-eliminated candidates by point estimate. K = 16 matches the number of 0.01-close differences the paper reports for SIMON32.

## 3. Outcomes

**Scoring stage (5a).** Each outcome is measured against an independent reference scoring with `t_reference = 2·10⁵` (a separate RNG stream).
- **Primary:** normalised shortlist regret, i.e. the gap in summed reference scores between the chosen shortlist and the reference top-K, divided by the latter.
- **Secondary:** ε-close precision (ε = 0.1, AutoND's criterion), top-K overlap, encryptions, and wall-clock time.

**Neural stage (5b).**
- **Training:** each shortlisted difference is trained with the identical DBitNet recipe: start round 8, target round 10, 10 epochs per round, 10⁷ training and 10⁶ selection samples, Adam with AMSGrad, MSE loss, batch size 5000. Neural seeds are 11–15.
- **Choosing each policy's finalist:** a policy's best candidate is the one with the highest *selection-set* accuracy.
- **Final test, run once:** only after every policy's finalist is fixed. It uses 5 fresh sets of 10⁶ samples from the frozen stream `test_seed = 990001`, which is never used for selection, tuning or early stopping.
- **Reported:** accuracy, advantage `2·acc − 1`, TPR/TNR, Wilson intervals, and the mean and t-interval across seeds.
- **Neural compute:** identical for every policy because K and the recipe are the same. A difference shortlisted by several policies is trained once and its result is shared. GPU time is logged.

## 4. Success criteria

Before the runs, "comparable quality" is defined as follows. On regret, the mean over seeds for B must be within **0.002** of A at full budget, or not worse than A by more than one pooled standard deviation, whichever is larger. On neural accuracy, the held-out accuracy of B's finalist must be within **0.002** of A's finalist (≈4σ of a 10⁶-sample evaluation).

H2/H3 are **supported** only if either:
1. B meets the comparable-quality criterion while using ≤ 50% of `cost(A)`, **and** C (uniform allocation at B's spend) is worse than B on regret by more than one pooled SD. Without the second condition, any saving belongs to stopping, not to adaptive allocation, and is reported that way.
2. At an equal budget fraction f < 1, B has lower mean regret than A in at least 4 of 5 seeds.

H1 (proxy reliability) is reported descriptively: the Spearman correlation between reference score and DBitNet selection accuracy across ≥ 24 candidates spanning the score range (Phase 3b), with a bootstrap confidence interval. A weak or negative correlation is a reportable result, not a failure.

## 5. What will not be claimed

See [claims_not_made.md](claims_not_made.md).

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-14 | Default interval changed from basic bootstrap to `bias_bounded` (before any main run) | Basic/percentile bootstrap under-covered badly (0–53% at nominal 90%) whenever unbiased bits are present (Phase 4 / tests). |
| 2026-11-14 | Tie tolerance applied to the noise width, 0.05·S_K (before any main run) | With a full-width rule the floor term prevented the tolerance from ever firing (demo profiling). |
