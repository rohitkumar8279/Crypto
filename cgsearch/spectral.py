"""Low-degree Walsh-spectrum scores for ranking input differences.

Setting. For an input difference delta and r rounds, let D = E_K(P) xor E_K(P xor delta)
in {0,1}^n with (P, K) uniform. For an output mask u, the correlation is
    c(u) = E[(-1)^{u.D}],      c(0) = 1.

Connections (proofs in docs/method_spectral.md):
  * AutoND's bias score is a degree-1, L1 Walsh quantity:
        b(delta) = (1/n) sum_j |1/2 - Pr[D_j = 1]| = (1/(2n)) sum_j |c(e_j)|.
  * Parseval: capacity  C(delta) = sum_{u != 0} c(u)^2 = 2^n * ||P_D - U||_2^2.
  * For the real-vs-random experiment restricted to the difference, the optimal single-pair
    accuracy is  acc* = 1/2 + TV(P_D, U)/2  (Gohr, Leander, Neumann 2022/1521), and
        1/2 + max_u |c(u)| / 4   <=   acc*   <=   1/2 + sqrt(C) / 4.
    The lower bound is *achieved* by the one-mask rule "say real iff u.D = [c(u) < 0]".
  * Truncating C to masks of weight <= d gives C_{<=d}; d = 1 is a squared version of the
    bias score, d = 2 adds the 2-bit differential-linear correlations (n(n-1)/2 masks).

Estimation from t samples (s_i(u) = (-1)^{u.D_i} in {-1, +1}):
  * c^(u) = mean_i s_i(u) is unbiased; E[c^^2] = c^2 + (1 - c^2)/t, so
        (t c^^2 - 1) / (t - 1)     is an UNBIASED estimator of c(u)^2
    -- no sample-size-dependent noise floor (unlike AutoND's |.|-based score).
  * Under H0 (D uniform), distinct masks give uncorrelated s(u), so  t * sum_{u in M} c^(u)^2
    is asymptotically chi^2 with |M| degrees of freedom: a test statistic of known
    distribution (AutoND Sec. 5.4 names this as future work).
  * Hoeffding (s in [-1, 1]): P(|c^ - c| >= eps) <= 2 exp(-t eps^2 / 2); with a union bound
    over |M| masks this certifies  acc* >= 1/2 + (max|c^| - eps)/4  with prob. >= 1 - alpha.
"""

from __future__ import annotations

import math

import numpy as np

CHUNK = 16  # candidates per einsum chunk (memory)


def walsh_low_degree(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bits (m, t, n) uint8 -> (c1 (m, n), c2 (m, n, n)) empirical correlations.

    c1[:, j]    = c^(e_j);   c2[:, i, j] = c^(e_i + e_j) for i != j (diagonal is 1).
    """
    m, t, n = bits.shape
    c1 = np.empty((m, n))
    c2 = np.empty((m, n, n))
    for s in range(0, m, CHUNK):
        S = 1.0 - 2.0 * bits[s:s + CHUNK].astype(np.float32)          # (c, t, n) in {-1, 1}
        c1[s:s + CHUNK] = S.mean(axis=1)
        c2[s:s + CHUNK] = np.einsum("ctn,ctk->cnk", S, S, optimize=True) / t
    return c1, c2


def unbiased_sq(c: np.ndarray, t: int) -> np.ndarray:
    return (t * c * c - 1.0) / (t - 1.0)


def _upper(c2: np.ndarray) -> np.ndarray:
    n = c2.shape[-1]
    iu = np.triu_indices(n, k=1)
    return c2[..., iu[0], iu[1]]                                       # (m, n(n-1)/2)


def spectral_features(bits: np.ndarray, alpha: float = 0.01) -> dict:
    """All per-candidate spectral quantities for one round. bits (m, t, n)."""
    m, t, n = bits.shape
    c1, c2 = walsh_low_degree(bits)
    p = _upper(c2)
    cap1 = unbiased_sq(c1, t).sum(axis=1)
    cap2 = unbiased_sq(p, t).sum(axis=1)
    k1, k2 = n, p.shape[1]
    stat1 = t * (c1 ** 2).sum(axis=1)
    stat12 = stat1 + t * (p ** 2).sum(axis=1)
    allc = np.concatenate([np.abs(c1), np.abs(p)], axis=1)
    best = allc.max(axis=1)
    arg = allc.argmax(axis=1)
    eps = math.sqrt(2.0 * math.log(2.0 * (k1 + k2) / alpha) / t)
    return {
        "t": t,
        "autond_bias": np.abs(c1).mean(axis=1) / 2.0,          # == AutoND bias score exactly
        "cap1": cap1,                                           # unbiased sum_j c(e_j)^2
        "cap2": cap2,                                           # unbiased sum_{i<j} c(e_i+e_j)^2
        "cap12": cap1 + cap2,                                   # unbiased C_{<=2}
        "chi2_stat1": stat1, "chi2_df1": k1,
        "chi2_stat12": stat12, "chi2_df12": k1 + k2,
        "max_abs_corr": best,
        "best_mask_index": arg,                                 # < n: single bit; else pair index
        "acc_lower_bound_plugin": 0.5 + best / 4.0,
        "acc_lower_bound_certified": 0.5 + np.maximum(best - eps, 0.0) / 4.0,
        "hoeffding_eps": eps,
        "acc_upper_proxy_cap12": 0.5 + np.sqrt(np.maximum(cap1 + cap2, 0.0)) / 4.0,
    }


def chi2_logsf(stat, df) -> np.ndarray:
    """log survival function of chi^2_df (log p-value), robust for tiny p."""
    from scipy.stats import chi2
    return chi2.logsf(stat, df)


def mask_name(index: int, n: int) -> str:
    """Human-readable output mask for spectral_features()['best_mask_index']."""
    if index < n:
        return f"bit{index}"
    iu = np.triu_indices(n, k=1)
    k = index - n
    return f"bit{iu[0][k]}^bit{iu[1][k]}"


def cap12_and_se(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased C_{<=2} and its standard error, per candidate. bits (m, t, n).

    Delta method on sum_u c^(u)^2: the per-sample influence is z_i = 2 * sum_u c^(u) s_i(u)
    = 2 * (c1 . s_i + 1/2 (s_i^T C2o s_i)), with C2o the pair-correlation matrix with zero
    diagonal. Var ~ Var(z)/t, plus the null-variance term 2|M|/t^2 that dominates when all
    c(u) ~ 0 (where the delta method degenerates).
    """
    m, t, n = bits.shape
    cap = np.empty(m)
    se = np.empty(m)
    n_masks = n + n * (n - 1) // 2
    for i in range(m):
        S = 1.0 - 2.0 * bits[i].astype(np.float64)              # (t, n)
        c1 = S.mean(0)
        G = S.T @ S / t
        C2o = G - np.diag(np.diag(G))
        iu = np.triu_indices(n, 1)
        cap[i] = unbiased_sq(c1, t).sum() + unbiased_sq(G[iu], t).sum()
        z = 2.0 * (S @ c1 + 0.5 * np.einsum("ti,ij,tj->t", S, C2o, S, optimize=True))
        se[i] = math.sqrt(z.var() / t + 2.0 * n_masks / t ** 2)
    return cap, se
