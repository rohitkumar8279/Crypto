"""Metrics for distinguishers and for candidate rankings."""

from __future__ import annotations

import math

import numpy as np


def advantage(acc: float) -> float:
    return 2.0 * acc - 1.0


def balanced_accuracy(tpr: float, tnr: float) -> float:
    return 0.5 * (tpr + tnr)


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def random_guess_sigma(n: int) -> float:
    """Std. dev. of the accuracy of a random guesser on n balanced samples (paper Sec. 6.1)."""
    return 1.0 / (2.0 * math.sqrt(n))


def mean_ci(values, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-interval over independent seeds."""
    from scipy import stats
    v = np.asarray(values, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.5 + conf / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))
    return m, m - h, m + h


def spearman(x, y) -> float:
    from scipy import stats
    return float(stats.spearmanr(x, y).statistic)


def kendall(x, y) -> float:
    from scipy import stats
    return float(stats.kendalltau(x, y).statistic)


def topk_overlap(a, b, k: int | None = None) -> float:
    k = k or max(len(a), len(b))
    return len(set(list(a)[:k]) & set(list(b)[:k])) / k


def shortlist_regret(chosen, reference_scores: dict, k: int) -> float:
    """Sum of reference scores of the true top-k minus that of the chosen shortlist,
    normalised by the former (0 = optimal shortlist, 1 = worst)."""
    ref_sorted = sorted(reference_scores.values(), reverse=True)[:k]
    best = sum(ref_sorted)
    got = sum(reference_scores.get(d, 0.0) for d in list(chosen)[:k])
    return (best - got) / best if best > 0 else 0.0
