import math

import numpy as np
import pytest

from cgsearch.autond_bridge import load_cipher, load_optimizer
from cgsearch.data import ints_to_bit_matrix, random_bits
from cgsearch.scoring import (AdaptiveConfig, AdaptiveScorer, CipherSampler, Ledger, SyntheticSampler,
                              batch_cost, bias_from_bits, intervals, score_fixed,
                              score_successive_halving)


@pytest.fixture(scope="module")
def simon():
    return load_cipher("simon3264")


# ---------------------------------------------------------------- equivalence with AutoND

def test_sampler_matches_autond_evaluate(simon):
    opt = load_optimizer()
    diffs = [0x400, 0x8000, 0x1, 0x0, 0x12345678]
    D = ints_to_bit_matrix(diffs, 32)
    for nr in (2, 5):
        rng = np.random.default_rng(nr)
        state = rng.bit_generator.state
        s = CipherSampler(simon, rng)
        ours = bias_from_bits(s.draw(D, 2000, nr))
        ours[np.asarray(diffs) == 0] = 0
        rng2 = np.random.default_rng(); rng2.bit_generator.state = state
        keys = random_bits(rng2, 2000, 64); pt0 = random_bits(rng2, 2000, 32)
        ref = opt.evaluate_multiple_differences(D, pt0, keys, simon.encrypt(pt0, keys, nr), nr, 32, 64, simon.encrypt)
        np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_fixed_chunking_is_exact_and_costed_once(simon):
    diffs = list(range(1, 40))
    r1 = score_fixed(CipherSampler(simon, np.random.default_rng(3)), diffs, [1, 2, 3], 500, 5, chunk_candidates=8)
    r2 = score_fixed(CipherSampler(simon, np.random.default_rng(3)), diffs, [1, 2, 3], 500, 5, chunk_candidates=1000)
    np.testing.assert_allclose(r1.scores, r2.scores)
    assert r1.ledger == r2.ledger
    assert r1.ledger["encryptions"] == batch_cost(len(diffs), 500, 3)
    assert r1.ledger["round_evaluations"] == 500 * (len(diffs) + 1) * (1 + 2 + 3)


def test_noise_floor_matches_theory():
    """For unbiased bits E|0.5 - p^| ~ sqrt(1/(2 pi t)); at t=1e4 this is ~0.004 (paper Sec. 5.4)."""
    rng = np.random.default_rng(0)
    for t in (1000, 10000):
        sc = bias_from_bits(rng.integers(0, 2, (200, t, 32), dtype=np.uint8))
        assert abs(sc.mean() - math.sqrt(1 / (2 * math.pi * t))) < 0.1 * math.sqrt(1 / (2 * math.pi * t))
    opt = load_optimizer()
    np.random.seed(0)
    assert abs(opt.empirical_threshold_estimation(10**4, 32).mean() - 0.004) < 0.0004


# ---------------------------------------------------------------- synthetic ground truth

def _toy(m=60, n=16, gap_pairs=True, seed=0):
    """Candidates 0..5 strongly biased (clear top), 6..7 near the K=6 boundary, rest noise."""
    rng = np.random.default_rng(seed)
    p = np.full((m, 1, n), 0.5)
    for i in range(6):
        p[i, 0, : 4 + i] = 0.5 + 0.30
    p[6, 0, :3] = 0.5 + 0.30            # close to candidate 0 (4 biased bits vs 3)
    p[7, 0, :2] = 0.5 + 0.30
    p[8:, 0, 0] = 0.5 + rng.uniform(0, 0.02, m - 8)  # weak
    return p


def _run_adaptive(p, cfg, seed=1, **kw):
    s = SyntheticSampler(p, [1], np.random.default_rng(seed))
    diffs = list(range(1, len(p) + 1))      # diffs must be non-zero; index = diff - 1
    D = np.arange(len(p))[:, None]
    res = AdaptiveScorer(s, diffs, [1], cfg, weights=[1.0], diff_bits=D,
                         boot_rng=np.random.default_rng(seed + 100), **kw).run()
    return s, res


def test_adaptive_recovers_true_topk_and_saves_budget():
    p = _toy()
    cfg = AdaptiveConfig(k=6, t0=50, t_max=6400, alpha=0.01, n_boot=200)
    s, res = _run_adaptive(p, cfg)
    exact = s.exact_scores([1.0])
    true_top = [i + 1 for i in np.argsort(-exact)[:6]]
    assert set(res.shortlist) == set(true_top)
    fixed_cost = batch_cost(len(p), cfg.t_max, 1)
    assert res.ledger["encryptions"] < 0.5 * fixed_cost
    # weak candidates stop early; close ones (indices 0, 5, 6) get more samples
    t = res.samples[:, 0]
    assert np.median(t[8:]) < t[[0, 6]].min()


def test_adaptive_respects_hard_budget():
    p = _toy()
    budget = 40_000
    cfg = AdaptiveConfig(k=6, t0=50, t_max=6400, budget=budget, n_boot=100)
    _, res = _run_adaptive(p, cfg)
    assert res.ledger["encryptions"] <= budget
    assert len(res.shortlist) == 6


def test_mcdiarmid_interval_contains_exact_score():
    p = _toy()
    cfg = AdaptiveConfig(k=6, t0=200, t_max=3200, interval="mcdiarmid", alpha=0.05)
    s, res = _run_adaptive(p, cfg)
    exact = s.exact_scores([1.0])
    assert np.all((res.lower <= exact + 1e-12) & (exact <= res.upper + 1e-12))


@pytest.mark.parametrize("n_biased", [0, 4, 16])
def test_bias_bounded_coverage_empirical(n_biased):
    """Empirical coverage at nominal 90% over repeated independent draws, including the
    hard case where many bits are unbiased (where basic/percentile bootstrap fail)."""
    from cgsearch.scoring import bootstrap_weighted
    rng = np.random.default_rng(5 + n_biased)
    p = np.full((1, 1, 16), 0.5); p[0, 0, :n_biased] = 0.56
    exact = np.mean(np.abs(0.5 - p))
    hits = {"bias_bounded": 0, "basic": 0}
    reps = 120
    for _ in range(reps):
        bits = (rng.random((1, 2000, 16)) < p[:, 0][:, None, :]).astype(np.uint8)
        pt = bias_from_bits(bits)
        boot = bootstrap_weighted([bits], [1.0], 200, rng)
        for m in hits:
            lo, hi = intervals(pt, boot, np.array([[2000]]), [1.0], m, 0.1)
            hits[m] += lo[0] <= exact <= hi[0]
    assert hits["bias_bounded"] / reps >= 0.85
    if n_biased < 16:  # documents why 'basic' is not the default
        assert hits["basic"] / reps < 0.75


def test_checkpoint_resume_gives_identical_result(tmp_path):
    p = _toy()
    cfg = AdaptiveConfig(k=6, t0=50, t_max=1600, n_boot=50)
    ck = str(tmp_path / "state.pkl")
    _, full = _run_adaptive(p, cfg, seed=4, checkpoint_path=ck)
    # resume from the saved final state: no more work, same answer
    _, again = _run_adaptive(p, cfg, seed=4, checkpoint_path=ck)
    assert full.shortlist == again.shortlist
    np.testing.assert_allclose(full.scores, again.scores)


def test_successive_halving_runs_and_keeps_k():
    p = _toy()
    s = SyntheticSampler(p, [1], np.random.default_rng(0))
    res = score_successive_halving(s, list(range(1, len(p) + 1)), [1],
                                   AdaptiveConfig(k=6, t0=50, t_max=3200), weights=[1.0],
                                   diff_bits=np.arange(len(p))[:, None])
    assert len(res.shortlist) == 6


def test_adaptive_on_cipher_smoke(simon):
    diffs = [0x400, 0x8000, 0x4000, 0x1, 0x3, 0xdeadbeef, 0x80000000, 0x10004]
    s = CipherSampler(simon, np.random.default_rng(0), Ledger())
    res = AdaptiveScorer(s, diffs, [1, 2, 3, 4], AdaptiveConfig(k=3, t0=100, t_max=1600, n_boot=100)).run()
    assert len(res.shortlist) == 3
    assert res.ledger["encryptions"] <= batch_cost(len(diffs), 1600, 4)


def test_adaptive_spectral_statistic_finds_pairwise_signal():
    """Candidates whose signal lives only in 2-bit masks: the spectral adaptive scorer
    recovers them; the per-bit statistic cannot see them."""
    rng = np.random.default_rng(0)
    m, n = 30, 10
    p = np.full((m, 1, n), 0.5)
    sam = SyntheticSampler(p, [1], np.random.default_rng(1))
    strengths = {0: 0.2, 1: 0.25, 2: 0.3}     # flip probability between bit0 and bit1

    class PairSampler(SyntheticSampler):
        def draw(self, diffs, batch, nr):
            b = super().draw(diffs, batch, nr)
            for row, i in enumerate(np.asarray(diffs)[:, 0].astype(int)):
                if i in strengths:
                    b[row, :, 1] = b[row, :, 0] ^ (self.rng.random(batch) < strengths[i])
            return b

    sam = PairSampler(p, [1], np.random.default_rng(1))
    cfg = AdaptiveConfig(k=3, t0=100, t_max=3200, statistic="spectral", alpha=0.01)
    res = AdaptiveScorer(sam, list(range(1, m + 1)), [1], cfg, weights=[1.0],
                         diff_bits=np.arange(m)[:, None]).run()
    assert set(res.shortlist) == {1, 2, 3}
    assert res.ledger["encryptions"] < batch_cost(m, 3200, 1) * 0.5
