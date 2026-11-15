"""Stopping-rule and edge-case tests for the spectral search and the adaptive scorer."""

import types

import numpy as np
import pytest

from cgsearch.autond_bridge import load_cipher
from cgsearch.scoring import AdaptiveConfig, AdaptiveScorer, SyntheticSampler, batch_cost
from cgsearch.spectral_search import group_orbits, orbit_canonical, rotate_words, spectral_optimize


def _random_oracle_cipher(seed=0):
    """A 'cipher' whose output is independent uniform bits: nothing is ever non-random."""
    rng = np.random.default_rng(seed)
    mod = types.SimpleNamespace(__name__="ciphers.random_oracle", plain_bits=16, key_bits=16)
    mod.encrypt = lambda p, k, r: rng.integers(0, 2, size=p.shape, dtype=np.uint8)
    return mod


@pytest.mark.parametrize("degree", [1, 2])
def test_chi2_stopping_controls_false_positives(degree):
    """On pure noise the fresh-sample chi^2 test must stop at round 1 (highest_round = 0)."""
    res = spectral_optimize(_random_oracle_cipher(degree), seed=degree, num_generations=2,
                            nb_samples=300, degree=degree)
    assert res["highest_round"] == 0
    assert "ranking" not in res


def test_spectral_search_detects_real_cipher():
    res = spectral_optimize(load_cipher("speck3264"), seed=1, num_generations=1, nb_samples=300, max_round=4)
    assert res["highest_round"] >= 3
    assert res["ranking"]["deepest_round_order"]


def test_degree_validation():
    with pytest.raises(ValueError):
        spectral_optimize(_random_oracle_cipher(), seed=0, degree=3)


def test_orbits():
    assert rotate_words(0x00010001, 1) == 0x00020002
    assert orbit_canonical(0x400) == orbit_canonical(0x1) == 0x1
    g = group_orbits([0x1, 0x400, 0x8000, 0x10000])
    assert len(g) == 2            # right-word single bits vs left-word single bit


# ------------------------------------------------------------- adaptive-scorer edge cases

def _run(p, cfg, seed=0):
    s = SyntheticSampler(p, [1], np.random.default_rng(seed))
    m = len(p)
    return AdaptiveScorer(s, list(range(1, m + 1)), [1], cfg, weights=[1.0],
                          diff_bits=np.arange(m)[:, None]).run()


@pytest.mark.parametrize("statistic", ["bias", "spectral"])
def test_all_tied_candidates_stop_within_budget(statistic):
    p = np.full((12, 1, 8), 0.5); p[:, 0, 0] = 0.7            # all identical
    budget = batch_cost(12, 800, 1)
    res = _run(p, AdaptiveConfig(k=3, t0=100, t_max=800, budget=budget, tol=0.05, statistic=statistic))
    assert res.ledger["encryptions"] <= budget and len(res.shortlist) == 3


def test_k_at_least_pool_size_returns_everyone_cheaply():
    p = np.full((4, 1, 8), 0.5); p[:, 0, 0] = [0.6, 0.7, 0.8, 0.9]
    res = _run(p, AdaptiveConfig(k=4, t0=100, t_max=1600))
    assert set(res.shortlist) == {1, 2, 3, 4}
    assert res.ledger["encryptions"] == batch_cost(4, 100, 1)   # decides after the first stage


def test_budget_too_small_for_first_stage():
    p = np.full((10, 1, 8), 0.5)
    res = _run(p, AdaptiveConfig(k=2, t0=100, t_max=800, budget=50))
    assert res.ledger["encryptions"] == 0 and res.config["stop_reason"] == "budget"
