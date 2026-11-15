import math

import numpy as np
import pytest

from cgsearch.autond_bridge import load_cipher, load_optimizer
from cgsearch.data import ints_to_bit_matrix, random_bits
from cgsearch.scoring import CipherSampler
from cgsearch.spectral import chi2_logsf, spectral_features, walsh_low_degree


def test_autond_bias_is_degree1_walsh():
    """AutoND's score equals (1/2n) sum_j |c(e_j)| exactly, on real cipher data."""
    simon = load_cipher("simon3264")
    opt = load_optimizer()
    D = ints_to_bit_matrix([0x400, 0x8000, 0x3], 32)
    rng = np.random.default_rng(0)
    st = rng.bit_generator.state
    bits = CipherSampler(simon, rng).draw(D, 3000, 5)
    f = spectral_features(bits)
    r2 = np.random.default_rng(); r2.bit_generator.state = st
    keys, pt0 = random_bits(r2, 3000, 64), random_bits(r2, 3000, 32)
    ref = opt.evaluate_multiple_differences(D, pt0, keys, simon.encrypt(pt0, keys, 5), 5, 32, 64, simon.encrypt)
    np.testing.assert_allclose(f["autond_bias"], ref, atol=1e-12)


def test_pair_correlations_match_bruteforce():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (2, 500, 8), dtype=np.uint8)
    c1, c2 = walsh_low_degree(bits)
    s = 1 - 2 * bits.astype(int)
    assert np.allclose(c1[1, 3], s[1, :, 3].mean())
    assert np.allclose(c2[0, 2, 5], (s[0, :, 2] * s[0, :, 5]).mean())


def test_unbiased_capacity_has_no_noise_floor():
    """On uniform data the capacity estimate averages ~0 for every t (AutoND's does not)."""
    rng = np.random.default_rng(2)
    for t in (200, 2000):
        caps = [spectral_features(rng.integers(0, 2, (8, t, 16), dtype=np.uint8))["cap12"] for _ in range(20)]
        caps = np.concatenate(caps)
        se = caps.std() / math.sqrt(len(caps))
        assert abs(caps.mean()) < 4 * se + 1e-9


def test_unbiased_capacity_recovers_known_value():
    """Planted pairwise correlation: bit1 = bit0 xor Bernoulli(q) gives c(e0+e1) = 1-2q."""
    rng = np.random.default_rng(3)
    t, q = 20000, 0.3
    b = rng.integers(0, 2, (1, t, 8), dtype=np.uint8)
    b[0, :, 1] = b[0, :, 0] ^ (rng.random(t) < q)
    f = spectral_features(b)
    assert abs(f["cap2"][0] - (1 - 2 * q) ** 2) < 0.02
    assert abs(f["cap1"][0]) < 0.01                 # invisible to the per-bit (AutoND) score
    assert f["autond_bias"][0] < 0.01
    assert abs(f["max_abs_corr"][0] - 0.4) < 0.02


def test_chi2_null_calibration():
    """Under H0, P(p < 0.05) ~ 0.05 for the degree<=2 statistic."""
    rng = np.random.default_rng(4)
    pv = []
    for _ in range(10):
        f = spectral_features(rng.integers(0, 2, (40, 1000, 16), dtype=np.uint8))
        pv.append(np.exp(chi2_logsf(f["chi2_stat12"], f["chi2_df12"])))
    rate = np.mean(np.concatenate(pv) < 0.05)
    assert 0.02 < rate < 0.09


def test_one_mask_rule_achieves_lower_bound():
    """The rule 'real iff parity matches sign of c' has accuracy 1/2 + |c|/4 vs uniform."""
    rng = np.random.default_rng(5)
    t, q = 200000, 0.2
    real = rng.integers(0, 2, (t, 4), dtype=np.uint8)
    real[:, 1] = real[:, 0] ^ (rng.random(t) < q)          # c(e0+e1) = 0.6
    rand = rng.integers(0, 2, (t, 4), dtype=np.uint8)
    acc = 0.5 * ((real[:, 0] ^ real[:, 1]) == 0).mean() + 0.5 * ((rand[:, 0] ^ rand[:, 1]) == 1).mean()
    assert abs(acc - (0.5 + 0.6 / 4)) < 0.005


@pytest.mark.parametrize("q", [None, 0.45, 0.3])
def test_cap12_se_coverage(q):
    """Normal 90% intervals cap +- 1.645 SE cover the true C_{<=2} at >= ~nominal rate."""
    from cgsearch.spectral import cap12_and_se
    rng = np.random.default_rng(10)
    n, t = 12, 3000
    true = 0.0 if q is None else (1 - 2 * q) ** 2
    hits = 0
    for _ in range(150):
        b = rng.integers(0, 2, (1, t, n), dtype=np.uint8)
        if q is not None:
            b[0, :, 1] = b[0, :, 0] ^ (rng.random(t) < q)
        cap, se = cap12_and_se(b)
        hits += abs(cap[0] - true) <= 1.645 * se[0]
    assert hits / 150 >= 0.84
