import numpy as np
import pytest

from cgsearch.autond_bridge import load_cipher
from cgsearch.data import bits_to_int, int_to_bits, make_pairs_dataset, seed_tree


@pytest.fixture(scope="module")
def simon():
    return load_cipher("simon3264")


def test_bit_conversion_roundtrip():
    for v in [0, 1, 0x400, 0x80000000, 0xDEADBEEF]:
        assert bits_to_int(int_to_bits(v, 32)) == v
    assert int_to_bits(0x400, 32)[21] == 1  # MSB-first, like AutoND


def test_format_and_balance(simon):
    X, Y = make_pairs_dataset(simon, 5, 0x400, 20000, np.random.default_rng(0))
    assert X.shape == (20000, 64) and X.dtype == np.uint8 and set(np.unique(X)) <= {0, 1}
    assert Y.shape == (20000,) and abs(Y.mean() - 0.5) < 0.02


def test_labels_one_round(simon):
    """After one SIMON round the new right word equals the old left word, so for real
    pairs the right-half ciphertext difference equals the left half of delta."""
    delta = 0x12340400
    X, Y = make_pairs_dataset(simon, 1, delta, 4000, np.random.default_rng(1))
    right_diff = np.array([bits_to_int(x[16:32] ^ x[48:64]) for x in X])
    assert np.all(right_diff[Y == 1] == delta >> 16)
    assert np.mean(right_diff[Y == 0] == delta >> 16) < 0.01


def test_real_pairs_are_encryptions(simon):
    X, Y, pt0, keys = make_pairs_dataset(simon, 3, 0x400, 500, np.random.default_rng(2), return_plaintexts=True)
    c0 = simon.encrypt(pt0, keys, 3)
    c1 = simon.encrypt(pt0 ^ int_to_bits(0x400, 32), keys, 3)
    real = Y == 1
    assert np.array_equal(X[real, :32], c0[real]) and np.array_equal(X[real, 32:], c1[real])


def test_seeded_reproducible_and_streams_independent(simon):
    a = make_pairs_dataset(simon, 4, 0x400, 1000, seed_tree(7, "train", 0x400, 4))
    b = make_pairs_dataset(simon, 4, 0x400, 1000, seed_tree(7, "train", 0x400, 4))
    c = make_pairs_dataset(simon, 4, 0x400, 1000, seed_tree(7, "test", 0x400, 4, 0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])
