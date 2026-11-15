"""Seeded data generation in AutoND's exact formats.

AutoND draws all randomness from ``os.urandom`` and therefore cannot be
replayed. These functions reproduce the same *format and distribution* with an
explicit ``numpy.random.Generator`` so every experiment can be re-run from its
seed. Equivalence of formats is checked in ``tests/test_data.py``.
"""

from __future__ import annotations

import numpy as np


def int_to_bits(value: int, nbits: int) -> np.ndarray:
    """MSB-first uint8 vector, AutoND's convention (main.integer_to_binary_array)."""
    if value < 0 or value >= (1 << nbits):
        raise ValueError(f"{value:#x} does not fit in {nbits} bits")
    return np.array([(value >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in np.asarray(bits).ravel():
        out = (out << 1) | int(b)
    return out


def ints_to_bit_matrix(values, nbits: int) -> np.ndarray:
    return np.stack([int_to_bits(int(v), nbits) for v in values]) if len(values) else np.zeros((0, nbits), np.uint8)


def random_bits(rng: np.random.Generator, n: int, nbits: int) -> np.ndarray:
    return rng.integers(0, 2, size=(n, nbits), dtype=np.uint8)


def make_pairs_dataset(cipher, nr: int, input_difference: int, n: int, rng: np.random.Generator,
                       return_plaintexts: bool = False):
    """Seeded single-key version of AutoND ``main.make_train_data``.

    Label 1: (E_K(P), E_K(P xor delta)) for a fresh random key per sample.
    Label 0: 2*plain_bits uniformly random bits -- this is what AutoND's code
    does (it overwrites the ciphertext pair with random bits), which differs
    from the paper's wording "random plaintext pairs". We keep AutoND's
    behaviour so results are comparable with the baseline.
    """
    pb, kb = cipher.plain_bits, cipher.key_bits
    delta = int_to_bits(input_difference, pb)
    keys = random_bits(rng, n, kb)
    pt0 = random_bits(rng, n, pb)
    pt1 = pt0 ^ delta
    c0 = cipher.encrypt(pt0, keys, nr)
    c1 = cipher.encrypt(pt1, keys, nr)
    X = np.hstack([c0, c1]).astype(np.uint8)
    Y = rng.integers(0, 2, size=n, dtype=np.uint8)
    n_rand = int(np.sum(Y == 0))
    X[Y == 0] = random_bits(rng, n_rand, X.shape[1])
    if return_plaintexts:
        return X, Y, pt0, keys
    return X, Y


def seed_tree(master_seed: int, *labels) -> np.random.Generator:
    """Derive an independent, named RNG stream from a master seed.

    ``seed_tree(1, "test", 3)`` and ``seed_tree(1, "train", 3)`` never share a stream,
    which keeps train / selection / test data independent by construction.
    """
    entropy = [int(master_seed)] + [_label_to_int(x) for x in labels]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _label_to_int(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    import hashlib
    return int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "big")
