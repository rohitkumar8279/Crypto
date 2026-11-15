import numpy as np
import pytest

from cgsearch import simon_ref as S
from cgsearch.autond_bridge import load_cipher
from cgsearch.data import bits_to_int, int_to_bits, random_bits


@pytest.fixture(scope="module")
def simon():
    return load_cipher("simon3264")


def _words_to_bits(words, n=16):
    return np.concatenate([int_to_bits(w, n) for w in words])[None]


def test_reference_published_vector():
    tv = S.TEST_VECTOR
    assert S.encrypt(tv["plaintext"], tv["key"]) == tv["ciphertext"]
    assert S.decrypt(tv["ciphertext"], tv["key"]) == tv["plaintext"]


def test_autond_published_vector(simon):
    tv = S.TEST_VECTOR
    c = simon.encrypt(_words_to_bits(tv["plaintext"]), _words_to_bits(tv["key"]), 32)
    assert bits_to_int(c[0]) == (tv["ciphertext"][0] << 16) | tv["ciphertext"][1]


def test_parameters(simon):
    assert (simon.plain_bits, simon.key_bits, simon.word_size) == (32, 64, 16)


@pytest.mark.parametrize("nr", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32])
def test_autond_matches_reference_reduced_rounds(simon, nr):
    rng = np.random.default_rng(nr)
    p, k = random_bits(rng, 64, 32), random_bits(rng, 64, 64)
    c = simon.encrypt(p, k, nr)
    for i in range(len(p)):
        kw = tuple(bits_to_int(k[i, 16 * j:16 * j + 16]) for j in range(4))
        pt = (bits_to_int(p[i, :16]), bits_to_int(p[i, 16:]))
        x, y = S.encrypt(pt, kw, nr)
        assert bits_to_int(c[i]) == (x << 16) | y
        assert S.decrypt((x, y), kw, nr) == pt


# ---------------------------------------------------------------- SPECK32/64

def test_speck_reference_published_vector():
    from cgsearch import speck_ref as P
    tv = P.TEST_VECTOR
    assert P.encrypt(tv["plaintext"], tv["key"]) == tv["ciphertext"]
    assert P.decrypt(tv["ciphertext"], tv["key"]) == tv["plaintext"]


@pytest.mark.parametrize("nr", [1, 2, 3, 5, 7, 8, 10, 22])
def test_speck_autond_matches_reference(nr):
    from cgsearch import speck_ref as P
    speck = load_cipher("speck3264")
    rng = np.random.default_rng(100 + nr)
    p, k = random_bits(rng, 64, 32), random_bits(rng, 64, 64)
    c = speck.encrypt(p, k, nr)
    for i in range(len(p)):
        kw = tuple(bits_to_int(k[i, 16 * j:16 * j + 16]) for j in range(4))
        pt = (bits_to_int(p[i, :16]), bits_to_int(p[i, 16:]))
        x, y = P.encrypt(pt, kw, nr)
        assert bits_to_int(c[i]) == (x << 16) | y
        assert P.decrypt((x, y), kw, nr) == pt
