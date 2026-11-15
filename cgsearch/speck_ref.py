"""Independent, integer-based reference implementation of SPECK32/64.

Written from the SPECK specification (Beaulieu et al., ePrint 2013/404), not from
AutoND's code, to cross-check AutoND's bit-matrix implementation (ciphers/speck3264.py).
"""

from __future__ import annotations

N = 16
M = 4
ALPHA, BETA = 7, 2
T_FULL = 22
MASK = (1 << N) - 1


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (N - r))) & MASK


def rotr(x: int, r: int) -> int:
    return ((x >> r) | (x << (N - r))) & MASK


def key_schedule(key_words: tuple[int, int, int, int], rounds: int) -> list[int]:
    """key_words = (l2, l1, l0, k0) in the spec's printed order."""
    l2, l1, l0, k0 = key_words
    k, l = [k0], [l0, l1, l2]
    for i in range(rounds - 1):
        l.append(((k[i] + rotr(l[i], ALPHA)) & MASK) ^ i)
        k.append(rotl(k[i], BETA) ^ l[i + M - 1])
    return k[:rounds]


def encrypt(pt: tuple[int, int], key_words: tuple[int, int, int, int], rounds: int = T_FULL) -> tuple[int, int]:
    x, y = pt
    for rk in key_schedule(key_words, rounds):
        x = ((rotr(x, ALPHA) + y) & MASK) ^ rk
        y = rotl(y, BETA) ^ x
    return x, y


def decrypt(ct: tuple[int, int], key_words: tuple[int, int, int, int], rounds: int = T_FULL) -> tuple[int, int]:
    x, y = ct
    for rk in reversed(key_schedule(key_words, rounds)):
        y = rotr(x ^ y, BETA)
        x = rotl(((x ^ rk) - y) & MASK, ALPHA)
    return x, y


# Published test vector (spec, Appendix C): Key 1918 1110 0908 0100,
# Plaintext 6574 694c -> Ciphertext a868 42f2.
TEST_VECTOR = {"key": (0x1918, 0x1110, 0x0908, 0x0100),
               "plaintext": (0x6574, 0x694C), "ciphertext": (0xA868, 0x42F2)}
