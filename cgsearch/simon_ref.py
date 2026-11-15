"""Independent, integer-based reference implementation of SIMON32/64.

Written from the SIMON specification (Beaulieu et al., ePrint 2013/404), not
from AutoND's code, so that it can be used to cross-check AutoND's bit-matrix
implementation in ``ciphers/simon3264.py``.
"""

from __future__ import annotations

N = 16          # word size
M = 4           # key words
T_FULL = 32     # full-round count
MASK = (1 << N) - 1
# z0 sequence as printed in the specification, z0[0] first.
Z0 = "11111010001001010110000111001101111101000100101011000011100110"
C = MASK ^ 3    # 2^n - 4


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (N - r))) & MASK


def rotr(x: int, r: int) -> int:
    return ((x >> r) | (x << (N - r))) & MASK


def f(x: int) -> int:
    return (rotl(x, 1) & rotl(x, 8)) ^ rotl(x, 2)


def key_schedule(key_words: tuple[int, int, int, int], rounds: int) -> list[int]:
    """key_words = (k3, k2, k1, k0) in the spec's printed order."""
    k = list(reversed(key_words))  # k[0] = k0
    for i in range(M, rounds):
        tmp = rotr(k[i - 1], 3) ^ k[i - 3]
        tmp ^= rotr(tmp, 1)
        k.append((C ^ int(Z0[(i - M) % 62]) ^ k[i - M] ^ tmp) & MASK)
    return k[:rounds]


def encrypt(pt: tuple[int, int], key_words: tuple[int, int, int, int], rounds: int = T_FULL) -> tuple[int, int]:
    x, y = pt
    for rk in key_schedule(key_words, max(rounds, M)):
        if rounds == 0:
            break
        x, y = y ^ f(x) ^ rk, x
        rounds -= 1
    return x, y


def decrypt(ct: tuple[int, int], key_words: tuple[int, int, int, int], rounds: int = T_FULL) -> tuple[int, int]:
    x, y = ct
    for rk in reversed(key_schedule(key_words, max(rounds, M))[:rounds]):
        x, y = y, x ^ f(y) ^ rk
    return x, y


# Published test vector (spec, Appendix B): Key 1918 1110 0908 0100,
# Plaintext 6565 6877 -> Ciphertext c69b e9bb.
TEST_VECTOR = {
    "key": (0x1918, 0x1110, 0x0908, 0x0100),
    "plaintext": (0x6565, 0x6877),
    "ciphertext": (0xC69B, 0xE9BB),
}
