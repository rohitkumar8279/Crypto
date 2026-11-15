"""AutoND search with the spectral score: drop-in fitness, chi^2 stopping, deepest-round
ranking, and rotation-orbit grouping for SIMON-like ciphers.

Uses AutoND's unmodified ``evo`` (evolutionary optimizer); only the fitness function
and the round-loop stopping rule differ from ``optimizer.optimize``.
"""

from __future__ import annotations

import math

import numpy as np

from .autond_bridge import load_optimizer
from .data import bits_to_int, ints_to_bit_matrix, random_bits, seed_tree
from .scoring import CipherSampler, Ledger
from .spectral import chi2_logsf, spectral_features


def _output_bits(x, pt0, keys, c0, nr, cipher):
    m, t = len(x), len(pt0)
    pt1 = (x[:, None, :cipher.plain_bits] ^ pt0[None]).reshape(-1, cipher.plain_bits)
    k1 = np.broadcast_to(keys[None], (m,) + keys.shape).reshape(-1, cipher.key_bits)
    return (cipher.encrypt(pt1, k1, nr).reshape(m, t, -1) ^ c0[None]).astype(np.uint8)


def spectral_fitness(x, pt0, keys, c0, nr, cipher, ledger: Ledger | None = None, key: str = "cap12"):
    """Drop-in replacement for AutoND's evaluate_multiple_differences (same inputs)."""
    if ledger is not None:
        ledger.add(len(x) * len(pt0), nr)
    out = np.empty(len(x))
    for s in range(0, len(x), 64):
        out[s:s + 64] = spectral_features(_output_bits(x[s:s + 64], pt0, keys, c0, nr, cipher))[key]
    out[np.sum(x, axis=1) == 0] = -np.inf
    return out


def spectral_optimize(cipher, seed: int, num_generations: int = 10, nb_samples: int = 1000,
                      alpha: float = 0.01, population: int = 32, max_round: int = 40,
                      rescore_t: int | None = None, verbose: bool = False, degree: int = 2) -> dict:
    """AutoND's round loop with spectral fitness and a chi^2 stopping rule.

    Round r counts as non-random if the round's best candidate, re-tested on FRESH samples,
    has a chi^2 log p-value below log(alpha). (One pre-selected candidate per round, so no
    multiplicity correction over the search is needed.)

    degree=2 uses C_{<=2} as fitness and the 528-df statistic; degree=1 uses the unbiased
    per-bit capacity C_1 and the n-df statistic. degree=1 is the ABLATION that isolates the
    calibrated stopping rule from the contribution of the 2-bit masks.
    """
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    fit_key = "cap12" if degree == 2 else "cap1"
    stat_key, df_key = ("chi2_stat12", "chi2_df12") if degree == 2 else ("chi2_stat1", "chi2_df1")
    opt = load_optimizer()
    pb, kb = cipher.plain_bits, cipher.key_bits
    rng = seed_tree(seed, "spectral_pool")
    np.random.seed(seed % (2**32))
    ledger = Ledger()
    diffs, all_diffs, per_round = None, [], []
    r = 1
    while True:
        keys, pt0 = random_bits(rng, nb_samples, kb), random_bits(rng, nb_samples, pb)
        c0 = cipher.encrypt(pt0, keys, r)
        ledger.add(nb_samples, r)
        n_eval = [0]

        def f(x, _r=r, _pt0=pt0, _k=keys, _c0=c0):
            n_eval[0] += len(x)
            return spectral_fitness(x, _pt0, _k, _c0, _r, cipher, ledger, key=fit_key)

        diffs, scores = opt.evo(f=f, n=num_generations, num_bits=pb, L=population, gen=diffs, verbose=int(verbose))
        best = diffs[-1:]
        fb = spectral_features(_output_bits(best, pt0, keys, c0, r, cipher))
        logp = float(chi2_logsf(fb[stat_key], fb[df_key])[0])
        # re-test the best on fresh samples to avoid winner's-curse selection bias
        k2, p2 = random_bits(rng, nb_samples, kb), random_bits(rng, nb_samples, pb)
        c2 = cipher.encrypt(p2, k2, r)
        ledger.add(nb_samples * 2, r)
        fv = spectral_features(_output_bits(best, p2, k2, c2, r, cipher))
        logp_fresh = float(chi2_logsf(fv[stat_key], fv[df_key])[0])
        all_diffs.append(diffs.copy())
        per_round.append({"round": r, "best": hex(bits_to_int(best[0])), "best_cap12": float(scores[-1]),
                          "logp_search": logp, "logp_fresh": logp_fresh, "n_evaluated": n_eval[0]})
        if verbose:
            print(per_round[-1], flush=True)
        # Fresh-sample test of one pre-selected candidate: no multiplicity over the search.
        if logp_fresh >= math.log(alpha) or r >= max_round:
            highest = r - 1
            break
        r += 1
    pool = sorted({bits_to_int(d) for d in np.unique(np.concatenate(all_diffs), axis=0)} - {0})
    res = {"cipher": cipher.__name__.split(".")[-1], "seed": seed, "degree": degree,
           "num_generations": num_generations,
           "nb_samples": nb_samples, "alpha": alpha, "highest_round": highest,
           "pool": [hex(x) for x in pool], "per_round": per_round, "ledger_search": ledger.as_dict()}
    if highest >= 1:
        res["ranking"] = rescore(cipher, pool, list(range(1, highest + 1)), rescore_t or 10 * nb_samples,
                                 seed_tree(seed, "spectral_rescore"))
    return res


def rescore(cipher, diffs: list[int], rounds: list[int], t: int, rng) -> dict:
    """Score candidates on fresh samples at every round; rank by the deepest round first
    (lexicographic: cap12 at R, then R-1, ...), and also report AutoND-style weighted scores."""
    D = ints_to_bit_matrix(diffs, cipher.plain_bits)
    caps, bias, maxc = [], [], []
    for r in rounds:
        cs, bs, ms = [], [], []
        for s in range(0, len(diffs), 32):
            f = spectral_features(CipherSampler(cipher, rng).draw(D[s:s + 32], t, r))
            cs.append(f["cap12"]); bs.append(f["autond_bias"]); ms.append(f["max_abs_corr"])
        caps.append(np.concatenate(cs)); bias.append(np.concatenate(bs)); maxc.append(np.concatenate(ms))
    caps, bias, maxc = np.stack(caps, 1), np.stack(bias, 1), np.stack(maxc, 1)
    w = np.asarray(rounds, float)
    order = np.lexsort([caps[:, i] for i in range(len(rounds))])[::-1]   # last key = deepest round
    return {"t": t, "rounds": rounds,
            "deepest_round_order": [hex(diffs[i]) for i in order],
            "cap12_deepest": {hex(diffs[i]): float(caps[i, -1]) for i in order},
            "autond_weighted_order": [hex(diffs[i]) for i in np.argsort(-(bias @ w), kind="stable")],
            "autond_weighted": {hex(d): float(v) for d, v in zip(diffs, bias @ w)},
            "max_abs_corr_deepest": {hex(d): float(v) for d, v in zip(diffs, maxc[:, -1])}}


# ----------------------------------------------------------------------------- orbits

def rotate_words(delta: int, k: int, word: int = 16, words: int = 2) -> int:
    mask = (1 << word) - 1
    out = 0
    for i in range(words):
        w = (delta >> (word * (words - 1 - i))) & mask
        w = ((w << k) | (w >> (word - k))) & mask
        out |= w << (word * (words - 1 - i))
    return out


def orbit_canonical(delta: int, word: int = 16, words: int = 2) -> int:
    return min(rotate_words(delta, k, word, words) for k in range(word))


def group_orbits(diffs: list[int], word: int = 16, words: int = 2) -> dict[int, list[int]]:
    g: dict[int, list[int]] = {}
    for d in diffs:
        g.setdefault(orbit_canonical(d, word, words), []).append(d)
    return g
