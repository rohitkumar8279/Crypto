"""Seeded candidate-pool generation with AutoND's unmodified evolutionary optimizer.

Mirrors the first loop of AutoND ``optimizer.optimize``: run ``evo`` round by round
(seeding each round with the previous round's population) until the best bias score
falls below threshold T. The union of returned populations is the candidate pool that
AutoND then re-scores; we freeze and save that pool so every scoring policy is compared
on exactly the same candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .autond_bridge import load_optimizer
from .data import random_bits, seed_tree, bits_to_int
from .scoring import Ledger


def build_candidate_pool(cipher, seed: int, num_generations: int = 5, nb_samples: int = 1000,
                         threshold: float = 0.05, population: int = 32, max_round: int = 40,
                         verbose: bool = False) -> dict:
    opt = load_optimizer()
    pb, kb = cipher.plain_bits, cipher.key_bits
    rng = seed_tree(seed, "pool")
    np.random.seed(seed % (2**32))  # AutoND's evo uses NumPy's global RNG
    ledger = Ledger()
    diffs, all_diffs, per_round = None, [], []
    r = 1
    while True:
        keys = random_bits(rng, nb_samples, kb)
        pt0 = random_bits(rng, nb_samples, pb)
        c0 = cipher.encrypt(pt0, keys, r)
        ledger.add(nb_samples, r)

        def f(x, _r=r, _pt0=pt0, _k=keys, _c0=c0):
            ledger.add(len(x) * nb_samples, _r)
            return opt.evaluate_multiple_differences(x, _pt0, _k, _c0, _r, pb, kb, cipher.encrypt,
                                                     scenario="single-key")

        diffs, scores = opt.evo(f=f, n=num_generations, num_bits=pb, L=population, gen=diffs,
                                verbose=int(verbose))
        all_diffs.append(diffs.copy())
        best = [(hex(bits_to_int(d)), float(s)) for d, s in zip(diffs[-5:], scores[-5:])]
        per_round.append({"round": r, "best_score": float(scores[-1]), "top5": best[::-1]})
        if scores[-1] < threshold or r >= max_round:
            highest = r - 1
            break
        r += 1
    pool = np.unique(np.concatenate(all_diffs), axis=0)
    pool_ints = sorted({bits_to_int(d) for d in pool} - {0})
    return {
        "cipher": cipher.__name__.split(".")[-1],
        "seed": seed,
        "num_generations": num_generations,
        "nb_samples": nb_samples,
        "threshold": threshold,
        "highest_round": highest,
        "rounds_scored": list(range(1, highest + 1)),
        "pool": [hex(x) for x in pool_ints],
        "per_round": per_round,
        "ledger": ledger.as_dict(),
    }


def save_pool(pool: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(pool, indent=2))


def load_pool(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["pool_ints"] = [int(x, 16) for x in d["pool"]]
    return d
