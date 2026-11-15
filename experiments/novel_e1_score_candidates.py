"""Novel E1 (part 1) -- build a pool with AutoND's own search and score every candidate
with both the AutoND bias score and the low-degree spectral scores, at high t.

Then select a stratified, pre-declared set of candidates for neural training (part 2):
  * top-k by AutoND weighted bias score,
  * top-k by spectral weighted capacity C_{<=2},
  * the largest rank disagreements between the two,
  * a random spread over the rest of the pool,
  * plus named reference differences.
The selection uses only proxy scores (never neural results). NumPy only.

Usage: python experiments/novel_e1_score_candidates.py --cipher speck3264
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import dump, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.candidates import build_candidate_pool, save_pool
from cgsearch.data import ints_to_bit_matrix, seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.scoring import CipherSampler
from cgsearch.spectral import spectral_features

REFERENCE = {"speck3264": [0x400000, 0x102000], "simon3264": [0x400, 0x8000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="speck3264")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--t", type=int, default=50_000)
    ap.add_argument("--max-round", type=int, default=8)
    ap.add_argument("--n-select", type=int, default=24)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--out", default=None, help="default: e1_<cipher>")
    args = ap.parse_args()
    cipher = load_cipher(args.cipher)
    n = cipher.plain_bits
    out = out_dir("novel", args.out or f"e1_{args.cipher}")

    pool = build_candidate_pool(cipher, seed=args.seed, num_generations=args.generations,
                                nb_samples=1000, threshold=0.05)
    diffs = sorted(set(pool["pool_ints"] if "pool_ints" in pool else [int(x, 16) for x in pool["pool"]])
                   | set(REFERENCE.get(args.cipher, [])))
    pool["pool"] = [hex(d) for d in diffs]
    save_pool(pool, out / "pool.json")
    print(f"pool: {len(diffs)} candidates; AutoND highest biased round {pool['highest_round']}", flush=True)

    D = ints_to_bit_matrix(diffs, n)
    rows = []
    rounds = list(range(1, args.max_round + 1))
    for r in rounds:
        rng = seed_tree(args.seed, "e1score", r)
        feats = {k: [] for k in ("autond_bias", "cap1", "cap2", "cap12", "max_abs_corr",
                                 "acc_lower_bound_certified", "chi2_stat12")}
        for s in range(0, len(diffs), 32):
            f = spectral_features(CipherSampler(cipher, rng).draw(D[s:s + 32], args.t, r))
            for k in feats:
                feats[k].append(np.atleast_1d(f[k]))
        for k in feats:
            feats[k] = np.concatenate(feats[k])
        for i, d in enumerate(diffs):
            rows.append({"difference": hex(d), "round": r, **{k: float(v[i]) for k, v in feats.items()}})
        print(f"round {r} scored", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "proxy_scores.csv", index=False)

    # weighted scores (AutoND weighting w_r = r) over the rounds where AutoND still saw a bias
    R = max(1, pool["highest_round"])
    w = df[df["round"] <= R].assign(wb=lambda x: x["round"] * x["autond_bias"],
                                    wc=lambda x: x["round"] * x["cap12"])
    agg = w.groupby("difference")[["wb", "wc"]].sum()
    deep = df[df["round"] == R + 1].set_index("difference")[["cap12", "max_abs_corr"]]
    agg = agg.join(deep.rename(columns={"cap12": "cap12_next_round", "max_abs_corr": "maxcorr_next_round"}))
    agg["rank_bias"] = agg["wb"].rank(ascending=False)
    agg["rank_spec"] = agg["cap12_next_round"].rank(ascending=False)
    agg = agg.sort_values("wb", ascending=False)

    rng = np.random.default_rng(args.seed)
    k = args.n_select // 4
    chosen = list(agg.index[:k])                                            # top by AutoND
    chosen += [d for d in agg.sort_values("cap12_next_round", ascending=False).index if d not in chosen][:k]
    dis = (agg["rank_bias"] - agg["rank_spec"]).abs().sort_values(ascending=False)
    chosen += [d for d in dis.index if d not in chosen][:k]
    rest = [d for d in agg.index if d not in chosen]
    chosen += list(rng.choice(rest, size=min(len(rest), args.n_select - len(chosen)), replace=False))
    for ref in REFERENCE.get(args.cipher, []):
        if hex(ref) not in chosen:
            chosen.append(hex(ref))
    agg.to_csv(out / "weighted_scores.csv")
    dump({"cipher": args.cipher, "args": vars(args), "autond_highest_round": pool["highest_round"],
          "weighted_rounds": list(range(1, R + 1)), "spectral_round": R + 1,
          "selected": chosen,
          "selection_rule": "k top-AutoND, k top-spectral(next round), k largest rank disagreement, rest random"},
         out / "selection.json")
    write_manifest(out, vars(args))
    pd.set_option("display.width", 200)
    print(agg.loc[chosen].to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
