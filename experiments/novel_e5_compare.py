"""Novel E5 -- allocation policies (fixed / adaptive / uniform-matched / successive halving)
under both statistics (AutoND bias, spectral C_{<=2}) on the frozen SPECK32 pool.
Design frozen in docs/novel_preregistration.md (E5). NumPy only.

Usage: python experiments/novel_e5_compare.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, Timer, dump, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.data import seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.metrics import shortlist_regret, topk_overlap
from cgsearch.scoring import (AdaptiveConfig, AdaptiveScorer, CipherSampler, max_uniform_t, score_fixed,
                              score_successive_halving, top_k)

K, T_FIXED, T_REF = 8, 10_000, 100_000


def setups(R):
    return {"bias": {"rounds": list(range(1, R + 1)), "weights": None},
            "spectral": {"rounds": [R + 1], "weights": [1.0]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--master-seed", type=int, default=20261117)
    ap.add_argument("--t-ref", type=int, default=T_REF)
    ap.add_argument("--t-fixed", type=int, default=T_FIXED)
    ap.add_argument("--pool-dir", default="e1_speck3264")
    ap.add_argument("--out", default="e5_compare")
    args = ap.parse_args()
    src = ROOT / "results/novel" / args.pool_dir
    pool = json.loads((src / "pool.json").read_text())
    diffs = [int(x, 16) for x in pool["pool"]]
    R = pool["highest_round"]
    cipher = load_cipher("speck3264")
    out = out_dir("novel", args.out)
    rows, shortlists, refs = [], {}, {}

    for stat, st in setups(R).items():
        rf = out / f"reference_{stat}.json"
        if rf.exists():
            ref = json.loads(rf.read_text())
        else:
            with Timer() as tm:
                r = score_fixed(CipherSampler(cipher, seed_tree(args.master_seed, "e5ref", stat)), diffs,
                                st["rounds"], args.t_ref, K, weights=st["weights"], chunk_candidates=8,
                                statistic=stat)
            ref = {"t": args.t_ref, "scores": dict(zip(map(hex, diffs), map(float, r.scores))), "seconds": tm.seconds}
            dump(ref, rf)
        refs[stat] = {int(k, 16): v for k, v in ref["scores"].items()}
        ref_top = top_k(diffs, np.array([refs[stat][d] for d in diffs]), K)

        def q(res):
            return {"regret": shortlist_regret(res.shortlist, refs[stat], K),
                    "topk_overlap_ref": topk_overlap(res.shortlist, ref_top, K),
                    "encryptions": res.ledger["encryptions"]}

        for seed in args.seeds:
            smp = lambda tag: CipherSampler(cipher, seed_tree(args.master_seed, "e5", stat, tag, seed))  # noqa: E731
            runs = {}
            with Timer() as tm:
                runs["A_fixed"] = score_fixed(smp("A"), diffs, st["rounds"], args.t_fixed, K, weights=st["weights"],
                                              statistic=stat)
            secs = {"A_fixed": tm.seconds}
            cfg = AdaptiveConfig(k=K, t0=min(250, args.t_fixed), growth=2.0, t_max=args.t_fixed, budget=runs["A_fixed"].ledger["encryptions"],
                                 alpha=0.01, tol=0.05, statistic=stat)
            with Timer() as tm:
                runs["B_adaptive"] = AdaptiveScorer(smp("B"), diffs, st["rounds"], cfg, weights=st["weights"],
                                                    boot_rng=seed_tree(args.master_seed, "e5boot", stat, seed)).run()
            secs["B_adaptive"] = tm.seconds
            tC = max_uniform_t(runs["B_adaptive"].ledger["encryptions"], len(diffs), len(st["rounds"]))
            with Timer() as tm:
                runs["C_uniform_matched"] = score_fixed(smp("C"), diffs, st["rounds"], max(tC, 2), K,
                                                        weights=st["weights"], statistic=stat,
                                                        policy_name="C_uniform_matched")
            secs["C_uniform_matched"] = tm.seconds
            with Timer() as tm:
                runs["C2_successive_halving"] = score_successive_halving(smp("C2"), diffs, st["rounds"], cfg,
                                                                         weights=st["weights"])
            secs["C2_successive_halving"] = tm.seconds
            for pol, res in runs.items():
                rows.append({"statistic": stat, "policy": pol, "seed": seed, "seconds": secs[pol],
                             "stop_reason": res.config.get("stop_reason", ""), **q(res)})
                shortlists[f"{stat}|{pol}|{seed}"] = [hex(x) for x in res.shortlist]
            print(stat, seed, {r["policy"]: (round(r["regret"], 4), r["encryptions"]) for r in rows[-4:]}, flush=True)
        shortlists[f"{stat}|reference_top{K}"] = [hex(x) for x in ref_top]

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    agg = (df.groupby(["statistic", "policy"])
             .agg(regret_mean=("regret", "mean"), regret_sd=("regret", "std"),
                  overlap_mean=("topk_overlap_ref", "mean"), encryptions_mean=("encryptions", "mean"),
                  seconds_mean=("seconds", "mean"), n=("seed", "count")).reset_index())
    agg.to_csv(out / "summary.csv", index=False)
    # agreement of the two statistics' reference shortlists
    dump({"shortlists": shortlists, "k": K, "R": R,
          "reference_overlap_bias_vs_spectral": topk_overlap(shortlists[f"bias|reference_top{K}"],
                                                             shortlists[f"spectral|reference_top{K}"], K)},
         out / "shortlists.json")
    write_manifest(out, vars(args))
    pd.set_option("display.width", 200)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
