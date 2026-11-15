"""Novel E3 -- AutoND search (bias score, empirical threshold) vs spectral search
(degree<=2 capacity, chi^2 stopping), same evo, seeds, generations and sample size.

Reports per run: highest round reached, search cost, rank of the known-best difference
under each ranking, top-5, and (SIMON) the rotation-orbit structure of the top-16.

Usage: python experiments/novel_e3_search.py [--seeds 1 2 3] [--generations 10]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common import Timer, dump, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.candidates import build_candidate_pool
from cgsearch.data import seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.spectral_search import group_orbits, orbit_canonical, rescore, spectral_optimize

KNOWN_BEST = {"speck3264": 0x400000, "simon3264": 0x400}


def rank_of(order: list[str], target: int, cname: str) -> int | None:
    if cname.startswith("simon"):   # any member of the target's rotation orbit counts
        canon = orbit_canonical(target)
        for i, d in enumerate(order):
            if orbit_canonical(int(d, 16)) == canon:
                return i + 1
        return None
    return order.index(hex(target)) + 1 if hex(target) in order else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--t", type=int, default=1000)
    ap.add_argument("--out", default="e3_search_with_ablation",
                    help="results subfolder (the original 2-method run is kept in results/novel/e3_search)")
    args = ap.parse_args()
    out = out_dir("novel", args.out)
    partial = out / "runs_partial.jsonl"            # resumable: one JSON row per finished method run
    rows = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()] if partial.exists() else []
    done = {(r["cipher"], r["seed"]) for r in rows}
    detail = json.loads((out / "detail.json").read_text()) if (out / "detail.json").exists() else {}
    for cname in ("speck3264", "simon3264"):
        cipher = load_cipher(cname)
        for seed in args.seeds:
            if (cname, seed) in done and sum(1 for r in rows if (r["cipher"], r["seed"]) == (cname, seed)) == 3:
                print(f"skip {cname} seed {seed} (done)", flush=True)
                continue
            with Timer() as ta:
                a = build_candidate_pool(cipher, seed=seed, num_generations=args.generations,
                                         nb_samples=args.t, threshold=0.05)
            a_pool = [int(x, 16) for x in a["pool"]]
            a_rank = rescore(cipher, a_pool, a["rounds_scored"], 10 * args.t, seed_tree(seed, "e3_rescore_a"))
            with Timer() as ts:
                s = spectral_optimize(cipher, seed=seed, num_generations=args.generations, nb_samples=args.t)
            with Timer() as t1:   # ablation: calibrated chi^2 stopping, but per-bit statistic only
                s1 = spectral_optimize(cipher, seed=seed, num_generations=args.generations, nb_samples=args.t, degree=1)
            tgt = KNOWN_BEST[cname]
            for name, res, rk, order_key, tm, cost in [
                ("autond_bias", a, a_rank, "autond_weighted_order", ta.seconds, a["ledger"]["encryptions"]),
                ("spectral_deg1_ablation", s1, s1.get("ranking", {}), "deepest_round_order", t1.seconds,
                 s1["ledger_search"]["encryptions"]),
                ("spectral", s, s.get("ranking", {}), "deepest_round_order", ts.seconds, s["ledger_search"]["encryptions"])]:
                order = rk.get(order_key, [])
                row = {"cipher": cname, "seed": seed, "method": name, "highest_round": res["highest_round"],
                       "pool_size": len(res["pool"]), "search_encryptions": cost, "seconds": tm,
                       "rank_known_best": rank_of(order, tgt, cname), "top5": " ".join(order[:5])}
                if cname.startswith("simon"):
                    orb = group_orbits([int(d, 16) for d in order[:16]])
                    row["top16_distinct_orbits"] = len(orb)
                rows.append(row)
                with partial.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(row, flush=True)
            detail[f"{cname}|{seed}"] = {"autond": {"per_round": a["per_round"], "highest_round": a["highest_round"]},
                                         "spectral": {"per_round": s["per_round"], "highest_round": s["highest_round"]},
                                         "spectral_deg1": {"per_round": s1["per_round"], "highest_round": s1["highest_round"]}}
            dump(detail, out / "detail.json")
    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    dump(detail, out / "detail.json")
    write_manifest(out, vars(args))
    pd.set_option("display.width", 250)
    print(df.drop(columns=["top5"]).to_string(index=False))


if __name__ == "__main__":
    main()
