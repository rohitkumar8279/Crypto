"""Novel E2 -- detection depth: per-bit bias (AutoND) vs low-degree Walsh spectrum.

For known-good differences, at equal sample size t, report for every round r:
  * AutoND bias score and its p-value against its own null distribution (simulated on
    uniform bits at the same t -- AutoND has no analytic null, its threshold is empirical);
  * degree-1 chi^2 test (calibrated version of the same information);
  * degree<=2 chi^2 test (adds n(n-1)/2 two-bit differential-linear masks);
  * strongest mask, unbiased capacities, and the certified accuracy lower bound.
"Detection depth" = deepest round with p < alpha (Bonferroni over rounds).
NumPy only.

Usage: python experiments/novel_e2_detection_depth.py [--t 100000] [--reps 3]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from common import Timer, dump, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.data import ints_to_bit_matrix, seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.scoring import CipherSampler
from cgsearch.spectral import chi2_logsf, mask_name, spectral_features

TARGETS = [("simon3264", 0x400, range(1, 14)), ("simon3264", 0x8000, range(1, 14)),
           ("speck3264", 0x400000, range(1, 10)), ("speck3264", 0x102000, range(1, 10))]
# neural reach from the AutoND paper (Table 6/8) for context
ND_REACH = {("simon3264", 0x400): 11, ("speck3264", 0x400000): 8}


def null_bias_logp(score: float, t: int, n: int, rng, sims: int = 4000) -> float:
    """Empirical log p-value of AutoND's bias score under uniform data (same t, n)."""
    # sum of t Bernoulli(1/2) per bit ~ Binomial; simulate the statistic exactly
    k = rng.binomial(t, 0.5, size=(sims, n))
    null = np.abs(0.5 - k / t).mean(axis=1)
    exceed = int((null >= score).sum())
    if exceed == 0:   # beyond simulation resolution; Gaussian tail approximation of the null
        z = (score - null.mean()) / null.std()
        from scipy.stats import norm
        return float(norm.logsf(z))
    return math.log(exceed / sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--out", default="e2_detection_depth")
    ap.add_argument("--max-rounds", type=int, default=None, help="smoke tests: cap rounds per target")
    args = ap.parse_args()
    out = out_dir("novel", args.out)
    rows = []
    for cname, diff, rounds in TARGETS:
        if args.max_rounds:
            rounds = range(1, min(max(rounds), args.max_rounds) + 1)
        cipher = load_cipher(cname)
        n = cipher.plain_bits
        D = ints_to_bit_matrix([diff], n)
        for rep in range(args.reps):
            rng = seed_tree(args.seed, "e2", cname, diff, rep)
            nrng = seed_tree(args.seed, "e2null", cname, diff, rep)
            for r in rounds:
                with Timer() as tm:
                    bits = CipherSampler(cipher, rng).draw(D, args.t, r)
                    f = spectral_features(bits, alpha=args.alpha)
                rows.append({
                    "cipher": cname, "difference": hex(diff), "rep": rep, "round": r, "t": args.t,
                    "autond_bias": f["autond_bias"][0],
                    "logp_autond_bias_null": null_bias_logp(f["autond_bias"][0], args.t, n, nrng),
                    "logp_chi2_deg1": float(chi2_logsf(f["chi2_stat1"], f["chi2_df1"])[0]),
                    "logp_chi2_deg12": float(chi2_logsf(f["chi2_stat12"], f["chi2_df12"])[0]),
                    "cap1": f["cap1"][0], "cap2": f["cap2"][0],
                    "max_abs_corr": f["max_abs_corr"][0],
                    "best_mask": mask_name(int(f["best_mask_index"][0]), n),
                    "acc_lb_certified": f["acc_lower_bound_certified"][0],
                    "seconds": tm.seconds})
            print(cname, hex(diff), "rep", rep, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_round.csv", index=False)

    thr = math.log(args.alpha / 13)  # Bonferroni over rounds
    summ = []
    for (cname, diff), g in df.groupby(["cipher", "difference"]):
        rec = {"cipher": cname, "difference": diff, "t": args.t,
               "nd_reach_paper": ND_REACH.get((cname, int(diff, 16)))}
        for col, name in [("logp_autond_bias_null", "depth_autond_bias"), ("logp_chi2_deg1", "depth_chi2_deg1"),
                          ("logp_chi2_deg12", "depth_chi2_deg12")]:
            depths = [int(gg.loc[gg[col] < thr, "round"].max()) if (gg[col] < thr).any() else 0
                      for _, gg in g.groupby("rep")]
            rec[name] = f"{min(depths)}-{max(depths)}" if min(depths) != max(depths) else str(depths[0])
        summ.append(rec)
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "summary.csv", index=False)
    dump({"args": vars(args), "summary": summ}, out / "summary.json")
    write_manifest(out, vars(args))
    show = df[df.rep == 0][["cipher", "difference", "round", "autond_bias", "logp_autond_bias_null",
                            "logp_chi2_deg1", "logp_chi2_deg12", "cap1", "cap2", "max_abs_corr", "best_mask",
                            "acc_lb_certified"]]
    pd.set_option("display.width", 250)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
