"""Phase 4 -- validate the adaptive scorer before it is used in the main comparison.

1. Interval coverage on synthetic candidates with *known* exact scores, for every interval
   method, several t and several numbers of unbiased bits (repeated independent trials).
2. Interval coverage on the real cipher: intervals from t samples vs. a high-t reference
   score (proxy for the exact score) on the frozen pool.
3. Allocation check on a synthetic pool: samples spent vs. distance to the top-K boundary,
   and whether the hard budget is respected.
NumPy only.

Usage:  python experiments/phase4_validate_scorer.py --config experiments/configs/simon32_demo.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, dump, load_config, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.candidates import load_pool
from cgsearch.data import seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.scoring import (AdaptiveConfig, AdaptiveScorer, CipherSampler, SyntheticSampler, batch_cost,
                              bias_from_bits, bootstrap_weighted, intervals)

METHODS = ["bias_bounded", "mcdiarmid", "basic", "percentile"]


def synthetic_coverage(rng, reps=200, alpha=0.1) -> pd.DataFrame:
    rows = []
    for t in (250, 1000, 4000):
        for n_biased in (0, 4, 16, 32):
            p = np.full((1, 1, 32), 0.5)
            p[0, 0, :n_biased] = 0.5 + rng.uniform(0.01, 0.1, n_biased)
            exact = float(np.mean(np.abs(0.5 - p)))
            hits = {m: 0 for m in METHODS}
            widths = {m: [] for m in METHODS}
            for _ in range(reps):
                bits = (rng.random((1, t, 32)) < p[:, 0][:, None, :]).astype(np.uint8)
                pt = bias_from_bits(bits)
                boot = bootstrap_weighted([bits], [1.0], 200, rng)
                for m in METHODS:
                    lo, hi = intervals(pt, boot, np.array([[t]]), [1.0], m, alpha)
                    hits[m] += lo[0] <= exact <= hi[0]
                    widths[m].append(hi[0] - lo[0])
            for m in METHODS:
                rows.append({"t": t, "n_biased_bits_of_32": n_biased, "method": m, "nominal": 1 - alpha,
                             "coverage": hits[m] / reps, "mean_width": float(np.mean(widths[m]))})
    return pd.DataFrame(rows)


def cipher_coverage(cfg, alpha=0.01, reps=20, t=None) -> pd.DataFrame | None:
    ref_file = ROOT / "results/phase3" / cfg["name"] / "reference_scores.json"
    if not ref_file.exists():
        return None
    ref = json.loads(ref_file.read_text())
    pool = load_pool(ROOT / "results/phase1" / cfg["name"] / "candidate_pool.json")
    diffs, rounds = pool["pool_ints"], pool["rounds_scored"]
    ref_scores = np.array(ref["scores"])
    cipher = load_cipher(cfg["cipher"])
    t = t or cfg["scoring"]["adaptive"]["t0"] * 4
    w = np.asarray(rounds, float)
    from cgsearch.data import ints_to_bit_matrix
    D = ints_to_bit_matrix(diffs, cipher.plain_bits)
    rows = []
    for rep in range(reps):
        s = CipherSampler(cipher, seed_tree(cfg["seed"], "phase4", rep))
        bits = [s.draw(D, t, nr) for nr in rounds]
        pr = np.stack([bias_from_bits(b) for b in bits], axis=1)
        pt = pr @ w
        boot = bootstrap_weighted(bits, w, 200, seed_tree(cfg["seed"], "phase4boot", rep))
        tt = np.full((len(diffs), len(rounds)), t)
        for m in METHODS:
            a = alpha / (len(diffs) * 6) if m == "mcdiarmid" else alpha
            lo, hi = intervals(pt, boot if m != "mcdiarmid" else None, tt, w, m, a)
            cover = (lo <= ref_scores) & (ref_scores <= hi)
            rows.append({"rep": rep, "t": t, "method": m, "coverage": float(cover.mean()),
                         "mean_width": float(np.mean(hi - lo))})
    df = pd.DataFrame(rows)
    df.attrs["note"] = (f"reference score uses t={ref['t']} and is itself noisy "
                        f"(floor ~ sum_r r*0.4/sqrt(t_ref)); coverage is of that proxy.")
    return df


def allocation_check(rng) -> dict:
    m, n = 80, 16
    p = np.full((m, 1, n), 0.5)
    strengths = np.sort(rng.uniform(0, 0.3, m))[::-1]
    for i, s in enumerate(strengths):
        p[i, 0, :4] = 0.5 + s
    sam = SyntheticSampler(p, [1], np.random.default_rng(1))
    K = 10
    budget = batch_cost(m, 4000, 1)
    cfg = AdaptiveConfig(k=K, t0=50, t_max=4000, budget=budget, tol=0.0)
    res = AdaptiveScorer(sam, list(range(1, m + 1)), [1], cfg, weights=[1.0],
                         diff_bits=np.arange(m)[:, None], boot_rng=np.random.default_rng(2)).run()
    exact = sam.exact_scores([1.0])
    kth = np.sort(exact)[::-1][K - 1]
    gap = np.abs(exact - kth)
    true_top = set(np.argsort(-exact)[:K] + 1)
    return {"k": K, "budget": budget, "spent": res.ledger["encryptions"],
            "budget_respected": res.ledger["encryptions"] <= budget,
            "shortlist_correct": set(res.shortlist) == true_top,
            "spearman_samples_vs_gap": float(pd.Series(res.samples[:, 0]).corr(pd.Series(-gap), method="spearman")),
            "per_candidate": [{"exact": float(e), "gap_to_kth": float(g), "samples": int(s), "status": st}
                              for e, g, s, st in zip(exact, gap, res.samples[:, 0], res.status)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = out_dir("phase4", cfg["name"])
    rng = seed_tree(cfg["seed"], "phase4", "synthetic")

    syn = synthetic_coverage(rng)
    syn.to_csv(out / "synthetic_coverage.csv", index=False)
    print(syn.pivot_table(index=["t", "n_biased_bits_of_32"], columns="method", values="coverage").to_string())

    cc = cipher_coverage(cfg)
    if cc is not None:
        cc.to_csv(out / "cipher_coverage.csv", index=False)
        print(cc.groupby("method")[["coverage", "mean_width"]].mean().to_string())
        print(cc.attrs["note"])

    alloc = allocation_check(rng)
    dump(alloc, out / "allocation_check.json")
    print({k: v for k, v in alloc.items() if k != "per_candidate"})
    _plot(alloc, out)
    write_manifest(out, cfg)
    print(f"Outputs in {out}")


def _plot(alloc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pc = pd.DataFrame(alloc["per_candidate"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for st, g in pc.groupby("status"):
        ax.scatter(g["gap_to_kth"], g["samples"], s=14, label=st)
    ax.set_xlabel("|exact score - K-th exact score|"); ax.set_ylabel("samples allocated")
    ax.set_yscale("log"); ax.set_title("Adaptive allocation concentrates on close decisions"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "allocation.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
