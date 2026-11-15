"""Phase 5a -- matched comparison of scoring policies (cipher evaluations only; NumPy).

Policies, all on the same frozen pool, rounds, weights, cost model and shortlist size K:
  A_fixed              AutoND fixed budget: t_fixed samples per candidate per round.
  B_adaptive           confidence-guided scorer, hard budget = cost(A), t_max = t_fixed.
  C_uniform_matched    ablation: uniform allocation with B's *actual* spend (t = budget_B / ...).
  C2_successive_halving ablation: B's checkpoints, keep top 1/2 by point estimate, no intervals.
Plus a budget sweep (fractions of cost(A)) for quality-vs-budget curves.

Quality is measured against an independent high-t reference scoring (Phase 3), never
against neural results or test data. Shortlists are saved for Phase 5b (neural).

Usage:  python experiments/phase5_scoring_compare.py --config experiments/configs/simon32_demo.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, Timer, dump, load_config, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.candidates import load_pool
from cgsearch.data import seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.metrics import shortlist_regret, topk_overlap
from cgsearch.scoring import (AdaptiveConfig, AdaptiveScorer, CipherSampler, max_uniform_t,
                              score_fixed, score_successive_halving, top_k)


def quality(res, ref: dict, ref_top, K, eps=0.1) -> dict:
    best = max(ref.values())
    sl = res.shortlist[:K]
    return {"regret": shortlist_regret(sl, ref, K),
            "topk_overlap_ref": topk_overlap(sl, ref_top, K),
            "eps_close_precision": float(np.mean([ref[d] >= (1 - eps) * best for d in sl])),
            "encryptions": res.ledger["encryptions"], "round_evaluations": res.ledger["round_evaluations"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["scoring"]
    K = sc["k"]
    pool = load_pool(ROOT / "results/phase1" / cfg["name"] / "candidate_pool.json")
    diffs, rounds = pool["pool_ints"], pool["rounds_scored"]
    ref_file = ROOT / "results/phase3" / cfg["name"] / "reference_scores.json"
    if not ref_file.exists():
        raise SystemExit("Run phase3_score_variability.py first (it creates the reference scoring).")
    ref_scores = json.loads(ref_file.read_text())["scores"]
    ref = dict(zip(diffs, ref_scores))
    ref_top = top_k(diffs, np.array(ref_scores), K)
    cipher = load_cipher(cfg["cipher"])
    out = out_dir("phase5", cfg["name"])
    acfg = sc["adaptive"]
    rows, shortlists = [], {}

    def run(policy, seed, frac, fn):
        with Timer() as tm:
            res = fn()
        q = quality(res, ref, ref_top, K)
        rows.append({"policy": policy, "seed": seed, "budget_fraction": frac, "seconds": tm.seconds,
                     "stop_reason": res.config.get("stop_reason", ""),
                     "t_per_round": int(res.config.get("t", 0)) or None, **q})
        shortlists[f"{policy}|seed={seed}|frac={frac}"] = [hex(x) for x in res.shortlist]
        return res

    for seed in sc["seeds"]:
        sampler = lambda tag: CipherSampler(cipher, seed_tree(cfg["seed"], "phase5", tag, seed))  # noqa: E731
        A = run("A_fixed", seed, 1.0, lambda: score_fixed(sampler("A"), diffs, rounds, sc["t_fixed"], K))
        cost_A = A.ledger["encryptions"]
        for frac in sc["budget_fractions"]:
            budget = int(frac * cost_A)
            ad = AdaptiveConfig(k=K, budget=budget, **acfg)
            B = run("B_adaptive", seed, frac, lambda: AdaptiveScorer(
                sampler(f"B{frac}"), diffs, rounds, ad, boot_rng=seed_tree(cfg["seed"], "boot", seed, frac)).run())
            if frac < 1.0:
                tA = max_uniform_t(budget, len(diffs), len(rounds))
                run("A_fixed", seed, frac, lambda: score_fixed(sampler(f"A{frac}"), diffs, rounds, tA, K))
            tC = max_uniform_t(B.ledger["encryptions"], len(diffs), len(rounds))
            r = run("C_uniform_matched", seed, frac, lambda: score_fixed(
                sampler(f"C{frac}"), diffs, rounds, tC, K, policy_name="C_uniform_matched"))
            rows[-1]["matched_to_B_encryptions"] = B.ledger["encryptions"]
            run("C2_successive_halving", seed, frac, lambda: score_successive_halving(
                sampler(f"C2{frac}"), diffs, rounds, ad))
            if frac == 1.0:
                dump(B.to_json(), out / f"B_adaptive_seed{seed}_detail.json")
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    agg = (df.groupby(["policy", "budget_fraction"])
             .agg(regret_mean=("regret", "mean"), regret_sd=("regret", "std"),
                  eps_close_precision=("eps_close_precision", "mean"),
                  topk_overlap_ref=("topk_overlap_ref", "mean"),
                  encryptions_mean=("encryptions", "mean"), n=("seed", "count"))
             .reset_index())
    agg.to_csv(out / "summary.csv", index=False)
    dump({"k": K, "pool_size": len(diffs), "rounds": rounds, "reference_topk": [hex(x) for x in ref_top],
          "shortlists": shortlists}, out / "shortlists.json")
    _plot(agg, out, K)
    write_manifest(out, cfg)
    print(agg.to_string(index=False))
    print(f"Outputs in {out}")


def _plot(agg, out, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pol, g in agg.groupby("policy"):
        g = g.sort_values("encryptions_mean")
        ax[0].errorbar(g["encryptions_mean"], g["regret_mean"], yerr=g["regret_sd"], marker="o", capsize=3, label=pol)
        ax[1].plot(g["encryptions_mean"], g["eps_close_precision"], marker="o", label=pol)
    ax[0].set_xscale("log"); ax[0].set_xlabel("cipher encryptions (scoring)"); ax[0].set_ylabel("shortlist regret vs reference")
    ax[0].set_title(f"Quality vs budget (K={K}; lower is better)"); ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("cipher encryptions (scoring)")
    ax[1].set_ylabel("fraction of shortlist 0.1-close to best"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "quality_vs_budget.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
