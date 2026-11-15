"""Phase 3a -- how noisy is AutoND's bias score, and how stable is its top-K ranking?

On the frozen candidate pool, repeat the fixed-budget scoring with independent seeds for
several sample sizes t, plus one high-t reference run. Reports per-candidate SD, rank
correlation between repetitions, top-K overlap between repetitions and with the reference,
and the cost of each t. NumPy only.

Usage:  python experiments/phase3_score_variability.py --config experiments/configs/simon32_demo.json
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import ROOT, Timer, dump, load_config, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.candidates import load_pool
from cgsearch.data import seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.metrics import kendall, spearman, topk_overlap
from cgsearch.scoring import CipherSampler, score_fixed, top_k


def reference_scores(cipher, cfg, diffs, rounds, out):
    path = out / "reference_scores.json"
    if path.exists():
        import json
        d = json.loads(path.read_text())
        return np.array(d["scores"]), np.array(d["per_round"])
    sc = cfg["scoring"]
    with Timer() as tm:
        ref = score_fixed(CipherSampler(cipher, seed_tree(cfg["seed"], "reference")), diffs, rounds,
                          sc["t_reference"], sc["k"], chunk_candidates=16)
    dump({"t": sc["t_reference"], "diffs": [hex(d) for d in diffs], "scores": ref.scores,
          "per_round": ref.per_round, "ledger": ref.ledger, "seconds": tm.seconds}, path)
    return ref.scores, ref.per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    pool = load_pool(ROOT / "results/phase1" / cfg["name"] / "candidate_pool.json")
    diffs, rounds = pool["pool_ints"], pool["rounds_scored"]
    cipher = load_cipher(cfg["cipher"])
    out = out_dir("phase3", cfg["name"])
    K = cfg["scoring"]["k"]

    ref_scores, ref_pr = reference_scores(cipher, cfg, diffs, rounds, out)
    ref_top = top_k(diffs, ref_scores, K)

    rows, summary = [], []
    for t in cfg["variability"]["t_values"]:
        reps = []
        for rep in range(cfg["variability"]["reps"]):
            with Timer() as tm:
                r = score_fixed(CipherSampler(cipher, seed_tree(cfg["seed"], "variability", t, rep)),
                                diffs, rounds, t, K, chunk_candidates=64)
            reps.append((r, tm.seconds))
        S = np.stack([r.scores for r, _ in reps])           # (reps, m)
        sd = S.std(axis=0, ddof=1)
        pair_sp, pair_kt, pair_ov = [], [], []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                pair_sp.append(spearman(S[i], S[j]))
                pair_kt.append(kendall(S[i], S[j]))
                pair_ov.append(topk_overlap(reps[i][0].shortlist, reps[j][0].shortlist, K))
        ref_ov = [topk_overlap(r.shortlist, ref_top, K) for r, _ in reps]
        ref_sp = [spearman(r.scores, ref_scores) for r, _ in reps]
        kth = np.sort(ref_scores)[::-1][K - 1]
        near = np.abs(ref_scores - kth) <= 2 * sd   # candidates whose rank vs the K-th is noise-level
        summary.append({
            "t": t, "encryptions_per_run": reps[0][0].ledger["encryptions"],
            "seconds_per_run": float(np.mean([s for _, s in reps])),
            "median_sd": float(np.median(sd)), "max_sd": float(sd.max()),
            "sd_top_k_median": float(np.median(sd[np.isin(diffs, ref_top)])),
            "rep_spearman_mean": float(np.mean(pair_sp)), "rep_kendall_mean": float(np.mean(pair_kt)),
            "rep_topk_overlap_mean": float(np.mean(pair_ov)),
            "ref_topk_overlap_mean": float(np.mean(ref_ov)), "ref_spearman_mean": float(np.mean(ref_sp)),
            "n_candidates_within_2sd_of_kth": int(near.sum())})
        for ci, d in enumerate(diffs):
            rows.append({"t": t, "difference": hex(d), "mean_score": S[:, ci].mean(), "sd": sd[ci],
                         "ref_score": ref_scores[ci], "in_ref_topk": d in ref_top})

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_candidate.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "summary.csv", index=False)
    dump({"k": K, "pool_size": len(diffs), "rounds": rounds, "reference_topk": [hex(x) for x in ref_top],
          "reference_scores_sorted": sorted([(hex(d), float(s)) for d, s in zip(diffs, ref_scores)],
                                            key=lambda x: -x[1])[:3 * K],
          "summary": summary}, out / "summary.json")
    _plots(df, sm, out, K)
    write_manifest(out, cfg)
    print(sm.to_string(index=False))
    print(f"Outputs in {out}")


def _plots(df, sm, out, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for t, g in df.groupby("t"):
        ax[0].scatter(g["ref_score"], g["sd"], s=10, label=f"t={t}")
    ax[0].set_xlabel("reference weighted score"); ax[0].set_ylabel("SD across repetitions")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Score noise by candidate")
    ax[1].plot(sm["t"], sm["rep_topk_overlap_mean"], "o-", label="top-K overlap between repetitions")
    ax[1].plot(sm["t"], sm["ref_topk_overlap_mean"], "s-", label="top-K overlap with reference")
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("samples per round t")
    ax[1].set_title(f"Ranking stability (K={K})"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "score_variability.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
