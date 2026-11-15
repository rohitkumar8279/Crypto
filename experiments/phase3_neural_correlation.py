"""Phase 3b -- H1: does AutoND's bias score predict DBitNet performance? (TensorFlow, GPU)

Picks `correlation_candidates` differences from the frozen pool spread over the reference
score range (stratified by quantile, always including the top), trains each with the fixed
recipe on the *selection* data only, and reports Spearman correlations (with bootstrap CIs)
between the selection accuracy and (i) the weighted score, (ii) the per-round scores.
The final test stream is never touched here.

Usage:  python experiments/phase3_neural_correlation.py --config experiments/configs/simon32_main.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, dump, load_config, out_dir  # noqa: E402

from cgsearch.candidates import load_pool
from cgsearch.manifest import write_manifest
from cgsearch.metrics import spearman
from cgsearch.neural import train_eval_dbitnet


def pick_stratified(diffs, scores, n, rng):
    order = np.argsort(-np.asarray(scores))
    top = list(order[: max(2, n // 6)])
    rest = order[len(top):]
    bins = np.array_split(rest, n - len(top))
    return [int(diffs[i]) for i in top] + [int(diffs[rng.choice(b)]) for b in bins if len(b)]


def boot_ci(x, y, rng, B=2000):
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 2:
            vals.append(spearman(x[i], y[i]))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:   # too few distinct values to resample (tiny runs): report undefined, do not crash
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["neural"]
    pool = load_pool(ROOT / "results/phase1" / cfg["name"] / "candidate_pool.json")
    ref = json.loads((ROOT / "results/phase3" / cfg["name"] / "reference_scores.json").read_text())
    diffs, scores, per_round = pool["pool_ints"], np.array(ref["scores"]), np.array(ref["per_round"])
    out = out_dir("phase3", cfg["name"], "neural")
    rng = np.random.default_rng(cfg["seed"])
    chosen = pick_stratified(diffs, scores, nc["correlation_candidates"], rng)

    rows = []
    for d in chosen:
        r = train_eval_dbitnet(cfg["cipher"], d, nc["start_round"], nc["target_round"], nc["seeds"][0],
                               out / "runs", epochs=nc["epochs"], n_train=nc["n_train"], n_select=nc["n_select"])
        i = diffs.index(d)
        rows.append({"difference": hex(d), "weighted_score": scores[i],
                     **{f"score_r{rr}": per_round[i, j] for j, rr in enumerate(pool["rounds_scored"])},
                     "selection_acc": r["selection_acc"],
                     **{f"sel_acc_r{x['round']}": x["best_val_acc"] for x in r["rounds"]},
                     "train_seconds": r["total_train_seconds"]})
        print(rows[-1]["difference"], rows[-1]["weighted_score"], rows[-1]["selection_acc"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "score_vs_neural.csv", index=False)
    brng = np.random.default_rng(1)
    res = {}
    for col in ["weighted_score"] + [c for c in df.columns if c.startswith("score_r")]:
        res[col] = {"spearman": spearman(df[col], df["selection_acc"]),
                    "ci95": boot_ci(df[col], df["selection_acc"], brng)}
    dump({"n": len(df), "target_round": nc["target_round"], "correlations": res}, out / "summary.json")
    write_manifest(out, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
