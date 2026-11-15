"""Phase 5b -- neural evaluation of the shortlists from Phase 5a (TensorFlow, GPU).

Stage `select`: every difference in any policy's full-budget shortlist (for the chosen
scoring seeds) is trained once with the fixed recipe and neural seed nc.seeds[0]; results
are shared between policies, so neural compute per policy is identical. Each policy's
finalist = highest *selection* accuracy within its shortlist.

Stage `final`: each finalist is retrained with every neural seed and evaluated ONCE on the
frozen test stream (nc.test_seed). A lock file prevents silent re-use of the test set.

Usage:
  python experiments/phase5_neural_eval.py --config ... --stage select
  python experiments/phase5_neural_eval.py --config ... --stage final
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ROOT, dump, load_config, out_dir  # noqa: E402

from cgsearch.manifest import write_manifest
from cgsearch.metrics import mean_ci
from cgsearch.neural import train_eval_dbitnet

POLICIES = ["A_fixed", "B_adaptive", "C_uniform_matched", "C2_successive_halving"]


def shortlists(cfg, scoring_seeds):
    d = json.loads((ROOT / "results/phase5" / cfg["name"] / "shortlists.json").read_text())["shortlists"]
    return {(p, s): [int(x, 16) for x in d[f"{p}|seed={s}|frac=1.0"]] for p in POLICIES for s in scoring_seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    ap.add_argument("--stage", choices=["select", "final"], required=True)
    ap.add_argument("--scoring-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--force-rerun-test", action="store_true", help="logged; invalidates 'evaluated once'")
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["neural"]
    seeds = args.scoring_seeds or cfg["scoring"]["seeds"]
    out = out_dir("phase5", cfg["name"], "neural")
    sl = shortlists(cfg, seeds)
    recipe = dict(epochs=nc["epochs"], n_train=nc["n_train"], n_select=nc["n_select"])

    if args.stage == "select":
        union = sorted({d for v in sl.values() for d in v})
        print(f"{len(union)} distinct differences to train for selection")
        sel = {}
        for d in union:
            r = train_eval_dbitnet(cfg["cipher"], d, nc["start_round"], nc["target_round"], nc["seeds"][0],
                                   out / "select", **recipe)
            sel[hex(d)] = {"selection_acc": r["selection_acc"], "train_seconds": r["total_train_seconds"]}
            print(hex(d), sel[hex(d)], flush=True)
        finalists = {}
        for (p, s), v in sl.items():
            best = max(v, key=lambda d: (sel[hex(d)]["selection_acc"], -d))
            finalists[f"{p}|seed={s}"] = {"finalist": hex(best), "selection_acc": sel[hex(best)]["selection_acc"],
                                          "shortlist_selection_accs": {hex(d): sel[hex(d)]["selection_acc"] for d in v}}
        dump({"selection": sel, "finalists": finalists}, out / "selection.json")
        write_manifest(out / "select", cfg)
        return

    lock = out / "final_test_done.json"
    if lock.exists() and not args.force_rerun_test:
        raise SystemExit(f"Final test already evaluated ({lock}). Use --force-rerun-test only with a logged reason.")
    sel = json.loads((out / "selection.json").read_text())
    finalists = {k: int(v["finalist"], 16) for k, v in sel["finalists"].items()}
    tested = {}
    for d in sorted(set(finalists.values())):
        tested[hex(d)] = [train_eval_dbitnet(cfg["cipher"], d, nc["start_round"], nc["target_round"], s,
                                             out / "final", n_test=nc["n_test"], n_test_sets=nc["n_test_sets"],
                                             test_seed=nc["test_seed"], evaluate_test=True, **recipe)["test"]
                          for s in nc["seeds"]]
    rows = []
    for key, d in finalists.items():
        p, s = key.split("|seed=")
        accs = [t["acc_mean"] for t in tested[hex(d)]]
        m, lo, hi = mean_ci(accs)
        rows.append({"policy": p, "scoring_seed": int(s), "finalist": hex(d), "test_acc_mean_over_neural_seeds": m,
                     "ci95_lo": lo, "ci95_hi": hi, "advantage": 2 * m - 1})
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_test.csv", index=False)
    summ = df.groupby("policy")["test_acc_mean_over_neural_seeds"].agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(out / "final_test_summary.csv", index=False)
    dump({"tested": tested, "rows": rows, "force_rerun": args.force_rerun_test}, lock)
    write_manifest(out / "final", cfg)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
