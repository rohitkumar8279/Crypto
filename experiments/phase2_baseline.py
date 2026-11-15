"""Phase 2 -- baseline reproduction (requires TensorFlow 2.15; use a GPU / Colab).

(a) Evaluate AutoND's released, polished 11-round SIMON32/64 DBitNet (difference 0x400) on
    fresh seeded test data. AutoND's README reports 0.5182 +- 0.0004; paper Table 6: 0.5179.
(b) Train DBitNet on 0x400 with AutoND's simple pipeline (start_round -> target_round), twice
    with different seeds, to check run-to-run reproducibility. Selection data only; no test.

Usage:  python experiments/phase2_baseline.py --config experiments/configs/simon32_demo.json [--skip-train]
"""

from __future__ import annotations

import argparse

from common import ROOT, dump, load_config, out_dir  # noqa: E402

from cgsearch.autond_bridge import AUTOND_DIR
from cgsearch.manifest import write_manifest
from cgsearch.neural import evaluate_saved_weights, train_eval_dbitnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--n-eval", type=int, default=10**6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    nc = cfg["neural"]
    out = out_dir("phase2", cfg["name"])
    report = {}

    w = AUTOND_DIR / "model_weights" / "simon3264_single-key_0x400_round11_polished.h5"
    ev = evaluate_saved_weights("simon3264", str(w), 0x400, 11, test_seed=nc["test_seed"] + 1,
                                n_test=args.n_eval, n_test_sets=5)
    ev["documented"] = {"autond_readme": "0.5182 +- 0.0004", "paper_table6_polished": 0.5179}
    report["released_weights_round11"] = ev
    print(f"Released 11-round model: acc {ev['acc_mean']:.4f} +- {ev['acc_std']:.4f} (README: 0.5182 +- 0.0004)")

    if not args.skip_train:
        runs = []
        for seed in nc["seeds"][:2]:
            r = train_eval_dbitnet(cfg["cipher"], int(cfg["reference_difference"], 16), nc["start_round"],
                                   nc["target_round"], seed, out / "train", epochs=nc["epochs"],
                                   n_train=nc["n_train"], n_select=nc["n_select"])
            runs.append({k: r[k] for k in ("seed", "selection_acc", "total_train_seconds", "total_examples_seen")}
                        | {"per_round": [(x["round"], x["best_val_acc"]) for x in r["rounds"]]})
            print(runs[-1])
        report["train_0x400"] = runs
        report["paper_reference_round_accuracies_dbitnet_simon32"] = {"8": 0.8335, "9": 0.6560, "10": 0.5616, "11": 0.5166}
    dump(report, out / "report.json")
    write_manifest(out, cfg)


if __name__ == "__main__":
    main()
