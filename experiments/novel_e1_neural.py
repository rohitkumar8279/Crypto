"""Novel E1 (part 2) -- train DBitNet (PyTorch port) on the pre-selected candidates and
evaluate once on a frozen test stream; then analyse proxy validity (see
docs/novel_preregistration.md). Resumable: finished candidates are skipped.

Usage:
  python experiments/novel_e1_neural.py --cipher speck3264 train
  python experiments/novel_e1_neural.py --cipher speck3264 analyse
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import dump, out_dir  # noqa: E402

from cgsearch.autond_bridge import load_cipher
from cgsearch.data import make_pairs_dataset, seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.metrics import spearman

RECIPE = {"speck3264": {"rounds": [5, 6, 7], "test_rounds": [6, 7]},
          "simon3264": {"rounds": [7, 8, 9], "test_rounds": [8, 9]}}
N_TRAIN, N_SELECT, N_TEST, EPOCHS, TEST_SEED = 200_000, 50_000, 200_000, 3, 770001
REPEAT_SEED2 = 4


def train_one(cipher, cname, diff, seed, out):
    import torch
    from cgsearch.torch_dbitnet import DBitNet, default_device, evaluate, fit
    path = out / "runs" / f"{diff:#x}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:                      # another worker is training this candidate
        lock.touch(exist_ok=False)
    except FileExistsError:
        return None
    torch.manual_seed(seed)
    model = DBitNet(2 * cipher.plain_bits).to(default_device())
    rec = {"difference": hex(diff), "seed": seed, "rounds": {}, "recipe": {
        "n_train": N_TRAIN, "n_select": N_SELECT, "n_test": N_TEST, "epochs": EPOCHS,
        "test_seed": TEST_SEED, "staged_rounds": RECIPE[cname]["rounds"]}}
    t0 = time.perf_counter()
    for r in RECIPE[cname]["rounds"]:
        X, Y = make_pairs_dataset(cipher, r, diff, N_TRAIN, seed_tree(seed, "e1train", diff, r))
        Xv, Yv = make_pairs_dataset(cipher, r, diff, N_SELECT, seed_tree(seed, "e1select", diff, r))
        h = fit(model, X, Y, Xv, Yv, epochs=EPOCHS, seed=seed)
        entry = {"best_val_acc": max(h["val_acc"]), "history": h}
        if r in RECIPE[cname]["test_rounds"]:
            Xt, Yt = make_pairs_dataset(cipher, r, diff, N_TEST, seed_tree(TEST_SEED, "e1test", diff, r))
            entry["test"] = evaluate(model, Xt, Yt)
        rec["rounds"][str(r)] = entry
        print(f"  {diff:#x} seed {seed} round {r}: val {entry['best_val_acc']:.4f}"
              + (f" test {entry['test']['acc']:.4f}" if "test" in entry else ""), flush=True)
    rec["seconds"] = time.perf_counter() - t0
    path.write_text(json.dumps(rec))
    lock.unlink(missing_ok=True)
    return rec


def boot_spearman(x, y, z=None, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = None if z is None else np.asarray(z, float)
    a, d = [], []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(y[i])) < 3:
            continue
        sa = spearman(x[i], y[i])
        a.append(sa)
        if z is not None:
            d.append(spearman(z[i], y[i]) - sa)
    def ci(v):
        v = [x for x in v if np.isfinite(x)]
        if not v:   # too few distinct values to resample (tiny smoke runs)
            return [float("nan"), float("nan")]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return ci(a), (ci(d) if z is not None else None)


def analyse(cname, out):
    sel = json.loads((out / "selection.json").read_text())
    px = pd.read_csv(out / "proxy_scores.csv")
    ws = pd.read_csv(out / "weighted_scores.csv").set_index("difference")
    runs = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*_seed1.json"))]
    rows = []
    for rec in runs:
        d = rec["difference"]
        row = {"difference": d, "autond_weighted": ws.loc[d, "wb"]}
        for r in RECIPE[cname]["test_rounds"]:
            row[f"test_acc_r{r}"] = rec["rounds"][str(r)]["test"]["acc"]
            p = px[(px.difference == d) & (px["round"] == r)].iloc[0]
            for k in ("autond_bias", "cap12", "max_abs_corr", "acc_lower_bound_certified"):
                row[f"{k}_r{r}"] = p[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "e1_table.csv", index=False)
    res = {"n_candidates": len(df)}
    for r in RECIPE[cname]["test_rounds"]:
        y = df[f"test_acc_r{r}"]
        block = {}
        for name in ["autond_weighted", f"autond_bias_r{r}", f"cap12_r{r}", f"max_abs_corr_r{r}"]:
            ci = boot_spearman(df[name], y)[0]
            dci = None if name == "autond_weighted" else boot_spearman(df["autond_weighted"], y, df[name])[1]
            block[name] = {"spearman": spearman(df[name], y), "ci95": ci, "minus_autond_weighted_ci95": dci}
        viol = df[df[f"acc_lower_bound_certified_r{r}"] > y + 1e-12]
        block["certified_bound_violations"] = viol["difference"].tolist()
        block["top_by_autond"] = {"difference": df.loc[df["autond_weighted"].idxmax(), "difference"],
                                  "test_acc": float(y[df["autond_weighted"].idxmax()])}
        block["top_by_spectral"] = {"difference": df.loc[df[f"cap12_r{r}"].idxmax(), "difference"],
                                    "test_acc": float(y[df[f"cap12_r{r}"].idxmax()])}
        block["best_trained"] = {"difference": df.loc[y.idxmax(), "difference"], "test_acc": float(y.max())}
        res[f"round_{r}"] = block
    noise = []
    for p in sorted((out / "runs").glob("*_seed2.json")):
        r2 = json.loads(p.read_text())
        r1 = json.loads((out / "runs" / p.name.replace("seed2", "seed1")).read_text())
        for r in RECIPE[cname]["test_rounds"]:
            noise.append({"difference": r2["difference"], "round": r,
                          "seed1": r1["rounds"][str(r)]["test"]["acc"], "seed2": r2["rounds"][str(r)]["test"]["acc"]})
    res["training_noise"] = noise
    dump(res, out / "e1_analysis.json")
    print(json.dumps(res, indent=1))
    _plot(df, cname, out)


def _plot(df, cname, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = RECIPE[cname]["test_rounds"]
    fig, ax = plt.subplots(len(rs), 2, figsize=(10, 4 * len(rs)))
    for i, r in enumerate(rs):
        y = df[f"test_acc_r{r}"]
        ax[i, 0].scatter(df["autond_weighted"], y)
        ax[i, 0].set_xlabel("AutoND weighted bias score"); ax[i, 0].set_ylabel(f"DBitNet test acc, round {r}")
        ax[i, 0].set_title(f"Spearman {spearman(df['autond_weighted'], y):.2f}")
        ax[i, 1].scatter(df[f"cap12_r{r}"], y)
        ax[i, 1].plot(np.sort(df[f"cap12_r{r}"]), np.full(len(df), 0.5), "k:", lw=0.5)
        ax[i, 1].scatter(df[f"cap12_r{r}"], df[f"acc_lower_bound_certified_r{r}"], marker="_", c="r", label="certified floor")
        ax[i, 1].set_xscale("symlog", linthresh=1e-3)
        ax[i, 1].set_xlabel(f"spectral capacity C<=2, round {r}"); ax[i, 1].legend()
        ax[i, 1].set_title(f"Spearman {spearman(df[f'cap12_r{r}'], y):.2f}")
    fig.tight_layout(); fig.savefig(out / "e1_scatter.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="speck3264")
    ap.add_argument("action", choices=["train", "analyse"])
    ap.add_argument("--reverse", action="store_true", help="second worker: process the list from the end")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=None, help="default: e1_<cipher>")
    ap.add_argument("--smoke", action="store_true", help="tiny data sizes, first 4 candidates, no seed-2 repeats")
    args = ap.parse_args()
    out = out_dir("novel", args.out or f"e1_{args.cipher}")
    if args.smoke:
        global N_TRAIN, N_SELECT, N_TEST, EPOCHS, REPEAT_SEED2
        N_TRAIN, N_SELECT, N_TEST, EPOCHS, REPEAT_SEED2 = 4000, 2000, 4000, 1, 0
    if args.action == "analyse":
        return analyse(args.cipher, out)
    import torch
    torch.set_num_threads(args.threads)
    cipher = load_cipher(args.cipher)
    sel = json.loads((out / "selection.json").read_text())["selected"]
    order = list(reversed(sel)) if args.reverse else sel
    if args.smoke:
        order = order[:4]
    for i, d in enumerate(order):
        print(f"[{i + 1}/{len(sel)}] {d}", flush=True)
        train_one(cipher, args.cipher, int(d, 16), 1, out)
    for d in sel[:REPEAT_SEED2]:
        train_one(cipher, args.cipher, int(d, 16), 2, out)
    write_manifest(out / "runs", {"recipe": RECIPE[args.cipher], "n_train": N_TRAIN, "epochs": EPOCHS})


if __name__ == "__main__":
    main()
