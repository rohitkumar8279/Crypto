"""Seeded DBitNet training / selection / final-test wrapper (requires TensorFlow 2.15).

The model (``dbitnet.make_model``), optimizer (Adam with AMSGrad), loss (MSE), batch
size (5000), checkpoint rule (best val_loss) and round-by-round "simple training
pipeline" follow AutoND's ``train_nets.py``. What differs:

* data come from seeded, *named* streams: 'train', 'select' and 'test' never overlap;
* the selection (validation) set drives checkpointing only; the final test sets are
  generated from a separate stream and are evaluated only when ``evaluate_test=True``;
* the run trains up to a fixed ``target_round`` (instead of AutoND's open-ended loop)
  so every candidate receives an identical neural-training budget;
* wall-clock time and examples seen are recorded for compute accounting.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .autond_bridge import load_cipher, load_dbitnet
from .data import make_pairs_dataset, seed_tree
from .metrics import advantage, wilson_ci

BATCHSIZE = 5000          # AutoND train_nets.BATCHSIZE


def _tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


def evaluate(model, X, Y, batch_size: int = BATCHSIZE) -> dict:
    Z = model.predict(X, batch_size=batch_size, verbose=0).flatten()
    Zb = Z > 0.5
    n, n1 = len(Y), int(np.sum(Y == 1))
    correct = int(np.sum(Zb == Y))
    tpr = float(np.sum(Zb[Y == 1]) / n1)
    tnr = float(np.sum(~Zb[Y == 0]) / (n - n1))
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    return {"acc": acc, "tpr": tpr, "tnr": tnr, "balanced_acc": 0.5 * (tpr + tnr),
            "advantage": advantage(acc), "acc_ci95": [lo, hi], "n": n}


def train_eval_dbitnet(cipher_name: str, input_difference: int, start_round: int, target_round: int,
                       seed: int, out_dir: str | Path, epochs: int = 10, n_train: int = 10**7,
                       n_select: int = 10**6, n_test: int = 10**6, n_test_sets: int = 5,
                       test_seed: int | None = None, evaluate_test: bool = False,
                       deterministic_ops: bool = False) -> dict:
    tf = _tf()
    cipher = load_cipher(cipher_name)
    dbitnet = load_dbitnet()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cipher_name}_{input_difference:#x}_r{start_round}-{target_round}_seed{seed}"
    result_path = out_dir / f"{tag}.json"
    if result_path.exists():
        prev = json.loads(result_path.read_text())
        if not evaluate_test or "test" in prev:
            return prev  # resumable: skip finished runs

    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    model = dbitnet.make_model(2 * cipher.plain_bits)
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss="mse", metrics=["acc"])

    rounds, t_start = [], time.perf_counter()
    weights_path = out_dir / f"{tag}_round{{r}}.h5"
    for r in range(start_round, target_round + 1):
        X, Y = make_pairs_dataset(cipher, r, input_difference, n_train, seed_tree(seed, "train", input_difference, r))
        Xv, Yv = make_pairs_dataset(cipher, r, input_difference, n_select, seed_tree(seed, "select", input_difference, r))
        ckpt = tf.keras.callbacks.ModelCheckpoint(str(weights_path).format(r=r), monitor="val_loss",
                                                  save_best_only=True, save_weights_only=True)
        t0 = time.perf_counter()
        h = model.fit(X, Y, epochs=epochs, batch_size=BATCHSIZE, validation_data=(Xv, Yv),
                      callbacks=[ckpt], verbose=2)
        model.load_weights(str(weights_path).format(r=r))
        rounds.append({"round": r, "best_val_acc": float(np.max(h.history["val_acc"])),
                       "history": {k: [float(x) for x in v] for k, v in h.history.items()},
                       "train_seconds": time.perf_counter() - t0,
                       "examples_seen": int(epochs * n_train)})
        del X, Y, Xv, Yv

    res = {"cipher": cipher_name, "difference": hex(input_difference), "start_round": start_round,
           "target_round": target_round, "seed": seed, "epochs": epochs, "n_train": n_train,
           "n_select": n_select, "rounds": rounds,
           "selection_acc": rounds[-1]["best_val_acc"],
           "total_train_seconds": time.perf_counter() - t_start,
           "total_examples_seen": int(sum(r["examples_seen"] for r in rounds)),
           "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]}

    if evaluate_test:
        if test_seed is None:
            raise ValueError("final test requires an explicit, frozen test_seed")
        evals = []
        for i in range(n_test_sets):
            Xt, Yt = make_pairs_dataset(cipher, target_round, input_difference, n_test,
                                        seed_tree(test_seed, "test", input_difference, target_round, i))
            evals.append(evaluate(model, Xt, Yt))
        accs = [e["acc"] for e in evals]
        res["test"] = {"test_seed": test_seed, "sets": evals, "acc_mean": float(np.mean(accs)),
                       "acc_std": float(np.std(accs)), "advantage_mean": advantage(float(np.mean(accs)))}
    result_path.write_text(json.dumps(res, indent=2))
    tf.keras.backend.clear_session()
    return res


def evaluate_saved_weights(cipher_name: str, weights_path: str, input_difference: int, nr: int,
                           test_seed: int, n_test: int = 10**6, n_test_sets: int = 5) -> dict:
    """Evaluate an existing AutoND .h5 model on fresh seeded data (cf. AutoND eval_nets.py)."""
    tf = _tf()
    cipher = load_cipher(cipher_name)
    model = load_dbitnet().make_model(2 * cipher.plain_bits)
    model.load_weights(weights_path)
    evals = []
    for i in range(n_test_sets):
        Xt, Yt = make_pairs_dataset(cipher, nr, input_difference, n_test,
                                    seed_tree(test_seed, "test", input_difference, nr, i))
        evals.append(evaluate(model, Xt, Yt))
    accs = [e["acc"] for e in evals]
    return {"weights": weights_path, "round": nr, "difference": hex(input_difference), "sets": evals,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs))}
