"""Phase 1 -- environment and implementation audit, verification evidence, smoke run.

Usage:  python experiments/phase1_verify.py --config experiments/configs/simon32_demo.json
Needs only NumPy (TensorFlow availability is probed in a subprocess and recorded).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

import numpy as np

from common import ROOT, Timer, dump, load_config, out_dir  # noqa: E402

from cgsearch import simon_ref as S
from cgsearch.autond_bridge import check_pinned, load_cipher, load_optimizer
from cgsearch.candidates import build_candidate_pool, save_pool
from cgsearch.data import bits_to_int, ints_to_bit_matrix, make_pairs_dataset, random_bits, seed_tree
from cgsearch.manifest import write_manifest
from cgsearch.scoring import CipherSampler, bias_from_bits


def probe_tensorflow() -> dict:
    code = "import tensorflow as tf; print(tf.__version__); print([d.name for d in tf.config.list_physical_devices()])"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        lines = r.stdout.strip().splitlines()
        return {"available": True, "version": lines[0], "devices": lines[-1]}
    err = [l for l in r.stderr.strip().splitlines() if l.strip()]
    return {"available": False, "error": err[-1] if err else "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/simon32_demo.json"))
    ap.add_argument("--verbose", action="store_true", help="print optimizer progress per generation")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = out_dir("phase1", cfg["name"])
    report: dict = {"checks": {}}
    commit = check_pinned(strict=False)
    report["autond_commit"] = commit

    cipher = load_cipher(cfg["cipher"])  # runs AutoND's own test-vector assertion on import
    report["checks"]["autond_import_testvector_assert"] = "passed (assert executed on import)"

    # 1. Published test vector, independent reference
    tv = S.TEST_VECTOR
    ok_ref = S.encrypt(tv["plaintext"], tv["key"]) == tv["ciphertext"]
    pb = np.concatenate([np.array([(w >> (15 - i)) & 1 for i in range(16)], np.uint8) for w in tv["plaintext"]])[None]
    kb = np.concatenate([np.array([(w >> (15 - i)) & 1 for i in range(16)], np.uint8) for w in tv["key"]])[None]
    ok_autond = bits_to_int(cipher.encrypt(pb, kb, 32)[0]) == (tv["ciphertext"][0] << 16 | tv["ciphertext"][1])
    report["checks"]["published_test_vector"] = {"reference": ok_ref, "autond": ok_autond, **{k: [hex(x) for x in v] for k, v in tv.items()}}

    # 2. Cross-check AutoND vs independent reference on random inputs, all round counts 1..32
    rng = seed_tree(cfg["seed"], "phase1", "crosscheck")
    n_per_round, mismatches = 300, 0
    for nr in range(1, 33):
        p, k = random_bits(rng, n_per_round, 32), random_bits(rng, n_per_round, 64)
        c = cipher.encrypt(p, k, nr)
        for i in range(n_per_round):
            kw = tuple(bits_to_int(k[i, 16 * j:16 * j + 16]) for j in range(4))
            x, y = S.encrypt((bits_to_int(p[i, :16]), bits_to_int(p[i, 16:])), kw, nr)
            mismatches += bits_to_int(c[i]) != ((x << 16) | y)
    report["checks"]["random_crosscheck"] = {"rounds": "1..32", "samples_per_round": n_per_round, "mismatches": int(mismatches)}

    # 3. Data labels: after 1 round, real pairs have right-half difference == left half of delta
    delta = int(cfg["reference_difference"], 16)
    X, Y = make_pairs_dataset(cipher, 1, delta, 20000, seed_tree(cfg["seed"], "phase1", "labels"))
    rd = np.array([bits_to_int(x[16:32] ^ x[48:64]) for x in X])
    report["checks"]["labels_round1"] = {
        "real_pairs_matching": float(np.mean(rd[Y == 1] == delta >> 16)),
        "random_pairs_matching": float(np.mean(rd[Y == 0] == delta >> 16)),
        "label_balance": float(Y.mean()),
        "note": "negative class = uniformly random 64-bit strings (AutoND main.make_train_data)"}

    # 4. Our sampler + bias score == AutoND evaluate_multiple_differences on identical inputs
    opt = load_optimizer()
    diffs = [delta, 0x8000, 0x1, 0x12345678]
    D = ints_to_bit_matrix(diffs, 32)
    r2 = np.random.default_rng(cfg["seed"])
    st = r2.bit_generator.state
    ours = bias_from_bits(CipherSampler(cipher, r2).draw(D, 2000, 4))
    r3 = np.random.default_rng(); r3.bit_generator.state = st
    keys, pt0 = random_bits(r3, 2000, 64), random_bits(r3, 2000, 32)
    ref = opt.evaluate_multiple_differences(D, pt0, keys, cipher.encrypt(pt0, keys, 4), 4, 32, 64, cipher.encrypt)
    report["checks"]["score_equivalence_max_abs_diff"] = float(np.max(np.abs(ours - ref)))

    # 5. Random baseline (noise floor) of the bias score
    np.random.seed(cfg["seed"])
    floor = {}
    for t in (10**3, 10**4):
        s = opt.empirical_threshold_estimation(t, 32)
        floor[str(t)] = {"mean": float(s.mean()), "std": float(s.std()), "max": float(s.max()),
                         "theory_mean_sqrt(1/(2 pi t))": math.sqrt(1 / (2 * math.pi * t))}
    report["checks"]["random_baseline_bias_score"] = floor

    # 6. Smoke run: AutoND optimizer (seeded) -> frozen candidate pool
    pc = cfg["pool"]
    with Timer() as tm:
        pool = build_candidate_pool(cipher, seed=cfg["seed"], num_generations=pc["num_generations"],
                                    nb_samples=pc["nb_samples"], threshold=pc["threshold"],
                                    verbose=args.verbose)
    pool["seconds"] = tm.seconds
    save_pool(pool, out / "candidate_pool.json")
    report["smoke_pool"] = {k: pool[k] for k in ("highest_round", "rounds_scored", "ledger", "seconds")}
    report["smoke_pool"]["pool_size"] = len(pool["pool"])
    report["smoke_pool"]["reference_difference_in_pool"] = cfg["reference_difference"] in pool["pool"]

    report["tensorflow"] = probe_tensorflow()
    dump(report, out / "report.json")
    write_manifest(out, cfg)

    ok = (ok_ref and ok_autond and mismatches == 0 and report["checks"]["labels_round1"]["real_pairs_matching"] == 1.0
          and report["checks"]["score_equivalence_max_abs_diff"] < 1e-12)
    print(f"Phase 1 verification {'PASSED' if ok else 'FAILED'}; pool size {len(pool['pool'])}, "
          f"highest biased round {pool['highest_round']}, {tm.seconds:.1f}s; TF available: {report['tensorflow']['available']}")
    print(f"Report: {out / 'report.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
