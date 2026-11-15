"""Regression tests for the PyTorch DBitNet port (fidelity to AutoND's Keras model)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cgsearch.autond_bridge import AUTOND_DIR, load_cipher  # noqa: E402
from cgsearch.data import make_pairs_dataset, seed_tree  # noqa: E402
from cgsearch.torch_dbitnet import DBitNet, evaluate, fit, get_dilation_rates, load_keras_h5  # noqa: E402

WEIGHTS = AUTOND_DIR / "model_weights" / "simon3264_single-key_0x400_round11_polished.h5"


def test_architecture_matches_paper():
    assert get_dilation_rates(64) == [31, 15, 7, 3]            # paper Table 4
    m = DBitNet(64)
    trainable = sum(p.numel() for p in m.parameters())
    bn_stats = sum(b.running_mean.numel() * 2 for b in m.modules() if isinstance(b, torch.nn.BatchNorm1d))
    assert trainable == 296_065 and trainable + bn_stats == 298_113   # paper Table 5: 298.11k


@pytest.mark.skipif(not WEIGHTS.exists(), reason="AutoND released weights not present")
def test_released_weights_reproduce_documented_accuracy():
    """AutoND README: 0.5182 +- 0.0004 (11-round SIMON32, 0x400). n=1e5 -> SE ~ 0.0016."""
    pytest.importorskip("h5py")
    m = load_keras_h5(DBitNet(64), str(WEIGHTS))
    X, Y = make_pairs_dataset(load_cipher("simon3264"), 11, 0x400, 100_000, seed_tree(3, "port_regression"))
    acc = evaluate(m, X, Y)["acc"]
    assert abs(acc - 0.5182) < 0.0065                          # ~4 SE
    assert acc > 0.505                                          # clearly above chance


def test_short_training_learns_easy_round():
    torch.manual_seed(0)
    c = load_cipher("speck3264")
    X, Y = make_pairs_dataset(c, 3, 0x400000, 20_000, seed_tree(1, "tr"))
    Xv, Yv = make_pairs_dataset(c, 3, 0x400000, 5_000, seed_tree(1, "va"))
    m = DBitNet(64)
    h = fit(m, X, Y, Xv, Yv, epochs=2, batch=1000)
    assert max(h["val_acc"]) > 0.9
