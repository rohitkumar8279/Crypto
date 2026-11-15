"""Checks the DBitNet wrapper's plumbing with a stand-in for TensorFlow.

This does NOT test TensorFlow/Keras behaviour (that needs a GPU run, Phase 2); it checks
data-stream separation, resumability, and that the test stream is only touched on request.
"""

import json
import sys
import types

import numpy as np
import pytest


class _FakeModel:
    def __init__(self, log):
        self.log = log

    def compile(self, **kw):
        pass

    def fit(self, X, Y, epochs, batch_size, validation_data, callbacks, verbose):
        self.log.append(("fit", X.shape, float(X[:50].mean()), float(validation_data[0][:50].mean())))
        return types.SimpleNamespace(history={"val_acc": [0.6] * epochs, "acc": [0.6] * epochs})

    def load_weights(self, path):
        pass

    def predict(self, X, batch_size, verbose):
        self.log.append(("predict", X.shape, float(X[:50].mean())))
        return np.full((len(X), 1), 0.7)


@pytest.fixture
def fake_tf(monkeypatch):
    log = []
    tf = types.ModuleType("tensorflow")
    tf.keras = types.SimpleNamespace(
        utils=types.SimpleNamespace(set_random_seed=lambda s: None),
        optimizers=types.SimpleNamespace(Adam=lambda **kw: None),
        callbacks=types.SimpleNamespace(ModelCheckpoint=lambda *a, **kw: None),
        backend=types.SimpleNamespace(clear_session=lambda: None))
    tf.config = types.SimpleNamespace(list_physical_devices=lambda *a: [],
                                      experimental=types.SimpleNamespace(enable_op_determinism=lambda: None))
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    import cgsearch.neural as neural
    monkeypatch.setattr(neural, "load_dbitnet", lambda: types.SimpleNamespace(make_model=lambda n: _FakeModel(log)))
    return neural, log


def test_selection_only_does_not_touch_test(fake_tf, tmp_path):
    neural, log = fake_tf
    r = neural.train_eval_dbitnet("simon3264", 0x400, 2, 3, seed=1, out_dir=tmp_path, epochs=2,
                                  n_train=1000, n_select=500)
    assert [x["round"] for x in r["rounds"]] == [2, 3]
    assert "test" not in r and not any(e[0] == "predict" for e in log)
    assert r["total_examples_seen"] == 2 * 2 * 1000


def test_final_test_requires_seed_and_is_resumable(fake_tf, tmp_path):
    neural, log = fake_tf
    with pytest.raises(ValueError):
        neural.train_eval_dbitnet("simon3264", 0x400, 2, 2, seed=1, out_dir=tmp_path, epochs=1,
                                  n_train=1000, n_select=500, evaluate_test=True)
    r = neural.train_eval_dbitnet("simon3264", 0x400, 2, 2, seed=1, out_dir=tmp_path / "b", epochs=1,
                                  n_train=1000, n_select=500, n_test=400, n_test_sets=3,
                                  test_seed=99, evaluate_test=True)
    assert len(r["test"]["sets"]) == 3 and r["test"]["test_seed"] == 99
    n_fit = sum(e[0] == "fit" for e in log)
    again = neural.train_eval_dbitnet("simon3264", 0x400, 2, 2, seed=1, out_dir=tmp_path / "b", epochs=1,
                                      n_train=1000, n_select=500, n_test=400, n_test_sets=3,
                                      test_seed=99, evaluate_test=True)
    assert sum(e[0] == "fit" for e in log) == n_fit  # resumed from JSON, no retraining
    assert again == json.loads(json.dumps(r))


def test_train_select_test_streams_differ(fake_tf, tmp_path):
    neural, log = fake_tf
    neural.train_eval_dbitnet("simon3264", 0x400, 2, 2, seed=1, out_dir=tmp_path, epochs=1, n_train=600,
                              n_select=600, n_test=600, n_test_sets=1, test_seed=1, evaluate_test=True)
    fit = next(e for e in log if e[0] == "fit")
    pred = next(e for e in log if e[0] == "predict")
    assert len({fit[2], fit[3], pred[2]}) == 3  # train, selection and test data are different draws
