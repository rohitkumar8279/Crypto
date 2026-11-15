"""Faithful PyTorch port of AutoND's DBitNet (dbitnet.make_model) + training recipe.

Why a port: TensorFlow cannot load on the development machine (Windows Application
Control blocks its DLLs), but PyTorch can. Fidelity is checked by loading AutoND's
released Keras weights into this port and reproducing their documented accuracy
(experiments/validate_port.py; tests/test_torch_dbitnet.py).

Layer-by-layer correspondence with dbitnet.py:
  input (x-0.5)/0.5 ; for each dilation d in get_dilation_rates(input_size):
     Conv1D(f, k=2, valid, dilation=d, relu) -> BN -> skip
     Conv1D(f, k=2, causal, dilation=1, relu) -> Add(skip) -> BN ; f += 16
  Flatten (Keras channels-last order) -> [Dense(256) BN ReLU] x2 -> Dense(64) BN ReLU -> Dense(1) sigmoid
  L2(1e-5) on the four Dense kernels only (as in dbitnet.py); BN epsilon 1e-3, momentum 0.99.
Training (train_nets.py): Adam(amsgrad=True, lr=1e-3, eps=1e-7), MSE loss, batch 5000,
keep the weights with the best validation loss.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

BN_EPS, BN_MOMENTUM_TORCH = 1e-3, 0.01   # Keras momentum 0.99 == torch momentum 0.01
L2 = 1e-5


def get_dilation_rates(input_size: int) -> list[int]:
    drs = []
    while input_size >= 8:
        drs.append(int(input_size / 2 - 1))
        input_size //= 2
    return drs


class _Block(nn.Module):
    def __init__(self, cin, f, d):
        super().__init__()
        self.c1 = nn.Conv1d(cin, f, 2, dilation=d)
        self.b1 = nn.BatchNorm1d(f, eps=BN_EPS, momentum=BN_MOMENTUM_TORCH)
        self.c2 = nn.Conv1d(f, f, 2, dilation=1)
        self.b2 = nn.BatchNorm1d(f, eps=BN_EPS, momentum=BN_MOMENTUM_TORCH)

    def forward(self, x):
        x = self.b1(torch.relu(self.c1(x)))
        skip = x
        x = torch.relu(self.c2(nn.functional.pad(x, (1, 0))))   # causal padding
        return self.b2(x + skip)


class DBitNet(nn.Module):
    def __init__(self, input_size: int = 64, n_filters: int = 32, n_add_filters: int = 16):
        super().__init__()
        blocks, cin, f, length = [], 1, n_filters, input_size
        for d in get_dilation_rates(input_size):
            blocks.append(_Block(cin, f, d))
            length -= d
            cin, f = f, f + n_add_filters
        self.blocks = nn.ModuleList(blocks)
        self.flat_dim = length * cin
        def dense(i, o):
            return nn.Linear(i, o)
        self.d0, self.d1, self.d2, self.out = dense(self.flat_dim, 256), dense(256, 256), dense(256, 64), dense(64, 1)
        self.n0 = nn.BatchNorm1d(256, eps=BN_EPS, momentum=BN_MOMENTUM_TORCH)
        self.n1 = nn.BatchNorm1d(256, eps=BN_EPS, momentum=BN_MOMENTUM_TORCH)
        self.n2 = nn.BatchNorm1d(64, eps=BN_EPS, momentum=BN_MOMENTUM_TORCH)
        self.input_size = input_size

    def forward(self, x):                      # x: (B, input_size) in {0,1}
        x = ((x.float() - 0.5) / 0.5).unsqueeze(1)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1)   # Keras flattens (length, channels)
        x = torch.relu(self.n0(self.d0(x)))
        x = torch.relu(self.n1(self.d1(x)))
        x = torch.relu(self.n2(self.d2(x)))
        return torch.sigmoid(self.out(x)).squeeze(1)

    def l2_penalty(self):
        return L2 * sum((m.weight ** 2).sum() for m in (self.d0, self.d1, self.d2, self.out))


def load_keras_h5(model: DBitNet, path: str) -> DBitNet:
    """Load weights saved by Keras ModelCheckpoint for dbitnet.make_model."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        def w(layer, name):
            return torch.tensor(np.array(g[layer][layer][f"{name}:0"]))
        def order(prefix):
            names = [k for k in g.keys() if k == prefix or k.startswith(prefix + "_")]
            return sorted(names, key=lambda k: 0 if k == prefix else int(k[len(prefix) + 1:]))
        convs, bns, dens = order("conv1d"), order("batch_normalization"), order("dense")
        conv_mods = [m for b in model.blocks for m in (b.c1, b.c2)]
        bn_mods = [m for b in model.blocks for m in (b.b1, b.b2)] + [model.n0, model.n1, model.n2]
        with torch.no_grad():
            for name, m in zip(convs, conv_mods):
                m.weight.copy_(w(name, "kernel").permute(2, 1, 0))      # (k,in,out) -> (out,in,k)
                m.bias.copy_(w(name, "bias"))
            for name, m in zip(bns, bn_mods):
                m.weight.copy_(w(name, "gamma")); m.bias.copy_(w(name, "beta"))
                m.running_mean.copy_(w(name, "moving_mean")); m.running_var.copy_(w(name, "moving_variance"))
            for name, m in zip(dens, (model.d0, model.d1, model.d2, model.out)):
                m.weight.copy_(w(name, "kernel").T); m.bias.copy_(w(name, "bias"))
    return model


def _dev(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def default_device() -> torch.device:
    """CUDA if available (e.g. Colab GPU), else CPU. Move the model with model.to(device)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch: int = 20000) -> np.ndarray:
    model.eval()
    d = _dev(model)
    out = [model(torch.from_numpy(X[i:i + batch]).to(d)).cpu().numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)


def evaluate(model, X, Y) -> dict:
    Z = predict(model, X) > 0.5
    n1 = int((Y == 1).sum())
    acc = float((Z == Y).mean())
    return {"acc": acc, "tpr": float(Z[Y == 1].mean()), "tnr": float((~Z[Y == 0]).mean()),
            "advantage": 2 * acc - 1, "n": len(Y), "n_pos": n1}


@torch.no_grad()
def recalibrate_bn(model: nn.Module, X: np.ndarray, n: int = 100_000, batch: int = 5000) -> None:
    """Set BatchNorm running statistics to exact averages over n training samples.

    With AutoND's full schedule (2000 steps/epoch, Keras momentum 0.99) the running averages
    converge; at reduced data sizes (tens of steps/epoch) they do not, and eval-mode accuracy
    collapses to ~0.5. Recomputing the statistics (cumulative average) recovers the limit the
    full-scale recipe reaches. Documented as a reduced-scale adaptation.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    for i in range(0, min(n, len(X)), batch):
        model(torch.from_numpy(X[i:i + batch]).to(_dev(model)))
    for m, mo in zip(bns, saved):
        m.momentum = mo
    model.eval()


def fit(model: DBitNet, X, Y, Xv, Yv, epochs: int, batch: int = 5000, seed: int = 0, lr: float = 1e-3,
        verbose: bool = False) -> dict:
    """AutoND recipe; returns history and restores the best-val-loss weights."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7, amsgrad=True)
    d = _dev(model)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y.astype(np.float32))
    hist = {"loss": [], "val_loss": [], "val_acc": [], "seconds": []}
    best, best_state = math.inf, None
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            p = model(Xt[idx].to(d))
            loss = ((p - Yt[idx].to(d)) ** 2).mean() + model.l2_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        recalibrate_bn(model, X)
        pv = predict(model, Xv)
        vl = float(((pv - Yv) ** 2).mean())
        va = float(((pv > 0.5) == Yv).mean())
        hist["loss"].append(tot / len(Xt)); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        hist["seconds"].append(time.perf_counter() - t0)
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss {hist['loss'][-1]:.4f} val_loss {vl:.4f} val_acc {va:.4f} "
                  f"({hist['seconds'][-1]:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return hist
