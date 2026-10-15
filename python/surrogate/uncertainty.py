"""Monte Carlo Dropout uncertainty (Gal & Ghahramani, 2016).

Normally dropout is switched off at inference. If we leave it ON and run the
same input through the network many times, each pass uses a different random
sub-network, so the predictions differ. The spread of those predictions is an
estimate of *epistemic* uncertainty: how unsure the model is because of
limited training data / model capacity. It grows where the model has seen
little data (e.g. the edges of the training ranges).

Raw MC-dropout spreads are usually too narrow or too wide by a roughly
constant factor, so we *calibrate*: on a validation set, find the factor s
per Greek such that |error| <= 1.96 * s * std for 95% of samples. That
factor is then applied to new predictions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def enable_dropout_only(model: nn.Module) -> None:
    """Eval mode everywhere except Dropout layers (so LayerNorm etc. stay deterministic)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 50, batch: int = 8192):
    """Returns (mean, std), each of shape (N, 5), from n_passes stochastic forward passes."""
    enable_dropout_only(model)
    means, stds = [], []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        # Stack passes: (n_passes, B, 5). Repeating the batch lets one big matmul do all passes.
        rep = xb.repeat(n_passes, 1)
        out = model(rep).view(n_passes, xb.shape[0], -1)
        means.append(out.mean(0))
        stds.append(out.std(0))
    model.eval()
    return torch.cat(means), torch.cat(stds)


def calibration_scale(abs_err: np.ndarray, std: np.ndarray, level: float = 0.95) -> np.ndarray:
    """Per-Greek factor s so that P(|err| <= z * s * std) = level on this data."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2)
    ratio = abs_err / np.maximum(std, 1e-12)  # how many stds each error was
    return np.quantile(ratio, level, axis=0) / z


def coverage(abs_err: np.ndarray, std: np.ndarray, scale=1.0, level: float = 0.95) -> np.ndarray:
    """Fraction of samples whose true value lies inside the predicted interval."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2)
    return (abs_err <= z * np.asarray(scale) * std).mean(axis=0)
