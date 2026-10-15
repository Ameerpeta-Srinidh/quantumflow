"""The neural Greeks surrogate.

Input : the 9 numbers that define a Heston European call
        (S0, K, T, r, v0, kappa, theta, xi, rho)
Output: the 5 Greeks (delta, gamma, theta, vega, rho) at once.

Design
------
* Feature engineering inside the model: option prices depend on K and S0 mostly
  through log(K / S0), and on time and variance mostly through their square
  roots, so we add those as extra inputs. Then every input is standardised
  with mean / std computed on the training set (stored as buffers so they are
  saved with the weights).
* Trunk: a linear "stem", then residual blocks. Each block is
  LayerNorm -> Linear -> SiLU -> Dropout -> Linear, added back onto its input.
  Residual connections let gradients flow straight through, which makes
  deeper nets train reliably.
* Heads: one small MLP per Greek on top of the shared trunk (multitask
  learning: the trunk learns features useful for all five).
* Dropout stays in the network so that, at inference time, we can keep it
  switched on and run many stochastic forward passes (MC Dropout) to get an
  uncertainty estimate. See uncertainty.py.
* Targets are standardised per Greek during training (they live on very
  different scales: gamma ~ 0.01, theta ~ -5) and un-standardised on output.
"""
from __future__ import annotations

import torch
from torch import nn

N_RAW_FEATURES = 9
N_GREEKS = 5


def engineer_features(x: torch.Tensor) -> torch.Tensor:
    """(N, 9) raw parameters -> (N, 12) model features."""
    S0, K, T, v0 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 4:5]
    log_m = torch.log(K / S0)
    sqrt_T = torch.sqrt(T)
    sqrt_v0 = torch.sqrt(v0)
    # standardised distance from the money, the quantity Black-Scholes d1 is built from
    d = log_m / (sqrt_v0 * sqrt_T)
    return torch.cat([x, log_m, sqrt_T, d], dim=1)


N_FEATURES = N_RAW_FEATURES + 3


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


class GreeksSurrogate(nn.Module):
    def __init__(self, width: int = 256, n_blocks: int = 4, head_width: int = 64, dropout: float = 0.05):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(N_FEATURES, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, head_width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(head_width, 1))
             for _ in range(N_GREEKS)]
        )
        # Normalisation statistics, filled in by fit_normalisation().
        self.register_buffer("x_mean", torch.zeros(N_FEATURES))
        self.register_buffer("x_std", torch.ones(N_FEATURES))
        self.register_buffer("y_mean", torch.zeros(N_GREEKS))
        self.register_buffer("y_std", torch.ones(N_GREEKS))
        self.config = dict(width=width, n_blocks=n_blocks, head_width=head_width, dropout=dropout)

    @torch.no_grad()
    def fit_normalisation(self, x_raw: torch.Tensor, y: torch.Tensor) -> None:
        f = engineer_features(x_raw)
        self.x_mean.copy_(f.mean(0))
        self.x_std.copy_(f.std(0).clamp_min(1e-8))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-12))

    def forward_normalised(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Predictions in standardised target units (what the loss sees)."""
        f = (engineer_features(x_raw) - self.x_mean) / self.x_std
        h = self.norm(self.blocks(self.stem(f)))
        return torch.cat([head(h) for head in self.heads], dim=1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Predictions in real units: (N, 5) = delta, gamma, theta, vega, rho."""
        return self.forward_normalised(x_raw) * self.y_std + self.y_mean


def save(model: GreeksSurrogate, path) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load(path, map_location="cpu") -> GreeksSurrogate:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = GreeksSurrogate(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
