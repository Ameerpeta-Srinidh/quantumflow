"""QuantumFlow neural Greeks surrogate: standalone Kaggle / Colab training script.

One file, no compiled code needed. Requirements: torch, numpy, pandas, pyarrow
(all preinstalled on Kaggle).

Kaggle: create a notebook, Settings > Accelerator > GPU T4 or P100,
"Add Input" the dataset holding greeks_dataset.parquet, then in a cell:
    !python train_greeks_kaggle.py --epochs 60
(or paste this file into a cell and run it). Outputs land in /kaggle/working:
greeks_surrogate.pt is a drop-in replacement for quantumflow/models/greeks_surrogate.pt.
"""
import numpy as np

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


# =====================================================================
# Training (GPU if available). Mirrors python/surrogate/train.py minus the
# parts that need the compiled C++ module (OOD labels, Monte Carlo timing).
# =====================================================================
import argparse, glob, json, os, time
import pandas as pd

FEATURES = ["S0", "K", "T", "r", "v0", "kappa", "theta", "xi", "rho"]
GREEKS = ["delta", "gamma", "theta_greek", "vega", "rho_greek"]


def find_dataset(path):
    if path and os.path.exists(path):
        return path
    hits = glob.glob("/kaggle/input/**/greeks_dataset.parquet", recursive=True) + glob.glob("greeks_dataset.parquet")
    if not hits:
        raise SystemExit("greeks_dataset.parquet not found: add the dataset to the notebook or pass --data")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--mc_passes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    args, _ = ap.parse_known_args()  # tolerate Jupyter's own argv

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    path = find_dataset(args.data)
    df = pd.read_parquet(path)
    X = torch.tensor(df[FEATURES].to_numpy(), dtype=torch.float32)
    Y = torch.tensor(df[GREEKS].to_numpy(), dtype=torch.float32)
    idx = np.random.default_rng(args.seed).permutation(len(df))
    a, b = int(0.7 * len(df)), int(0.8 * len(df))
    tr, va, te = idx[:a], idx[a:b], idx[b:]
    print(f"{path}: {len(df):,} rows  train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    model = GreeksSurrogate(width=args.width, n_blocks=args.blocks, dropout=args.dropout)
    model.fit_normalisation(X[tr], Y[tr])
    model.to(dev)
    Xtr, Ytr, Xva, Yva = X[tr].to(dev), Y[tr].to(dev), X[va].to(dev), Y[va].to(dev)
    w = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0], device=dev)  # delta, gamma weighted x2
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    Ytr_n = (Ytr - model.y_mean) / model.y_std
    best, best_state, log = float("inf"), None, []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        run = 0.0
        for i in range(0, len(tr), args.batch):
            j = perm[i:i + args.batch]
            loss = (((model.forward_normalised(Xtr[j]) - Ytr_n[j]) ** 2) * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            vl = sum(((((model.forward_normalised(Xva[i:i + 16384]) - (Yva[i:i + 16384] - model.y_mean) / model.y_std) ** 2) * w)
                      .mean(1).sum().item()) for i in range(0, len(va), 16384)) / len(va)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log.append(dict(epoch=ep, train_loss=run / len(tr), val_loss=vl))
        print(f"epoch {ep:3d}  train {run / len(tr):.3e}  val {vl:.3e}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    train_s = time.perf_counter() - t0

    Xte = X[te].to(dev)
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i + 16384]) for i in range(0, len(te), 16384)]).cpu().numpy()
    truth = Y[te].numpy()
    err = np.abs(pred - truth)
    metrics = {"device": str(dev), "train_seconds": train_s, "epochs": args.epochs, "per_greek": {}}
    for k, g in enumerate(GREEKS):
        metrics["per_greek"][g] = {"mae": float(err[:, k].mean()), "p99_abs_err": float(np.quantile(err[:, k], 0.99))}
        print(f"{g:12s} MAE {err[:, k].mean():.5f}   p99 |err| {np.quantile(err[:, k], 0.99):.5f}")

    vs = torch.tensor(va[:10000])
    mu_v, sd_v = mc_dropout_predict(model, X[vs].to(dev), args.mc_passes)
    scale = calibration_scale(np.abs(mu_v.cpu().numpy() - Y[vs].numpy()), sd_v.cpu().numpy())
    n_uq = min(20000, len(te))
    mu_t, sd_t = mc_dropout_predict(model, Xte[:n_uq], args.mc_passes)
    e_t = np.abs(mu_t.cpu().numpy() - truth[:n_uq])
    cov = coverage(e_t, sd_t.cpu().numpy(), scale)
    metrics["uncertainty"] = {"calibration_scale": dict(zip(GREEKS, map(float, scale))),
                              "coverage95_calibrated": dict(zip(GREEKS, map(float, cov)))}
    print("calibrated 95% coverage:", np.round(cov, 3))

    xb = Xte[:10000]
    with torch.no_grad():
        model(xb)
        if dev.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(20): model(xb)
        if dev.type == "cuda": torch.cuda.synchronize()
    metrics["batched_per_option_us"] = (time.perf_counter() - t1) / 20 / len(xb) * 1e6
    print(f"batched inference: {metrics['batched_per_option_us']:.3f} us per option")

    model.cpu()
    torch.save({"config": model.config, "state_dict": model.state_dict()}, os.path.join(args.out, "greeks_surrogate.pt"))
    json.dump(metrics, open(os.path.join(args.out, "surrogate_metrics_kaggle.json"), "w"), indent=2)
    pd.DataFrame(log).to_csv(os.path.join(args.out, "surrogate_training.csv"), index=False)
    print("saved greeks_surrogate.pt, surrogate_metrics_kaggle.json, surrogate_training.csv to", args.out)


if __name__ == "__main__":
    main()
