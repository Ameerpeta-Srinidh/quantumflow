"""Train the neural Greeks surrogate and measure it.

    python python/surrogate/train.py --data data/greeks_dataset.parquet

What it does
  1. splits the dataset 70% train / 10% validation / 20% test (100K test rows
     for the default 500K dataset), using a fixed seed
  2. trains the multitask MLP with a weighted MSE (delta and gamma weighted x2)
  3. keeps the weights from the epoch with the best validation loss
  4. reports test-set accuracy per Greek, MC-dropout uncertainty calibration,
     an out-of-distribution check, and inference speed vs Monte Carlo bumping
Outputs
  models/greeks_surrogate.pt         weights + normalisation statistics
  results/surrogate_metrics.json     every number quoted in the README
  results/surrogate_test_preds.npz   predictions used by the plotting script
  results/surrogate_training.csv     loss curve
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qf_paths import FEATURES, GREEKS, MODELS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from surrogate.model import GreeksSurrogate, save  # noqa: E402
from surrogate.uncertainty import calibration_scale, coverage, mc_dropout_predict  # noqa: E402

LOSS_WEIGHTS = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0])  # delta, gamma weighted higher (most used for hedging)


def split(n: int, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.7 * n), int(0.8 * n)
    return idx[:a], idx[a:b], idx[b:]


def evaluate_loss(model, x, y, batch=16384):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch):
            pred = model.forward_normalised(x[i:i + batch])
            tgt = (y[i:i + batch] - model.y_mean) / model.y_std
            total += (((pred - tgt) ** 2) * LOSS_WEIGHTS).mean(1).sum().item()
    return total / len(x)


def train(model, x_tr, y_tr, x_va, y_va, epochs, batch, lr, log_path):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * ((len(x_tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    y_tr_n = (y_tr - model.y_mean) / model.y_std
    best, best_state, rows = float("inf"), None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(x_tr))
        run = 0.0
        for i in range(0, len(x_tr), batch):
            j = perm[i:i + batch]
            pred = model.forward_normalised(x_tr[j])
            loss = (((pred - y_tr_n[j]) ** 2) * LOSS_WEIGHTS).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(j)
        tr_loss = run / len(x_tr)
        va_loss = evaluate_loss(model, x_va, y_va)
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rows.append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss, seconds=time.perf_counter() - t0))
        print(f"epoch {ep:3d}  train {tr_loss:.3e}  val {va_loss:.3e}  ({rows[-1]['seconds']:.1f}s)", flush=True)
    pd.DataFrame(rows).to_csv(log_path, index=False)
    model.load_state_dict(best_state)
    model.eval()
    return best


def timing(fn, n_repeat):
    fn()  # warm-up
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/greeks_dataset.parquet"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--mc_passes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_parquet(args.data)
    X = torch.tensor(df[FEATURES].to_numpy(), dtype=torch.float32)
    Y = torch.tensor(df[GREEKS].to_numpy(), dtype=torch.float32)
    tr, va, te = split(len(df), args.seed)
    print(f"dataset {len(df):,} rows: train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    model = GreeksSurrogate(width=args.width, n_blocks=args.blocks, dropout=args.dropout)
    model.fit_normalisation(X[tr], Y[tr])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    t0 = time.perf_counter()
    train(model, X[tr], Y[tr], X[va], Y[va], args.epochs, args.batch, args.lr, RESULTS_DIR / "surrogate_training.csv")
    train_seconds = time.perf_counter() - t0
    save(model, MODELS_DIR / "greeks_surrogate.pt")

    # ---------------- accuracy on the held-out test set ----------------
    with torch.no_grad():
        pred = torch.cat([model(X[te][i:i + 16384]) for i in range(0, len(te), 16384)]).numpy()
    truth = Y[te].numpy()
    err = np.abs(pred - truth)
    metrics = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_params": n_params,
               "train_seconds": train_seconds, "epochs": args.epochs, "per_greek": {}}
    for j, g in enumerate(GREEKS):
        ss_res = ((pred[:, j] - truth[:, j]) ** 2).sum()
        ss_tot = ((truth[:, j] - truth[:, j].mean()) ** 2).sum()
        metrics["per_greek"][g] = {"mae": float(err[:, j].mean()), "p99_abs_err": float(np.quantile(err[:, j], 0.99)),
                                   "r2": float(1 - ss_res / ss_tot), "truth_std": float(truth[:, j].std())}
        print(f"{g:12s} MAE {err[:, j].mean():.5f}   p99 |err| {np.quantile(err[:, j], 0.99):.5f}   "
              f"R^2 {metrics['per_greek'][g]['r2']:.6f}")

    # ---------------- MC dropout uncertainty + calibration ----------------
    n_uq = min(20000, len(te))
    va_sub = va[:10000]
    mu_va, sd_va = mc_dropout_predict(model, X[va_sub], args.mc_passes)
    scale = calibration_scale(np.abs(mu_va.numpy() - Y[va_sub].numpy()), sd_va.numpy())
    mu_te, sd_te = mc_dropout_predict(model, X[te[:n_uq]], args.mc_passes)
    err_te = np.abs(mu_te.numpy() - truth[:n_uq])
    cov_raw = coverage(err_te, sd_te.numpy())
    cov_cal = coverage(err_te, sd_te.numpy(), scale)
    metrics["uncertainty"] = {"mc_passes": args.mc_passes, "n_test": n_uq,
                              "coverage95_raw": dict(zip(GREEKS, map(float, cov_raw))),
                              "calibration_scale": dict(zip(GREEKS, map(float, scale))),
                              "coverage95_calibrated": dict(zip(GREEKS, map(float, cov_cal)))}
    print("95% interval coverage, raw MC dropout:", np.round(cov_raw, 3))
    print("95% interval coverage, calibrated    :", np.round(cov_cal, 3), " scale", np.round(scale, 2))

    # ---------------- out-of-distribution check ----------------
    # Deliberately sample far outside the training ranges and see whether the
    # uncertainty rises together with the error.
    import qf_core
    rng = np.random.default_rng(1)
    n_ood = 4000
    S0 = rng.uniform(80, 120, n_ood)
    logm = rng.uniform(-0.6, 0.6, n_ood)
    xi = rng.uniform(0.1, 1.5, n_ood)
    Xo = np.column_stack([S0, S0 * np.exp(logm), rng.uniform(0.1, 2.0, n_ood), rng.uniform(0, 0.06, n_ood),
                          rng.uniform(0.01, 0.16, n_ood), rng.uniform(0.5, 5, n_ood), rng.uniform(0.01, 0.16, n_ood),
                          xi, rng.uniform(-0.9, 0, n_ood)])
    Yo = qf_core.heston_greeks_batch(Xo)[:, 1:]
    mu_o, sd_o = mc_dropout_predict(model, torch.tensor(Xo, dtype=torch.float32), args.mc_passes)
    inside = (np.abs(logm) <= 0.3) & (xi <= 0.9)
    e_o = np.abs(mu_o.numpy() - Yo)
    metrics["ood"] = {"delta_mae_inside": float(e_o[inside, 0].mean()), "delta_mae_outside": float(e_o[~inside, 0].mean()),
                      "delta_std_inside": float(sd_o.numpy()[inside, 0].mean()),
                      "delta_std_outside": float(sd_o.numpy()[~inside, 0].mean())}
    print("OOD:", metrics["ood"])

    # ---------------- speed: surrogate vs Monte Carlo bump ----------------
    torch.set_grad_enabled(False)
    x1 = X[te[:1]]
    xb = X[te[:10000]]
    t_single = timing(lambda: model(x1), 2000)
    t_batch = timing(lambda: model(xb), 20) / len(xb)
    t_uq_single = timing(lambda: mc_dropout_predict(model, x1, args.mc_passes), 200)
    model.eval()
    p = qf_core.HestonParams()
    mc_runs = [qf_core.mc_greeks(p, n_paths=1 << 16, n_steps=128, backend="auto")["elapsed_ms"] / 1e3 for _ in range(3)]
    t_mc = float(np.median(mc_runs))
    metrics["speed"] = {
        "surrogate_single_call_us": t_single * 1e6,
        "surrogate_batched_per_option_us": t_batch * 1e6,
        "surrogate_mc_dropout_single_us": t_uq_single * 1e6,
        "mc_bump_greeks_ms": t_mc * 1e3,
        "mc_bump_config": "9 runs x 65,536 paths x 128 steps, backend=" + ("cuda" if qf_core.cuda_available() else "cpu"),
        "speedup_single_call": t_mc / t_single,
        "speedup_batched": t_mc / t_batch,
        "torch_threads": torch.get_num_threads(),
    }
    print(json.dumps(metrics["speed"], indent=2))

    with open(RESULTS_DIR / "surrogate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(RESULTS_DIR / "surrogate_test_preds.npz", x=X[te[:n_uq]].numpy(), truth=truth[:n_uq],
                        pred=pred[:n_uq], mc_mean=mu_te.numpy(), mc_std=sd_te.numpy(), scale=scale,
                        ood_x=Xo, ood_truth=Yo, ood_mean=mu_o.numpy(), ood_std=sd_o.numpy())
    print(f"saved model to {MODELS_DIR / 'greeks_surrogate.pt'} and metrics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
