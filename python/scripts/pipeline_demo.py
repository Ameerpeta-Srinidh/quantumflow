"""End-to-end QuantumFlow pipeline (all five components together).

    python python/scripts/pipeline_demo.py

  1. Order book (C++): plays a synthetic market-data feed and samples the
     live mid price every `--every` messages.
  2. Monte Carlo engine (CUDA if available, else CPU): prices an at-the-money
     call on that mid price.
  3. Neural surrogate (PyTorch): predicts all five Greeks with MC-dropout
     uncertainty. If the uncertainty is too high, we fall back to Monte Carlo
     bump Greeks, which is what a risk system would do in production.
  4. Everything is timed; results go to results/pipeline_demo.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qf_paths import GREEKS, MODELS_DIR, RESULTS_DIR  # noqa: E402
import qf_core  # noqa: E402
from surrogate.model import load  # noqa: E402
from surrogate.uncertainty import mc_dropout_predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", type=int, default=200_000)
    ap.add_argument("--every", type=int, default=20_000)
    ap.add_argument("--fallback_delta_std", type=float, default=0.01,
                    help="if calibrated delta std exceeds this, recompute Greeks with Monte Carlo")
    args = ap.parse_args()

    backend = "cuda" if qf_core.cuda_available() else "cpu"
    print(f"Monte Carlo backend: {backend} ({qf_core.cuda_device_name()})")
    model = load(MODELS_DIR / "greeks_surrogate.pt")
    import json
    scale = json.load(open(RESULTS_DIR / "surrogate_metrics.json"))["uncertainty"]["calibration_scale"]["delta"]

    # The simulated book quotes around tick 8192 with a 0.01 tick: mid ~ 81.9.
    # Rescale so the option sees a spot around 100, as in the training data.
    book = qf_core.OrderBook(1 << 14, 0.01)
    t0 = time.perf_counter()
    mids, ns = book.run_feed(args.messages, 7, args.every)
    print(f"order book: {args.messages:,} messages in {ns / 1e6:.1f} ms ({ns / args.messages:.0f} ns/msg avg)")
    spots = mids[mids > 0] * (100.0 / 81.92)

    rows = []
    for S in spots:
        p = qf_core.HestonParams(S0=float(S), K=100.0, T=0.5, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        mc = qf_core.mc_price(p, n_paths=1 << 16, n_steps=128, backend=backend)
        x = torch.tensor([[p.S0, p.K, p.T, p.r, p.v0, p.kappa, p.theta, p.xi, p.rho]], dtype=torch.float32)
        t1 = time.perf_counter()
        mean, std = mc_dropout_predict(model, x, 50)
        t_sur = (time.perf_counter() - t1) * 1e6
        g = mean.numpy()[0]
        delta_std = float(std[0, 0]) * scale
        source = "surrogate"
        if delta_std > args.fallback_delta_std:
            mg = qf_core.mc_greeks(p, n_paths=1 << 16, n_steps=128, backend=backend)
            g = np.array([mg[k] for k in ["delta", "gamma", "theta", "vega", "rho"]])
            source = "monte_carlo_fallback"
        exact = qf_core.heston_greeks(p)
        rows.append(dict(spot=S, mc_price=mc["price"], mc_std_error=mc["std_error"], mc_ms=mc["elapsed_ms"],
                         exact_price=exact["price"], surrogate_us=t_sur, delta_std=delta_std, source=source,
                         **{f"{k}_pred": v for k, v in zip(GREEKS, g)}, delta_exact=exact["delta"]))
        r = rows[-1]
        print(f"spot {S:7.3f}  MC price {r['mc_price']:.4f} +- {r['mc_std_error']:.4f} (exact {r['exact_price']:.4f}, "
              f"{r['mc_ms']:.0f} ms)  delta {r['delta_pred']:.4f} +- {delta_std:.4f} (exact {r['delta_exact']:.4f}, "
              f"{t_sur:.0f} us, {source})")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "pipeline_demo.csv", index=False)
    print(f"total {time.perf_counter() - t0:.2f}s; wrote {RESULTS_DIR / 'pipeline_demo.csv'}")


if __name__ == "__main__":
    main()
