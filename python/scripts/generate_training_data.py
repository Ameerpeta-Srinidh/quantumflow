"""Generate the Greeks training set.

    python python/scripts/generate_training_data.py \
        --n_samples 500000 \
        --heston_params configs/heston_params.yaml \
        --output data/greeks_dataset.parquet

Each row is one random Heston option (inputs drawn uniformly from the ranges in
the YAML file) plus its price and five Greeks.

Label engines (--engine):
  analytic  (default) Greeks from the semi-closed-form Heston price by finite
            differences. Exact to ~1e-8, fast on a CPU. The C++ code is
            validated against the Monte Carlo engine in tests/, so these are
            the same numbers MC converges to, minus the MC noise.
  mc        Greeks by bump-and-reprice Monte Carlo through the same engine the
            design document describes (CUDA if available, else CPU). Much
            slower and noisy; useful to show the surrogate also learns from
            noisy labels.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qf_paths import FEATURES, GREEKS, ROOT  # noqa: E402
import qf_core  # noqa: E402


def sample_parameters(n: int, ranges: dict, rng: np.random.Generator) -> np.ndarray:
    def U(key):
        lo, hi = ranges[key]
        return rng.uniform(lo, hi, n)

    S0 = U("S0")
    K = S0 * np.exp(U("log_moneyness"))
    cols = dict(S0=S0, K=K, T=U("T"), r=U("r"), v0=U("v0"), kappa=U("kappa"), theta=U("theta"), xi=U("xi"),
                rho=U("rho"))
    return np.column_stack([cols[f] for f in FEATURES])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_samples", type=int, default=500_000)
    ap.add_argument("--heston_params", default=str(ROOT / "configs/heston_params.yaml"))
    ap.add_argument("--output", default=str(ROOT / "data/greeks_dataset.parquet"))
    ap.add_argument("--engine", choices=["analytic", "mc"], default="analytic")
    ap.add_argument("--mc_paths", type=int, default=1 << 15)
    ap.add_argument("--mc_steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    ranges = yaml.safe_load(open(args.heston_params))
    rng = np.random.default_rng(args.seed)
    X = sample_parameters(args.n_samples, ranges, rng)

    t0 = time.perf_counter()
    chunk = 20_000
    parts = []
    for i in range(0, len(X), chunk):
        xb = np.ascontiguousarray(X[i:i + chunk])
        if args.engine == "analytic":
            parts.append(qf_core.heston_greeks_batch(xb))
        else:
            parts.append(qf_core.mc_greeks_batch(xb, n_paths=args.mc_paths, n_steps=args.mc_steps, seed=args.seed))
        done = min(i + chunk, len(X))
        rate = done / (time.perf_counter() - t0)
        print(f"  {done:>8}/{len(X)}  ({rate:,.0f} options/s)", flush=True)
    Y = np.vstack(parts)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(X, columns=FEATURES)
    df["price"] = Y[:, 0]
    for j, g in enumerate(GREEKS):
        df[g] = Y[:, j + 1]
    bad = ~np.isfinite(df.to_numpy()).all(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} rows with non-finite values")
        df = df[~bad]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df):,} rows to {args.output} in {elapsed:.1f}s using engine={args.engine}")
    print(df.describe().T[["mean", "std", "min", "max"]].to_string(float_format=lambda v: f"{v:10.4f}"))


if __name__ == "__main__":
    main()
