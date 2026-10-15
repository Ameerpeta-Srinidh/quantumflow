# Train the Greeks surrogate on Kaggle

1. Kaggle > Datasets > New Dataset: upload `greeks_dataset.parquet` (500,000 options, 64 MB).
2. Kaggle > Code > New Notebook. Settings: Accelerator = GPU T4 x2 or P100. Add Input: your dataset.
3. Upload `train_greeks_kaggle.py` to the notebook (or paste it into a cell) and run:
   `!python train_greeks_kaggle.py --epochs 60`
4. Download from /kaggle/working: `greeks_surrogate.pt`, `surrogate_metrics_kaggle.json`,
   `surrogate_training.csv`. Put the .pt file in `quantumflow/models/` to use it in the pipeline.

The dataset's columns: S0, K, T, r, v0, kappa, theta, xi, rho (inputs), price, delta, gamma,
theta_greek, vega, rho_greek (exact analytic Heston labels). To regenerate it yourself:
`python python/scripts/generate_training_data.py --n_samples 500000` (needs the built qf_core).
