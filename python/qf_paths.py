"""Locations used by every script, so they work from any working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "configs"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))  # so `import qf_core` finds the compiled module

FEATURES = ["S0", "K", "T", "r", "v0", "kappa", "theta", "xi", "rho"]
GREEKS = ["delta", "gamma", "theta_greek", "vega", "rho_greek"]
