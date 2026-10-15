#!/usr/bin/env bash
# QuantumFlow: build, test, benchmark, train and plot in one go.
#   ./run_all.sh            everything
#   ./run_all.sh build      only configure + compile
set -euo pipefail
cd "$(dirname "$0")"
BUILD=${BUILD:-build}
PY=${PY:-python3}

step() { printf '\n==== %s ====\n' "$1"; }

step "configure + build (CUDA is used automatically if nvcc is found)"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v $PY)"
cmake --build "$BUILD" -j
[[ "${1:-}" == "build" ]] && exit 0

step "C++ tests"
ctest --test-dir "$BUILD" --output-on-failure
step "Python validation tests (vs QuantLib)"
$PY -m pytest tests/test_validation.py -q

mkdir -p results
step "order book latency benchmark"
"$BUILD/ob_latency_bench" results
step "CPU Monte Carlo benchmark"
"$BUILD/mc_cpu_bench" results
if [[ -x "$BUILD/cuda_bench" ]]; then
  step "CUDA benchmark"
  "$BUILD/cuda_bench" results
fi

step "generate training data"
$PY python/scripts/generate_training_data.py --n_samples "${N_SAMPLES:-500000}"
step "train surrogate"
$PY python/surrogate/train.py
step "end-to-end pipeline demo"
$PY python/scripts/pipeline_demo.py
step "plots + dashboard"
$PY python/dashboard/make_plots.py
