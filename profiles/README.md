# GPU profiles (Nsight)

This folder is where the Nsight Systems / Nsight Compute output belongs. It is
empty in the first version of the repo because the project was built on a
machine with no NVIDIA GPU: the CUDA kernel was compiled for sm_61, sm_75 and
sm_86 (ptxas reported 64 registers per thread in float, 94 in double), but it
has not been run or profiled yet.

## How to fill it (free, on Google Colab)

Open `notebooks/colab_cuda.ipynb` in Colab with a T4 GPU runtime, or run the
same commands on any Linux machine with an NVIDIA GPU and CUDA 12:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda_bench results                        # CSVs for the dashboard

# Timeline: kernel launches, memcpy, gaps
nsys profile -o profiles/qf_timeline --force-overwrite true ./build/cuda_bench results

# Kernel deep dive: occupancy, memory throughput, roofline, warp stalls
ncu --set full --kernel-name regex:heston_kernel --launch-count 3 \
    -o profiles/qf_kernel --force-overwrite true ./build/cuda_bench results
```

Open the `.nsys-rep` / `.ncu-rep` files in the Nsight GUIs on your laptop
(the GUIs are free and run without a GPU) and save screenshots here as
`nsight_screenshot.png` and `roofline_chart.png`.

## What to record for the README

| Metric (Nsight Compute section) | Where to look |
|---|---|
| Achieved occupancy | Occupancy |
| Registers per thread / theoretical occupancy | Launch Statistics, Occupancy |
| Memory throughput (GB/s) | Memory Workload Analysis |
| Compute (SM) throughput | GPU Speed Of Light |
| L2 hit rate | Memory Workload Analysis |
| Warp execution efficiency / divergence | Source Counters, Warp State Statistics |
| Roofline position | GPU Speed Of Light > Roofline |

Experiments already wired into `cuda_bench` for before/after tables:
block size sweep (32 to 1024), float vs double, and SoA vs AoS path storage.
