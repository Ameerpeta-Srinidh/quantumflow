// QuantumFlow - host-side interface to the CUDA Heston Monte Carlo engine.
// This header is plain C++ so it can be included from non-CUDA files
// (the pybind11 bridge, the benchmarks).
#pragma once

#include "../common/heston_params.hpp"

namespace qf::cuda {

// How the kernel writes full price paths to global memory (for the
// coalescing experiment). kNone only keeps the terminal values in registers.
enum class PathLayout : int { kNone = 0, kSoA = 1, kAoS = 2 };

struct KernelStats {
    double kernel_ms = 0.0;      // GPU time of the kernel alone (cudaEvent)
    double total_ms = 0.0;       // including allocation and copies
    int block_size = 0;
    int grid_size = 0;
    double paths_per_sec = 0.0;
};

bool device_available();
const char* device_name();

// Price one European call. If stats is non-null it receives timing details.
McResult heston_mc_cuda(const HestonParams& p, const McConfig& cfg, int block_size = 256,
                        PathLayout layout = PathLayout::kNone, KernelStats* stats = nullptr);

}  // namespace qf::cuda
