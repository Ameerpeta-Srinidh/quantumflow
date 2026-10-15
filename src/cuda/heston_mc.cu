// QuantumFlow - CUDA Heston stochastic-volatility Monte Carlo kernel.
//
// Design (see the teaching guide for the "why" of each point):
//  * one thread = one sample (a path, or an antithetic pair of paths)
//  * random numbers generated on the GPU with cuRAND Philox, in registers
//  * the path state (ln S, v, ln X) lives in registers; nothing is written to
//    global memory per step unless we ask for paths (layout experiment)
//  * per-thread sums -> warp shuffle -> shared memory -> 1 atomicAdd per block
//  * the Heston step itself is the same function the CPU engine uses
//    (src/common/heston_path.hpp), so CPU and GPU agree by construction.
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "../common/heston_path.hpp"
#include "curand_wrapper.cuh"
#include "heston_mc.cuh"
#include "variance_reduction.cuh"

namespace qf::cuda {

#define QF_CUDA_CHECK(call)                                                                  \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess)                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__));        \
    } while (0)

template <PathLayout Layout, class R>
__global__ void heston_kernel(PathConstantsT<R> c, unsigned long long seed, unsigned long long n_samples,
                              bool antithetic, double* __restrict__ global_sums, float* __restrict__ paths) {
    double acc[kNumSums] = {0, 0, 0, 0, 0};

    // Grid-stride loop: works for any n_samples with any grid size.
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n_samples;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        RngState rng;
        rng_init(&rng, seed, i);

        PathStateT<R> a, b;
        path_init(a, c);
        path_init(b, c);
        // The mirror path uses sign = -1; computed arithmetically, no branch.
        const R sign = antithetic ? R(-1) : R(1);

        for (int t = 0; t < c.n_steps; ++t) {
            const auto z = Normal2<R>::draw(&rng);
            path_step(a, c, z.x, z.y);
            path_step(b, c, sign * z.x, sign * z.y);
            if constexpr (Layout == PathLayout::kSoA) {
                // Step t of all paths is contiguous: consecutive threads write
                // consecutive addresses -> one coalesced transaction per warp.
                paths[(unsigned long long)t * n_samples + i] = (float)exp(a.lnS);
            } else if constexpr (Layout == PathLayout::kAoS) {
                // Each path is contiguous: consecutive threads write addresses
                // n_steps floats apart -> 32 separate transactions per warp.
                paths[i * (unsigned long long)c.n_steps + t] = (float)exp(a.lnS);
            }
        }
        // When antithetic is off, b is an exact copy of a, so the average is a.
        const double y = 0.5 * ((double)disc_call_payoff(a.lnS, c) + (double)disc_call_payoff(b.lnS, c));
        const double x = 0.5 * ((double)disc_call_payoff(a.lnX, c) + (double)disc_call_payoff(b.lnX, c));
        acc[0] += y;
        acc[1] += y * y;
        acc[2] += x;
        acc[3] += x * x;
        acc[4] += x * y;
    }
    block_reduce_and_commit(acc, global_sums);
}

template <class R>
void launch(PathLayout layout, int grid, int block, const PathConstantsT<R>& c, const McConfig& cfg,
            unsigned long long n, double* d_sums, float* d_paths) {
    switch (layout) {
        case PathLayout::kNone:
            heston_kernel<PathLayout::kNone, R><<<grid, block>>>(c, cfg.seed, n, cfg.antithetic, d_sums, d_paths);
            break;
        case PathLayout::kSoA:
            heston_kernel<PathLayout::kSoA, R><<<grid, block>>>(c, cfg.seed, n, cfg.antithetic, d_sums, d_paths);
            break;
        case PathLayout::kAoS:
            heston_kernel<PathLayout::kAoS, R><<<grid, block>>>(c, cfg.seed, n, cfg.antithetic, d_sums, d_paths);
            break;
    }
}

bool device_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

const char* device_name() {
    static char name[256] = "none";
    cudaDeviceProp prop{};
    if (device_available() && cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        std::snprintf(name, sizeof(name), "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    return name;
}

McResult heston_mc_cuda(const HestonParams& p, const McConfig& cfg, int block_size, PathLayout layout,
                        KernelStats* stats) {
    cudaEvent_t start_all, start_k, stop_k;
    QF_CUDA_CHECK(cudaEventCreate(&start_all));
    QF_CUDA_CHECK(cudaEventCreate(&start_k));
    QF_CUDA_CHECK(cudaEventCreate(&stop_k));
    QF_CUDA_CHECK(cudaEventRecord(start_all));

    const unsigned long long n = cfg.antithetic ? cfg.n_paths / 2 : cfg.n_paths;

    double* d_sums = nullptr;
    float* d_paths = nullptr;
    QF_CUDA_CHECK(cudaMalloc(&d_sums, kNumSums * sizeof(double)));
    QF_CUDA_CHECK(cudaMemset(d_sums, 0, kNumSums * sizeof(double)));
    if (layout != PathLayout::kNone)
        QF_CUDA_CHECK(cudaMalloc(&d_paths, n * (size_t)cfg.n_steps * sizeof(float)));

    // Enough blocks to fill the GPU several times over, capped so that each
    // thread loops over a few samples (amortises the reduction).
    int device = 0, sm_count = 0;
    QF_CUDA_CHECK(cudaGetDevice(&device));
    QF_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    const unsigned long long needed = (n + block_size - 1) / block_size;
    const unsigned long long cap = (unsigned long long)sm_count * 32;
    const int grid = (int)(needed < cap ? needed : cap);

    QF_CUDA_CHECK(cudaEventRecord(start_k));
    if (cfg.gpu_single_precision)
        launch<float>(layout, grid, block_size, make_path_constants<float>(p, cfg.n_steps), cfg, n, d_sums, d_paths);
    else
        launch<double>(layout, grid, block_size, make_path_constants<double>(p, cfg.n_steps), cfg, n, d_sums, d_paths);
    QF_CUDA_CHECK(cudaGetLastError());
    QF_CUDA_CHECK(cudaEventRecord(stop_k));

    double h_sums[kNumSums];
    QF_CUDA_CHECK(cudaMemcpy(h_sums, d_sums, sizeof(h_sums), cudaMemcpyDeviceToHost));
    QF_CUDA_CHECK(cudaEventSynchronize(stop_k));
    float k_ms = 0, all_ms = 0;
    QF_CUDA_CHECK(cudaEventElapsedTime(&k_ms, start_k, stop_k));
    QF_CUDA_CHECK(cudaEventElapsedTime(&all_ms, start_all, stop_k));

    cudaFree(d_sums);
    if (d_paths) cudaFree(d_paths);
    cudaEventDestroy(start_all);
    cudaEventDestroy(start_k);
    cudaEventDestroy(stop_k);

    const McSums sums{h_sums[0], h_sums[1], h_sums[2], h_sums[3], h_sums[4]};
    const double EX = bs_call(p.S0, p.K, p.T, p.r, control_variate_sigma(p));
    McResult res = finalize(sums, n, EX, cfg.control_variate);
    res.elapsed_ms = all_ms;
    if (stats) {
        stats->kernel_ms = k_ms;
        stats->total_ms = all_ms;
        stats->block_size = block_size;
        stats->grid_size = grid;
        stats->paths_per_sec = cfg.n_paths / (k_ms * 1e-3);
    }
    return res;
}

}  // namespace qf::cuda
