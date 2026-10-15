// QuantumFlow - device-side variance reduction and reduction helpers.
//
// Antithetic variates: every thread simulates a path with shocks Z and its
// "mirror" path with shocks -Z, from the same random numbers, and averages the
// two payoffs. Doing both in one thread (rather than thread i and thread
// i + N/2) means no data exchange between threads and no branch on thread
// index, so warps never diverge. The averaged pair is one sample for the
// variance estimate, which keeps the standard error honest.
//
// Control variate: alongside the Heston path we evolve a geometric Brownian
// motion X driven by the same Z1 shocks. E[X payoff] is known exactly (a
// Black-Scholes price), so Y - beta (X - E[X]) has the same mean as Y but much
// lower variance. The device only accumulates the five sums needed to compute
// beta = Cov(X,Y)/Var(X) on the host afterwards.
#pragma once

namespace qf::cuda {

constexpr int kNumSums = 5;  // sum y, sum y^2, sum x, sum x^2, sum x*y
constexpr unsigned kFullMask = 0xffffffffu;

// Innermost reduction: 32 lanes of a warp add their values in 5 shuffle steps
// with no shared memory and no __syncthreads.
__device__ __forceinline__ double warp_reduce_sum(double v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(kFullMask, v, offset);
    return v;
}

// Block reduction: warp shuffle -> one partial per warp in shared memory ->
// the first warp reduces those -> thread 0 does ONE atomicAdd per sum.
// Global atomics drop from blockDim.x per block to 1 per block.
__device__ __forceinline__ void block_reduce_and_commit(double (&vals)[kNumSums], double* global_sums) {
    __shared__ double partial[32][kNumSums];  // up to 1024 threads = 32 warps
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;

#pragma unroll
    for (int k = 0; k < kNumSums; ++k) {
        const double s = warp_reduce_sum(vals[k]);
        if (lane == 0) partial[warp][k] = s;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < kNumSums; ++k) {
            double s = lane < n_warps ? partial[lane][k] : 0.0;
            s = warp_reduce_sum(s);
            if (lane == 0) atomicAdd(&global_sums[k], s);  // double atomicAdd needs sm_60+
        }
    }
}

}  // namespace qf::cuda
