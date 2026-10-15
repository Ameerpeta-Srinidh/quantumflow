// QuantumFlow - on-GPU random number setup (cuRAND device API).
//
// Every thread owns a Philox4_32_10 generator. Philox is counter based, so
// curand_init is cheap (no large skip-ahead tables like XORWOW needs) and
// sample i always gets subsequence i: results do not depend on how many
// threads or blocks we launch. The random numbers are generated in registers
// and consumed immediately, so nothing ever crosses the PCIe bus.
#pragma once

#include <curand_kernel.h>

namespace qf::cuda {

using RngState = curandStatePhilox4_32_10_t;

__device__ __forceinline__ void rng_init(RngState* s, unsigned long long seed, unsigned long long sample_id) {
    // (seed, subsequence, offset): one independent stream per sample.
    curand_init(seed, sample_id, 0ULL, s);
}

// Two independent standard normals (Z1 for price, Z2 for variance), in the
// precision the kernel runs in.
template <class R>
struct Normal2;
template <>
struct Normal2<float> {
    __device__ __forceinline__ static float2 draw(RngState* s) { return curand_normal2(s); }
};
template <>
struct Normal2<double> {
    __device__ __forceinline__ static double2 draw(RngState* s) { return curand_normal2_double(s); }
};

}  // namespace qf::cuda
