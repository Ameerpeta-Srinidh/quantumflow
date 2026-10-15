// QuantumFlow - CPU reference Monte Carlo engine for Heston (OpenMP parallel).
//
// This does exactly what the CUDA kernel does, path for path, so it serves as
//  (1) a correctness reference for the GPU kernel,
//  (2) a fallback backend on machines without an NVIDIA GPU, and
//  (3) the CPU baseline in the benchmarks.
#pragma once

#include <chrono>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/heston_params.hpp"
#include "../common/heston_path.hpp"
#include "../common/philox.hpp"

namespace qf {

// Simulates one sample: a single path, or an antithetic pair averaged together.
// Returns the discounted Heston payoff (y) and the discounted control payoff (x).
inline void simulate_sample(const PathConstants& c, const Philox4x32& rng, std::uint64_t sample_id,
                            bool antithetic, double& y, double& x) {
    PathState a, b;
    path_init(a, c);
    path_init(b, c);
    double z[4];
    for (int t = 0; t < c.n_steps; t += 2) {
        // One Philox call gives 4 normals = 2 time steps x (Z1, Z2).
        rng.normal4(sample_id, static_cast<std::uint64_t>(t), z);
        path_step(a, c, z[0], z[1]);
        if (antithetic) path_step(b, c, -z[0], -z[1]);
        if (t + 1 < c.n_steps) {
            path_step(a, c, z[2], z[3]);
            if (antithetic) path_step(b, c, -z[2], -z[3]);
        }
    }
    if (antithetic) {
        y = 0.5 * (disc_call_payoff(a.lnS, c) + disc_call_payoff(b.lnS, c));
        x = 0.5 * (disc_call_payoff(a.lnX, c) + disc_call_payoff(b.lnX, c));
    } else {
        y = disc_call_payoff(a.lnS, c);
        x = disc_call_payoff(a.lnX, c);
    }
}

inline McResult heston_mc_cpu(const HestonParams& p, const McConfig& cfg, int n_threads = 0) {
    const auto t0 = std::chrono::steady_clock::now();
    const PathConstants c = make_path_constants(p, cfg.n_steps);
    const Philox4x32 rng(cfg.seed);
    // With antithetic sampling each sample is a pair of paths, so the number of
    // simulated paths stays n_paths either way.
    const std::int64_t n = static_cast<std::int64_t>(cfg.antithetic ? cfg.n_paths / 2 : cfg.n_paths);

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif
    double sy = 0, syy = 0, sx = 0, sxx = 0, sxy = 0;
#pragma omp parallel for reduction(+ : sy, syy, sx, sxx, sxy) schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        double y, x;
        simulate_sample(c, rng, static_cast<std::uint64_t>(i), cfg.antithetic, y, x);
        sy += y; syy += y * y; sx += x; sxx += x * x; sxy += x * y;
    }
    const McSums sums{sy, syy, sx, sxx, sxy};
    const double EX = bs_call(p.S0, p.K, p.T, p.r, c.cv_sigma);
    McResult res = finalize(sums, static_cast<std::uint64_t>(n), EX, cfg.control_variate);
    res.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace qf
