// Shared benchmark helpers: a cycle-accurate timer and percentile summaries.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace qf::bench {

// Reads the CPU timestamp counter. lfence stops the CPU from reordering the
// read around the code being measured. ~20 cycles of overhead, far less than
// std::chrono::steady_clock (which goes through the vDSO).
inline std::uint64_t ticks() {
#if defined(__x86_64__)
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

// Nanoseconds per timestamp-counter tick, measured against steady_clock.
inline double ns_per_tick() {
#if defined(__x86_64__)
    const auto c0 = std::chrono::steady_clock::now();
    const std::uint64_t t0 = ticks();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::uint64_t t1 = ticks();
    const auto c1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(c1 - c0).count() / double(t1 - t0);
#else
    return 1.0;
#endif
}

inline double timer_overhead_ticks() {
    std::vector<std::uint64_t> d(100000);
    for (auto& x : d) {
        const auto a = ticks();
        x = ticks() - a;
    }
    std::sort(d.begin(), d.end());
    return double(d[d.size() / 2]);
}

struct Percentiles {
    double p50, p90, p95, p99, p999, max, mean;
};

inline Percentiles percentiles(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, std::size_t(q * v.size()))]; };
    double s = 0;
    for (double x : v) s += x;
    return {at(0.50), at(0.90), at(0.95), at(0.99), at(0.999), v.back(), s / v.size()};
}

inline std::string results_dir(int argc, char** argv) {
    return argc > 1 ? std::string(argv[1]) : std::string("results");
}

}  // namespace qf::bench
