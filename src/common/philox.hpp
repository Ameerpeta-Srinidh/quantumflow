// QuantumFlow - Philox4x32-10 counter-based random number generator (CPU side).
//
// Philox (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC'11)
// is the same family cuRAND uses on the GPU. It is "counter based": the random
// numbers are a pure function of (key, counter). Path p, step t always gets
// the same numbers regardless of which thread runs it, so CPU results are
// reproducible for any thread count.
#pragma once

#include <cmath>
#include <cstdint>

#include "heston_params.hpp"

namespace qf {

struct Philox4x32 {
    std::uint32_t key0, key1;

    QF_HD explicit Philox4x32(std::uint64_t seed)
        : key0(static_cast<std::uint32_t>(seed)), key1(static_cast<std::uint32_t>(seed >> 32)) {}

    QF_HD static inline void mulhilo(std::uint32_t a, std::uint32_t b, std::uint32_t& hi, std::uint32_t& lo) {
        const std::uint64_t prod = static_cast<std::uint64_t>(a) * b;
        hi = static_cast<std::uint32_t>(prod >> 32);
        lo = static_cast<std::uint32_t>(prod);
    }

    // Returns 4 independent uniformly distributed 32-bit integers for this counter.
    QF_HD inline void operator()(std::uint64_t c_lo, std::uint64_t c_hi, std::uint32_t out[4]) const {
        std::uint32_t c[4] = {static_cast<std::uint32_t>(c_lo), static_cast<std::uint32_t>(c_lo >> 32),
                              static_cast<std::uint32_t>(c_hi), static_cast<std::uint32_t>(c_hi >> 32)};
        std::uint32_t k0 = key0, k1 = key1;
        for (int round = 0; round < 10; ++round) {
            std::uint32_t hi0, lo0, hi1, lo1;
            mulhilo(0xD2511F53u, c[0], hi0, lo0);
            mulhilo(0xCD9E8D57u, c[2], hi1, lo1);
            const std::uint32_t n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
            c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
    }

    // Four standard normals via Box-Muller.
    QF_HD inline void normal4(std::uint64_t c_lo, std::uint64_t c_hi, double z[4]) const {
        std::uint32_t u[4];
        (*this)(c_lo, c_hi, u);
        const double inv = 1.0 / 4294967296.0;
        for (int k = 0; k < 4; k += 2) {
            const double u1 = (u[k] + 0.5) * inv;  // in (0,1), never 0 so log is safe
            const double u2 = (u[k + 1] + 0.5) * inv;
            const double rad = sqrt(-2.0 * log(u1));
            z[k] = rad * cos(6.283185307179586 * u2);
            z[k + 1] = rad * sin(6.283185307179586 * u2);
        }
    }
};

}  // namespace qf
