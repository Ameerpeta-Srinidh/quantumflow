// QuantumFlow - AVX2 scan for the next non-empty price level.
//
// When the best level of the book empties (all its orders filled or
// cancelled), we must find the next price that still has quantity. The level
// quantities live in a dense int32 array indexed by price tick, so this is
// "find the first non-zero element after index i". AVX2 checks 8 int32 lanes
// per instruction: compare against zero, turn the 8 results into an 8-bit
// mask, and use count-trailing-zeros to jump straight to the first hit.
//
// A scalar fallback is used when the compiler is not targeting AVX2, so the
// code builds everywhere; check godbolt.org or `objdump -d` to confirm the
// vpcmpeqd / vpmovmskb instructions are really emitted.
#pragma once

#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace qf::ob {

// First index in [from, to) with a[index] != 0, or -1 if there is none.
inline std::int32_t scan_forward_nonzero(const std::int32_t* a, std::int32_t from, std::int32_t to) {
    std::int32_t i = from;
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 8 <= to; i += 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i is_zero = _mm256_cmpeq_epi32(v, zero);
        // One bit per lane that is NOT zero.
        const unsigned mask = ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(is_zero))) & 0xFFu;
        if (mask) return i + __builtin_ctz(mask);
    }
#endif
    for (; i < to; ++i)
        if (a[i] != 0) return i;
    return -1;
}

// Last index in [from, to) with a[index] != 0 (searching downwards), or -1.
inline std::int32_t scan_backward_nonzero(const std::int32_t* a, std::int32_t from, std::int32_t to) {
    std::int32_t i = to;  // exclusive upper bound
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    for (; i - 8 >= from; i -= 8) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i - 8));
        const __m256i is_zero = _mm256_cmpeq_epi32(v, zero);
        const unsigned mask = ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(is_zero))) & 0xFFu;
        if (mask) return i - 8 + (31 - __builtin_clz(mask));
    }
#endif
    for (; i > from; --i)
        if (a[i - 1] != 0) return i - 1;
    return -1;
}

}  // namespace qf::ob
