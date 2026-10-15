// Tiny test helper: CHECK(cond) records a failure instead of aborting, so one
// run reports every broken expectation.
#pragma once
#include <cmath>
#include <cstdio>

inline int g_failures = 0;
inline int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                    \
    do {                                                                         \
        ++g_checks;                                                              \
        const double a_ = (a), b_ = (b);                                         \
        if (!(std::fabs(a_ - b_) <= (tol))) {                                    \
            ++g_failures;                                                        \
            std::printf("  FAIL %s:%d  %s = %.10g vs %s = %.10g (tol %g)\n",     \
                        __FILE__, __LINE__, #a, a_, #b, b_, (double)(tol));      \
        }                                                                        \
    } while (0)

inline int report(const char* name) {
    std::printf("%s: %d checks, %d failures\n", name, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
