// Pricing tests: the analytic Heston price, the CPU Monte Carlo engine, the
// variance-reduction techniques, and the Greeks.
#include <cmath>

#include "check.hpp"
#include "common/heston_analytic.hpp"
#include "common/mc_greeks.hpp"
#include "cpu/heston_mc_cpu.hpp"

using namespace qf;

// 1) When vol-of-vol -> 0 and v0 = theta, variance is constant and Heston must
//    collapse to Black-Scholes. This checks the characteristic-function code.
//    rho = 0 matters: with rho != 0 the price differs from BS at first order
//    in xi (that difference *is* the volatility skew), so it converges slowly.
static void test_bs_limit() {
    for (double K : {80.0, 100.0, 120.0})
        for (double T : {0.25, 1.0, 2.0}) {
            HestonParams p;
            p.K = K; p.T = T; p.v0 = p.theta = 0.04; p.xi = 1e-3; p.rho = 0.0;
            CHECK_NEAR(heston_call_analytic(p), bs_call(p.S0, K, T, p.r, 0.2), 2e-4);
        }
}

// 2) Analytic Greeks in the same limit must match the Black-Scholes Greeks.
//    kappa is also ~0 here: vega bumps today's variance v0 only, and with
//    strong mean reversion that bump fades before expiry (so Heston vega is
//    legitimately smaller than BS vega). With kappa ~ 0 it persists, like BS.
static void test_bs_greeks_limit() {
    HestonParams p;
    p.v0 = p.theta = 0.04; p.xi = 1e-3; p.rho = 0.0; p.kappa = 1e-3; p.K = 105; p.T = 0.75;
    const Greeks g = heston_greeks_analytic(p);
    const double s = 0.2, sT = s * std::sqrt(p.T);
    const double d1 = (std::log(p.S0 / p.K) + (p.r + 0.5 * s * s) * p.T) / sT, d2 = d1 - sT;
    const double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2 * M_PI);
    CHECK_NEAR(g.delta, norm_cdf(d1), 1e-4);
    CHECK_NEAR(g.gamma, pdf / (p.S0 * sT), 1e-4);
    CHECK_NEAR(g.vega, p.S0 * pdf * std::sqrt(p.T) * 0.01, 1e-3);
    CHECK_NEAR(g.rho, p.K * p.T * std::exp(-p.r * p.T) * norm_cdf(d2) * 0.01, 1e-3);
    const double bs_theta = -p.S0 * pdf * s / (2 * std::sqrt(p.T)) - p.r * p.K * std::exp(-p.r * p.T) * norm_cdf(d2);
    CHECK_NEAR(g.theta, bs_theta, 2e-3);
}

// 3) Monte Carlo agrees with the analytic price within its error bars
//    (plus a small allowance for time-discretisation bias).
static void test_mc_vs_analytic() {
    const HestonParams cases[] = {
        {100, 100, 1.0, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7},
        {100, 90, 0.5, 0.01, 0.09, 1.5, 0.06, 0.8, -0.5},
        {100, 115, 2.0, 0.05, 0.02, 3.0, 0.05, 0.3, -0.3},
    };
    for (const auto& p : cases) {
        McConfig cfg;
        cfg.n_paths = 1 << 20;
        cfg.n_steps = 200;
        const McResult r = heston_mc_cpu(p, cfg);
        const double exact = heston_call_analytic(p);
        std::printf("  MC %.5f +- %.5f   analytic %.5f   diff/SE %.2f\n", r.price, r.std_error, exact,
                    (r.price - exact) / r.std_error);
        CHECK_NEAR(r.price, exact, 4 * r.std_error + 0.01);
    }
}

// 4) Each variance-reduction technique must shrink the standard error at the
//    same number of simulated paths.
static void test_variance_reduction() {
    HestonParams p;
    McConfig cfg;
    cfg.n_paths = 1 << 18;
    cfg.n_steps = 100;
    cfg.antithetic = false; cfg.control_variate = false;
    const double se_plain = heston_mc_cpu(p, cfg).std_error;
    cfg.antithetic = true;
    const double se_anti = heston_mc_cpu(p, cfg).std_error;
    cfg.control_variate = true;
    const double se_both = heston_mc_cpu(p, cfg).std_error;
    std::printf("  std error: plain %.5f  antithetic %.5f  antithetic+CV %.5f  (variance ratio %.1fx)\n",
                se_plain, se_anti, se_both, (se_plain * se_plain) / (se_both * se_both));
    CHECK(se_anti < se_plain);
    CHECK(se_both < se_anti);
}

// 5) Counter-based RNG: identical result for any thread count.
static void test_determinism() {
    HestonParams p;
    McConfig cfg;
    cfg.n_paths = 1 << 14;
    cfg.n_steps = 50;
    const double a = heston_mc_cpu(p, cfg, 1).price;
    const double b = heston_mc_cpu(p, cfg, 4).price;
    CHECK_NEAR(a, b, 1e-9);
}

// 6) Bump-and-reprice Monte Carlo Greeks land near the analytic Greeks.
static void test_mc_greeks() {
    HestonParams p;
    McConfig cfg;
    cfg.n_paths = 1 << 18;
    cfg.n_steps = 100;
    const McGreeks mg = mc_bump_greeks(p, cfg, [](const HestonParams& q, const McConfig& c) { return heston_mc_cpu(q, c); });
    const Greeks ag = heston_greeks_analytic(p);
    std::printf("  delta MC %.4f / exact %.4f   gamma %.4f / %.4f   vega %.4f / %.4f   theta %.3f / %.3f   rho %.4f / %.4f\n",
                mg.g.delta, ag.delta, mg.g.gamma, ag.gamma, mg.g.vega, ag.vega, mg.g.theta, ag.theta, mg.g.rho, ag.rho);
    CHECK_NEAR(mg.g.delta, ag.delta, 0.01);
    CHECK_NEAR(mg.g.gamma, ag.gamma, 0.004);
    CHECK_NEAR(mg.g.vega, ag.vega, 0.03);
    CHECK_NEAR(mg.g.theta, ag.theta, 0.3);
    CHECK_NEAR(mg.g.rho, ag.rho, 0.02);
}

int main() {
    test_bs_limit();
    test_bs_greeks_limit();
    test_mc_vs_analytic();
    test_variance_reduction();
    test_determinism();
    test_mc_greeks();
    return report("test_heston");
}
