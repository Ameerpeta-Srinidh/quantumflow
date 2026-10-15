// QuantumFlow - shared model types and the Black-Scholes closed form.
//
// Everything here is plain C++17 and marked QF_HD so the same code compiles
// for the CPU and, when nvcc is used, for the GPU.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define QF_HD __host__ __device__
#else
#define QF_HD
#endif

namespace qf {

// All the inputs that define one European call under Heston.
//   S0    spot price today
//   K     strike
//   T     time to expiry in years
//   r     continuously compounded risk-free rate
//   v0    variance today (vol = sqrt(v0); v0 = 0.04 means 20% vol)
//   kappa speed at which variance is pulled back to theta
//   theta long-run variance
//   xi    volatility of variance ("vol of vol")
//   rho   correlation between the price shock and the variance shock
struct HestonParams {
    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.03;
    double v0 = 0.04;
    double kappa = 2.0;
    double theta = 0.04;
    double xi = 0.5;
    double rho = -0.7;
};

// Settings for one Monte Carlo run.
struct McConfig {
    std::uint64_t n_paths = 1 << 20;  // number of antithetic *pairs* is n_paths / 2 when antithetic=true
    int n_steps = 256;                // time steps per path
    std::uint64_t seed = 42;
    bool antithetic = true;
    bool control_variate = true;
    // GPU only: simulate paths in float (fast on every GPU) instead of double
    // (1/32 to 1/64 speed on consumer cards). Payoff sums stay in double.
    bool gpu_single_precision = true;
};

// Result of one Monte Carlo run.
struct McResult {
    double price = 0.0;     // estimated option value
    double std_error = 0.0; // standard error of that estimate
    double beta = 0.0;      // control-variate coefficient that was used (0 if off)
    std::uint64_t n_samples = 0;  // independent samples (pairs, if antithetic)
    double elapsed_ms = 0.0;
};

QF_HD inline double norm_cdf(double x) { return 0.5 * erfc(-x * 0.70710678118654752440); }

// Black-Scholes price of a European call.
QF_HD inline double bs_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return S > K ? S - K : 0.0;
    const double sd = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / sd;
    const double d2 = d1 - sd;
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
}

// The control variate is a geometric Brownian motion driven by the same price
// shocks as the Heston path, with a constant volatility equal to the
// *expected average* Heston variance over [0, T]. Its exact expectation is a
// Black-Scholes price, which is what lets us subtract it.
QF_HD inline double control_variate_sigma(const HestonParams& p) {
    const double kT = p.kappa * p.T;
    const double w = kT > 1e-8 ? (1.0 - exp(-kT)) / kT : 1.0;
    const double avg_var = p.theta + (p.v0 - p.theta) * w;
    return sqrt(avg_var > 1e-12 ? avg_var : 1e-12);
}

// Turn the running sums from a Monte Carlo run into a price and a standard
// error, applying the optimal control-variate coefficient beta = Cov(Y,X)/Var(X).
// Y = discounted Heston payoff, X = discounted control payoff, EX = its known mean.
struct McSums {
    double sy = 0, syy = 0, sx = 0, sxx = 0, sxy = 0;
};

inline McResult finalize(const McSums& s, std::uint64_t n, double EX, bool use_cv) {
    McResult out;
    const double dn = static_cast<double>(n);
    const double my = s.sy / dn;
    const double vy = (s.syy - dn * my * my) / (dn - 1.0);
    out.n_samples = n;
    if (!use_cv) {
        out.price = my;
        out.std_error = std::sqrt(vy > 0 ? vy / dn : 0.0);
        return out;
    }
    const double mx = s.sx / dn;
    const double vx = (s.sxx - dn * mx * mx) / (dn - 1.0);
    const double cxy = (s.sxy - dn * mx * my) / (dn - 1.0);
    const double beta = vx > 0 ? cxy / vx : 0.0;
    // Var(Y - beta X) = Var(Y) - 2 beta Cov + beta^2 Var(X) = Var(Y) - Cov^2/Var(X)
    const double vcv = vy - beta * cxy;
    out.price = my - beta * (mx - EX);
    out.std_error = std::sqrt(vcv > 0 ? vcv / dn : 0.0);
    out.beta = beta;
    return out;
}

}  // namespace qf
