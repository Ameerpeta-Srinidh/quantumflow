// QuantumFlow - semi-closed-form Heston (1993) price for a European call.
//
// This is the "answer key" for the project: the Monte Carlo engines are tested
// against it, and the Greeks surrogate is trained on Greeks computed from it.
//
// We use the single-integral form of Lewis (2000), which needs one
// characteristic-function evaluation per point instead of two and whose
// integrand decays like 1/u^2 (the classic Heston P1/P2 form decays like 1/u):
//
//   C = S0 - sqrt(K) e^{-rT} / pi * Int_0^inf Re[ e^{-iu ln K} phi(u - i/2) ] / (u^2 + 1/4) du
//
// phi(u) = E[exp(iu ln S_T)] is the characteristic function of the log price,
// written in the numerically stable "little Heston trap" form of Albrecher,
// Mayer, Schoutens & Tistaert (2007).
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "heston_params.hpp"

namespace qf {

namespace detail {

// Gauss-Legendre nodes/weights on [-1, 1], computed once by Newton iteration.
template <int N>
struct GaussLegendre {
    std::array<double, N> x{}, w{};
    GaussLegendre() {
        for (int i = 0; i < (N + 1) / 2; ++i) {
            double z = std::cos(M_PI * (i + 0.75) / (N + 0.5));
            double pp = 0;
            for (int it = 0; it < 100; ++it) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1);
                }
                pp = N * (z * p1 - p2) / (z * z - 1.0);
                const double z1 = z;
                z = z1 - p1 / pp;
                if (std::abs(z - z1) < 1e-15) break;
            }
            x[i] = -z;
            x[N - 1 - i] = z;
            w[i] = w[N - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
        }
    }
};

inline const GaussLegendre<64>& gl64() {
    static const GaussLegendre<64> g;
    return g;
}

using cd = std::complex<double>;

// Characteristic function of ln(S_T) under Heston ("little trap" form).
inline cd heston_cf(cd u, const HestonParams& p) {
    const cd i(0.0, 1.0);
    const double xi2 = p.xi * p.xi;
    const cd a = p.kappa - p.rho * p.xi * i * u;
    const cd d = std::sqrt(a * a + xi2 * (i * u + u * u));
    const cd g = (a - d) / (a + d);
    const cd edt = std::exp(-d * p.T);
    const cd C = (p.kappa * p.theta / xi2) * ((a - d) * p.T - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
    const cd D = ((a - d) / xi2) * (1.0 - edt) / (1.0 - g * edt);
    return std::exp(i * u * (std::log(p.S0) + p.r * p.T) + C + D * p.v0);
}

}  // namespace detail

// Price of a European call under Heston, accurate to ~1e-10 in absolute terms.
// The integral is taken panel by panel (64-point Gauss-Legendre on each
// 20-wide panel) until a panel adds nothing measurable: short-dated, low-vol
// options have slowly decaying integrands and automatically get more panels.
inline double heston_call_analytic(const HestonParams& p) {
    using detail::cd;
    const auto& gl = detail::gl64();
    const cd i(0.0, 1.0);
    const double lnK = std::log(p.K);
    const double width = 20.0;
    const int max_panels = 200;
    double I = 0.0;
    for (int k = 0; k < max_panels; ++k) {
        const double a = width * k, half = 0.5 * width, mid = a + half;
        double panel = 0.0;
        for (int j = 0; j < 64; ++j) {
            const double u = mid + half * gl.x[j];
            const cd f = std::exp(-i * u * lnK) * detail::heston_cf(cd(u, -0.5), p) / (u * u + 0.25);
            panel += gl.w[j] * half * f.real();
        }
        I += panel;
        if (k >= 1 && std::abs(panel) < 1e-13) break;
    }
    return p.S0 - std::sqrt(p.K) * std::exp(-p.r * p.T) / M_PI * I;
}

// The five Greeks, using the conventions traders quote:
//   delta = dC/dS                       (per $1 move in spot)
//   gamma = d2C/dS2                     (change in delta per $1)
//   theta = dC/dt = -dC/dT              (per year of calendar time; divide by 365 for per day)
//   vega  = dC/d(sigma0) * 0.01         (per 1 vol point, sigma0 = sqrt(v0))
//   rho   = dC/dr * 0.01                (per 1 percentage point of rate)
struct Greeks {
    double price, delta, gamma, theta, vega, rho;
};

// Greeks by central finite differences of the analytic price. Because the
// analytic price is smooth and accurate to ~1e-8, small bumps are safe here
// (unlike with noisy Monte Carlo prices).
inline Greeks heston_greeks_analytic(const HestonParams& p) {
    auto price = [](HestonParams q) { return heston_call_analytic(q); };
    Greeks g{};
    g.price = price(p);

    const double hS = 1e-3 * p.S0;
    HestonParams up = p, dn = p;
    up.S0 += hS; dn.S0 -= hS;
    const double cu = price(up), cdn = price(dn);
    g.delta = (cu - cdn) / (2 * hS);
    g.gamma = (cu - 2 * g.price + cdn) / (hS * hS);

    const double hT = 1e-4;
    up = p; dn = p; up.T += hT; dn.T -= hT;
    g.theta = -(price(up) - price(dn)) / (2 * hT);

    const double s0 = std::sqrt(p.v0), hs = 1e-4;
    up = p; dn = p;
    up.v0 = (s0 + hs) * (s0 + hs);
    dn.v0 = (s0 - hs) * (s0 - hs);
    g.vega = (price(up) - price(dn)) / (2 * hs) * 0.01;

    const double hr = 1e-5;
    up = p; dn = p; up.r += hr; dn.r -= hr;
    g.rho = (price(up) - price(dn)) / (2 * hr) * 0.01;
    return g;
}

}  // namespace qf
