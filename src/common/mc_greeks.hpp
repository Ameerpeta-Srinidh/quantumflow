// QuantumFlow - Greeks by "bump and reprice" Monte Carlo.
//
// This is the slow, expensive method the neural surrogate replaces: every
// Greek needs extra full Monte Carlo runs with one input nudged. We use
// *common random numbers* (the same seed for every bumped run) so the noise
// largely cancels in the differences; without that, the Greeks would be
// swamped by Monte Carlo noise.
//
// Cost: 1 base run + 2 (spot) + 2 (time) + 2 (vol) + 2 (rate) = 9 runs.
#pragma once

#include <chrono>
#include <cmath>

#include "heston_analytic.hpp"
#include "heston_params.hpp"

namespace qf {

struct McGreeks {
    Greeks g;
    double elapsed_ms;
    int n_runs;
};

template <class Pricer>  // Pricer: McResult(const HestonParams&, const McConfig&)
McGreeks mc_bump_greeks(const HestonParams& p, const McConfig& cfg, Pricer&& price_fn) {
    const auto t0 = std::chrono::steady_clock::now();
    auto px = [&](const HestonParams& q) { return price_fn(q, cfg).price; };
    McGreeks out{};
    Greeks& g = out.g;
    g.price = px(p);

    // Bigger bumps than the analytic version: MC noise would dominate tiny ones.
    const double hS = 0.01 * p.S0;
    HestonParams up = p, dn = p;
    up.S0 += hS; dn.S0 -= hS;
    const double cu = px(up), cdn = px(dn);
    g.delta = (cu - cdn) / (2 * hS);
    g.gamma = (cu - 2 * g.price + cdn) / (hS * hS);

    const double hT = 1.0 / 365.0;
    up = p; dn = p; up.T += hT; dn.T -= hT;
    g.theta = -(px(up) - px(dn)) / (2 * hT);

    const double s0 = std::sqrt(p.v0), hs = 0.01;
    up = p; dn = p;
    up.v0 = (s0 + hs) * (s0 + hs);
    dn.v0 = (s0 - hs) * (s0 - hs);
    g.vega = (px(up) - px(dn)) / (2 * hs) * 0.01;

    const double hr = 1e-3;
    up = p; dn = p; up.r += hr; dn.r -= hr;
    g.rho = (px(up) - px(dn)) / (2 * hr) * 0.01;

    out.n_runs = 9;
    out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace qf
