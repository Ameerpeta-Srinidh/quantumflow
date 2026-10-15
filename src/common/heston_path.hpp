// QuantumFlow - the per-path Heston simulation, shared by the CPU and CUDA engines.
//
// Discretisation: "full truncation" Euler for variance (Lord, Koekkoek & van
// Dijk 2010) and log-Euler for price. Variance can go slightly negative in a
// discrete step; full truncation uses max(v, 0) everywhere v appears, which is
// the scheme with the smallest bias among the simple Euler fixes.
//
//   v+       = max(v, 0)
//   ln S    += (r - v+/2) dt + sqrt(v+ dt) * Z1
//   v       += kappa (theta - v+) dt + xi sqrt(v+ dt) * (rho Z1 + sqrt(1-rho^2) Z2)
//
// The control variate path uses the same Z1 shocks with a constant volatility:
//   ln X    += (r - s^2/2) dt + s sqrt(dt) * Z1
// which is exact for geometric Brownian motion, so E[e^{-rT}(X_T-K)+] equals
// the Black-Scholes price with volatility s.
#pragma once

#include "heston_params.hpp"

namespace qf {

// Templated on the floating-point type R so the GPU can run in float
// (consumer GPUs do double-precision math 32-64x slower than float). The CPU
// engine and all tests use double; sums of payoffs are always kept in double.
template <class R>
struct PathConstantsT {
    R dt, sqdt, lnS0, kappa, theta, xi, rho, rho_bar, r;
    R cv_drift, cv_vol, disc, K, v0;
    double cv_sigma;
    int n_steps;
};
using PathConstants = PathConstantsT<double>;

template <class R = double>
QF_HD inline PathConstantsT<R> make_path_constants(const HestonParams& p, int n_steps) {
    PathConstantsT<R> c{};
    const double dt = p.T / n_steps;
    c.dt = R(dt);
    c.sqdt = R(sqrt(dt));
    c.lnS0 = R(log(p.S0));
    c.kappa = R(p.kappa);
    c.theta = R(p.theta);
    c.xi = R(p.xi);
    c.rho = R(p.rho);
    c.rho_bar = R(sqrt(1.0 - p.rho * p.rho));
    c.r = R(p.r);
    c.cv_sigma = control_variate_sigma(p);
    c.cv_drift = R((p.r - 0.5 * c.cv_sigma * c.cv_sigma) * dt);
    c.cv_vol = R(c.cv_sigma * sqrt(dt));
    c.disc = R(exp(-p.r * p.T));
    c.K = R(p.K);
    c.v0 = R(p.v0);
    c.n_steps = n_steps;
    return c;
}

// State of one path while it is being simulated.
template <class R>
struct PathStateT {
    R lnS, v, lnX;
};
using PathState = PathStateT<double>;

template <class R>
QF_HD inline void path_init(PathStateT<R>& s, const PathConstantsT<R>& c) {
    s.lnS = c.lnS0;
    s.v = c.v0;
    s.lnX = c.lnS0;
}

// Advance one path by one time step given two independent standard normals.
template <class R>
QF_HD inline void path_step(PathStateT<R>& s, const PathConstantsT<R>& c, R z1, R z2) {
    using std::sqrt;
    const R vp = s.v > R(0) ? s.v : R(0);
    const R sv = sqrt(vp * c.dt);
    s.lnS += (c.r - R(0.5) * vp) * c.dt + sv * z1;
    s.v += c.kappa * (c.theta - vp) * c.dt + c.xi * sv * (c.rho * z1 + c.rho_bar * z2);
    s.lnX += c.cv_drift + c.cv_vol * z1;
}

template <class R>
QF_HD inline R disc_call_payoff(R lnS, const PathConstantsT<R>& c) {
    using std::exp;
    const R ST = exp(lnS);
    return c.disc * (ST > c.K ? ST - c.K : R(0));
}

}  // namespace qf
