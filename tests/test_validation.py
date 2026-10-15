"""Validate the QuantumFlow pricers against QuantLib (an independent,
widely used open-source library) and check the pybind11 bridge.

    python -m pytest tests/test_validation.py -v

QuantLib reference: AnalyticHestonEngine (Heston 1993 semi-closed form,
QuantLib's own implementation). Tests that need QuantLib are skipped if it is
not installed (pip install QuantLib).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import qf_core  # noqa: E402

ql = pytest.importorskip("QuantLib")

CASES = [
    # S0,  K,    T,    r,    v0,   kappa, theta, xi,  rho
    (100, 100, 1.00, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7),
    (100, 80, 0.50, 0.01, 0.09, 1.5, 0.06, 0.8, -0.5),
    (100, 120, 2.00, 0.05, 0.02, 3.0, 0.05, 0.3, -0.3),
    (95, 105, 0.25, 0.02, 0.16, 4.0, 0.09, 0.9, -0.9),
    (110, 100, 1.50, 0.00, 0.01, 0.5, 0.16, 0.2, 0.0),
]


def quantlib_heston_call(S0, K, T, r, v0, kappa, theta, xi, rho):
    today = ql.Date(15, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    maturity = today + int(round(T * 365))
    T_exact = dc.yearFraction(today, maturity)
    rate = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc, ql.Continuous))
    div = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc, ql.Continuous))
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    process = ql.HestonProcess(rate, div, spot, v0, kappa, theta, xi, rho)
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(maturity))
    opt.setPricingEngine(engine)
    return opt.NPV(), T_exact


@pytest.mark.parametrize("case", CASES)
def test_analytic_matches_quantlib(case):
    ref, T_exact = quantlib_heston_call(*case)
    S0, K, _, r, v0, kappa, theta, xi, rho = case
    ours = qf_core.heston_price(qf_core.HestonParams(S0, K, T_exact, r, v0, kappa, theta, xi, rho))
    assert ours == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("case", CASES[:3])
def test_monte_carlo_matches_quantlib(case):
    ref, T_exact = quantlib_heston_call(*case)
    S0, K, _, r, v0, kappa, theta, xi, rho = case
    res = qf_core.mc_price(qf_core.HestonParams(S0, K, T_exact, r, v0, kappa, theta, xi, rho),
                           n_paths=1 << 20, n_steps=200, backend="cpu")
    # within 4 standard errors, plus a small allowance for time-step bias
    assert abs(res["price"] - ref) < 4 * res["std_error"] + 0.01


def test_black_scholes_matches_quantlib():
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.02, 0.25
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    ref = ql.blackFormula(ql.Option.Call, K, S * np.exp(r * T), sigma * np.sqrt(T), np.exp(-r * T))
    assert qf_core.bs_call(S, K, T, r, sigma) == pytest.approx(ref, abs=1e-10)
    assert np.isfinite(d1)


def test_batch_api_is_zero_copy_and_consistent():
    X = np.array([list(c) for c in CASES], dtype=np.float64)
    out = qf_core.heston_greeks_batch(X)
    assert out.shape == (len(CASES), 6)
    assert not out.flags["OWNDATA"]  # memory owned by C++ (capsule), not copied into NumPy
    for row, case in zip(out, CASES):
        assert row[0] == pytest.approx(qf_core.heston_price(qf_core.HestonParams(*case)), abs=1e-12)
        assert 0.0 <= row[1] <= 1.0  # call delta
        assert row[2] >= 0.0         # gamma
        assert row[4] >= 0.0         # vega


def test_order_book_through_bridge():
    b = qf_core.OrderBook(1000, 0.01)
    b.add(1, qf_core.Side.BUY, 500, 10)
    b.add(2, qf_core.Side.SELL, 502, 10)
    assert b.best_bid() == 500 and b.best_ask() == 502
    assert b.mid_price() == pytest.approx(5.01)
    assert b.add(3, qf_core.Side.BUY, 502, 4) == 4  # crosses and trades
    assert b.top().ask_qty == 6
