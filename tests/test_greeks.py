"""Analytical Greeks: reference formulas, finite differences, PDE, parity relations, limits."""

from __future__ import annotations

import math

import numpy as np
import pytest
import reference as ref
from conftest import GRID, OPTION_TYPES, STRIKE

import bspricer as bs

K = STRIKE


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_matches_independent_reference(S, T, r, sigma, q, opt):
    expected = ref.greeks(S, K, T, r, sigma, opt == "call", q)
    got = bs.bs_greeks(S, K, T, r, sigma, opt, q)
    for name in bs.GREEK_NAMES:
        assert got[name] == pytest.approx(expected[name], rel=1e-10, abs=1e-12), name


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_against_central_finite_differences(S, T, r, sigma, q, opt):
    """Central differences with steps of 1e-3 standard deviation (spot), 1e-4 T, 1e-4 sigma, 1e-4 r.

    The truncation error is O(h^2); the largest error observed on this grid is 3.6e-7 of the
    Greek's natural scale (delta: 1, gamma: 1/(S sigma sqrt T), theta: S sigma / sqrt T,
    vega: S sqrt T, rho: K T). The tolerance 1e-5 of that scale leaves a margin of ~30.
    """

    def price(S_=S, T_=T, r_=r, s_=sigma):
        return bs.bs_price(S_, K, T_, r_, s_, opt, q)

    h_s, h_t, h_v, h_r = 1e-3 * S * sigma * math.sqrt(T), 1e-4 * T, 1e-4 * sigma, 1e-4
    fd = {
        "delta": (price(S_=S + h_s) - price(S_=S - h_s)) / (2 * h_s),
        "gamma": (price(S_=S + h_s) - 2 * price() + price(S_=S - h_s)) / h_s**2,
        "theta": -(price(T_=T + h_t) - price(T_=T - h_t)) / (2 * h_t),
        "vega": (price(s_=sigma + h_v) - price(s_=sigma - h_v)) / (2 * h_v),
        "rho": (price(r_=r + h_r) - price(r_=r - h_r)) / (2 * h_r),
    }
    scale = {
        "delta": 1.0,
        "gamma": 1.0 / (S * sigma * math.sqrt(T)),
        "theta": S * sigma / math.sqrt(T),
        "vega": S * math.sqrt(T),
        "rho": K * T,
    }
    analytic = bs.bs_greeks(S, K, T, r, sigma, opt, q)
    for name, value in fd.items():
        assert abs(value - analytic[name]) <= 1e-5 * scale[name], name


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_black_scholes_pde(S, T, r, sigma, q, opt):
    """Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0 (max observed 1.8e-17, scaled)."""
    residual = bs.bs_pde_residual(S, K, T, r, sigma, opt, q)
    assert abs(residual) <= 1e-12 * max(S, K) * (1.0 + 1.0 / T)


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
def test_call_put_relations(S, T, r, sigma, q):
    """Differentiating put-call parity: equal gamma and vega, fixed delta/theta/rho gaps."""
    c = bs.bs_greeks(S, K, T, r, sigma, "call", q)
    p = bs.bs_greeks(S, K, T, r, sigma, "put", q)
    tol = 1e-10 * max(S, K)
    assert c["delta"] - p["delta"] == pytest.approx(math.exp(-q * T), abs=1e-12)
    assert c["gamma"] == pytest.approx(p["gamma"], rel=1e-12)
    assert c["vega"] == pytest.approx(p["vega"], rel=1e-12, abs=1e-12)
    theta_gap = q * S * math.exp(-q * T) - r * K * math.exp(-r * T)
    assert abs(c["theta"] - p["theta"] - theta_gap) <= tol
    assert abs(c["rho"] - p["rho"] - K * T * math.exp(-r * T)) <= tol


def test_gamma_peaks_at_closed_form_spot():
    """argmax_S Gamma = K exp(-(r - q + 3 sigma^2 / 2) T), not exactly the strike."""
    T, r, sigma, q = 0.5, 0.03, 0.4, 0.01
    spots = np.linspace(70.0, 130.0, 600_001)
    gamma = bs.bs_greeks(spots, K, T, r, sigma, "call", q)["gamma"]
    expected = K * math.exp(-(r - q + 1.5 * sigma**2) * T)
    assert spots[np.argmax(gamma)] == pytest.approx(expected, abs=2e-4)


def test_vega_scales_with_sqrt_maturity_at_the_forward():
    """At F = K with r = q = 0, vega = S sqrt(T) phi(sigma sqrt(T) / 2)."""
    for T in (0.1, 0.5, 2.0):
        expected = 100.0 * math.sqrt(T) * ref.norm_pdf(0.5 * 0.2 * math.sqrt(T))
        assert bs.bs_greeks(100.0, 100.0, T, 0.0, 0.2)["vega"] == pytest.approx(expected, rel=1e-12)


def test_degenerate_greeks_at_expiry():
    g = bs.bs_greeks(np.array([90.0, 100.0, 110.0]), K, 0.0, 0.05, 0.2, "call")
    np.testing.assert_array_equal(g["delta"], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(g["gamma"], [0.0, math.inf, 0.0])
    np.testing.assert_array_equal(g["vega"], [0.0, 0.0, 0.0])
    gp = bs.bs_greeks(np.array([90.0, 100.0, 110.0]), K, 0.0, 0.05, 0.2, "put")
    np.testing.assert_array_equal(gp["delta"], [-1.0, -0.5, 0.0])


def test_zero_vol_greeks_match_limit_price():
    """At sigma = 0 the price is max(S e^{-qT} - K e^{-rT}, 0); its derivatives are the Greeks."""
    S, T, r, q = 98.0, 1.0, 0.05, 0.01  # forward above the strike
    g = bs.bs_greeks(S, K, T, r, 0.0, "call", q)
    assert g["delta"] == pytest.approx(math.exp(-q * T))
    assert g["gamma"] == 0.0 and g["vega"] == 0.0
    assert g["theta"] == pytest.approx(q * S * math.exp(-q * T) - r * K * math.exp(-r * T))
    assert g["rho"] == pytest.approx(K * T * math.exp(-r * T))


def test_vectorised_matches_scalar():
    spots = np.array([80.0, 100.0, 125.0])
    vec = bs.bs_greeks(spots, K, 0.3, 0.02, np.array([0.15, 0.2, 0.3]), "put", 0.01)
    for i, (s, v) in enumerate(zip(spots, [0.15, 0.2, 0.3], strict=True)):
        scalar = bs.bs_greeks(s, K, 0.3, 0.02, v, "put", 0.01)
        for name in bs.GREEK_NAMES:
            assert vec[name][i] == scalar[name]


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        bs.bs_greeks(100.0, 100.0, 1.0, 0.0, 0.2, "Call")
