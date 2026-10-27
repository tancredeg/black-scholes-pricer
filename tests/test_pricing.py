"""Closed-form prices: identities, no-arbitrage bounds, limits and input validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import reference as ref
from conftest import GRID, OPTION_TYPES, STRIKE

import bspricer as bs

K = STRIKE


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
def test_matches_independent_reference(S, T, r, sigma, q):
    for opt in OPTION_TYPES:
        expected = ref.price(S, K, T, r, sigma, opt == "call", q)
        assert bs.bs_price(S, K, T, r, sigma, opt, q) == pytest.approx(
            expected, rel=1e-11, abs=1e-11
        )


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
def test_put_call_parity(S, T, r, sigma, q):
    """C - P = S e^{-qT} - K e^{-rT} (model-free)."""
    lhs = bs.bs_price(S, K, T, r, sigma, "call", q) - bs.bs_price(S, K, T, r, sigma, "put", q)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) <= 1e-12 * max(S, K)


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_no_arbitrage_bounds(S, T, r, sigma, q, opt):
    """max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}, and the put analogue."""
    value = bs.bs_price(S, K, T, r, sigma, opt, q)
    lower, upper = bs.no_arbitrage_bounds(S, K, T, r, opt, q)
    tol = 1e-12 * max(S, K)
    assert lower - tol <= value <= upper + tol


@pytest.mark.parametrize(("T", "r", "sigma", "q"), [(0.5, 0.03, 0.2, 0.0), (2.0, 0.05, 0.6, 0.03)])
def test_call_decreasing_and_convex_in_strike(T, r, sigma, q):
    strikes = np.linspace(50.0, 200.0, 301)
    calls = bs.bs_price(100.0, strikes, T, r, sigma, "call", q)
    assert np.all(np.diff(calls) <= 1e-12)
    assert np.all(calls[:-2] - 2 * calls[1:-1] + calls[2:] >= -1e-12)
    # slope bounded by the discount factor: -e^{-rT} <= dC/dK <= 0
    assert np.all(np.diff(calls) / np.diff(strikes) >= -math.exp(-r * T) - 1e-12)


@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_increasing_in_volatility(opt):
    vols = np.linspace(0.01, 1.5, 200)
    values = bs.bs_price(100.0, 110.0, 0.5, 0.03, vols, opt, 0.01)
    assert np.all(np.diff(values) > 0)


def test_call_increasing_in_maturity_without_dividends():
    maturities = np.linspace(0.01, 5.0, 200)
    values = bs.bs_price(100.0, 105.0, maturities, 0.03, 0.2, "call")
    assert np.all(np.diff(values) > 0)


# --- limits ---------------------------------------------------------------------------------


@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_expiry_returns_payoff(S, opt):
    payoff = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    assert bs.bs_price(S, K, 0.0, 0.05, 0.2, opt) == payoff


@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_short_maturity_converges_to_payoff(S, opt):
    """|V(T) - payoff| <= S sigma sqrt(T): the time value vanishes like sqrt(T)."""
    payoff = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    for T in (1e-4, 1e-8, 1e-12):
        assert abs(bs.bs_price(S, K, T, 0.05, 0.2, opt) - payoff) <= S * 0.2 * math.sqrt(T) + 1e-12


# Cases where the sign of S - K differs from the sign of the forward moneyness: these are the
# cases where the pre-v1.0 code returned a wrong (even negative) price (docs/AUDIT.md, C3).
ZERO_VOL_CASES = [
    (105.0, 100.0, 1.0, 0.00, 0.00),
    (98.0, 100.0, 1.0, 0.05, 0.00),  # spot below strike, forward above
    (100.0, 100.0, 1.0, 0.00, 0.03),  # spot at strike, forward below
    (102.0, 100.0, 1.0, 0.00, 0.05),  # spot above strike, forward below
    (100.0, 100.0, 2.0, 0.04, 0.04),  # forward exactly at the strike
]


@pytest.mark.parametrize(("S", "Kz", "T", "r", "q"), ZERO_VOL_CASES)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_zero_vol_limit_is_discounted_forward_intrinsic(S, Kz, T, r, q, opt):
    fwd_gap = S * math.exp(-q * T) - Kz * math.exp(-r * T)
    limit = max(fwd_gap, 0.0) if opt == "call" else max(-fwd_gap, 0.0)
    for sigma in (0.0, 1e-300, 1e-14, 1e-10, 1e-8, 1e-6):
        value = bs.bs_price(S, Kz, T, r, sigma, opt, q)
        assert value >= 0.0
        assert abs(value - limit) <= 1e-10 + S * sigma * math.sqrt(T)


@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_spot_to_zero(opt):
    """S -> 0: the call is worthless and the put tends to K e^{-rT} - S (deep in the money)."""
    S = 1e-10
    value = bs.bs_price(S, K, 1.0, 0.05, 0.2, opt)
    expected = 0.0 if opt == "call" else K * math.exp(-0.05) - S
    assert value == pytest.approx(expected, abs=1e-12)


def test_spot_to_infinity():
    S = 1e9
    call = bs.bs_price(S, K, 1.0, 0.05, 0.2, "call", 0.01)
    assert call == pytest.approx(S * math.exp(-0.01) - K * math.exp(-0.05), rel=1e-14)
    assert bs.bs_price(S, K, 1.0, 0.05, 0.2, "put", 0.01) == pytest.approx(0.0, abs=1e-12)


# --- API ------------------------------------------------------------------------------------


def test_scalar_inputs_return_float_and_arrays_broadcast():
    assert isinstance(bs.bs_price(100.0, 100.0, 1.0, 0.0, 0.2), float)
    out = bs.bs_price(np.array([90.0, 100.0]), np.array([[95.0], [105.0]]), 1.0, 0.0, 0.2)
    assert isinstance(out, np.ndarray) and out.shape == (2, 2)
    assert out[1, 0] == pytest.approx(bs.bs_price(90.0, 105.0, 1.0, 0.0, 0.2))


def test_d1_d2_and_forward():
    d1, d2 = bs.d1_d2(100.0, 100.0, 1.0, 0.0, 0.2)
    assert d1 == pytest.approx(0.1) and d2 == pytest.approx(-0.1)
    assert bs.d1_d2(100.0, 100.0, 1.0, 0.0, 0.0) == (0.0, 0.0)
    assert bs.d1_d2(100.0, 101.0, 1.0, 0.0, 0.0) == (-math.inf, -math.inf)
    assert bs.forward(100.0, 2.0, 0.05, 0.01) == pytest.approx(100.0 * math.exp(0.08))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S": 0.0},
        {"S": -1.0},
        {"K": 0.0},
        {"T": -0.1},
        {"sigma": -0.2},
        {"r": math.nan},
        {"q": math.inf},
        {"S": np.array([100.0, math.nan])},
    ],
)
def test_invalid_inputs_raise(kwargs):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "sigma": 0.2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        bs.bs_price(**args)


@pytest.mark.parametrize("bad", ["CALL", "straddle", ""])
def test_invalid_option_type_raises(bad):
    with pytest.raises(ValueError, match="option_type"):
        bs.bs_price(100.0, 100.0, 1.0, 0.03, 0.2, bad)
