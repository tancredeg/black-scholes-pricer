"""Implied volatility: round trips and explicit failure modes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import GRID, OPTION_TYPES, STRIKE

import bspricer as bs

K = STRIKE


@pytest.mark.parametrize(("S", "T", "r", "sigma", "q"), GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_round_trip(S, T, r, sigma, q, opt):
    """IV(price(sigma)) = sigma, to the precision the price can carry.

    A price is stored with ~1e-16 relative precision, so sigma is determined to about
    eps * price / vega. When the time value is below the price resolution the solver
    must refuse (ImpliedVolError) rather than return an arbitrary volatility.
    """
    price = bs.bs_price(S, K, T, r, sigma, opt, q)
    lower, _ = bs.no_arbitrage_bounds(S, K, T, r, opt, q)
    vega = bs.bs_greeks(S, K, T, r, sigma, opt, q)["vega"]
    if price - lower <= 1e-12 * max(S, K):
        with pytest.raises(bs.ImpliedVolError, match="not identifiable"):
            bs.bs_implied_vol(price, S, K, T, r, opt, q)
        return
    implied = bs.bs_implied_vol(price, S, K, T, r, opt, q)
    resolution = 1e-12 * max(S, K) / vega if vega > 0 else math.inf
    assert abs(implied - sigma) <= max(1e-9, 10 * resolution)


def test_deep_itm_short_dated_is_rejected_not_misreported():
    """Pre-v1.0 code returned sigma = 1e-6 for this price, generated with sigma = 0.19 (AUDIT M2)."""
    S, Kd, T, r = 420.0, 310.0, 5 / 252, 0.05
    price = bs.bs_price(S, Kd, T, r, 0.19, "call")
    with pytest.raises(bs.ImpliedVolError, match="not identifiable"):
        bs.bs_implied_vol(price, S, Kd, T, r, "call")


@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_prices_outside_bounds_raise(opt):
    S, T, r = 100.0, 0.5, 0.03
    lower, upper = bs.no_arbitrage_bounds(S, K, T, r, opt)
    with pytest.raises(bs.ImpliedVolError, match="outside the no-arbitrage bounds"):
        bs.bs_implied_vol(upper + 1e-6, S, K, T, r, opt)
    with pytest.raises(bs.ImpliedVolError, match="outside the no-arbitrage bounds"):
        bs.bs_implied_vol(lower - 1e-6, S, K, T, r, opt)
    with pytest.raises(bs.ImpliedVolError, match="upper bound"):
        bs.bs_implied_vol(upper, S, K, T, r, opt)


def test_volatility_above_search_bracket_raises():
    price = bs.bs_price(100.0, 100.0, 1.0, 0.0, 3.0)
    assert bs.bs_implied_vol(price, 100.0, 100.0, 1.0, 0.0) == pytest.approx(3.0, abs=1e-10)
    with pytest.raises(bs.ImpliedVolError, match="sigma_max"):
        bs.bs_implied_vol(price, 100.0, 100.0, 1.0, 0.0, sigma_max=2.0)


def test_implied_vol_is_monotone_in_price():
    prices = np.linspace(2.5, 20.0, 50)  # lower bound is 100 (1 - e^{-0.02}) = 1.98
    vols = [bs.bs_implied_vol(p, 100.0, 100.0, 1.0, 0.02) for p in prices]
    assert np.all(np.diff(vols) > 0)


@pytest.mark.parametrize("kwargs", [{"T": 0.0}, {"S": -1.0}, {"price": math.nan}])
def test_invalid_inputs_raise(kwargs):
    args = {"price": 5.0, "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        bs.bs_implied_vol(**args)
