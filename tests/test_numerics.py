"""Numerical methods converge to the closed form: CRR tree (order 1) and Monte Carlo (order 1/2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import bspricer as bs

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


# --- GBM simulation -------------------------------------------------------------------------


def test_gbm_shape_reproducibility_and_moments():
    paths = bs.simulate_gbm(S, 0.08, SIGMA, T, 12, 100_000, seed=0)
    assert paths.shape == (100_000, 13) and np.all(paths[:, 0] == S)
    np.testing.assert_array_equal(paths, bs.simulate_gbm(S, 0.08, SIGMA, T, 12, 100_000, seed=0))
    terminal = paths[:, -1]
    se = terminal.std(ddof=1) / math.sqrt(terminal.size)
    assert abs(terminal.mean() - S * math.exp(0.08 * T)) < 4 * se
    log_inc = np.diff(np.log(paths), axis=1)
    assert log_inc.var(ddof=1) == pytest.approx(SIGMA**2 / 12, rel=0.01)


def test_gbm_antithetic_pairs():
    paths = bs.simulate_gbm(S, 0.0, SIGMA, T, 4, 10, seed=1, antithetic=True)
    drift = -0.5 * SIGMA**2 * T / 4
    inc = np.diff(np.log(paths), axis=1) - drift
    np.testing.assert_allclose(inc[:5], -inc[5:], atol=1e-14)
    with pytest.raises(ValueError):
        bs.simulate_gbm(S, 0.0, SIGMA, T, 4, 9, seed=1, antithetic=True)


# --- Monte Carlo pricing ------------------------------------------------------------------------


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_within_four_standard_errors(opt, antithetic):
    est = bs.mc_price(S, 110.0, T, R, SIGMA, 200_000, seed=42, option_type=opt, q=0.02, antithetic=antithetic)
    exact = bs.bs_price(S, 110.0, T, R, SIGMA, opt, 0.02)
    assert abs(est.price - exact) < 4 * est.std_error


def test_antithetic_reduces_standard_error():
    plain = bs.mc_price(S, K, T, R, SIGMA, 100_000, seed=1)
    anti = bs.mc_price(S, K, T, R, SIGMA, 100_000, seed=1, antithetic=True)
    assert anti.std_error < 0.8 * plain.std_error


@pytest.mark.slow
def test_mc_error_decays_like_inverse_sqrt_paths():
    """RMSE over 200 seeds for 1k ... 64k paths; log-log slope must be close to -1/2 (measured -0.49)."""
    exact = bs.bs_price(S, K, T, R, SIGMA)
    sizes = np.array([1_000, 4_000, 16_000, 64_000])
    rmse = [
        math.sqrt(np.mean([(bs.mc_price(S, K, T, R, SIGMA, int(n), seed=s).price - exact) ** 2 for s in range(200)]))
        for n in sizes
    ]
    slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
    assert -0.6 < slope < -0.4


# --- CRR binomial tree ------------------------------------------------------------------------


@pytest.mark.slow
def test_crr_converges_at_order_one_at_the_money():
    """|CRR(n) - BS| ~ c / n: log-log slope -1 for even n at the money (measured -0.999)."""
    exact = bs.bs_price(S, K, T, R, SIGMA)
    steps = np.array([50, 100, 200, 400, 800, 1600, 3200])
    errors = [abs(bs.crr_price(S, K, T, R, SIGMA, int(n)) - exact) for n in steps]
    slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert -1.1 < slope < -0.9


@pytest.mark.parametrize(("strike", "opt", "q"), [(110.0, "put", 0.02), (90.0, "put", 0.03), (120.0, "call", 0.0)])
def test_crr_error_bounded_by_c_over_n_away_from_the_money(strike, opt, q):
    """Away from the money the error oscillates with n, but stays below 2 S sigma^2 T / n here."""
    exact = bs.bs_price(S, strike, T, R, SIGMA, opt, q)
    for n in (100, 400, 1600):
        assert abs(bs.crr_price(S, strike, T, R, SIGMA, n, opt, q) - exact) < 2 * S * SIGMA**2 * T / n


def test_crr_american_options():
    euro_put = bs.crr_price(S, K, T, R, SIGMA, 1000, "put")
    amer_put = bs.crr_price(S, K, T, R, SIGMA, 1000, "put", american=True)
    assert amer_put > euro_put + 0.1  # early-exercise premium with r > 0
    euro_call = bs.crr_price(S, K, T, R, SIGMA, 1000, "call")
    amer_call = bs.crr_price(S, K, T, R, SIGMA, 1000, "call", american=True)
    assert amer_call == pytest.approx(euro_call, abs=1e-12)  # never optimal to exercise early if q = 0


def test_crr_rejects_invalid_trees():
    with pytest.raises(ValueError, match="probability"):
        bs.crr_price(S, K, 10.0, 0.5, 0.01, 2)
    with pytest.raises(ValueError):
        bs.crr_price(S, K, T, R, SIGMA, 0)
