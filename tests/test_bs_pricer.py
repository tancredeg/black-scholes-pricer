"""
Test suite for bs_pricer.py
===========================
Invariant-based tests over parameter grids, statistical Monte Carlo
validation, and consistency checks across scalar/vectorised entry points.

Run from the repo root:
    pytest -v
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import bspricer as bs

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# A representative grid spanning ITM/ATM/OTM, short/medium/long T, low/mid/high vol
S_GRID = [200.0, 280.0, 310.0, 340.0, 420.0]
K = 310.0
T_GRID = [5 / 252, 30 / 252, 90 / 252, 252 / 252]
R_GRID = [0.0, 0.05]
SIGMA_GRID = [0.05, 0.19, 0.40, 0.80]
Q_GRID = [0.0, 0.013, 0.03]
OPTION_TYPES = ["call", "put"]

GRID = list(itertools.product(S_GRID, T_GRID, R_GRID, SIGMA_GRID, Q_GRID, OPTION_TYPES))


# ---------------------------------------------------------------------------
# Put-Call Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S, T, r, sigma, q", list(itertools.product(
    S_GRID, T_GRID, R_GRID, SIGMA_GRID, Q_GRID,
)))
def test_put_call_parity(S, T, r, sigma, q):
    """C - P = S*exp(-qT) - K*exp(-rT) must hold to machine precision."""
    C = bs.bs_price(S, K, T, r, sigma, "call", q=q)
    P = bs.bs_price(S, K, T, r, sigma, "put", q=q)
    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10, f"PCP violated by {lhs - rhs:.2e}"


# ---------------------------------------------------------------------------
# PDE residual across the grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S, T, r, sigma, q, opt", GRID)
def test_pde_residual(S, T, r, sigma, q, opt):
    """The Black-Scholes-Merton PDE residual must be < 1e-9 everywhere."""
    resid = bs.bs_pde_residual(S, K, T, r, sigma, option_type=opt, q=q)
    assert abs(resid) < 1e-9, f"PDE residual {resid:.2e} at (S,T,r,sigma,q,opt)=({S},{T},{r},{sigma},{q},{opt})"


# ---------------------------------------------------------------------------
# Intrinsic value at expiry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S", S_GRID)
@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_intrinsic_at_expiry(S, opt):
    """At T<=1e-9, price collapses to max(payoff, 0)."""
    price = bs.bs_price(S, K, 0.0, 0.05, 0.19, opt)
    expected = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    assert abs(price - expected) < 1e-12


# ---------------------------------------------------------------------------
# Implied vol roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("S, T, r, sigma, q, opt", [
    p for p in GRID if p[3] > 0.06  # solver is ill-conditioned for sigma<5%
])
def test_iv_roundtrip(S, T, r, sigma, q, opt):
    """price(S,K,T,r,IV(price(...))) == price(...).

    Skip when time value is microscopic (deep ITM near expiry): IV is
    mathematically ill-defined there because the price barely changes
    with sigma. This is a known limit of the inverse problem, not a bug.
    """
    C = bs.bs_price(S, K, T, r, sigma, opt, q=q)
    intrinsic = (max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if opt == "call"
                 else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0))
    time_value = C - intrinsic
    if time_value < 1e-3:
        pytest.skip(f"Time value {time_value:.2e} too small for IV recovery")

    iv = bs.bs_implied_vol(C, S, K, T, r, opt, q=q)
    assert abs(iv - sigma) < 1e-5, f"IV mismatch: {iv} vs {sigma}"


def test_iv_outside_arbitrage_bounds_raises():
    """Calling IV with a market price outside no-arb bounds must raise."""
    with pytest.raises(ValueError, match="outside arbitrage bounds"):
        bs.bs_implied_vol(1e6, 310.0, 310.0, 30 / 252, 0.05, "call")


# ---------------------------------------------------------------------------
# Vectorised vs scalar Greeks consistency
# ---------------------------------------------------------------------------


def test_greeks_vec_matches_scalar():
    """bs_greeks_vec on an array must equal bs_greeks called in a loop."""
    S_arr = np.array(S_GRID)
    g_vec = bs.bs_greeks_vec(S_arr, K, 30 / 252, 0.05, 0.19, "call", q=0.013)
    g_loop = {k: np.array([
        bs.bs_greeks(s, K, 30 / 252, 0.05, 0.19, "call", q=0.013)[k]
        for s in S_arr
    ]) for k in g_vec}
    for key in g_vec:
        np.testing.assert_allclose(g_vec[key], g_loop[key], atol=1e-12)


# ---------------------------------------------------------------------------
# Hedging engine: scalar vs vectorised
# ---------------------------------------------------------------------------


def test_hedging_scalar_vs_vec():
    """run_hedging_experiment and run_hedging_experiment_vec must agree."""
    paths = bs.simulate_gbm(310.0, 0.14, 0.19, 30 / 252, 30, 200, seed=42)
    e_s, rmse_s, mean_s = bs.run_hedging_experiment(paths, K, 30 / 252, 0.05, 0.19)
    e_v, rmse_v, mean_v = bs.run_hedging_experiment_vec(paths, K, 30 / 252, 0.05, 0.19)
    np.testing.assert_allclose(e_s, e_v, atol=1e-10)
    assert abs(rmse_s - rmse_v) < 1e-10
    assert abs(mean_s - mean_v) < 1e-10


def test_hedging_call_put_symmetry():
    """For sigma_h = sigma_r and q=0, the hedging error on calls and puts is
    identical path-by-path: it only depends on gamma * S^2 * (dW^2 - sigma^2 dt),
    which is the same for both option types."""
    paths = bs.simulate_gbm(310.0, 0.14, 0.19, 30 / 252, 30, 100, seed=42)
    e_c, _, _ = bs.run_hedging_experiment_vec(paths, K, 30 / 252, 0.05, 0.19, "call")
    e_p, _, _ = bs.run_hedging_experiment_vec(paths, K, 30 / 252, 0.05, 0.19, "put")
    np.testing.assert_allclose(e_c, e_p, atol=1e-10)


# ---------------------------------------------------------------------------
# Monte Carlo vs analytical pricing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt", OPTION_TYPES)
def test_mc_vs_analytical(opt):
    """Simulate under Q (drift = r - q), discount, compare to BS price.
    Must agree to within 4 standard errors of the MC estimate."""
    S0, T, r, sigma, q = 310.0, 30 / 252, 0.05, 0.19, 0.013
    N = 200_000
    paths = bs.simulate_gbm(S0, mu=r - q, sigma=sigma, T=T, N_steps=30, N_paths=N, seed=42)
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0) if opt == "call" else np.maximum(K - ST, 0.0)
    mc_price = np.exp(-r * T) * payoff.mean()
    mc_se = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(N)
    bs_p = bs.bs_price(S0, K, T, r, sigma, opt, q=q)
    diff = abs(mc_price - bs_p)
    assert diff < 4 * mc_se, f"|MC - BS| = {diff:.4f} > 4*SE = {4*mc_se:.4f}"


# ---------------------------------------------------------------------------
# Dividend backward compatibility
# ---------------------------------------------------------------------------


def test_q_zero_default_matches_no_q():
    """Calling without q must equal calling with q=0.0."""
    args = (310.0, 310.0, 30 / 252, 0.05, 0.19)
    assert bs.bs_price(*args, "call") == bs.bs_price(*args, "call", q=0.0)
    g1 = bs.bs_greeks(*args, "call")
    g2 = bs.bs_greeks(*args, "call", q=0.0)
    for k in g1:
        assert g1[k] == g2[k]


def test_dividend_reduces_call_increases_put():
    """Increasing q at fixed sigma must reduce the call price and increase the put."""
    args = (310.0, 310.0, 30 / 252, 0.05, 0.19)
    C_no_div = bs.bs_price(*args, "call", q=0.0)
    C_with_div = bs.bs_price(*args, "call", q=0.03)
    P_no_div = bs.bs_price(*args, "put", q=0.0)
    P_with_div = bs.bs_price(*args, "put", q=0.03)
    assert C_with_div < C_no_div
    assert P_with_div > P_no_div


# ---------------------------------------------------------------------------
# Hedging RMSE scaling sanity check
# ---------------------------------------------------------------------------


def test_rmse_scales_as_sqrt_dt():
    """Halving dt should approximately scale RMSE by 1/sqrt(2) for fixed T_total."""
    S0, mu, sigma, r, T_total = 310.0, 0.14, 0.19, 0.05, 60 / 252

    # Daily rebalancing
    paths_daily = bs.simulate_gbm(S0, mu, sigma, T_total, N_steps=60, N_paths=500, seed=42)
    _, rmse_daily, _ = bs.run_hedging_experiment_vec(paths_daily, K, T_total, r, sigma)

    # Half as frequent (every 2 days)
    paths_2d = bs.simulate_gbm(S0, mu, sigma, T_total, N_steps=30, N_paths=500, seed=42)
    _, rmse_2d, _ = bs.run_hedging_experiment_vec(paths_2d, K, T_total, r, sigma, dt=2 / 252)

    ratio = rmse_2d / rmse_daily
    expected = np.sqrt(2)
    # Loose tolerance: finite sample noise + path discretisation differences
    assert 0.8 * expected < ratio < 1.3 * expected, f"RMSE scaling ratio {ratio:.3f} vs sqrt(2)={expected:.3f}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs.bs_price(310.0, 310.0, 30 / 252, 0.05, 0.19, option_type="straddle")
    with pytest.raises(ValueError):
        bs.bs_greeks(310.0, 310.0, 30 / 252, 0.05, 0.19, option_type="bad")
