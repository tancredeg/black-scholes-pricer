"""Discrete delta hedging: oracle agreement, unbiasedness, exact attribution, convergence."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import reference as ref

import bspricer as bs

S0, K, T, R, SIGMA, MU = 100.0, 100.0, 0.25, 0.03, 0.20, 0.07


@pytest.fixture(scope="module")
def paths() -> np.ndarray:
    return bs.simulate_gbm(S0, MU, SIGMA, T, 63, 20_000, seed=11)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_matches_scalar_reference(opt):
    """Vectorised engine == independent scalar loop (dividends and costs included)."""
    sample = bs.simulate_gbm(S0, MU - 0.02, SIGMA, T, 21, 40, seed=3)
    res = bs.delta_hedge(sample, 105.0, T, R, 0.25, opt, q=0.02, cost=5e-4)
    expected = [ref.hedge_path(list(p), 105.0, T, R, 0.25, opt == "call", 0.02, 5e-4) for p in sample]
    np.testing.assert_allclose(res.pnl, expected, rtol=0, atol=1e-9)


def test_unbiased_when_hedging_at_the_true_volatility(paths):
    res = bs.delta_hedge(paths, K, T, R, SIGMA)
    assert abs(res.mean) < 4 * res.std_error


@pytest.mark.parametrize("q", [0.013, 0.03])
def test_dividends_are_credited(q):
    """Regression for AUDIT C4: with q > 0 the old engine was biased by -0.29 (t = -31.7)."""
    p = bs.simulate_gbm(S0, MU - q, SIGMA, T, 63, 20_000, seed=7)
    res = bs.delta_hedge(p, K, T, R, SIGMA, q=q)
    assert abs(res.mean) < 4 * res.std_error


def test_time_step_comes_from_the_grid():
    """Regression for AUDIT M1: there is no dt argument that could disagree with the paths."""
    assert "dt" not in inspect.signature(bs.delta_hedge).parameters
    p = bs.simulate_gbm(S0, MU, SIGMA, 60 / 252, 30, 20_000, seed=5)
    res = bs.delta_hedge(p, K, 60 / 252, R, SIGMA)
    assert res.dt == pytest.approx(2 / 252) and res.n_steps == 30
    assert abs(res.mean) < 4 * res.std_error


def test_call_and_put_hedges_coincide_without_dividends(paths):
    """Short call - short put = short forward, statically hedged by one share when q = 0."""
    sub = paths[:500]
    call = bs.delta_hedge(sub, K, T, R, 0.25, "call")
    put = bs.delta_hedge(sub, K, T, R, 0.25, "put")
    np.testing.assert_allclose(call.pnl, put.pnl, atol=1e-10)


def test_decomposition_is_exact(paths):
    res = bs.delta_hedge(paths[:2_000], 95.0, T, R, 0.18, "put", q=0.01, sigma_price=0.22, cost=2e-4, decompose=True)
    assert res.decomposition is not None
    np.testing.assert_allclose(res.decomposition.total(), res.pnl, atol=1e-10)
    np.testing.assert_allclose(res.decomposition.costs, res.costs)


def test_gamma_theta_carry_equal_dollar_gamma_times_variance_gap(paths):
    """Gamma + theta + carry = sum 1/2 Gamma S^2 (sigma_h^2 dt - dS^2/S^2) up to O(dt^2)."""
    sub = paths[:2_000]
    d = bs.delta_hedge(sub, K, T, R, 0.25, decompose=True).decomposition
    assert d is not None
    proxy = bs.variance_pnl(sub, K, T, R, 0.25)
    assert np.max(np.abs(d.gamma + d.theta + d.carry - proxy)) < 5e-3 * proxy.std()


def test_selling_rich_shifts_pnl_by_the_premium_difference(paths):
    sub = paths[:1_000]
    fair = bs.delta_hedge(sub, K, T, R, SIGMA)
    rich = bs.delta_hedge(sub, K, T, R, SIGMA, sigma_price=0.25)
    edge = (bs.bs_price(S0, K, T, R, 0.25) - bs.bs_price(S0, K, T, R, SIGMA)) * math.exp(R * T)
    np.testing.assert_allclose(rich.pnl - fair.pnl, edge, atol=1e-10)


def test_transaction_costs(paths):
    sub = paths[:1_000]
    free = bs.delta_hedge(sub, K, T, R, SIGMA)
    paid = bs.delta_hedge(sub, K, T, R, SIGMA, cost=1e-3)
    np.testing.assert_allclose(free.pnl - paid.pnl, paid.costs, rtol=1e-9)
    assert np.all(paid.costs > 0)
    coarse = bs.delta_hedge(sub[:, ::9], K, T, R, SIGMA, cost=1e-3)
    assert coarse.costs.mean() < paid.costs.mean()  # fewer rebalances, less turnover


def test_hedging_at_realised_vol_locks_in_the_vol_spread():
    """Ahmad-Wilmott (2005): sell at 25 %, realised 20 %, rebalance 126 and 504 times.

    Hedging with the realised vol, the P&L converges to (V(25 %) - V(20 %)) e^{rT} and its
    sd is pure discretisation noise, halving when N is multiplied by 4 (measured 0.306 ->
    0.155). Hedging with the implied vol, a path-dependent part remains (0.455 -> 0.368).
    """
    sell = 0.25
    fine = bs.simulate_gbm(S0, MU, SIGMA, T, 504, 10_000, seed=12)
    target = (bs.bs_price(S0, K, T, R, sell) - bs.bs_price(S0, K, T, R, SIGMA)) * math.exp(R * T)
    implied = {n: bs.delta_hedge(fine[:, :: 504 // n], K, T, R, sell) for n in (126, 504)}
    realised = {n: bs.delta_hedge(fine[:, :: 504 // n], K, T, R, SIGMA, sigma_price=sell) for n in (126, 504)}
    assert abs(realised[504].mean - target) < 4 * realised[504].std_error
    assert realised[504].std < 0.5 * implied[504].std
    assert 0.4 < realised[504].std / realised[126].std < 0.6
    assert implied[504].std / implied[126].std > 0.75
    expected = bs.variance_pnl(fine, K, T, R, sell, sigma_realized=SIGMA).mean()
    assert abs(implied[504].mean - expected) < 4 * implied[504].std_error


@pytest.mark.parametrize("sigma_hedge", [0.15, 0.25])
def test_vol_mismatch_pnl_is_explained_by_the_gamma_weighted_variance_gap(paths, sigma_hedge):
    sub = paths[:5_000]
    res = bs.delta_hedge(sub, K, T, R, sigma_hedge)
    proxy = bs.variance_pnl(sub, K, T, R, sigma_hedge)
    slope, _ = np.polyfit(proxy, res.pnl, 1)
    r2 = np.corrcoef(proxy, res.pnl)[0, 1] ** 2
    assert 0.9 < slope < 1.1 and r2 > 0.9


def test_kamal_derman_formula():
    vega = 20.0
    assert bs.kamal_derman_std(vega, 0.2, 100) == pytest.approx(math.sqrt(math.pi / 4) * 0.2 * vega / 10)
    with pytest.raises(ValueError):
        bs.kamal_derman_std(vega, 0.2, 0)


@pytest.mark.slow
def test_replication_error_decays_like_inverse_sqrt_n():
    """Common random numbers: the same Brownian paths observed at N = 8 ... 512 dates.

    Fitting log sd(P&L) = a + b log N, theory (Kamal-Derman 1999) predicts b = -1/2.
    """
    fine = bs.simulate_gbm(S0, MU, SIGMA, T, 512, 10_000, seed=2026)
    vega = bs.bs_greeks(S0, K, T, R, SIGMA)["vega"]
    ns = np.array([8, 16, 32, 64, 128, 256, 512])
    stds = np.array([bs.delta_hedge(fine[:, :: 512 // n], K, T, R, SIGMA).std for n in ns])
    slope = np.polyfit(np.log(ns), np.log(stds), 1)[0]
    assert -0.55 < slope < -0.45
    ratio = stds / np.array([bs.kamal_derman_std(vega, SIGMA, int(n)) for n in ns])
    assert np.all((ratio > 0.85) & (ratio < 1.1))


def test_input_validation():
    with pytest.raises(ValueError):
        bs.delta_hedge(np.array([[100.0, -1.0]]), K, T, R, SIGMA)
    with pytest.raises(ValueError):
        bs.delta_hedge(np.array([[100.0]]), K, T, R, SIGMA)
    with pytest.raises(ValueError):
        bs.delta_hedge(np.array([[100.0, 101.0]]), K, T, R, 0.0)
    with pytest.raises(ValueError):
        bs.delta_hedge(np.array([[100.0, 101.0]]), K, T, R, SIGMA, cost=-1e-4)
    one_path = bs.delta_hedge(np.array([100.0, 101.0, 99.0]), K, T, R, SIGMA, keep_deltas=True)
    assert one_path.pnl.shape == (1,) and one_path.deltas is not None
    assert one_path.deltas.shape == (1, 2)
    assert one_path.deltas[0, 0] == pytest.approx(bs.bs_greeks(S0, K, T, R, SIGMA)["delta"])
