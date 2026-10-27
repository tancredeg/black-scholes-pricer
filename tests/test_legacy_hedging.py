"""Legacy hedging-engine tests, kept until the engine is rewritten (next commit)."""
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


