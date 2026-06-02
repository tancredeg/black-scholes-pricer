"""
bs_pricer.py
============
Black-Scholes option pricing, Greeks analytics, GBM simulation,
delta-hedging engine, and implied volatility solver.

All functions are vectorised over (S, K) where applicable. ``bs_greeks_vec``
provides a fully vectorised Greeks computation; ``bs_greeks`` is the scalar
wrapper kept for backward compatibility.

Conventions
-----------
* Theta is expressed per TRADING DAY (divided by 252) so that
  ``theta_pnl = theta * dt`` with ``dt = 1/252`` directly.
* Vega is per 1-point (100%) move in sigma; multiply by 0.01 for a vol point.
* Rho is per 1-point (100%) move in r; multiply by 0.01 for a vol point.
* Continuous dividend yield ``q`` defaults to 0.0 (no dividend). For SPY,
  pass ``q ≈ 0.013`` (typical historical dividend yield).
* RNG: ``np.random.default_rng`` only — the legacy ``np.random.seed`` is
  forbidden throughout this codebase.

Mathematical references
-----------------------
Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. Journal of Political Economy, 81(3), 637-654.

Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of
Economics and Management Science, 4(1), 141-183. (Dividend yield extension.)

Hull, J. C. (2022). Options, Futures, and Other Derivatives (11th ed.).
Pearson. Chapters 15-19.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute the Black-Scholes-Merton d1 and d2 intermediate quantities.

    With continuous dividend yield ``q``:

        d1 = [ ln(S/K) + (r - q + sigma^2 / 2) * T ] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Parameters
    ----------
    S, K : float or np.ndarray
        Spot and strike prices. Must be positive.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float or np.ndarray
        Volatility (annualised). Must be non-negative.
    q : float, default 0.0
        Continuous dividend yield (annualised).

    Returns
    -------
    d1, d2 : float or np.ndarray

    Notes
    -----
    Degenerate cases (T <= 1e-9 or sigma <= 1e-9) return (+inf, +inf) when
    S >= K and (-inf, -inf) when S < K. This maps through N() to the
    correct deep-ITM / deep-OTM limit of delta.

    Examples
    --------
    >>> d1, d2 = _d1_d2(310.0, 310.0, 30/252, 0.05, 0.19)
    >>> round(float(d1), 6), round(float(d2), 6)
    (0.123576, 0.05802)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T_val = float(T)

    # Scalar degenerate fast-path
    if sigma.ndim == 0 and (T_val <= 1e-9 or float(sigma) <= 1e-9):
        itm = S >= K
        inf_val = np.where(itm, np.inf, -np.inf)
        return inf_val, inf_val

    denom = sigma * np.sqrt(T_val)
    itm = S >= K
    inf_val = np.where(itm, np.inf, -np.inf)
    safe = np.where(denom > 1e-14, denom, np.nan)
    d1_raw = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T_val) / safe
    d2_raw = d1_raw - denom
    d1 = np.where(np.isnan(d1_raw), inf_val, d1_raw)
    d2 = np.where(np.isnan(d2_raw), inf_val, d2_raw)

    if d1.ndim == 0:
        return float(d1), float(d2)
    return d1, d2


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Black-Scholes-Merton closed-form European option price.

    Call:  C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    Put:   P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

    Parameters
    ----------
    S, K : float or np.ndarray
        Spot and strike prices.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Implied volatility (annualised).
    option_type : {'call', 'put'}, default 'call'
    q : float, default 0.0
        Continuous dividend yield.

    Returns
    -------
    price : float or np.ndarray
        Shape matches broadcast(S, K). Returns a Python float for scalar input.

    Notes
    -----
    If T <= 1e-9 the function returns intrinsic value directly:
    max(S - K, 0) for calls and max(K - S, 0) for puts.

    Examples
    --------
    >>> round(float(bs_price(310.0, 310.0, 30/252, 0.05, 0.19, 'call')), 3)
    9.035
    >>> round(float(bs_price(310.0, 310.0, 30/252, 0.05, 0.19, 'put')), 3)
    7.195
    >>> # Dividend yield reduces the call price
    >>> round(float(bs_price(310.0, 310.0, 30/252, 0.05, 0.19, 'call', q=0.013)), 3)
    8.774
    """
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    otype = option_type.lower()

    if T <= 1e-9:
        if otype == "call":
            out = np.maximum(S_arr - K_arr, 0.0)
        elif otype == "put":
            out = np.maximum(K_arr - S_arr, 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        return float(out) if out.ndim == 0 else out

    d1, d2 = _d1_d2(S_arr, K_arr, T, r, sigma, q=q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if otype == "call":
        out = S_arr * disc_q * norm.cdf(d1) - K_arr * disc_r * norm.cdf(d2)
    elif otype == "put":
        out = K_arr * disc_r * norm.cdf(-d2) - S_arr * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


def bs_greeks_vec(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float | np.ndarray,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Vectorised Black-Scholes Greeks. Accepts arrays for S, K, sigma.

    Returns a dict whose values are np.ndarray (or 0-d ndarray for scalar input
    callers; use ``bs_greeks`` for a dict of Python floats).

    See ``bs_greeks`` for full documentation of conventions and edge cases.

    Examples
    --------
    >>> import numpy as np
    >>> g = bs_greeks_vec(np.array([280., 310., 340.]), 310.0, 30/252, 0.05, 0.19)
    >>> g['delta'].round(4).tolist()
    [0.0765, 0.5492, 0.9373]
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    otype = option_type.lower()
    if otype not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Broadcast all inputs to a common shape so the result shape is stable
    bshape = np.broadcast_shapes(S.shape, K.shape, sigma_arr.shape)
    S_b = np.broadcast_to(S, bshape).astype(float, copy=True)
    K_b = np.broadcast_to(K, bshape).astype(float, copy=True)
    sigma_b = np.broadcast_to(sigma_arr, bshape).astype(float, copy=True)

    expired_mask = T <= 1e-9
    if expired_mask:
        if otype == "call":
            delta = np.where(S_b >= K_b, 1.0, 0.0)
        else:
            delta = np.where(S_b < K_b, -1.0, 0.0)
        zeros = np.zeros(bshape)
        return {"delta": delta, "gamma": zeros, "theta": zeros, "vega": zeros, "rho": zeros}

    d1, d2 = _d1_d2(S_b, K_b, T, r, sigma_b, q=q)
    pdf_d1 = norm.pdf(d1)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    sqrt_T = np.sqrt(T)

    if otype == "call":
        delta = disc_q * norm.cdf(d1)
        raw_theta_annual = (
            -(S_b * disc_q * pdf_d1 * sigma_b) / (2.0 * sqrt_T)
            - r * K_b * disc_r * norm.cdf(d2)
            + q * S_b * disc_q * norm.cdf(d1)
        )
        rho = K_b * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)  # = -disc_q * N(-d1)
        raw_theta_annual = (
            -(S_b * disc_q * pdf_d1 * sigma_b) / (2.0 * sqrt_T)
            + r * K_b * disc_r * norm.cdf(-d2)
            - q * S_b * disc_q * norm.cdf(-d1)
        )
        rho = -K_b * T * disc_r * norm.cdf(-d2)

    gamma = (disc_q * pdf_d1) / (S_b * sigma_b * sqrt_T)
    vega = S_b * disc_q * sqrt_T * pdf_d1
    theta = raw_theta_annual / 252.0  # per trading day

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """
    Scalar Black-Scholes Greeks. Use ``bs_greeks_vec`` for array inputs.

    Parameters
    ----------
    S, K : float
    T : float (years)
    r : float (annualised continuous rate)
    sigma : float (annualised vol)
    option_type : {'call', 'put'}, default 'call'
    q : float, default 0.0
        Continuous dividend yield.

    Returns
    -------
    greeks : dict with keys 'delta', 'gamma', 'theta', 'vega', 'rho'.

    Notes
    -----
    THETA CONVENTION: per TRADING DAY (/252). The raw annual partial
    derivative is divided by 252 so ``theta_pnl = theta * dt`` with
    ``dt = 1/252`` works directly. Theta is negative: the option loses
    value as calendar time advances.

    VEGA: per 1-unit move in sigma. Multiply by 0.01 for a 1 vol-point move.

    RHO: per 1-unit move in r. Multiply by 0.01 for a 100 bp move.

    AT EXPIRY (T <= 1e-9): returns delta = 1 (call ITM) / 0 (call OTM) /
    -1 (put ITM) / 0 (put OTM); all other Greeks are zero.

    Examples
    --------
    >>> g = bs_greeks(310.0, 310.0, 30/252, 0.05, 0.19, 'call')
    >>> round(g['delta'], 4)
    0.5492
    >>> round(g['gamma'], 6)
    0.019481
    """
    g = bs_greeks_vec(S, K, T, r, sigma, option_type=option_type, q=q)
    return {k: float(v) for k, v in g.items()}


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------


def bs_implied_vol(
    C_market: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
    xtol: float = 1e-8,
) -> float:
    """
    Solve for the Black-Scholes implied volatility via Brent's method.

    Parameters
    ----------
    C_market : float
        Observed market price of the option.
    S, K, T, r, q, option_type : see ``bs_price``.
    sigma_lo, sigma_hi : float
        Bracket for Brent's root finder. Defaults span 1e-6 to 500% vol.
    xtol : float
        Absolute tolerance on sigma.

    Returns
    -------
    sigma : float
        Implied volatility such that ``bs_price(...) == C_market`` to ``xtol``.

    Raises
    ------
    ValueError
        If C_market is outside the arbitrage bounds [intrinsic, S*exp(-qT)]
        for calls or [intrinsic, K*exp(-rT)] for puts, or if Brent cannot
        bracket a root in [sigma_lo, sigma_hi].

    Notes
    -----
    Brent's method is chosen over Newton-Raphson because:
    * It is globally convergent provided the bracket contains a root.
    * Near-the-money options have vega that is well-behaved, but deep OTM
      or near-expiry options have vega -> 0, which breaks Newton iterations.

    Newton-Raphson is faster (typically 4-6 iterations vs 20-40 for Brent)
    but more fragile; for production a hybrid Newton-with-Brent-fallback is
    the standard. Here we prioritise robustness for a research-grade module.

    Examples
    --------
    >>> # Roundtrip: price an option then recover its vol
    >>> S, K, T, r, sigma_true = 310.0, 310.0, 30/252, 0.05, 0.19
    >>> C = bs_price(S, K, T, r, sigma_true, 'call')
    >>> sigma_iv = bs_implied_vol(C, S, K, T, r, 'call')
    >>> round(sigma_iv, 6) == round(sigma_true, 6)
    True
    """
    otype = option_type.lower()
    if otype not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if T <= 1e-9:
        raise ValueError("Cannot compute implied vol at or past expiry (T <= 0).")

    # Arbitrage bounds
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if otype == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r

    if C_market < lower - 1e-10 or C_market > upper + 1e-10:
        raise ValueError(
            f"Market price {C_market:.6f} outside arbitrage bounds "
            f"[{lower:.6f}, {upper:.6f}] for a {otype}."
        )

    def objective(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, otype, q=q)) - C_market

    f_lo = objective(sigma_lo)
    f_hi = objective(sigma_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Root not bracketed: f({sigma_lo})={f_lo:.6e}, "
            f"f({sigma_hi})={f_hi:.6e}. Widen the bracket."
        )

    return float(brentq(objective, sigma_lo, sigma_hi, xtol=xtol))


# ---------------------------------------------------------------------------
# PDE residual (internal consistency check)
# ---------------------------------------------------------------------------


def bs_pde_residual(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """
    Compute the Black-Scholes-Merton PDE residual.

    With dividends, the PDE is:
        Theta + 1/2 * sigma^2 * S^2 * Gamma + (r - q) * S * Delta - r * C = 0

    Theta here is the raw ANNUAL form (per-trading-day stored theta * 252).

    Residuals must be < 1e-10 across the valid parameter range. Larger
    residuals indicate a bug in one of the Greek formulas.

    Examples
    --------
    >>> abs(bs_pde_residual(310.0, 310.0, 30/252, 0.05, 0.19)) < 1e-10
    True
    >>> abs(bs_pde_residual(310.0, 310.0, 30/252, 0.05, 0.19, q=0.013)) < 1e-10
    True
    >>> abs(bs_pde_residual(310.0, 290.0, 90/252, 0.05, 0.25, 'put', q=0.02)) < 1e-10
    True
    """
    g = bs_greeks(S, K, T, r, sigma, option_type=option_type, q=q)
    C = bs_price(S, K, T, r, sigma, option_type=option_type, q=q)

    theta_annual = g["theta"] * 252.0
    return float(
        theta_annual
        + 0.5 * sigma ** 2 * S ** 2 * g["gamma"]
        + (r - q) * S * g["delta"]
        - r * C
    )


# ---------------------------------------------------------------------------
# GBM simulation
# ---------------------------------------------------------------------------


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate GBM paths under the log-exact discretisation:

        S_{t+dt} = S_t * exp[(mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z]

    Parameters
    ----------
    S0 : float
    mu : float
        Drift. Use the real-world drift to model observed dynamics under P,
        or use (r - q) to simulate under the risk-neutral measure Q (e.g.,
        for Monte Carlo pricing validation).
    sigma : float
    T : float
    N_steps : int
    N_paths : int
    seed : int

    Returns
    -------
    paths : np.ndarray, shape (N_paths, N_steps + 1)

    Notes
    -----
    The log-exact discretisation has zero discretisation bias for the
    log-normal marginal at each step. Euler-Maruyama on dS/S has O(dt) bias.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 5, seed=42)
    >>> paths.shape
    (5, 31)
    >>> bool(np.all(paths[:, 0] == 310.0))
    True
    """
    rng = np.random.default_rng(seed)
    dt = T / N_steps
    Z = rng.standard_normal((N_paths, N_steps))
    log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.hstack([np.zeros((N_paths, 1)), log_inc])
    log_paths = np.cumsum(log_paths, axis=1)
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Delta hedging
# ---------------------------------------------------------------------------


def delta_hedge_single_path(
    S_path: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    dt: float = 1.0 / 252,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Discrete delta-hedging on one price path (option seller's perspective).

    Strategy:
    * t=0: sell the option (receive C0), buy delta_0 shares (long delta_0
           when call, short -delta_0 when put), invest/borrow net cash at r.
    * t=k (k=1..N-1): recompute delta_k, rebalance, cash grows by exp(r*dt).
    * t=T: liquidate the share position, settle the option payoff.
           hedging_error = portfolio_value - payoff.

    Positive hedging_error means the hedge over-performed.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 1, seed=42)
    >>> _, _, err = delta_hedge_single_path(paths[0], 310.0, 30/252, 0.05, 0.19)
    >>> bool(abs(err) < 5.0)
    True
    """
    N_steps = len(S_path) - 1
    cash = np.zeros(N_steps + 1)
    deltas = np.zeros(N_steps + 1)

    # t=0
    T_rem = T_total
    C0 = float(bs_price(S_path[0], K, T_rem, r, sigma, option_type, q=q))
    delta_0 = float(bs_greeks(S_path[0], K, T_rem, r, sigma, option_type, q=q)["delta"])
    deltas[0] = delta_0
    cash[0] = C0 - delta_0 * S_path[0]

    # t=1..N-1
    for k in range(1, N_steps):
        T_rem = max(T_total - k * dt, 1e-9)
        cash[k] = cash[k - 1] * np.exp(r * dt)
        d_new = float(bs_greeks(S_path[k], K, T_rem, r, sigma, option_type, q=q)["delta"])
        cash[k] -= (d_new - deltas[k - 1]) * S_path[k]
        deltas[k] = d_new

    # Settlement at t=T
    cash_final = cash[N_steps - 1] * np.exp(r * dt)
    delta_last = deltas[N_steps - 1]
    portfolio_value = cash_final + delta_last * S_path[N_steps]

    if option_type.lower() == "call":
        payoff = max(S_path[N_steps] - K, 0.0)
    else:
        payoff = max(K - S_path[N_steps], 0.0)

    hedging_error = portfolio_value - payoff
    cash[N_steps] = cash_final
    deltas[N_steps] = delta_last

    return cash, deltas, hedging_error


def run_hedging_experiment(
    S_paths: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    dt: float = 1.0 / 252,
    q: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """
    Scalar-loop wrapper around ``delta_hedge_single_path``. O(N_paths * N_steps)
    Python calls. Kept for backward compatibility and as a reference
    implementation that ``run_hedging_experiment_vec`` is asserted against.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 100, seed=42)
    >>> errors, rmse, _ = run_hedging_experiment(paths, 310.0, 30/252, 0.05, 0.19)
    >>> bool(rmse < 3.0)
    True
    """
    N_paths = S_paths.shape[0]
    errors = np.zeros(N_paths)
    for i in range(N_paths):
        _, _, err = delta_hedge_single_path(
            S_paths[i], K, T_total, r, sigma, option_type, dt, q=q
        )
        errors[i] = err
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return errors, rmse, float(np.mean(errors))


def run_hedging_experiment_vec(
    S_paths: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    dt: float = 1.0 / 252,
    q: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """
    Vectorised delta-hedging experiment. Loops over time steps but
    vectorises across paths -> ~100x speedup vs ``run_hedging_experiment``.

    All paths are propagated simultaneously; at each step we compute the
    delta vector for every path in one ``bs_greeks_vec`` call.

    Returns the same triple ``(errors, rmse, mean_error)`` as the scalar
    version, to within floating-point noise.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 100, seed=42)
    >>> e_scalar, rmse_s, _ = run_hedging_experiment(paths, 310.0, 30/252, 0.05, 0.19)
    >>> e_vec, rmse_v, _ = run_hedging_experiment_vec(paths, 310.0, 30/252, 0.05, 0.19)
    >>> bool(np.allclose(e_scalar, e_vec, atol=1e-10))
    True
    """
    N_paths, n_plus_one = S_paths.shape
    N_steps = n_plus_one - 1
    otype = option_type.lower()

    # t=0
    S0 = S_paths[:, 0]
    T_rem = T_total
    C0 = bs_price(S0, K, T_rem, r, sigma, otype, q=q)
    d_prev = bs_greeks_vec(S0, K, T_rem, r, sigma, option_type=otype, q=q)["delta"]
    cash = C0 - d_prev * S0  # shape (N_paths,)
    growth = np.exp(r * dt)

    # t=1..N-1
    for k in range(1, N_steps):
        T_rem = max(T_total - k * dt, 1e-9)
        cash = cash * growth
        Sk = S_paths[:, k]
        d_new = bs_greeks_vec(Sk, K, T_rem, r, sigma, option_type=otype, q=q)["delta"]
        cash = cash - (d_new - d_prev) * Sk
        d_prev = d_new

    # Settlement at t=T
    cash = cash * growth
    ST = S_paths[:, -1]
    portfolio = cash + d_prev * ST
    if otype == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
    errors = portfolio - payoff
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return errors, rmse, float(np.mean(errors))


# ---------------------------------------------------------------------------
# Empirical volatility
# ---------------------------------------------------------------------------


def rolling_realized_vol(
    prices: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Annualised rolling realised volatility (close-to-close estimator).

    Parameters
    ----------
    prices : pd.Series
        Adjusted close prices (datetime-indexed).
    windows : list of int, default [30, 60, 252]

    Returns
    -------
    vol_df : pd.DataFrame
        Columns 'vol_{w}d', annualised by sqrt(252).

    Notes
    -----
    This is the simplest realised-vol estimator. For OHLC data, range-based
    estimators (Parkinson, Garman-Klass, Yang-Zhang) are ~5x more efficient
    for the same sample size and are preferred in practice. They are not
    implemented here because the SPY CSV used in the notebook is close-only.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    >>> vols = rolling_realized_vol(prices, windows=[30, 60])
    >>> vols.columns.tolist()
    ['vol_30d', 'vol_60d']
    """
    if windows is None:
        windows = [30, 60, 252]
    log_ret = np.log(prices / prices.shift(1))
    vol_dict: dict[str, pd.Series] = {}
    for w in windows:
        vol_dict[f"vol_{w}d"] = log_ret.rolling(window=w).std(ddof=1) * np.sqrt(252)
    return pd.DataFrame(vol_dict, index=prices.index)
