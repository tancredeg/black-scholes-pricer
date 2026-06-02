"""
bs_pricer.py
============
Black-Scholes option pricing, Greeks analytics, GBM simulation,
and delta-hedging engine.

All functions are vectorized where applicable (numpy arrays accepted for S, K).
Theta convention: per TRADING DAY (÷252) throughout.
RNG: np.random.default_rng only — legacy np.random.seed is forbidden.

Mathematical references
-----------------------
Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. Journal of Political Economy, 81(3), 637–654.

Hull, J. C. (2022). Options, Futures, and Other Derivatives (11th ed.).
Pearson. Chapters 15–19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute the Black-Scholes d1 and d2 intermediate quantities.

    Parameters
    ----------
    S : float or np.ndarray
        Current spot price(s). Must be positive.
    K : float or np.ndarray
        Strike price(s). Must be positive.
    T : float
        Time to expiry in years (T > 0 expected; see Notes for T ≤ 0).
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Volatility (annualised). Must be non-negative.

    Returns
    -------
    d1 : float or np.ndarray
        d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 : float or np.ndarray
        d2 = d1 − σ·√T

    Notes
    -----
    **Edge cases (T ≤ 1e-9 or sigma ≤ 1e-9)**
    The denominator σ·√T collapses to zero in both cases.
    The function returns (+∞, +∞) when S ≥ K and (−∞, −∞) when S < K.
    This maps through N(·) to delta = 1 / 0 respectively, which is the
    correct deep-ITM / deep-OTM limit.

    **ATM at expiry (S = K, T → 0)**
    d1 = d2 = +∞ by this convention, giving delta = N(+∞) = 1.
    This is a known limitation: the true delta at expiry is a Heaviside
    step function (0 below K, 1 above K), and at S = K exactly it is
    undefined (depending on whether T ↓ 0 from above or the payoff
    convention).  Callers requiring precise at-expiry analytics should
    treat this case separately.

    Examples
    --------
    >>> d1, d2 = _d1_d2(310.0, 310.0, 30/252, 0.05, 0.19)
    >>> round(float(d1), 6), round(float(d2), 6)
    (0.123576, 0.05802)
    """
    S     = np.asarray(S,     dtype=float)
    K     = np.asarray(K,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T_val = float(T)

    # Degenerate check applies only when sigma and T are scalars
    if sigma.ndim == 0 and (T_val <= 1e-9 or float(sigma) <= 1e-9):
        itm = S >= K
        inf_val = np.where(itm, np.inf, -np.inf)
        return inf_val, inf_val

    denom = sigma * np.sqrt(T_val)

    # For array inputs, guard individual zero-denominator elements
    itm = S >= K
    inf_val = np.where(itm, np.inf, -np.inf)
    safe = np.where(denom > 1e-14, denom, np.nan)
    d1_raw = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_val) / safe
    d2_raw = d1_raw - denom
    d1 = np.where(np.isnan(d1_raw), inf_val, d1_raw)
    d2 = np.where(np.isnan(d2_raw), inf_val, d2_raw)

    # Unwrap to scalar when all inputs were scalar
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
    option_type: str = "call",
) -> float | np.ndarray:
    """
    Black-Scholes closed-form option price.

    Parameters
    ----------
    S : float or np.ndarray
        Current spot price(s).
    K : float or np.ndarray
        Strike price(s).
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Implied / hedging volatility (annualised).
    option_type : {'call', 'put'}, default 'call'
        Type of European vanilla option.

    Returns
    -------
    price : float or np.ndarray
        Black-Scholes price.  Shape matches broadcast(S, K).

    Notes
    -----
    If T ≤ 1e-9 the function returns intrinsic value directly:
    - Call: max(S − K, 0)
    - Put:  max(K − S, 0)
    This avoids NaN from division by zero and matches the payoff
    boundary condition of the PDE.

    The function is vectorized: S and K may be numpy arrays of compatible
    shape.  r, sigma, T must be scalars.

    Examples
    --------
    >>> round(float(bs_price(310.0, 310.0, 30/252, 0.05, 0.19, 'call')), 3)
    9.035
    >>> round(float(bs_price(310.0, 310.0, 30/252, 0.05, 0.19, 'put')), 3)
    7.195
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    otype = option_type.lower()

    if T <= 1e-9:
        if otype == "call":
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if otype == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    elif otype == "put":
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Black-Scholes Greeks for a European vanilla option.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Implied / hedging volatility (annualised).
    option_type : {'call', 'put'}, default 'call'
        Type of European vanilla option.

    Returns
    -------
    greeks : dict
        Keys: 'delta', 'gamma', 'theta', 'vega', 'rho'.

    Notes
    -----
    **Theta convention: per TRADING DAY (÷252).**
    Theta is defined as ∂C/∂t divided by 252 so that
    ``theta_pnl = theta * dt`` with ``dt = 1/252`` directly,
    without any additional conversion factor.  This is consistent
    with dt = 1/252 used in the hedging simulation throughout this
    codebase.  The raw ∂C/∂τ (where τ is time-to-expiry in years)
    has the opposite sign and is expressed per year; dividing by 252
    converts to per-trading-day and corrects the sign convention
    (theta is a negative number: the option loses value as time passes).

    **Vega convention:** expressed per 1-point (100%) move in sigma,
    i.e., ∂C/∂σ.  To get the P&L from a 1 vol-point (0.01) move,
    multiply by 0.01.

    **Rho convention:** expressed per 1-point (100%) move in r.
    Multiply by 0.01 for a 1 bp (0.0001) effect.

    **At expiry (T ≤ 1e-9):**
    Returns delta = 1 (call ITM) or 0 (call OTM), gamma = 0,
    theta = 0, vega = 0, rho = 0.  The delta discontinuity at
    S = K is handled by the +∞ convention in _d1_d2.

    Examples
    --------
    >>> g = bs_greeks(310.0, 310.0, 30/252, 0.05, 0.19, 'call')
    >>> round(g['delta'], 4)
    0.5492
    >>> round(g['gamma'], 6)
    0.019481
    """
    otype = option_type.lower()

    if T <= 1e-9:
        # At expiry: delta is Heaviside, all other Greeks are zero
        if otype == "call":
            delta = 1.0 if S >= K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    # Delta
    if otype == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0  # = -N(-d1)

    # Gamma (same for call and put by symmetry)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Theta: ∂C/∂t (note: ∂C/∂τ = -∂C/∂t where τ = T-t)
    # Raw annual theta (∂C/∂t, time moving forward → option value decreasing):
    raw_theta_annual: float
    if otype == "call":
        raw_theta_annual = (
            -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * discount * norm.cdf(d2)
        )
    else:
        raw_theta_annual = (
            -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * discount * norm.cdf(-d2)
        )
    # Convention: per trading day (÷252)
    theta = raw_theta_annual / 252.0

    # Vega (same for call and put)
    vega = S * sqrt_T * pdf_d1

    # Rho
    if otype == "call":
        rho = K * T * discount * norm.cdf(d2)
    else:
        rho = -K * T * discount * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------


def bs_pde_residual(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Compute the Black-Scholes PDE residual for a European call.

    The BS PDE is:  Θ + ½σ²S²Γ + rSΔ − rC = 0

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Volatility (annualised).

    Returns
    -------
    residual : float
        PDE residual.  Expected to be < 1e-10 for any valid inputs,
        confirming internal consistency of the pricer and Greeks.

    Notes
    -----
    Theta here is used in its raw ANNUAL form (∂C/∂t per year) for
    the PDE, so we multiply the stored per-trading-day theta by 252
    before substituting.  This is the only place in the codebase where
    the annual form is needed.

    The PDE is derived under the risk-neutral measure and holds exactly
    in continuous time.  Residuals > 1e-8 indicate a bug in one of the
    Greek formulas.

    Examples
    --------
    >>> abs(bs_pde_residual(310.0, 310.0, 30/252, 0.05, 0.19)) < 1e-10
    True
    """
    greeks = bs_greeks(S, K, T, r, sigma, option_type="call")
    C = bs_price(S, K, T, r, sigma, option_type="call")

    # Restore annual theta from the per-trading-day convention
    theta_annual = greeks["theta"] * 252.0
    delta = greeks["delta"]
    gamma = greeks["gamma"]

    residual = theta_annual + 0.5 * sigma ** 2 * S ** 2 * gamma + r * S * delta - r * C
    return float(residual)


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
    Simulate Geometric Brownian Motion paths using the log-exact discretization.

    The exact discretization is:
        S_{t+dt} = S_t · exp[(μ − σ²/2)·dt + σ·√dt·Z],   Z ~ N(0,1)

    This avoids the Euler-Maruyama discretization error and preserves
    the log-normal marginal distribution exactly at each step.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    mu : float
        Real-world (physical-measure P) annualised drift.
    sigma : float
        Annualised volatility.
    T : float
        Total time horizon in years.
    N_steps : int
        Number of time steps.  dt = T / N_steps.
    N_paths : int
        Number of Monte Carlo paths.
    seed : int
        Seed for np.random.default_rng.  Guarantees reproducibility.

    Returns
    -------
    paths : np.ndarray, shape (N_paths, N_steps + 1)
        Simulated price paths.  Column 0 is S0 for all paths.

    Notes
    -----
    **Real-world drift mu, not risk-neutral rate r.**
    This function simulates paths under the physical measure P.  Using
    real-world drift is intentional and correct for modeling observed
    price paths in the hedging simulation.  Pricing (i.e., computing
    expected payoffs as option prices) requires drift r under Q and is
    handled separately by bs_price().

    **RNG:** uses np.random.default_rng(seed) exclusively.  The legacy
    np.random.seed() is not used anywhere in this codebase.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 5, seed=42)
    >>> paths.shape
    (5, 31)
    >>> paths[:, 0].tolist()
    [310.0, 310.0, 310.0, 310.0, 310.0]
    """
    rng = np.random.default_rng(seed)
    dt = T / N_steps

    # Draw all random increments at once: shape (N_paths, N_steps)
    Z = rng.standard_normal((N_paths, N_steps))

    # Log-exact increment per step
    increment = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z

    # Build log-price paths via cumsum, then exponentiate
    log_increments = np.hstack([
        np.zeros((N_paths, 1)),   # t=0: log(S0/S0) = 0
        increment,
    ])
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(log_paths)
    return paths


# ---------------------------------------------------------------------------
# Delta-hedging engine
# ---------------------------------------------------------------------------


def delta_hedge_single_path(
    S_path: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    dt: float = 1.0 / 252,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simulate the discrete delta-hedging replication strategy on one price path.

    Strategy (from the perspective of the OPTION SELLER):
    - t=0: sell the option (receive C0), buy delta_0 shares,
           invest/borrow the net cash at rate r.
    - t=k: recompute delta_k, rebalance share position,
           cash account earns r continuously (cash *= exp(r·dt)).
    - t=T: hedging error = portfolio value − option payoff.

    Parameters
    ----------
    S_path : np.ndarray, shape (N_steps + 1,)
        Price path including t=0.
    K : float
        Strike price.
    T_total : float
        Total time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Hedging volatility (annualised).  May differ from realized vol.
    option_type : {'call', 'put'}, default 'call'
        Type of option being hedged.
    dt : float, default 1/252
        Time step in years.  Must satisfy N_steps * dt ≈ T_total.

    Returns
    -------
    cash_account : np.ndarray, shape (N_steps + 1,)
        Cash position at each step (before final settlement).
    deltas : np.ndarray, shape (N_steps + 1,)
        Delta of the hedging portfolio at each step.
    hedging_error : float
        Final portfolio value minus option payoff.
        Positive: hedge over-performed (hedger profits).
        Negative: hedge under-performed (hedger losses).

    Notes
    -----
    **Deep ITM case (delta ≈ 1 at t=0):**
    When the option is deep ITM, delta_0 ≈ 1 and the share purchase
    cost (delta_0 · S_path[0]) can exceed the received premium C0.
    In this case, cash[0] = C0 − delta_0 · S_path[0] < 0.
    This is a borrowing position at rate r, which is economically
    correct and NOT an error.  The strategy remains self-financing.

    **Hedging error sources under discrete rebalancing:**
    The error is O(√dt) and arises from the path-wise discretization
    of the continuous Itô integral.  The theoretical expected error
    is zero (BS replication is exact in continuous time); empirically,
    RMSE scales as √dt as rebalancing frequency decreases.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 1, seed=42)
    >>> cash, deltas, err = delta_hedge_single_path(
    ...     paths[0], 310.0, 30/252, 0.05, 0.19)
    >>> bool(abs(err) < 5.0)   # error should be small relative to premium
    True
    """
    N_steps = len(S_path) - 1
    cash_account = np.zeros(N_steps + 1)
    deltas = np.zeros(N_steps + 1)

    # ---- t = 0: initiate hedge ----
    T_remaining = T_total
    C0 = bs_price(S_path[0], K, T_remaining, r, sigma, option_type)
    delta_0 = bs_greeks(S_path[0], K, T_remaining, r, sigma, option_type)["delta"]

    deltas[0] = delta_0
    # cash = option premium received − cost of initial share position
    # Negative cash is borrowing (deep ITM case): economically valid.
    cash_account[0] = C0 - delta_0 * S_path[0]

    # ---- t = 1, …, N_steps − 1: rebalance ----
    for k in range(1, N_steps):
        T_remaining = T_total - k * dt
        T_remaining = max(T_remaining, 1e-9)  # guard against floating-point drift

        # Accrue interest on cash
        cash_account[k] = cash_account[k - 1] * np.exp(r * dt)

        # New delta at updated S and T
        delta_new = bs_greeks(S_path[k], K, T_remaining, r, sigma, option_type)["delta"]

        # Rebalance: sell (delta_new − delta_old) shares
        cash_account[k] -= (delta_new - deltas[k - 1]) * S_path[k]
        deltas[k] = delta_new

    # ---- t = T: settle ----
    # Accrue interest on penultimate cash balance
    cash_final = cash_account[N_steps - 1] * np.exp(r * dt)

    # Liquidate the share position
    delta_last = deltas[N_steps - 1]
    portfolio_value = cash_final + delta_last * S_path[N_steps]

    # Option payoff (from option seller's perspective: we owe this)
    if option_type.lower() == "call":
        payoff = max(S_path[N_steps] - K, 0.0)
    else:
        payoff = max(K - S_path[N_steps], 0.0)

    hedging_error = portfolio_value - payoff

    # Store final values for diagnostics
    cash_account[N_steps] = cash_final
    deltas[N_steps] = delta_last

    return cash_account, deltas, hedging_error


def run_hedging_experiment(
    S_paths: np.ndarray,
    K: float,
    T_total: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    dt: float = 1.0 / 252,
) -> tuple[np.ndarray, float, float]:
    """
    Run the delta-hedging replication experiment across all Monte Carlo paths.

    Parameters
    ----------
    S_paths : np.ndarray, shape (N_paths, N_steps + 1)
        Simulated price paths from simulate_gbm().
    K : float
        Strike price.
    T_total : float
        Total time to expiry in years.
    r : float
        Continuously compounded risk-free rate (annualised).
    sigma : float
        Hedging volatility (annualised).
    option_type : {'call', 'put'}, default 'call'
        Type of option.
    dt : float, default 1/252
        Time step in years.

    Returns
    -------
    errors : np.ndarray, shape (N_paths,)
        Hedging error on each path.
    rmse : float
        Root-mean-squared hedging error across all paths.
    mean_error : float
        Mean hedging error.  Expected ≈ 0 when sigma_hedge = sigma_realized.

    Notes
    -----
    RMSE scales approximately as √dt as rebalancing frequency decreases.
    This is consistent with the Gamma P&L formula:
        Hedging error ≈ ½ · Γ · S² · (σ_r² − σ_h²) · dt · (path terms)

    For sigma_hedge = sigma_realized, the mean error converges to zero
    as N_paths → ∞, confirming the BS replication argument.

    Examples
    --------
    >>> paths = simulate_gbm(310.0, 0.14, 0.19, 30/252, 30, 100, seed=42)
    >>> errors, rmse, mean_err = run_hedging_experiment(
    ...     paths, 310.0, 30/252, 0.05, 0.19)
    >>> rmse < 3.0  # RMSE < 3 USD for ATM call at 19% vol
    True
    """
    N_paths = S_paths.shape[0]
    errors = np.zeros(N_paths)

    for i in range(N_paths):
        _, _, err = delta_hedge_single_path(
            S_paths[i], K, T_total, r, sigma, option_type, dt
        )
        errors[i] = err

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_error = float(np.mean(errors))
    return errors, rmse, mean_error


# ---------------------------------------------------------------------------
# Empirical volatility
# ---------------------------------------------------------------------------


def rolling_realized_vol(
    prices: pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute annualised realized volatility on rolling windows.

    Parameters
    ----------
    prices : pd.Series
        Time series of adjusted close prices (any frequency, assumed daily).
        Index should be datetime-like for meaningful output labelling.
    windows : list of int, default [30, 60, 252]
        Rolling window lengths in trading days.

    Returns
    -------
    vol_df : pd.DataFrame
        Columns: ['vol_30d', 'vol_60d', 'vol_252d'] (or matching window names).
        Values are annualised standard deviations of log returns.
        Annualization factor: √252 (trading-day convention).

    Notes
    -----
    Log returns: r_t = ln(P_t / P_{t-1}).
    Rolling std uses ddof=1 (sample standard deviation).
    Annualisation: σ_annualised = σ_daily · √252.

    The first (window − 1) rows will be NaN because the rolling window
    is not yet full.  This is expected behaviour, not a bug.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    >>> vols = rolling_realized_vol(prices, windows=[30, 60])
    >>> vols.columns.tolist()
    ['vol_30d', 'vol_60d']
    >>> bool(vols['vol_30d'].dropna().between(0.05, 0.30).all())
    True
    """
    if windows is None:
        windows = [30, 60, 252]

    log_returns = np.log(prices / prices.shift(1))
    vol_dict: dict[str, pd.Series] = {}
    for w in windows:
        col = f"vol_{w}d"
        vol_dict[col] = log_returns.rolling(window=w).std(ddof=1) * np.sqrt(252)

    return pd.DataFrame(vol_dict, index=prices.index)
