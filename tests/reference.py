"""Independent reference implementations used as test oracles.

Written with the ``math`` module only (no numpy, no code shared with
``bspricer``) so that agreement with the package is evidence, not tautology.
"""

from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def price(S: float, K: float, T: float, r: float, sigma: float, call: bool, q: float = 0.0) -> float:
    """Black-Scholes-Merton price (Hull 2022, eq. 17.4 and 17.5)."""
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q) * T) / v + 0.5 * v
    d2 = d1 - v
    if call:
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, sigma: float, call: bool, q: float = 0.0) -> dict[str, float]:
    """Delta, gamma, theta (per year), vega, rho — textbook formulas (Hull 2022, ch. 19)."""
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q) * T) / v + 0.5 * v
    d2 = d1 - v
    eq, er = math.exp(-q * T), math.exp(-r * T)
    gamma = eq * norm_pdf(d1) / (S * v)
    vega = S * eq * math.sqrt(T) * norm_pdf(d1)
    decay = -S * eq * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if call:
        return {
            "delta": eq * norm_cdf(d1),
            "gamma": gamma,
            "theta": decay - r * K * er * norm_cdf(d2) + q * S * eq * norm_cdf(d1),
            "vega": vega,
            "rho": K * T * er * norm_cdf(d2),
        }
    return {
        "delta": -eq * norm_cdf(-d1),
        "gamma": gamma,
        "theta": decay + r * K * er * norm_cdf(-d2) - q * S * eq * norm_cdf(-d1),
        "vega": vega,
        "rho": -K * T * er * norm_cdf(-d2),
    }


def hedge_path(
    path: list[float],
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    q: float = 0.0,
    cost: float = 0.0,
) -> float:
    """P&L at T of a short option delta-hedged on one path, scalar loop."""
    n = len(path) - 1
    dt = T / n
    premium = price(path[0], K, T, r, sigma, call, q)
    shares = greeks(path[0], K, T, r, sigma, call, q)["delta"]
    cash = premium - shares * path[0] - cost * abs(shares) * path[0]
    for k in range(1, n + 1):
        cash = cash * math.exp(r * dt) + shares * q * dt * 0.5 * (path[k - 1] + path[k])
        target = greeks(path[k], K, T - k * dt, r, sigma, call, q)["delta"] if k < n else 0.0
        cash -= (target - shares) * path[k] + cost * abs(target - shares) * path[k]
        shares = target
    payoff = max(path[-1] - K, 0.0) if call else max(K - path[-1], 0.0)
    return cash - payoff
