"""Black-Scholes-Merton implied volatility by bracketed root finding.

The price is strictly increasing in sigma on ``(0, inf)`` (vega > 0), from the
lower no-arbitrage bound at sigma = 0 to the upper bound as sigma -> inf, so
any price strictly inside the bounds has a unique implied volatility.
Brent's method is used because it is guaranteed to converge on a bracket,
whereas Newton-Raphson diverges when vega is close to zero (deep in or out of
the money, or close to expiry).

Reference: Brent, R. P. (1973). *Algorithms for Minimization without
Derivatives*, ch. 4. Prentice-Hall.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from bspricer._validation import OptionType, as_float_array, check_option_type
from bspricer.pricing import bs_price, no_arbitrage_bounds

__all__ = ["ImpliedVolError", "bs_implied_vol"]


class ImpliedVolError(ValueError):
    """The price has no implied volatility, or it cannot be identified numerically."""


def bs_implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    *,
    sigma_max: float = 10.0,
    xtol: float = 1e-12,
    price_tol: float | None = None,
) -> float:
    """Implied volatility of a European option price.

    Parameters
    ----------
    price : observed option price.
    S, K, T, r, option_type, q : as in :func:`bspricer.bs_price`; ``T`` must be > 0.
    sigma_max : upper end of the search bracket.
    xtol : absolute tolerance on sigma passed to Brent's method.
    price_tol : price resolution, default ``1e-12 * max(S, K)``. A price whose
        time value (distance to the lower bound) is below this resolution does not
        identify sigma: the function raises instead of returning an arbitrary value.

    Raises
    ------
    ImpliedVolError
        If the price violates the no-arbitrage bounds, if its time value is below
        ``price_tol`` (sigma not identifiable), or if the implied vol exceeds
        ``sigma_max``.

    Examples
    --------
    >>> from bspricer import bs_price
    >>> p = bs_price(100.0, 110.0, 0.5, 0.03, 0.25, "call")
    >>> round(bs_implied_vol(p, 100.0, 110.0, 0.5, 0.03, "call"), 12)
    0.25
    """
    otype = check_option_type(option_type)
    for name, value in (("price", price), ("r", r), ("q", q)):
        as_float_array(name, value)
    as_float_array("S", S, positive=True)
    as_float_array("K", K, positive=True)
    as_float_array("T", T, positive=True)
    if not sigma_max > 0.0:
        raise ValueError("sigma_max must be > 0")

    lower, upper = (float(b) for b in no_arbitrage_bounds(S, K, T, r, otype, q))
    tol = 1e-12 * max(S, K) if price_tol is None else float(price_tol)

    if price < lower - tol or price > upper + tol:
        raise ImpliedVolError(
            f"price {price:.10g} outside the no-arbitrage bounds [{lower:.10g}, {upper:.10g}]"
        )
    if price - lower <= tol:
        raise ImpliedVolError(
            f"time value {price - lower:.3g} is below the price resolution {tol:.3g}: "
            "implied volatility is not identifiable"
        )
    if upper - price <= tol:
        raise ImpliedVolError("price equals the upper bound: implied volatility is infinite")

    def objective(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, otype, q)) - price

    if objective(sigma_max) < 0.0:
        raise ImpliedVolError(f"implied volatility above sigma_max = {sigma_max}")
    # f(0) = lower - price < 0 by the checks above, so [0, sigma_max] brackets the root.
    root = brentq(objective, 0.0, sigma_max, xtol=xtol, maxiter=500)
    sigma = float(root)
    if not math.isfinite(sigma) or sigma <= 0.0:  # pragma: no cover - guarded by the bracket
        raise ImpliedVolError("root finder returned a non-positive volatility")
    return sigma
