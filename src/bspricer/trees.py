r"""Cox-Ross-Rubinstein binomial tree.

With :math:`\Delta t = T/n`, :math:`u = e^{\sigma\sqrt{\Delta t}}`, :math:`d = 1/u` and the
risk-neutral up-probability :math:`p = (e^{(r-q)\Delta t} - d)/(u - d)`, the option value is
obtained by backward induction :math:`V_i = e^{-r\Delta t}[p V_{i+1}^{u} + (1-p) V_{i+1}^{d}]`,
with early exercise :math:`V_i = \max(V_i, \text{payoff}(S_i))` for American options.

For European options the price converges to Black-Scholes-Merton at rate :math:`O(1/n)`,
with an oscillation between odd and even ``n`` caused by the position of the strike
relative to the terminal nodes.

Reference: Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option Pricing: A
Simplified Approach. *Journal of Financial Economics*, 7(3), 229-263.
"""

from __future__ import annotations

import numpy as np

from bspricer._validation import FloatArray, OptionType, as_float_array, check_option_type

__all__ = ["crr_price"]


def _payoff(spot: FloatArray, K: float, option_type: OptionType) -> FloatArray:
    if option_type == "call":
        return np.maximum(spot - K, 0.0)
    return np.maximum(K - spot, 0.0)


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: OptionType = "call",
    q: float = 0.0,
    *,
    american: bool = False,
) -> float:
    """Price a European (default) or American option on a CRR binomial tree.

    Raises ``ValueError`` if the risk-neutral probability leaves ``(0, 1)``, i.e.
    when ``sigma * sqrt(dt) <= |r - q| * dt`` (step too coarse for the drift).

    Examples
    --------
    >>> round(crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 2000), 4)
    10.4496
    """
    otype = check_option_type(option_type)
    as_float_array("S", S, positive=True)
    as_float_array("K", K, positive=True)
    as_float_array("T", T, positive=True)
    as_float_array("sigma", sigma, positive=True)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (float(np.exp((r - q) * dt)) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1): increase n_steps")
    disc = float(np.exp(-r * dt))

    exponents = 2.0 * np.arange(n_steps + 1) - n_steps  # S u^j d^(n-j) = S u^(2j-n)
    values = _payoff(np.asarray(S * u**exponents, dtype=np.float64), K, otype)
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            spots = np.asarray(S * u ** (2.0 * np.arange(i + 1) - i), dtype=np.float64)
            values = np.maximum(values, _payoff(spots, K, otype))
    return float(values[0])
