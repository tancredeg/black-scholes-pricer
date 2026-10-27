r"""Black-Scholes-Merton closed-form prices for European options.

Model: under the risk-neutral measure the underlying follows
:math:`dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t`, with constant rate ``r``,
continuous dividend yield ``q`` and volatility ``sigma``.

.. math::

    C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
    P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),

    d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad
    d_2 = d_1 - \sigma\sqrt{T}.

Units: ``T`` in years, ``sigma`` per unit (0.20 = 20 %), ``r`` and ``q``
continuously compounded. All functions broadcast over array inputs.

References
----------
Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate
Liabilities. *Journal of Political Economy*, 81(3), 637-654.
Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of
Economics and Management Science*, 4(1), 141-183.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from bspricer._validation import FloatArray, OptionType, as_float_array, check_option_type, to_output

__all__ = ["OptionType", "bs_price", "d1_d2", "forward", "no_arbitrage_bounds"]


def _inputs(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate and broadcast the six model inputs to a common shape."""
    arrays = np.broadcast_arrays(
        as_float_array("S", S, positive=True),
        as_float_array("K", K, positive=True),
        as_float_array("T", T, nonnegative=True),
        as_float_array("r", r),
        as_float_array("sigma", sigma, nonnegative=True),
        as_float_array("q", q),
    )
    S_, K_, T_, r_, s_, q_ = (np.asarray(a, dtype=np.float64) for a in arrays)
    return S_, K_, T_, r_, s_, q_


def _d1_d2(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, sigma: FloatArray, q: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """d1, d2 on validated arrays, with the exact limit when sigma*sqrt(T) = 0.

    When ``sigma * sqrt(T) == 0`` the price is the discounted forward intrinsic
    value, so d1 = d2 = +inf if the forward is above the strike, -inf if below
    and 0 at the money (forward). This is the limit sigma -> 0+ and T -> 0+.
    """
    total_vol = sigma * np.sqrt(T)
    log_fwd_moneyness = np.log(S / K) + (r - q) * T  # ln(F / K)
    degenerate = total_vol == 0.0
    safe_vol = np.where(degenerate, 1.0, total_vol)
    d1 = (log_fwd_moneyness + 0.5 * total_vol**2) / safe_vol
    limit = np.where(log_fwd_moneyness > 0.0, np.inf, np.where(log_fwd_moneyness < 0.0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = d1 - total_vol
    return d1, d2


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[float | FloatArray, float | FloatArray]:
    r"""Black-Scholes-Merton :math:`d_1` and :math:`d_2`.

    .. math::
        d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}

    For ``sigma * sqrt(T) == 0`` returns ``(+inf, +inf)`` if :math:`S e^{(r-q)T} > K`,
    ``(-inf, -inf)`` if below and ``(0, 0)`` at the forward.

    Examples
    --------
    >>> d1, d2 = d1_d2(100.0, 100.0, 0.25, 0.03, 0.20)
    >>> round(d1, 6), round(d2, 6)
    (0.125, 0.025)
    """
    d1, d2 = _d1_d2(*_inputs(S, K, T, r, sigma, q))
    return to_output(d1), to_output(d2)


def forward(S: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0) -> float | FloatArray:
    r"""Forward price :math:`F = S e^{(r - q)T}`."""
    S_ = as_float_array("S", S, positive=True)
    T_ = as_float_array("T", T, nonnegative=True)
    return to_output(S_ * np.exp((as_float_array("r", r) - as_float_array("q", q)) * T_))


def _bs_price(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
    option_type: OptionType,
) -> FloatArray:
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_spot = S * np.exp(-q * T)
    disc_strike = K * np.exp(-r * T)
    if option_type == "call":
        return np.asarray(disc_spot * ndtr(d1) - disc_strike * ndtr(d2), dtype=np.float64)
    return np.asarray(disc_strike * ndtr(-d2) - disc_spot * ndtr(-d1), dtype=np.float64)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> float | FloatArray:
    r"""Black-Scholes-Merton price of a European call or put.

    .. math::
        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)

    Parameters
    ----------
    S, K : spot and strike, > 0.
    T : time to expiry in years, >= 0. ``T = 0`` returns the payoff.
    r : continuously compounded risk-free rate.
    sigma : volatility, >= 0. ``sigma = 0`` returns the discounted forward intrinsic value
        :math:`\max(S e^{-qT} - K e^{-rT}, 0)` for a call.
    option_type : ``"call"`` or ``"put"``.
    q : continuous dividend yield.

    Returns
    -------
    Python float for scalar inputs, otherwise an array of the broadcast shape.

    Examples
    --------
    >>> round(bs_price(100.0, 100.0, 0.25, 0.03, 0.20, "call"), 6)
    4.357619
    >>> round(bs_price(100.0, 100.0, 0.25, 0.03, 0.20, "put"), 6)
    3.610425
    """
    otype = check_option_type(option_type)
    return to_output(_bs_price(*_inputs(S, K, T, r, sigma, q), otype))


def no_arbitrage_bounds(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> tuple[float | FloatArray, float | FloatArray]:
    r"""Model-free bounds on a European option price.

    Call: :math:`\max(S e^{-qT} - K e^{-rT}, 0) \le C \le S e^{-qT}`.
    Put: :math:`\max(K e^{-rT} - S e^{-qT}, 0) \le P \le K e^{-rT}`.

    Reference: Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed., ch. 11.
    """
    otype = check_option_type(option_type)
    S_, K_, T_, r_, _, q_ = _inputs(S, K, T, r, 0.0, q)
    disc_spot = S_ * np.exp(-q_ * T_)
    disc_strike = K_ * np.exp(-r_ * T_)
    if otype == "call":
        return to_output(np.maximum(disc_spot - disc_strike, 0.0)), to_output(disc_spot)
    return to_output(np.maximum(disc_strike - disc_spot, 0.0)), to_output(disc_strike)
