r"""Analytical Black-Scholes-Merton Greeks.

All Greeks are raw partial derivatives in natural units:

* ``delta`` :math:`= \partial V/\partial S`
* ``gamma`` :math:`= \partial^2 V/\partial S^2`
* ``theta`` :math:`= \partial V/\partial t` **per year** (calendar time t, so theta < 0 for a long vanilla
  in most cases). Divide by 252 for "per trading day".
* ``vega``  :math:`= \partial V/\partial \sigma` per unit of vol. Multiply by 0.01 for "per vol point".
* ``rho``   :math:`= \partial V/\partial r` per unit of rate. Multiply by 0.0001 for "per basis point".

With :math:`\varphi` the standard normal density and :math:`\tau = T`:

.. math::
    \Delta_C = e^{-q\tau} N(d_1), \quad \Delta_P = -e^{-q\tau} N(-d_1), \quad
    \Gamma = \frac{e^{-q\tau}\varphi(d_1)}{S\sigma\sqrt{\tau}}, \quad
    \mathcal{V} = S e^{-q\tau}\sqrt{\tau}\,\varphi(d_1),

    \Theta_C = -\frac{S e^{-q\tau}\varphi(d_1)\sigma}{2\sqrt{\tau}} - rKe^{-r\tau}N(d_2) + qSe^{-q\tau}N(d_1),

    \rho_C = K\tau e^{-r\tau}N(d_2), \quad \rho_P = -K\tau e^{-r\tau}N(-d_2).

They satisfy the Black-Scholes-Merton PDE
:math:`\Theta + \tfrac12\sigma^2S^2\Gamma + (r-q)S\Delta - rV = 0` (see :func:`bs_pde_residual`).

Reference: Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed., ch. 19.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from bspricer._validation import FloatArray, OptionType, check_option_type, to_output
from bspricer.pricing import _bs_price, _d1_d2, _inputs

__all__ = ["GREEK_NAMES", "bs_greeks", "bs_pde_residual"]

GREEK_NAMES = ("delta", "gamma", "theta", "vega", "rho")
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _safe_ratio(num: FloatArray, den: FloatArray) -> FloatArray:
    """num / den, with the limit +inf (num > 0) or 0 (num == 0) where den == 0."""
    safe_den = np.where(den > 0.0, den, 1.0)
    limit = np.where(num > 0.0, np.inf, 0.0)
    return np.where(den > 0.0, num / safe_den, limit)


def _bs_greeks(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
    option_type: OptionType,
) -> dict[str, FloatArray]:
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    pdf = np.exp(-0.5 * d1**2) * _INV_SQRT_2PI  # 0 at d1 = +-inf
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    gamma = _safe_ratio(disc_q * pdf, S * sigma * sqrt_t)
    vega = S * disc_q * sqrt_t * pdf
    time_decay = -_safe_ratio(S * disc_q * pdf * sigma, 2.0 * sqrt_t)

    if option_type == "call":
        delta = disc_q * ndtr(d1)
        theta = time_decay - r * K * disc_r * ndtr(d2) + q * S * disc_q * ndtr(d1)
        rho = K * T * disc_r * ndtr(d2)
    else:
        delta = -disc_q * ndtr(-d1)
        theta = time_decay + r * K * disc_r * ndtr(-d2) - q * S * disc_q * ndtr(-d1)
        rho = -K * T * disc_r * ndtr(-d2)
    return {
        "delta": np.asarray(delta, dtype=np.float64),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "theta": np.asarray(theta, dtype=np.float64),
        "vega": np.asarray(vega, dtype=np.float64),
        "rho": np.asarray(rho, dtype=np.float64),
    }


def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> dict[str, float | FloatArray]:
    """Delta, gamma, theta (per year), vega (per unit vol) and rho (per unit rate).

    Inputs broadcast like :func:`bspricer.bs_price`. In the degenerate case
    ``sigma * sqrt(T) == 0`` the Greeks are the exact limits: delta is the step
    function of the forward moneyness (1/2 at the forward), gamma and vega are 0
    away from the forward, and gamma is ``+inf`` exactly at the forward.

    Examples
    --------
    >>> g = bs_greeks(100.0, 100.0, 0.25, 0.03, 0.20, "call")
    >>> round(g["delta"], 6), round(g["gamma"], 6), round(g["vega"], 6)
    (0.549738, 0.039584, 19.791884)
    >>> round(g["theta"], 6), round(g["theta"] / 252, 6)  # per year, per trading day
    (-9.43524, -0.037441)
    """
    otype = check_option_type(option_type)
    greeks = _bs_greeks(*_inputs(S, K, T, r, sigma, q), otype)
    return {name: to_output(value) for name, value in greeks.items()}


def bs_pde_residual(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> float | FloatArray:
    r"""Residual of the Black-Scholes-Merton PDE evaluated with the analytical Greeks.

    .. math:: \Theta + \tfrac12\sigma^2 S^2 \Gamma + (r - q) S \Delta - r V

    Zero up to rounding for ``T > 0`` and ``sigma > 0``; a non-zero value flags an
    inconsistency between :func:`bs_price` and :func:`bs_greeks`.

    Examples
    --------
    >>> abs(bs_pde_residual(100.0, 90.0, 0.5, 0.03, 0.25, "put", q=0.01)) < 1e-12
    True
    """
    otype = check_option_type(option_type)
    S_, K_, T_, r_, s_, q_ = _inputs(S, K, T, r, sigma, q)
    g = _bs_greeks(S_, K_, T_, r_, s_, q_, otype)
    value = _bs_price(S_, K_, T_, r_, s_, q_, otype)
    residual = g["theta"] + 0.5 * s_**2 * S_**2 * g["gamma"] + (r_ - q_) * S_ * g["delta"] - r_ * value
    return to_output(np.asarray(residual, dtype=np.float64))
