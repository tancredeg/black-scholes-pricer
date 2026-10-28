r"""Discrete delta hedging of a short European option.

Book: at :math:`t_0` the hedger **sells** one option at the Black-Scholes price
computed with ``sigma_price`` and buys :math:`\Delta_0` shares computed with
``sigma_hedge``; the cash account accrues at the rate ``r`` and receives the
dividends :math:`q\,\Delta_k S\,\Delta t` paid on the shares held. The hedge is
rebalanced at every grid date :math:`t_k = k\,\Delta t`, :math:`\Delta t = T/N`,
paying a proportional cost ``cost * |trade| * S``. At :math:`T` the shares are
sold and the payoff is paid. The result is the P&L at :math:`T` (positive = the
short-option hedger makes money).

Exact decomposition. With :math:`\Pi_k = \text{cash}_k + \Delta_k S_k - V_k`
(the book marked with ``sigma_hedge``) and :math:`V_N` = payoff,

.. math::
    \text{P\&L} = e^{rT}(V_0^{\text{price}} - V_0^{\text{hedge}})
      + \sum_{k=0}^{N-1} e^{r(T - t_{k+1})}\Big[
        \underbrace{-\tfrac12\Gamma_k \delta S_k^2}_{\text{gamma}}
        \underbrace{-\Theta_k \Delta t}_{\text{theta}}
        + \underbrace{(e^{r\Delta t}-1)(V_k - \Delta_k S_k) + D_k}_{\text{carry}}
        + \varepsilon_k \Big] - \text{costs},

where :math:`\varepsilon_k` collects the third- and higher-order terms of the
Taylor expansion of :math:`V`. Using the Black-Scholes-Merton PDE, gamma +
theta + carry equals :math:`\tfrac12\Gamma_k S_k^2[\sigma_h^2\Delta t - (\delta S_k/S_k)^2]`
up to :math:`O(\Delta t^2)`: the hedger earns the gap between the hedging
variance and the realised variance, weighted by dollar gamma
(see :func:`variance_pnl`).

References
----------
Black, F. and Scholes, M. (1973), *Journal of Political Economy*, 81(3), 637-654.
Kamal, M. and Derman, E. (1999). Correcting Black-Scholes. *Risk*, 12(1), 82-85.
Ahmad, R. and Wilmott, P. (2005). Which Free Lunch Would You Like Today, Sir? Delta
Hedging, Volatility Arbitrage and Optimal Portfolios. *Wilmott Magazine*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from bspricer._validation import FloatArray, OptionType, as_float_array, check_option_type
from bspricer.greeks import _bs_greeks
from bspricer.pricing import _bs_price

__all__ = ["HedgeResult", "PnLDecomposition", "delta_hedge", "kamal_derman_std", "variance_pnl"]


@dataclass(frozen=True)
class PnLDecomposition:
    """Per-path P&L attribution, every term compounded to the option expiry."""

    initial_mark: FloatArray
    gamma: FloatArray
    theta: FloatArray
    carry: FloatArray
    residual: FloatArray
    costs: FloatArray

    def total(self) -> FloatArray:
        """Sum of the terms; equals :attr:`HedgeResult.pnl` to rounding."""
        return self.initial_mark + self.gamma + self.theta + self.carry + self.residual - self.costs


@dataclass(frozen=True)
class HedgeResult:
    """Outcome of :func:`delta_hedge` (arrays have one entry per path)."""

    pnl: FloatArray
    premium: FloatArray
    costs: FloatArray
    n_steps: int
    dt: float
    decomposition: PnLDecomposition | None = None
    deltas: FloatArray | None = None

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1)) if self.pnl.size > 1 else 0.0

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.pnl**2)))

    @property
    def std_error(self) -> float:
        return self.std / float(np.sqrt(self.pnl.size))


def _validate_paths(paths: ArrayLike) -> FloatArray:
    arr = as_float_array("paths", paths, positive=True)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1) with n_steps >= 1")
    return arr


def _payoff(spot: FloatArray, K: float, option_type: OptionType) -> FloatArray:
    if option_type == "call":
        return np.maximum(spot - K, 0.0)
    return np.maximum(K - spot, 0.0)


def delta_hedge(
    paths: ArrayLike,
    K: float,
    T: float,
    r: float,
    sigma_hedge: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    *,
    sigma_price: float | None = None,
    cost: float = 0.0,
    decompose: bool = False,
    keep_deltas: bool = False,
) -> HedgeResult:
    """Simulate the discrete delta hedge of a short option on each path.

    Parameters
    ----------
    paths : array of shape ``(n_paths, N + 1)``, prices on the uniform grid
        ``t_k = k T / N`` (a 1-d array is treated as one path).
    K, T, r, q, option_type : option and market parameters; the time step is
        ``dt = T / N`` (no separate ``dt`` argument, so the grid and the option
        maturity cannot disagree).
    sigma_hedge : volatility used for the deltas (and the marks in the decomposition).
    sigma_price : volatility at which the option is sold; defaults to ``sigma_hedge``.
    cost : proportional transaction cost per unit of traded notional (1e-4 = 1 bp),
        charged on the initial trade, every rebalance and the final unwind.
    decompose : also return the exact gamma / theta / carry / residual attribution.
    keep_deltas : also return the hedge ratios, shape ``(n_paths, N)``.

    Examples
    --------
    >>> from bspricer.montecarlo import simulate_gbm
    >>> paths = simulate_gbm(100.0, 0.05, 0.2, 0.25, 63, 2_000, seed=0)
    >>> res = delta_hedge(paths, 100.0, 0.25, 0.03, 0.2)
    >>> bool(abs(res.mean) < 4 * res.std_error), round(float(res.std / res.premium[0]), 2)
    (True, 0.1)
    """
    otype = check_option_type(option_type)
    spots = _validate_paths(paths)
    as_float_array("K", K, positive=True)
    as_float_array("T", T, positive=True)
    as_float_array("r", r)
    as_float_array("q", q)
    as_float_array("sigma_hedge", sigma_hedge, positive=True)
    as_float_array("cost", cost, nonnegative=True)
    sig_p = (
        sigma_hedge
        if sigma_price is None
        else float(as_float_array("sigma_price", sigma_price, positive=True))
    )

    n_paths, n_cols = spots.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    growth = float(np.exp(r * dt))
    K_, r_, q_, sh_ = (np.asarray(x, dtype=np.float64) for x in (K, r, q, sigma_hedge))

    def greeks_at(spot: FloatArray, tau: float) -> dict[str, FloatArray]:
        return _bs_greeks(spot, K_, np.asarray(tau), r_, sh_, q_, otype)

    s0 = spots[:, 0]
    premium = _bs_price(s0, K_, np.asarray(T), r_, np.asarray(sig_p), q_, otype)
    g = greeks_at(s0, T)
    delta = g["delta"]
    initial_cost = cost * np.abs(delta) * s0
    cash = premium - delta * s0 - initial_cost
    costs_fv = initial_cost * np.exp(r * T)

    deltas = np.empty((n_paths, n_steps)) if keep_deltas else None
    value = _bs_price(s0, K_, np.asarray(T), r_, sh_, q_, otype) if decompose else premium
    initial_mark = (premium - value) * np.exp(r * T)
    gamma_pnl = np.zeros(n_paths)
    theta_pnl = np.zeros(n_paths)
    carry_pnl = np.zeros(n_paths)
    residual = np.zeros(n_paths)

    for k in range(n_steps):
        if deltas is not None:
            deltas[:, k] = delta
        s_now, s_next = spots[:, k], spots[:, k + 1]
        dividends = delta * q * dt * 0.5 * (s_now + s_next)
        cash = cash * growth + dividends
        compound = float(np.exp(r * (T - (k + 1) * dt)))
        last = k + 1 == n_steps
        next_value: FloatArray | None
        if last:
            new_delta = np.zeros(n_paths)
            next_greeks = None
            next_value = _payoff(s_next, K, otype)
        else:
            tau = T - (k + 1) * dt
            next_greeks = greeks_at(s_next, tau)
            new_delta = next_greeks["delta"]
            next_value = (
                _bs_price(s_next, K_, np.asarray(tau), r_, sh_, q_, otype) if decompose else None
            )
        trade = new_delta - delta
        trade_cost = cost * np.abs(trade) * s_next
        cash = cash - trade * s_next - trade_cost
        costs_fv = costs_fv + trade_cost * compound

        if decompose:
            assert next_value is not None
            ds = s_next - s_now
            gamma_step = -0.5 * g["gamma"] * ds**2
            theta_step = -g["theta"] * dt
            carry_step = (growth - 1.0) * (value - delta * s_now) + dividends
            taylor = next_value - value - delta * ds - g["theta"] * dt - 0.5 * g["gamma"] * ds**2
            gamma_pnl += compound * gamma_step
            theta_pnl += compound * theta_step
            carry_pnl += compound * carry_step
            residual += compound * -taylor
            value = next_value
        if next_greeks is not None:
            g = next_greeks
        delta = new_delta

    pnl = cash - _payoff(spots[:, -1], K, otype)
    decomposition = (
        PnLDecomposition(initial_mark, gamma_pnl, theta_pnl, carry_pnl, residual, costs_fv)
        if decompose
        else None
    )
    return HedgeResult(
        pnl=np.asarray(pnl, dtype=np.float64),
        premium=np.asarray(premium, dtype=np.float64),
        costs=np.asarray(costs_fv, dtype=np.float64),
        n_steps=n_steps,
        dt=dt,
        decomposition=decomposition,
        deltas=deltas,
    )


def variance_pnl(
    paths: ArrayLike,
    K: float,
    T: float,
    r: float,
    sigma_hedge: float,
    option_type: OptionType = "call",
    q: float = 0.0,
    *,
    sigma_realized: float | None = None,
) -> FloatArray:
    r"""Leading-order P&L of the short, delta-hedged option ("gamma times variance gap").

    .. math::
        \sum_{k=0}^{N-1} e^{r(T-t_{k+1})}\,\tfrac12\,\Gamma_k S_k^2
        \big[\sigma_h^2\Delta t - x_k\big],
        \qquad x_k = \Big(\frac{S_{k+1}-S_k}{S_k}\Big)^2 \ \text{or}\ \sigma_r^2\Delta t,

    with :math:`\Gamma_k` computed at ``sigma_hedge`` along the path. With
    ``sigma_realized=None`` the realised squared returns are used (path-wise
    predictor); otherwise :math:`\sigma_r^2 \Delta t` (expected-variance predictor).
    """
    otype = check_option_type(option_type)
    spots = _validate_paths(paths)
    n_steps = spots.shape[1] - 1
    dt = T / n_steps
    K_, r_, q_, sh_ = (np.asarray(x, dtype=np.float64) for x in (K, r, q, sigma_hedge))
    out = np.zeros(spots.shape[0])
    for k in range(n_steps):
        s_now = spots[:, k]
        gamma = _bs_greeks(s_now, K_, np.asarray(T - k * dt), r_, sh_, q_, otype)["gamma"]
        if sigma_realized is None:
            realized = (spots[:, k + 1] / s_now - 1.0) ** 2
        else:
            realized = np.full_like(s_now, sigma_realized**2 * dt)
        out += (
            np.exp(r * (T - (k + 1) * dt))
            * 0.5
            * gamma
            * s_now**2
            * (sigma_hedge**2 * dt - realized)
        )
    return out


def kamal_derman_std(vega: float, sigma: float, n_steps: int) -> float:
    r"""Kamal-Derman (1999) approximation of the hedging-error standard deviation.

    For an at-the-money option hedged ``n_steps`` times at the correct volatility,

    .. math::
        \operatorname{sd}(\text{P\&L}) \approx \sqrt{\pi/4}\;\frac{\sigma\,\mathcal V}{\sqrt N},

    with :math:`\mathcal V` the Black-Scholes vega (per unit vol) at inception.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return float(np.sqrt(np.pi / 4.0) * sigma * vega / np.sqrt(n_steps))
