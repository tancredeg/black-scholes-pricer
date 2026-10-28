r"""Short-volatility backtest on a real price history.

Every ``horizon`` trading days the hedger sells an at-the-money option of maturity
``horizon / 252`` at an implied volatility read from the market (the VIX, by default),
delta-hedges it once per trading day on the *observed* closes, and holds it to expiry.
Trades do not overlap. Each trade is run through :func:`bspricer.delta_hedge` on the
price path rescaled to a spot of 100, so P&L figures are per 100 of notional.

The P&L of trade *i* is, to leading order (see :mod:`bspricer.hedging`),

.. math::
    \text{P\&L}_i \approx \sum_k \tfrac12\,\Gamma_k S_k^2\big[\sigma_{\text{imp}}^2\Delta t
        - (\delta S_k / S_k)^2\big],

i.e. the gap between implied and realised variance weighted by the dollar gamma along
the path. A positive average P&L is the variance risk premium.

Reference: Carr, P. and Wu, L. (2009). Variance Risk Premiums. *Review of Financial
Studies*, 22(3), 1311-1341.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bspricer._validation import OptionType, check_option_type
from bspricer.hedging import delta_hedge, variance_pnl
from bspricer.vol import TRADING_DAYS, realized_vol

__all__ = ["align_market_data", "discount_yield_to_continuous", "short_vol_backtest"]


def discount_yield_to_continuous(discount_percent: float, days: int = 91) -> float:
    r"""Convert a T-bill rate quoted on a discount basis (percent) to a continuous rate.

    The bill price is :math:`P = 1 - d \cdot \text{days}/360`; the continuously
    compounded rate on an actual/365 basis is :math:`r = -\ln P \cdot 365/\text{days}`.

    Examples
    --------
    >>> round(discount_yield_to_continuous(5.0), 6)
    0.051018
    """
    price = 1.0 - discount_percent / 100.0 * days / 360.0
    if price <= 0.0:
        raise ValueError("discount rate too large for the bill tenor")
    return -math.log(price) * 365.0 / days


def align_market_data(
    prices: pd.Series, implied_vol_points: pd.Series, tbill_discount_percent: pd.Series
) -> pd.DataFrame:
    """Align implied vol (in vol points) and T-bill rates on the trading days of ``prices``.

    Values are carried forward from the last available observation (FRED series have
    holiday gaps); dates before the first observation are dropped. Returns columns
    ``spot``, ``implied_vol`` (per unit) and ``rate`` (continuous).
    """
    frame = pd.DataFrame({"spot": prices.astype(float)})
    frame["implied_vol"] = implied_vol_points.reindex(frame.index).ffill() / 100.0
    discount = tbill_discount_percent.reindex(frame.index).ffill()
    frame["rate"] = [
        discount_yield_to_continuous(d) if np.isfinite(d) else np.nan for d in discount
    ]
    return frame.dropna()


def short_vol_backtest(
    market: pd.DataFrame,
    horizon: int = 21,
    *,
    offset: int = 0,
    vol_shift: float = 0.0,
    cost: float = 0.0,
    option_type: OptionType = "call",
) -> pd.DataFrame:
    """Sell and delta-hedge one ATM option every ``horizon`` trading days.

    Parameters
    ----------
    market : output of :func:`align_market_data` (columns spot, implied_vol, rate).
    horizon : option maturity and spacing between trades, in trading days.
    offset : index of the first trade date (0 <= offset < horizon), to test the
        sensitivity to the arbitrary start of the calendar.
    vol_shift : added to the implied vol at inception (e.g. -0.02 sells 2 vol points
        below the VIX, a rough allowance for VIX exceeding at-the-money implied vol).
    cost : proportional transaction cost per unit of traded notional.
    option_type : the P&L is identical for calls and puts when ``q = 0`` (the engine is
        put-call parity consistent), so this only changes the premium column.

    Returns
    -------
    One row per trade: dates, implied and realised vol (zero-mean, annualised by 252),
    rate, premium, P&L and its attribution (gamma, theta, carry, residual, costs), the
    dollar-gamma variance proxy and the underlying return over the trade.
    """
    otype = check_option_type(option_type)
    if horizon < 1 or not 0 <= offset < horizon:
        raise ValueError("need horizon >= 1 and 0 <= offset < horizon")
    spots = market["spot"].to_numpy(dtype=np.float64)
    maturity = horizon / TRADING_DAYS
    rows: list[dict[str, object]] = []
    for start in range(offset, len(market) - horizon, horizon):
        window = spots[start : start + horizon + 1]
        path = 100.0 * window / window[0]
        sigma = float(market["implied_vol"].iloc[start]) + vol_shift
        rate = float(market["rate"].iloc[start])
        res = delta_hedge(path, 100.0, maturity, rate, sigma, otype, cost=cost, decompose=True)
        attribution = res.decomposition
        assert attribution is not None
        rows.append(
            {
                "start": market.index[start],
                "end": market.index[start + horizon],
                "implied_vol": sigma,
                "realized_vol": realized_vol(window),
                "rate": rate,
                "premium": float(res.premium[0]),
                "pnl": float(res.pnl[0]),
                "gamma": float(attribution.gamma[0]),
                "theta": float(attribution.theta[0]),
                "carry": float(attribution.carry[0]),
                "residual": float(attribution.residual[0]),
                "costs": float(attribution.costs[0]),
                "variance_proxy": float(variance_pnl(path, 100.0, maturity, rate, sigma, otype)[0]),
                "underlying_return": float(window[-1] / window[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)
