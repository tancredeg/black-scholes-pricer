r"""Realised volatility estimators and loaders for the frozen market data.

Close-to-close estimators on log returns :math:`x_t = \ln(S_t / S_{t-1})`:

* rolling sample standard deviation, annualised by :math:`\sqrt{252}`:
  :math:`\hat\sigma_t = \sqrt{252}\,\sqrt{\tfrac{1}{w-1}\sum_{i=t-w+1}^{t}(x_i - \bar x)^2}`;
* zero-mean realised volatility :math:`\sqrt{\tfrac{252}{n}\sum x_i^2}`, the convention of
  variance swaps and of the VIX, used to compare realised with implied volatility.

Reference: Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed., ch. 15.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "load_fred_series",
    "load_price_series",
    "log_returns",
    "realized_vol",
    "rolling_realized_vol",
]

TRADING_DAYS = 252


def load_price_series(path: str | Path, column: str = "Close") -> pd.Series:
    """Load a daily price CSV (date index in the first column) as a clean float Series."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = frame[column].astype(float).dropna().sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"duplicate dates in {path}")
    if (series <= 0).any():
        raise ValueError(f"non-positive prices in {path}")
    series.index.name = "date"
    return series


def load_fred_series(path: str | Path) -> pd.Series:
    """Load a FRED ``fredgraph.csv`` download; empty values (holidays) are dropped."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().sort_index()
    series.index.name = "date"
    return series.astype(float)


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns, first observation dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def rolling_realized_vol(
    prices: pd.Series,
    windows: Sequence[int] = (30, 60, 252),
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Annualised rolling close-to-close volatility (sample std, ``ddof=1``).

    Columns are ``vol_{w}d``; the first ``w`` rows of each column are NaN.

    Examples
    --------
    >>> idx = pd.bdate_range("2024-01-01", periods=300)
    >>> prices = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.001))), index=idx)
    >>> rolling_realized_vol(prices, windows=[30]).columns.tolist()
    ['vol_30d']
    """
    returns = np.log(prices / prices.shift(1))
    data = {
        f"vol_{w}d": returns.rolling(window=w).std(ddof=1) * np.sqrt(periods_per_year)
        for w in windows
    }
    return pd.DataFrame(data, index=prices.index)


def realized_vol(prices: pd.Series | np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r"""Zero-mean annualised realised volatility :math:`\sqrt{\tfrac{252}{n}\sum_i x_i^2}` of a price path."""
    values = np.asarray(prices, dtype=np.float64)
    returns = np.diff(np.log(values))
    if returns.size == 0:
        raise ValueError("need at least two prices")
    return float(np.sqrt(periods_per_year * np.mean(returns**2)))
