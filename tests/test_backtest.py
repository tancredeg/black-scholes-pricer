"""Short-vol backtest on real SPY data: data alignment, accounting and a synthetic control."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import bspricer as bs
from bspricer.backtest import align_market_data, discount_yield_to_continuous, short_vol_backtest


@pytest.fixture(scope="module")
def market(data_dir) -> pd.DataFrame:
    return align_market_data(
        bs.load_price_series(data_dir / "SPY_2020_2024.csv"),
        bs.load_fred_series(data_dir / "VIXCLS_2020_2024.csv"),
        bs.load_fred_series(data_dir / "DTB3_2020_2024.csv"),
    )


def test_discount_yield_conversion():
    assert discount_yield_to_continuous(0.0) == 0.0
    price = 1 - 0.05 * 91 / 360
    assert discount_yield_to_continuous(5.0) == pytest.approx(-math.log(price) * 365 / 91)
    assert discount_yield_to_continuous(-0.05) < 0.0  # March 2020 bills traded above par
    with pytest.raises(ValueError):
        discount_yield_to_continuous(500.0)


def test_market_alignment(market):
    assert len(market) == 1257 and not market.isna().any().any()
    assert market.loc["2020-04-08", "implied_vol"] == pytest.approx(0.4335)
    assert market["rate"].between(-0.01, 0.06).all()


def test_backtest_accounting(market):
    trades = short_vol_backtest(market)
    assert len(trades) == 59
    assert trades["start"].iloc[0] == pd.Timestamp("2020-01-02")
    np.testing.assert_array_equal(
        trades["start"].iloc[1:].to_numpy(), trades["end"].iloc[:-1].to_numpy()
    )
    attribution = trades[["gamma", "theta", "carry", "residual"]].sum(axis=1) - trades["costs"]
    np.testing.assert_allclose(attribution, trades["pnl"], atol=1e-10)
    assert (trades["premium"] > 0).all()


def test_call_and_put_backtests_have_the_same_pnl(market):
    calls = short_vol_backtest(market, option_type="call")
    puts = short_vol_backtest(market, option_type="put")
    np.testing.assert_allclose(calls["pnl"], puts["pnl"], atol=1e-10)


def test_costs_and_vol_shift_reduce_pnl(market):
    base = short_vol_backtest(market)
    assert (short_vol_backtest(market, cost=1e-4)["pnl"] < base["pnl"]).all()
    # Selling 2 points cheaper lowers the premium, but it also changes the hedge ratios, so a
    # trade can gain: here only March 2020 (realised 90 % vs implied 32 %) does.
    cheaper = short_vol_backtest(market, vol_shift=-0.02)
    assert cheaper["pnl"].mean() < base["pnl"].mean()
    assert (cheaper["pnl"] < base["pnl"]).sum() == len(base) - 1


def test_synthetic_control_without_premium():
    """On a simulated GBM history with implied = realised vol the mean P&L is zero."""
    n_trades, horizon = 600, 21
    path = bs.simulate_gbm(
        100.0, 0.07, 0.2, n_trades * horizon / 252, n_trades * horizon, 1, seed=4
    )[0]
    index = pd.bdate_range("2000-01-03", periods=path.size)
    market = pd.DataFrame({"spot": path, "implied_vol": 0.2, "rate": 0.03}, index=index)
    fair = short_vol_backtest(market, horizon)
    se = fair["pnl"].std(ddof=1) / math.sqrt(len(fair))
    assert len(fair) == n_trades and abs(fair["pnl"].mean()) < 4 * se
    rich = short_vol_backtest(market, horizon, vol_shift=0.05)
    assert rich["pnl"].mean() > 10 * se
    assert rich["pnl"].mean() == pytest.approx(rich["variance_proxy"].mean(), rel=0.05)


def test_invalid_schedule_raises(market):
    with pytest.raises(ValueError):
        short_vol_backtest(market, horizon=21, offset=21)
