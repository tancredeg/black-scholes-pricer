"""bspricer — Black-Scholes-Merton pricing, Greeks, numerical checks and discrete delta hedging.

Conventions: ``T`` in years, ``sigma`` per unit, ``r`` and ``q`` continuously
compounded; Greeks are raw partial derivatives (theta per year, vega per unit
vol, rho per unit rate). The hedging engine is written from the point of view
of the option seller.
"""

from bspricer.backtest import align_market_data, discount_yield_to_continuous, short_vol_backtest
from bspricer.greeks import GREEK_NAMES, bs_greeks, bs_pde_residual
from bspricer.hedging import (
    HedgeResult,
    PnLDecomposition,
    delta_hedge,
    kamal_derman_std,
    variance_pnl,
)
from bspricer.implied_vol import ImpliedVolError, bs_implied_vol
from bspricer.montecarlo import MCEstimate, mc_price, simulate_gbm
from bspricer.pricing import OptionType, bs_price, d1_d2, forward, no_arbitrage_bounds
from bspricer.trees import crr_price
from bspricer.vol import (
    TRADING_DAYS,
    load_fred_series,
    load_price_series,
    log_returns,
    realized_vol,
    rolling_realized_vol,
)

__version__ = "0.9.0"

__all__ = [
    "GREEK_NAMES",
    "TRADING_DAYS",
    "HedgeResult",
    "ImpliedVolError",
    "MCEstimate",
    "OptionType",
    "PnLDecomposition",
    "__version__",
    "align_market_data",
    "bs_greeks",
    "bs_implied_vol",
    "bs_pde_residual",
    "bs_price",
    "crr_price",
    "d1_d2",
    "delta_hedge",
    "discount_yield_to_continuous",
    "forward",
    "kamal_derman_std",
    "load_fred_series",
    "load_price_series",
    "log_returns",
    "mc_price",
    "no_arbitrage_bounds",
    "realized_vol",
    "rolling_realized_vol",
    "short_vol_backtest",
    "simulate_gbm",
    "variance_pnl",
]
