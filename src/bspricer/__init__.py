"""bspricer — Black-Scholes-Merton pricing, Greeks, numerical cross-checks and discrete delta hedging.

Conventions: ``T`` in years, ``sigma`` per unit, ``r`` and ``q`` continuously
compounded; Greeks are raw partial derivatives (theta per year, vega per unit
vol, rho per unit rate). The hedging engine is written from the point of view
of the option seller.
"""

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
from bspricer.core import rolling_realized_vol

__version__ = "0.9.0"

__all__ = [
    "GREEK_NAMES",
    "HedgeResult",
    "ImpliedVolError",
    "MCEstimate",
    "OptionType",
    "PnLDecomposition",
    "__version__",
    "bs_greeks",
    "bs_implied_vol",
    "bs_pde_residual",
    "bs_price",
    "crr_price",
    "d1_d2",
    "delta_hedge",
    "forward",
    "kamal_derman_std",
    "mc_price",
    "no_arbitrage_bounds",
    "rolling_realized_vol",
    "simulate_gbm",
    "variance_pnl",
]
