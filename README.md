# Black-Scholes Options Pricer and Delta Hedging Simulator

A rigorously implemented Python library for Black-Scholes-Merton option
pricing, Greeks analytics, GBM Monte Carlo simulation, discrete delta-hedging,
and implied volatility solving. Includes empirical analysis on real SPY data
2020-2024.

Built to demonstrate desk-level command of vanilla option mechanics: pricing,
hedging cost decomposition, P&L attribution, and the effect of model
assumptions (sigma mismatch, dividend yield, rebalancing frequency) on the
replication error.

---

## Repo structure

```
.
├── bs_pricer.py                  # Importable module: pricing, Greeks, IV, hedging
├── black_scholes.ipynb           # Notebook: 9 sections, empirical analysis on SPY
├── tests/test_bs_pricer.py       # 2000+ pytest assertions over a parameter grid
├── .github/workflows/ci.yml      # CI: doctest + pytest on Python 3.10-3.12
├── requirements.txt
├── README.md
└── data/
    ├── .gitkeep                  # SPY.csv is gitignored; populated on first run
    └── SPY.csv                   # cached after first yfinance call
```

---

## Installation and validation

```bash
pip install -r requirements.txt

# Run the full test suite (35 doctests + 2000+ pytest assertions)
python -m doctest bs_pricer.py
python -m pytest tests/ -v
```

Requires Python 3.10+.

---

## Module API

### Pricing

```python
import bs_pricer as bs

# Plain call/put
C = bs.bs_price(S=310.0, K=310.0, T=30/252, r=0.05, sigma=0.19, option_type='call')
# 9.0351

# With continuous dividend yield (SPY is ~1.3%)
C_div = bs.bs_price(S=310.0, K=310.0, T=30/252, r=0.05, sigma=0.19,
                    option_type='call', q=0.013)
# 8.7741  (lower because dividends reduce the forward price)
```

### Greeks (scalar and vectorised)

```python
# Scalar dict, includes dividend yield support
g = bs.bs_greeks(S=310.0, K=310.0, T=30/252, r=0.05, sigma=0.19,
                 option_type='call', q=0.013)
# {'delta': 0.5446, 'gamma': 0.01946, 'theta': -0.1666, 'vega': 42.30, 'rho': 19.19}

# Vectorised: pass np.ndarray for S, K, or sigma
import numpy as np
S_grid = np.linspace(280, 340, 7)
g_vec = bs.bs_greeks_vec(S_grid, 310.0, 30/252, 0.05, 0.19, 'call')
# g_vec['delta'] is a length-7 array
```

### Implied volatility

```python
# Brent's method, globally convergent within [1e-6, 5.0] vol bracket
sigma_iv = bs.bs_implied_vol(C_market=9.0351, S=310.0, K=310.0,
                             T=30/252, r=0.05, option_type='call', q=0.0)
# 0.19  (recovers the input vol exactly)
```

### GBM simulation and delta hedging

```python
# Simulate 1000 GBM paths, 30 daily steps, under the physical measure
paths = bs.simulate_gbm(S0=310.0, mu=0.14, sigma=0.19, T=30/252,
                        N_steps=30, N_paths=1000, seed=42)

# Vectorised hedging experiment: ~570x faster than the scalar version
errors, rmse, mean_err = bs.run_hedging_experiment_vec(
    S_paths=paths, K=310.0, T_total=30/252, r=0.05, sigma=0.19
)
# RMSE: $1.2092 (13.4% of premium), mean: ~0
```

### Realised volatility

```python
import yfinance as yf
prices = yf.download("SPY", start="2020-01-01", end="2024-12-31",
                     auto_adjust=True)["Close"].dropna()
vol_df = bs.rolling_realized_vol(prices, windows=[30, 60, 252])
```

---

## Mathematical background

### Black-Scholes-Merton pricing (with continuous dividend yield q)

For a European call:

    C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)

    d1 = [ ln(S/K) + (r - q + sigma^2 / 2) * T ] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

Put price by put-call parity:

    P = C - S * exp(-qT) + K * exp(-rT)

### Greeks

| Greek | Formula (call, with q)                              | Interpretation                          |
|-------|------------------------------------------------------|------------------------------------------|
| Delta | exp(-qT) * N(d1)                                     | Shares per option in the replication     |
| Gamma | exp(-qT) * N'(d1) / (S * sigma * sqrt(T))            | Delta sensitivity per $1 move in S       |
| Theta | per trading day                                      | Value lost per trading day               |
| Vega  | S * exp(-qT) * sqrt(T) * N'(d1)                      | Sensitivity to a 1-unit vol change       |
| Rho   | K * T * exp(-rT) * N(d2)                             | Sensitivity to a 1-unit rate change      |

**Theta convention:** per trading day (divided by 252). With `dt = 1/252`,
the theta P&L over one trading day is exactly `theta * dt`.

### PDE (Black-Scholes-Merton)

    Theta + 0.5 * sigma^2 * S^2 * Gamma + (r - q) * S * Delta - r * C = 0

Verified to residual < 1e-10 across a grid of 600+ parameter combinations
in the pytest suite.

### Delta hedging

The BS price equals the cost of the continuous replication strategy. In
practice, daily rebalancing introduces an error that scales as sqrt(dt).
The per-step P&L decomposes as:

    epsilon_t ~ (1/2) * Gamma_t * S_t^2 * [ (dW_t)^2 - sigma_hedge^2 * dt ]

Taking expectations under the physical measure P (where E[(dW_t)^2] = sigma_realized^2 * dt):

    E[epsilon_t] = (1/2) * Gamma * S^2 * (sigma_realized^2 - sigma_hedge^2) * dt

This is the **mean P&L** — it is zero when sigma_hedge = sigma_realized.
The **variance** of the error remains positive even when the vols match,
which is why the RMSE > 0 even in the "correct" hedging regime.

---

## Key results

All numbers reproduce with `S0=310, K=310, T=30/252, r=0.05, sigma=0.19,
mu=0.14, q=0, N_PATHS=1000, SEED=42` and `run_hedging_experiment_vec`.

### Pricing and Greeks (call, q=0)

| Quantity   | Value      | Units              |
|------------|------------|--------------------|
| Call price | 9.0351     | USD                |
| Put price  | 7.1954     | USD                |
| d1 / d2    | 0.1236 / 0.0580 |               |
| Delta      | 0.5492     | shares/option      |
| Gamma      | 0.019481   | 1/USD              |
| Theta      | -0.1661    | USD/trading day    |
| Vega       | 42.346     | USD/unit vol       |
| Rho        | 19.192     | USD/unit rate      |
| PDE residual | -2.16e-15 | (machine epsilon) |

### Monte Carlo cross-check (simulated under Q with drift r-q)

| N paths | MC price | MC 95% CI       | BS analytical | Within CI? |
|---------|----------|------------------|----------------|------------|
| 200,000 | 9.0594   | +/- 0.057       | 9.0351         | Yes        |

This validation is implemented in the test suite (`test_mc_vs_analytical`).

### Hedging RMSE by rebalancing frequency (T_total = 30/252 nominal)

| Strategy             | dt       | N_steps | T_eff     | C0    | RMSE  | RMSE/C0 |
|----------------------|----------|---------|-----------|-------|-------|---------|
| Daily rebalancing    | 1/252    | 30      | 30/252    | 9.04  | 1.21  | 13.4%   |
| Weekly rebalancing   | 5/252    | 6       | 30/252    | 9.04  | 2.67  | 29.6%   |
| Buy-and-hold (no rebalance) | 21/252 | 1   | 21/252    | 7.43  | 5.14  | 69.2%   |

Note on the "buy-and-hold" row: with `N_steps=1`, the loop performs zero
rebalances. The effective maturity becomes 21/252 (a single hedge cannot
span 30 trading days at monthly cadence in this implementation). Both the
GBM simulation and the hedging engine use the same T_eff to preserve
time-accounting consistency.

Scaling: weekly/daily RMSE ratio = 2.21 vs theoretical sqrt(5) = 2.24,
consistent with the sqrt(dt) prediction of the Gamma P&L formula.

### Vol mismatch P&L (short call hedger, sigma_realized = 0.19)

| sigma_hedge | Mean P&L | % paths profitable |
|-------------|----------|---------------------|
| 0.15        | -1.68    | 6.8%                |
| 0.19        | -0.01    | 49.3%               |
| 0.23        | +1.66    | 91.6%               |

Breakeven at sigma_hedge = sigma_realized, as the Gamma P&L formula predicts.

---

## Caveats and known limitations

* **Realised volatility estimator.** The module uses close-to-close log
  returns. For OHLC data, range-based estimators (Parkinson, Garman-Klass,
  Yang-Zhang) have ~5x higher efficiency for the same sample size. Not
  implemented because the SPY CSV used here is close-only, but a production
  module on intraday data would prefer them.
* **Dividend yield.** Defaults to `q=0` for backward compatibility. For
  SPY-specific calculations, use `q ≈ 0.013` to match the empirical
  dividend yield over 2020-2024.
* **At-expiry edge case.** At `S = K` and `T -> 0`, delta is a Heaviside
  step function and not uniquely defined at the kink. The implementation
  returns delta = 1 (call) / -1 (put) for `S >= K` exactly. Callers
  needing precise at-expiry analytics should treat this case separately.
* **Implied vol solver.** Brent's method is robust but slower than
  Newton-Raphson (~20-40 iterations vs 4-6). For deep ITM options near
  expiry the time value is negligible and IV is mathematically
  ill-defined; the solver returns the lower bracket in those cases.
* **No transaction costs.** The hedging RMSE in this analysis ignores
  bid-ask spreads, commissions, and slippage. In practice the optimal
  rebalancing frequency is a trade-off between Gamma P&L variance and
  cumulative friction (Leland 1985).
* **Constant volatility assumption.** All experiments assume the same
  sigma over the option's life. In reality, vol is stochastic (cf. the
  82.7% peak vs 6.8% trough in the SPY realised vol time series).

---

## Data sourcing

This project uses **real SPY adjusted-close data only**. The notebook never
falls back to synthetic data; if no source is reachable, Section 1 raises
a `RuntimeError`.

### Automatic (recommended)

Section 1 downloads SPY via `yfinance` and caches to `data/SPY.csv`:

```python
yfinance.download("SPY", start="2020-01-01", end="2024-12-31", auto_adjust=True)
```

### Manual fallback

If `yfinance` is unavailable, download adjusted closes manually from
Yahoo Finance, Stooq, or any vendor and save as `data/SPY.csv` with
columns `Date,Close`. Section 1 detects the cache and skips the download.

---

## Test coverage

```
$ python -m pytest tests/
2060 passed, 120 skipped in ~9s
```

Coverage highlights:
* Put-call parity over a 5x4x2x4x3 = 480-point parameter grid.
* Black-Scholes-Merton PDE residual < 1e-9 over the same grid (call + put).
* Implied vol roundtrip (skipping cases where the time value < 1e-3 USD
  and IV is ill-defined).
* `bs_greeks_vec` matches `bs_greeks` in a loop to within 1e-12.
* `run_hedging_experiment_vec` matches `run_hedging_experiment` to within
  1e-10 absolute error per path.
* Hedging RMSE scales as sqrt(dt) (statistical test).
* MC pricing under Q with drift `r-q` matches BS analytical to within 4
  standard errors of the MC estimate.

---

## Project motivation

Options pricing is the quantification of a hedging cost. A practitioner who
cannot explain *why* the BS price equals the replication cost, or *how* the
P&L decomposes into Gamma and Theta contributions, has not understood the
model. This project demonstrates that understanding numerically, end to end,
from empirical volatility estimation through to per-path hedging error
attribution, with documented invariants and full reproducibility.

---

## References

* Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy*, 81(3), 637-654.
* Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal*,
  4(1), 141-183.
* Leland, H. E. (1985). Option Pricing and Replication with Transaction
  Costs. *Journal of Finance*, 40(5), 1283-1301.
* Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.).
