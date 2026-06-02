# Black-Scholes Options Pricer and Delta Hedging Simulator

A production-quality Python implementation of Black-Scholes pricing theory,
Greeks analytics, GBM Monte Carlo simulation, and discrete delta-hedging.
Built to demonstrate desk-level command of vanilla options mechanics to
quantitative recruiters evaluating candidates for equity derivatives,
structured products, or risk management roles.

---

## Repo Structure

```
/
├── black_scholes.ipynb   ← Main notebook (Sections 0–8, fully executable)
├── bs_pricer.py          ← Importable module: all pricing and hedging logic
├── README.md
├── requirements.txt
└── data/
    ├── .gitkeep          ← Empty placeholder (SPY.csv is gitignored)
    └── SPY.csv           ← Populated on first run via yfinance
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

### Price a vanilla option in three lines

```python
import bs_pricer as bs

call_price = bs.bs_price(S=310.0, K=310.0, T=30/252, r=0.05, sigma=0.19, option_type='call')
greeks     = bs.bs_greeks(S=310.0, K=310.0, T=30/252, r=0.05, sigma=0.19, option_type='call')
print(f"Call: ${call_price:.4f} | Delta: {greeks['delta']:.4f} | Gamma: {greeks['gamma']:.6f}")
# Call: $9.0351 | Delta: 0.5492 | Gamma: 0.019481
```

### Run a delta-hedging experiment

```python
import bs_pricer as bs

# Simulate 1000 GBM paths, 30 daily steps
paths = bs.simulate_gbm(S0=310.0, mu=0.14, sigma=0.19, T=30/252,
                        N_steps=30, N_paths=1000, seed=42)

# Run the daily delta-hedging replication
errors, rmse, mean_err = bs.run_hedging_experiment(
    S_paths=paths, K=310.0, T_total=30/252, r=0.05, sigma=0.19
)
print(f"RMSE: ${rmse:.4f} ({rmse / 9.0351 * 100:.1f}% of premium)")
# RMSE: $1.2092 (13.4% of premium)
```

### Compute rolling realized volatility

```python
import yfinance as yf
import bs_pricer as bs

prices = yf.download("SPY", start="2020-01-01", end="2024-12-31",
                     auto_adjust=True)["Close"].dropna()
vol_df = bs.rolling_realized_vol(prices, windows=[30, 60, 252])
print(vol_df.tail(3))
```

---

## Mathematical Background

### Black-Scholes Pricing Formula

For a European call on a non-dividend-paying stock under the risk-neutral measure Q:

    C = S * N(d1) - K * exp(-r*T) * N(d2)

    d1 = [ ln(S/K) + (r + sigma^2/2) * T ] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

Put price by put-call parity:

    P = C - S + K * exp(-r*T)

### Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta | N(d1) | Shares held per option in the replicating portfolio |
| Gamma | N'(d1) / (S * sigma * sqrt(T)) | Rate of change of delta per $1 move in S |
| Theta | [per trading day] | Option value lost per trading day |
| Vega  | S * sqrt(T) * N'(d1) | Price sensitivity to a 1-unit vol change |
| Rho   | K * T * exp(-r*T) * N(d2) | Price sensitivity to a 1-unit rate change |

**Theta convention throughout this codebase:** per trading day (divided by 252),
so that `theta_pnl = theta * dt` with `dt = 1/252` directly.

### Delta Hedging

The BS price is the exact cost of the continuous replication strategy:
sell the option, buy delta shares, finance the difference at rate r,
and rebalance continuously. In practice, daily rebalancing introduces
a hedging error that scales as sqrt(dt) — quantified in Section 5 and
Experiment 1.

---

## Key Results

All numbers produced by running `black_scholes.ipynb` with the default parameters:
`S0=310, K=310, T=30/252, r=0.05, sigma=0.19, mu=0.14, N_PATHS=1000, SEED=42`.

### Default Parameters — Pricing

| Quantity    | Value     |
|-------------|-----------|
| Call price  | $9.0351   |
| Put price   | $7.1954   |
| d1          | 0.1236    |
| d2          | 0.0580    |
| N(d1)       | 0.5492    |
| N(d2)       | 0.5231    |

### Greeks Table

| Greek | Value     | Units               |
|-------|-----------|---------------------|
| Delta | 0.5492    | shares/option       |
| Gamma | 0.019481  | 1/USD               |
| Theta | -0.1661   | USD/trading day     |
| Vega  | 42.346    | USD/unit vol        |
| Rho   | 19.192    | USD/unit rate       |

### Hedging RMSE by Rebalancing Frequency

| Frequency | dt       | N_steps | T_eff       | C0 ($)  | RMSE ($) | RMSE/C0 (%) |
|-----------|----------|---------|-------------|---------|----------|-------------|
| Daily     | 1/252    | 30      | 30/252      | 9.0351  | 1.2092   | 13.38%      |
| Weekly    | 5/252    | 6       | 30/252      | 9.0351  | 2.6727   | 29.58%      |
| Monthly   | 21/252   | 1       | 21/252 (*)  | 7.4323  | 5.1404   | 69.16%      |

(*) For the monthly case, N_steps = max(1, round(T/dt)) = 1, so the effective
maturity becomes 21/252 (a single rebalance cannot cleanly span a 30-trading-day
option at monthly cadence). Both the GBM simulation and hedging engine use the
same T_eff, preserving time-accounting consistency.

RMSE scaling: weekly/daily ratio = 2.21 (theoretical sqrt(5) = 2.24). Consistent
with the Gamma P&L formula's sqrt(dt) prediction.

### Vol Mismatch P&L — Short Call Hedger (sigma_realized = 0.19)

| sigma_h | Mean P&L ($) | % Profitable |
|---------|-------------|--------------|
| 0.15    | -1.6797     | 6.8%         |
| 0.17    | -0.8484     | 21.5%        |
| 0.19    | -0.0136     | 49.3%        |
| 0.21    | +0.8242     | 75.1%        |
| 0.23    | +1.6643     | 91.6%        |

Breakeven at sigma_h = sigma_realized, as the Gamma P&L formula predicts.

---

## Data Sourcing

This project uses **real SPY adjusted-close data only**. The notebook never
falls back to synthetic data; if no source is reachable, Section 1 raises a
`RuntimeError` with explicit instructions.

### Automatic (recommended)

Section 1 downloads SPY automatically via `yfinance` and caches to `data/SPY.csv`
for offline reruns:

```python
yfinance.download("SPY", start="2020-01-01", end="2024-12-31", auto_adjust=True)
```

Requires internet access on first run. Subsequent runs use the local cache.

### Manual fallback

If `yfinance` is unavailable (corporate firewall, rate limit, API change), download
SPY adjusted closes manually from any of:

- **Yahoo Finance**: https://finance.yahoo.com/quote/SPY/history/ → set range
  2020-01-01 to 2024-12-31 → "Download"
- **Stooq**: https://stooq.com/q/d/?s=spy.us → daily, full history
- **Bloomberg / Refinitiv / any vendor**: export adjusted closes

Save the file as `data/SPY.csv` with this schema:

```
Date,Close
2020-01-02,323.54
2020-01-03,321.55
...
```

Then rerun the notebook — Section 1 detects the cache and skips the download.

### Why no synthetic data

For a portfolio project, fabricated price paths in plots labelled "SPY 2020–2024"
would be a material misrepresentation. The codebase refuses to substitute
generated data for the empirical analysis in Section 1 and Section 8.

---

## Project Motivation

Options pricing is not just formula application — it is the quantification
of a hedging cost. A practitioner who cannot explain *why* the BS price
equals the replication cost, or *how* the P&L decomposes into gamma and
theta, has not understood the model. This project demonstrates that
understanding numerically, end to end, from empirical volatility estimation
through to per-path hedging error attribution.

---

## Requirements

See `requirements.txt`. Key dependencies:

```
numpy==1.26.4
scipy==1.13.0
pandas==2.2.2
matplotlib==3.9.0
yfinance==0.2.44
jupyter==1.0.0
```
