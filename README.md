# Black-Scholes pricer and delta-hedging simulator

[![CI](https://github.com/tancredeg/black-scholes-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/tancredeg/black-scholes-pricer/actions/workflows/ci.yml)

> **v1.0 rework in progress (October 2026).** The library has been rewritten and tested; the
> quantitative results and the full README are generated next by `scripts/run_results.py`.
> The Phase 0 audit of the previous version is in [`docs/AUDIT.md`](docs/AUDIT.md) (French).

`bspricer` prices European options in closed form (Black-Scholes-Merton with a dividend yield),
cross-checks the formulas against a CRR tree, Monte Carlo and finite differences, and simulates
the discrete delta hedge of a short option with dividends, transaction costs and a misspecified
volatility, with an exact gamma / theta / carry attribution of the P&L.

```bash
pip install -e ".[dev]"   # Python >= 3.12
python -m pytest          # 3,200+ checks: identities, limits, convergence, regression on frozen data
```

```python
import bspricer as bs

bs.bs_price(100.0, 100.0, 0.25, 0.03, 0.20, "call")  # 4.3576
paths = bs.simulate_gbm(100.0, 0.07, 0.20, 0.25, n_steps=63, n_paths=10_000, seed=0)
res = bs.delta_hedge(paths, K=100.0, T=0.25, r=0.03, sigma_hedge=0.20, decompose=True)
res.mean, res.std  # hedging P&L of the short call, daily rebalancing
```
