# Changelog

## v1.0.0 — 2026-10-28

Rewrite after the Phase 0 audit (`docs/AUDIT.md`, in French). The public API changed; v0.2.0
remains available through its tag-less commit `6339479`.

### Fixed
* **σ → 0 limit** (audit C3): the degenerate branch used `S >= K` instead of the forward
  moneyness and returned wrong, even negative, prices below σ = 1e-9.
* **Dividends in the hedge** (C4): `q · Δ · S · dt` is now credited to the cash account; the old
  engine was biased by −0.29 on a 30-day ATM call on a 310 spot (t = −31.7) at q = 1.3 %.
* **Time grid** (M1): the hedging step is `T / n_steps`; the separate `dt` argument, which could
  silently disagree with the paths, is gone.
* **Implied vol** (M2): raises `ImpliedVolError` when the time value is below the price resolution
  instead of returning the lower bracket.
* **Input validation** (M3): S, K > 0, T, σ ≥ 0, finite values, exact `"call"` / `"put"`.
* **Theta units** (C1): every Greek is now a raw partial derivative (theta per year), which removes
  the factor-252 error of the old gamma / theta decomposition.
* README and notebook figures that did not match the code (M4), and a non-reproduced number in
  the removed `NOTEBOOK_PATCHES.md` (M9).

### Added
* Installable package `src/bspricer` (`pip install -e .`), pinned dependencies, MIT license.
* Exact P&L attribution of the hedge (initial mark, gamma, theta, carry, residual, costs) (C2),
  proportional transaction costs, separate selling and hedging volatilities, `variance_pnl`
  (dollar gamma × variance gap) and `kamal_derman_std`.
* `crr_price` (European and American CRR tree) and `mc_price` (exact sampling, antithetic option).
* `bspricer.backtest`: monthly short ATM option on real SPY closes, sold at the VIX.
* Frozen public data in `data/sample/` (SPY, VIX, 3-month T-bill) with SHA256 checksums.
* `scripts/run_results.py`: four results (replication error vs N, optimal rebalancing with costs,
  misspecified volatility, SPY backtest) written to `results/`, and the README result blocks
  generated from them.
* Test suite: identities, no-arbitrage bounds, limits, finite differences, PDE, CRR and Monte Carlo
  convergence orders, hedging oracle and attribution, regression on the frozen data, README and
  results consistency, notebook execution. ruff, mypy --strict and CI on Python 3.12–3.14.
* `notebooks/demo.ipynb`, an offline guided tour.

### Removed
* `bs_pricer.py` (split into modules), `bs_greeks_vec` (use `bs_greeks` with arrays),
  `delta_hedge_single_path` / `run_hedging_experiment(_vec)` (use `delta_hedge`),
  `requirements.txt`, `conftest.py`, `NOTEBOOK_PATCHES.md`, the legacy notebook and its HTML export.

## v0.2.0 — Post-audit improvements

### Added
* **Continuous dividend yield `q`** as a parameter to all pricing, Greeks,
  and hedging functions (default `q=0.0` preserves backward compatibility).
  Required for accurate SPY-based pricing — the q=0.013 dividend yield
  corrects a ~2.9% systematic mispricing on ATM 30-day calls.
* **`bs_implied_vol`** — implied volatility solver via Brent's method.
  Globally convergent, with explicit arbitrage-bound checks. Roundtrip
  recovers input vol to ~1e-8.
* **`bs_greeks_vec`** — fully vectorised Greeks accepting numpy arrays for
  S, K, sigma. The scalar `bs_greeks` is preserved as a thin wrapper.
* **`run_hedging_experiment_vec`** — vectorised hedging engine. Loops over
  time steps but parallelises across paths. **~570x faster** than the
  scalar version (12.4s -> 22ms for 1000 paths x 30 steps). Numerically
  identical to the scalar version (max abs diff < 1e-10 per path).
* **`tests/test_bs_pricer.py`** — 2000+ pytest assertions including:
  put-call parity, Black-Scholes-Merton PDE residual < 1e-9 over a
  parameter grid, MC vs analytical pricing under Q, IV roundtrip,
  vec/scalar consistency, dividend monotonicity, hedging symmetry.
* **`.github/workflows/ci.yml`** — CI on Python 3.10, 3.11, 3.12 running
  doctests + pytest.

### Changed
* `bs_pricer.py` module docstring extended to document the dividend
  convention and reference Merton (1973) for the dividend-yield extension.
* `bs_pde_residual` now accepts `option_type` and `q` and verifies the
  generalised PDE: `Theta + 0.5*sigma^2*S^2*Gamma + (r-q)*S*Delta - r*C = 0`.
* `bs_price` now coerces 0-d numpy results to Python `float` (was leaking
  `np.float64`, contradicting the type hint).
* `option_type` typed as `Literal["call", "put"]` for static checker support.
* README rewritten: aligned with `requirements.txt` versions, documented
  the new API surface, added a "Caveats and known limitations" section
  (dividend yield, OHLC range estimators, transaction costs, constant-vol
  assumption), framed the project as "rigorously implemented" rather than
  "production-quality".
* `requirements.txt` switched from exact pins to `>=` constraints; pytest
  added.

### Fixed
* Module docstring previously claimed "All functions are vectorized where
  applicable" — `bs_greeks` was not. Now true with `bs_greeks_vec`.
* README and `requirements.txt` had divergent version numbers. Now aligned.

### Notebook patches (apply manually — see `NOTEBOOK_PATCHES.md`)
* Add MC vs analytical validation cell in Section 4.
* Switch Section 5 hedging call to `run_hedging_experiment_vec` (~500x faster).
* Rename Experiment 1 "Monthly rebalancing" row to "Buy-and-hold (N=1)"
  with a clarifying note (the loop runs zero rebalances at N_steps=1).
* Add put/call hedging symmetry sanity test in Section 6.
* Add IV roundtrip demonstration in a new Section 8.5.
* Add Parkinson/Garman-Klass/Yang-Zhang note in the realised-vol section.
