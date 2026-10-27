# Frozen market data (public sources)

These files are committed so that every result in the README can be regenerated
offline and bit-for-bit. They are never overwritten by the code; refreshing them
is a deliberate step (`python scripts/download_data.py`, then update the hashes in
`SHA256SUMS` and re-run `scripts/run_results.py`).

| File | Series | Source | Retrieved | Rows |
|---|---|---|---|---|
| `SPY_2020_2024.csv` | SPDR S&P 500 ETF, daily close adjusted for splits and dividends (`auto_adjust=True`) | Yahoo Finance via `yfinance` | 2026-10-12 | 1,257 trading days, 2020-01-02 to 2024-12-30 |
| `VIXCLS_2020_2024.csv` | CBOE Volatility Index (VIX), daily close, in vol points | FRED, series `VIXCLS` | 2026-10-28 | 1,304 business days, 2020-01-02 to 2024-12-31 |
| `DTB3_2020_2024.csv` | 3-Month Treasury Bill secondary market rate, discount basis, in percent | FRED, series `DTB3` | 2026-10-28 | 1,304 business days, 2020-01-02 to 2024-12-31 |

Notes

* The Yahoo request used `end="2024-12-31"`, which is exclusive, so the SPY series stops on 2024-12-30.
* Yahoo rescales the whole adjusted history whenever a new dividend is paid. A download made in June 2026
  differs from this file by a constant factor of 0.99496 on every price; log returns agree to 9e-7.
  This is why the file is frozen rather than re-downloaded on every run.
* FRED leaves the value empty on US holidays; the loaders drop those rows and align the series on SPY trading days.
* Because SPY is adjusted for dividends, it is a total-return series: hedging it with the dividend yield set
  to `q = 0` is the consistent choice.
* The data are redistributed only to make the research reproducible. Each series remains subject to its
  provider's terms: Yahoo Finance for SPY; FRED for DTB3 (U.S. Treasury data); for VIXCLS, FRED cites the
  Chicago Board Options Exchange as the copyright holder.
