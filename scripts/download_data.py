"""Re-download the public market data into ``data/sample/`` (optional, needs network).

The committed files are the reference for every published number. Running this
script overwrites them; check the result with ``sha256sum -c data/sample/SHA256SUMS``
and re-run ``scripts/run_results.py`` before committing new data.

    pip install yfinance
    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"


def download_spy(start: str, end: str) -> pd.Series:
    import yfinance as yf  # optional dependency

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data for SPY")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna()
    close.index.name = "Date"
    close.name = "Close"
    return close


def download_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    return pd.read_csv(FRED.format(sid=series_id, start=start, end=end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    download_spy(args.start, args.end).to_csv(SAMPLE / "SPY_2020_2024.csv")
    for sid in ("VIXCLS", "DTB3"):
        download_fred(sid, args.start, args.end).to_csv(
            SAMPLE / f"{sid}_2020_2024.csv", index=False
        )
    print(f"Wrote data to {SAMPLE}. Verify with: sha256sum -c data/sample/SHA256SUMS")


if __name__ == "__main__":
    main()
