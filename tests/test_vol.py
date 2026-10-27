"""Realised-volatility estimators and regression tests on the frozen market data."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

import bspricer as bs


@pytest.fixture(scope="module")
def spy(data_dir) -> pd.Series:
    return bs.load_price_series(data_dir / "SPY_2020_2024.csv")


def test_frozen_files_match_their_checksums(data_dir):
    """data/sample is the reference for every published number; it must never change silently."""
    for line in (data_dir / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == digest, name


def test_spy_series_integrity(spy):
    assert len(spy) == 1257
    assert spy.index[0] == pd.Timestamp("2020-01-02") and spy.index[-1] == pd.Timestamp("2024-12-30")
    assert spy.index.is_monotonic_increasing and not spy.index.has_duplicates
    assert np.isfinite(spy.to_numpy()).all() and (spy > 0).all()


def test_fred_loaders_drop_holidays(data_dir):
    vix = bs.load_fred_series(data_dir / "VIXCLS_2020_2024.csv")
    tbill = bs.load_fred_series(data_dir / "DTB3_2020_2024.csv")
    assert not vix.isna().any() and not tbill.isna().any()
    assert vix.loc["2020-04-08"] == 43.35  # implied vol on the day realised 30d vol peaked
    assert tbill.min() == -0.05 and tbill.idxmin() == pd.Timestamp("2020-03-26")  # briefly negative in March 2020
    assert tbill.max() == 5.36


def test_rolling_vol_regression(spy):
    """Section 1 figures of the original notebook, recomputed on the frozen file."""
    vol = bs.rolling_realized_vol(spy, windows=[30, 60, 252])
    v30, v252 = vol["vol_30d"].dropna(), vol["vol_252d"].dropna()
    assert v30.max() == pytest.approx(0.8269172667052918, rel=1e-10)
    assert v30.idxmax() == pd.Timestamp("2020-04-08")
    assert v30.min() == pytest.approx(0.06844494082406055, rel=1e-10)
    assert v30.idxmin() == pd.Timestamp("2024-07-16")
    assert v252.index[0] == pd.Timestamp("2020-12-31")  # the 252d estimate starts a year in
    gap = (v30[v252.index] - v252).abs()
    assert gap.max() == pytest.approx(0.2399008468885961, rel=1e-10)
    assert gap.idxmax() == pd.Timestamp("2021-01-14")


def test_realized_vol_estimators(spy):
    zero_mean = bs.realized_vol(spy)
    sample = bs.log_returns(spy).std(ddof=1) * math.sqrt(252)
    assert zero_mean == pytest.approx(0.2109995716507011, rel=1e-10)
    assert abs(zero_mean - sample) < 1e-3  # the daily mean is negligible against the daily vol


def test_realized_vol_of_a_known_path():
    returns = np.array([0.01, -0.01, 0.02, -0.02])
    path = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    assert bs.realized_vol(path) == pytest.approx(math.sqrt(252 * np.mean(returns**2)))
    with pytest.raises(ValueError):
        bs.realized_vol(np.array([100.0]))


def test_loader_rejects_bad_files(tmp_path):
    bad = tmp_path / "dup.csv"
    bad.write_text("Date,Close\n2024-01-02,1\n2024-01-02,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        bs.load_price_series(bad)
    neg = tmp_path / "neg.csv"
    neg.write_text("Date,Close\n2024-01-02,1\n2024-01-03,-2\n")
    with pytest.raises(ValueError, match="non-positive"):
        bs.load_price_series(neg)
