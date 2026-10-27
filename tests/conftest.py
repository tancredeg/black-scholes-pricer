"""Shared fixtures and parameter grids."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make tests/reference.py importable

SPOTS = [60.0, 85.0, 100.0, 115.0, 160.0]
STRIKE = 100.0
MATURITIES = [1 / 252, 0.1, 0.5, 2.0]
RATES = [0.0, 0.05]
VOLS = [0.05, 0.2, 0.6]
YIELDS = [0.0, 0.03]
OPTION_TYPES = ["call", "put"]

GRID = list(itertools.product(SPOTS, MATURITIES, RATES, VOLS, YIELDS))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA
