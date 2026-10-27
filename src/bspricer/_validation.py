"""Input validation shared by all public functions (internal)."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

OptionType: TypeAlias = Literal["call", "put"]
FloatArray: TypeAlias = NDArray[np.float64]


def check_option_type(option_type: str) -> OptionType:
    """Return ``option_type`` if it is exactly ``"call"`` or ``"put"``, else raise."""
    if option_type == "call":
        return "call"
    if option_type == "put":
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def as_float_array(
    name: str,
    value: ArrayLike,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    """Convert ``value`` to a float64 array and check that it is finite (and signed as required)."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return arr


def to_output(arr: FloatArray) -> float | FloatArray:
    """Return a Python float for 0-d results and the array otherwise."""
    return float(arr) if arr.ndim == 0 else arr
