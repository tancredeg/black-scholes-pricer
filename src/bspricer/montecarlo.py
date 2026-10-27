r"""Geometric Brownian motion simulation and Monte Carlo pricing.

Paths use the exact log-normal transition

.. math:: S_{t+\Delta t} = S_t \exp\big[(\mu - \tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\big],
          \quad Z \sim \mathcal N(0, 1),

so the marginals are exact for any step size (no Euler bias). ``mu`` is the
drift of the *price*: use the real-world drift for hedging experiments and
``r - q`` for risk-neutral pricing.

Reference: Glasserman, P. (2003). *Monte Carlo Methods in Financial
Engineering*, ch. 3.2 and 4.2. Springer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from bspricer._validation import FloatArray, OptionType, as_float_array, check_option_type

__all__ = ["MCEstimate", "mc_price", "simulate_gbm"]


class MCEstimate(NamedTuple):
    """Monte Carlo estimate with its standard error."""

    price: float
    std_error: float
    n_paths: int


def _normals(rng: np.random.Generator, shape: tuple[int, int], antithetic: bool) -> FloatArray:
    if not antithetic:
        return rng.standard_normal(shape)
    if shape[0] % 2:
        raise ValueError("n_paths must be even when antithetic=True")
    half = rng.standard_normal((shape[0] // 2, shape[1]))
    return np.concatenate([half, -half], axis=0)


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    *,
    antithetic: bool = False,
) -> FloatArray:
    """Simulate GBM paths on the uniform grid ``t_k = k T / n_steps``.

    Returns an array of shape ``(n_paths, n_steps + 1)``; column 0 equals ``S0``.
    With ``antithetic=True`` the second half of the paths uses the negated normals
    of the first half (``n_paths`` must be even).

    Examples
    --------
    >>> paths = simulate_gbm(100.0, 0.05, 0.2, 1.0, 12, 4, seed=0)
    >>> paths.shape, bool((paths[:, 0] == 100.0).all())
    ((4, 13), True)
    """
    as_float_array("S0", S0, positive=True)
    as_float_array("sigma", sigma, nonnegative=True)
    as_float_array("T", T, positive=True)
    as_float_array("mu", mu)
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = _normals(rng, (n_paths, n_steps), antithetic)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return np.asarray(S0 * np.exp(log_paths), dtype=np.float64)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int,
    option_type: OptionType = "call",
    q: float = 0.0,
    *,
    antithetic: bool = False,
) -> MCEstimate:
    r"""Risk-neutral Monte Carlo price of a European option.

    Samples :math:`S_T` exactly under :math:`\mathbb Q` (drift ``r - q``) and
    returns the discounted mean payoff with its standard error. With
    ``antithetic=True`` the standard error is computed on the pair averages,
    which are i.i.d.

    Examples
    --------
    >>> est = mc_price(100.0, 100.0, 1.0, 0.03, 0.2, 100_000, seed=1)
    >>> est.n_paths
    100000
    """
    otype = check_option_type(option_type)
    as_float_array("K", K, positive=True)
    terminal = simulate_gbm(S, r - q, sigma, T, 1, n_paths, seed, antithetic=antithetic)[:, -1]
    payoff = np.maximum(terminal - K, 0.0) if otype == "call" else np.maximum(K - terminal, 0.0)
    discounted = np.exp(-r * T) * payoff
    if antithetic:
        half = n_paths // 2
        samples = 0.5 * (discounted[:half] + discounted[half:])
    else:
        samples = discounted
    std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return MCEstimate(price=float(samples.mean()), std_error=std_error, n_paths=n_paths)
