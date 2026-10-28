"""Regenerate every table and figure of the README from the code and the frozen data.

    python scripts/run_results.py             # full run (about 1 minute), rewrites results/ and README
    python scripts/run_results.py --quick     # small sample, for smoke tests
    python scripts/run_results.py --quick --out /tmp/results --no-readme

Results
-------
R1  Replication error of a daily-to-intraday delta hedge versus the number of rebalances N
    (Monte Carlo, common random numbers), log-log slope against the 1/sqrt(N) law.
R2  Proportional transaction costs: RMSE of the total hedged P&L versus N, optimal N*.
R3  Misspecified volatility: P&L when selling and hedging at sigma_i != sigma_r, its
    explanation by dollar gamma x variance gap, path dependence, choice of hedge vol.
R4  Real data: monthly short ATM option on SPY 2020-2024 sold at the VIX and delta-hedged
    daily on the observed closes.

Every random draw uses a fixed seed; every number printed in the README is read from
results/summary.json, which this script writes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import bspricer as bs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_block  # noqa: E402

# ---- visual identity (reference palette of the dataviz method, light mode) -------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"  # diverging poles (blue / red)


@dataclass(frozen=True)
class Config:
    """Base case shared by the Monte Carlo results (inputs, not results)."""

    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25  # 63 trading days
    r: float = 0.03
    sigma: float = 0.20  # realised (simulation) volatility
    mu: float = 0.07  # real-world drift of the simulated paths
    n_paths: int = 20_000
    n_fine: int = 1008  # finest grid: 16 rebalances per trading day
    seed: int = 20261028
    sigma_rich: float = 0.24  # selling vol used for path-dependence and hedge-vol choice

    @property
    def daily_step(self) -> int:
        return self.n_fine // 63


FULL = Config()
QUICK = Config(n_paths=2_000, n_fine=252)
R1_GRID = (4, 7, 9, 12, 21, 42, 63, 126, 252, 504, 1008)
COSTS_BP = (0, 1, 2, 5, 10, 20)
SIGMA_GRID = (0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )


def _save(fig: Any, path: Path, title: str, description: str) -> None:
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "Tancrède Geli",
            "Description": description,
            "Copyright": "Tancrède Geli, MIT License",
            "Software": f"bspricer {bs.__version__} (scripts/run_results.py)",
        },
    )
    plt.close(fig)


def _divisors(n: int, minimum: int = 4) -> list[int]:
    return [d for d in range(minimum, n + 1) if n % d == 0]


# ---- R1 ---------------------------------------------------------------------------------------


def r1_replication(fine: np.ndarray, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    grid = [n for n in R1_GRID if cfg.n_fine % n == 0]
    premium = float(bs.bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma))
    vega = float(bs.bs_greeks(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma)["vega"])
    pnl = np.column_stack(
        [
            bs.delta_hedge(fine[:, :: cfg.n_fine // n], cfg.K, cfg.T, cfg.r, cfg.sigma).pnl
            for n in grid
        ]
    )
    sd = pnl.std(axis=0, ddof=1)
    kd = np.array([bs.kamal_derman_std(vega, cfg.sigma, n) for n in grid])
    table = pd.DataFrame(
        {
            "n_rebalances": grid,
            "rebalances_per_day": np.array(grid) / 63.0,
            "pnl_mean": pnl.mean(axis=0),
            "pnl_mean_se": sd / math.sqrt(pnl.shape[0]),
            "pnl_sd": sd,
            "pnl_sd_pct_premium": 100.0 * sd / premium,
            "kamal_derman_sd": kd,
            "sd_over_kamal_derman": sd / kd,
        }
    )
    log_n = np.log(grid)
    asym = np.array(grid) >= 63  # daily or finer: the regime of the asymptotic N^(-1/2) law
    slope_all = np.polyfit(log_n, np.log(sd), 1)[0]
    slope = np.polyfit(log_n[asym], np.log(sd[asym]), 1)[0]
    rng = np.random.default_rng(cfg.seed + 1)
    boot_all, boot = [], []
    for _ in range(400):
        resampled = np.log(pnl[rng.integers(0, pnl.shape[0], pnl.shape[0])].std(axis=0, ddof=1))
        boot_all.append(np.polyfit(log_n, resampled, 1)[0])
        boot.append(np.polyfit(log_n[asym], resampled[asym], 1)[0])
    daily = table.loc[table["n_rebalances"] == 63].iloc[0]
    summary = {
        "premium": premium,
        "vega": vega,
        "slope": float(slope),
        "slope_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "slope_all": float(slope_all),
        "slope_all_ci95": [float(v) for v in np.percentile(boot_all, [2.5, 97.5])],
        "coarsest_n": int(grid[0]),
        "coarsest_ratio_kd": float(table["sd_over_kamal_derman"].iloc[0]),
        "n_paths": int(pnl.shape[0]),
        "daily_sd": float(daily["pnl_sd"]),
        "daily_sd_pct_premium": float(daily["pnl_sd_pct_premium"]),
        "daily_ratio_kd": float(daily["sd_over_kamal_derman"]),
        "finest_n": int(grid[-1]),
        "finest_sd_pct_premium": float(table["pnl_sd_pct_premium"].iloc[-1]),
        "max_abs_mean_t": float(np.max(np.abs(table["pnl_mean"] / table["pnl_mean_se"]))),
    }
    # The same paths are reused at every N, so the mean-P&L t-stats above are not independent.
    # Independent check of unbiasedness: fresh paths, daily hedge.
    check = bs.simulate_gbm(
        cfg.S0, cfg.mu, cfg.sigma, cfg.T, 63, 10 * cfg.n_paths, seed=cfg.seed + 2
    )
    res = bs.delta_hedge(check, cfg.K, cfg.T, cfg.r, cfg.sigma)
    summary["independent_n_paths"] = int(check.shape[0])
    summary["independent_mean"] = res.mean
    summary["independent_mean_t"] = res.mean / res.std_error
    return table, summary


def fig_r1(table: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    n = table["n_rebalances"].to_numpy()
    ax.plot(
        n,
        table["kamal_derman_sd"],
        color=SERIES[1],
        label=r"Kamal-Derman: $\sqrt{\pi/4}\,\sigma\,\mathcal{V}/\sqrt{N}$",
    )
    ax.plot(
        n,
        table["pnl_sd"],
        color=SERIES[0],
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"Simulated, {summary['n_paths']:,} paths",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(v)) for v in n])
    ax.minorticks_off()
    for label, x in (("weekly", 12), ("daily", 63), ("16 / day", 1008)):
        if x in n:
            y = float(table.loc[table["n_rebalances"] == x, "pnl_sd"].iloc[0])
            ax.annotate(
                label, (x, y), xytext=(6, 8), textcoords="offset points", color=INK_2, fontsize=8.5
            )
    ax.set_xlabel("Rebalances over the option life, N (log scale)")
    ax.set_ylabel("Std. dev. of hedged P&L, per 100 notional (log)")
    ax.set_title("R1. The replication error decays like $N^{-1/2}$")
    ax.text(
        0.98,
        0.96,
        f"log-log slope {summary['slope']:.3f} for N ≥ 63\n{summary['slope_all']:.3f} over all N",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK_2,
        fontsize=9,
    )
    ax.legend(loc="lower left")
    _save(
        fig,
        path,
        "R1 replication error versus rebalancing frequency",
        "Std. dev. of the P&L of a delta-hedged short ATM call versus the number of rebalances, against Kamal-Derman.",
    )


# ---- R2 ---------------------------------------------------------------------------------------


def r2_costs(fine: np.ndarray, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    grid = _divisors(cfg.n_fine)
    rows = []
    for n in grid:
        # With cost = 1 the engine returns the compounded traded notional ("turnover");
        # the P&L at any cost c is then pnl_0 - c * turnover exactly (costs are linear in c).
        res = bs.delta_hedge(fine[:, :: cfg.n_fine // n], cfg.K, cfg.T, cfg.r, cfg.sigma, cost=1.0)
        turnover = res.costs
        pnl0 = res.pnl + turnover
        for c_bp in COSTS_BP:
            pnl = pnl0 - c_bp * 1e-4 * turnover
            rows.append(
                {
                    "cost_bp": c_bp,
                    "n_rebalances": n,
                    "mean_cost": float(c_bp * 1e-4 * turnover.mean()),
                    "pnl_mean": float(pnl.mean()),
                    "pnl_sd": float(pnl.std(ddof=1)),
                    "pnl_rmse": float(np.sqrt(np.mean(pnl**2))),
                    "turnover_mean": float(turnover.mean()),
                }
            )
    table = pd.DataFrame(rows)
    # Leading-order model. Hedge error: sd^2 ~ a / N (R1). Traded notional: a fixed part F
    # (buying the initial delta, selling the final one) plus a rebalancing part b sqrt(N)
    # (sum of |Gamma dS| ~ sqrt(N)). RMSE^2 ~ a / N + c^2 (F + b sqrt(N))^2 is minimal where
    # a / N^2 = c^2 b (F / sqrt(N) + b); for F << b sqrt(N), N* = sqrt(a) / (b c), i.e. N* ~ 1/c.
    base = table[table["cost_bp"] == 0]
    daily_or_finer = base[base["n_rebalances"] >= 63]
    a = float(np.mean(daily_or_finer["pnl_sd"] ** 2 * daily_or_finer["n_rebalances"]))
    b, fixed = (
        float(v) for v in np.polyfit(np.sqrt(base["n_rebalances"]), base["turnover_mean"], 1)
    )
    fit_error = float(
        np.max(
            np.abs(base["turnover_mean"] - fixed - b * np.sqrt(base["n_rebalances"]))
            / base["turnover_mean"]
        )
    )

    def n_star_theory(c: float) -> float:
        def foc(log_n: float) -> float:
            n = math.exp(log_n)
            return -a / n**2 + c**2 * b * (fixed / math.sqrt(n) + b)

        return math.exp(brentq(foc, math.log(1.0), math.log(1e7)))

    optimum = []
    for c_bp in COSTS_BP[1:]:
        sub = table[table["cost_bp"] == c_bp].reset_index(drop=True)
        best = sub.loc[sub["pnl_rmse"].idxmin()]
        optimum.append(
            {
                "cost_bp": c_bp,
                "n_star": int(best["n_rebalances"]),
                "n_star_theory": n_star_theory(c_bp * 1e-4),
                "rmse_at_n_star": float(best["pnl_rmse"]),
                "mean_cost_at_n_star": float(best["mean_cost"]),
                "rmse_daily": float(sub.loc[sub["n_rebalances"] == 63, "pnl_rmse"].iloc[0]),
                "at_grid_edge": bool(best["n_rebalances"] == grid[-1]),
            }
        )
    interior = [o for o in optimum if not o["at_grid_edge"]]
    exponent = np.polyfit(
        np.log([o["cost_bp"] for o in interior]), np.log([o["n_star"] for o in interior]), 1
    )[0]
    return table, {
        "a": a,
        "b": b,
        "fixed_turnover": fixed,
        "turnover_fit_max_rel_error": fit_error,
        "initial_trade_notional": float(
            bs.bs_greeks(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma)["delta"]
        )
        * cfg.S0,
        "n_star_cost_exponent": float(exponent),
        "grid": grid,
        "optimum": optimum,
    }


def fig_r2(table: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    shown = (0, 2, 5, 20)
    best = {o["cost_bp"]: o for o in summary["optimum"]}
    for color, c_bp in zip(SERIES, shown, strict=True):
        sub = table[table["cost_bp"] == c_bp]
        ax.plot(
            sub["n_rebalances"],
            sub["pnl_rmse"],
            color=color,
            label=f"{c_bp} bp" if c_bp else "no cost",
        )
        if c_bp in best and not best[c_bp]["at_grid_edge"]:
            o = best[c_bp]
            ax.plot(
                o["n_star"],
                o["rmse_at_n_star"],
                "o",
                color=color,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            ax.annotate(
                f"N* = {o['n_star']}",
                (o["n_star"], o["rmse_at_n_star"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color=INK_2,
                fontsize=8.5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [n for n in (4, 12, 21, 63, 126, 252, 504, 1008) if n <= summary["grid"][-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Rebalances over the option life, N (log scale)")
    ax.set_ylabel("RMSE of total hedged P&L incl. costs (log)")
    ax.set_title("R2. Proportional costs create an optimal rebalancing frequency")
    ax.legend(title="one-way cost", title_fontsize=8.5, loc="upper center", ncols=4)
    _save(
        fig,
        path,
        "R2 optimal rebalancing frequency with transaction costs",
        "RMSE of the P&L of a delta-hedged short ATM call including proportional costs, versus the number of rebalances.",
    )


# ---- R3 ---------------------------------------------------------------------------------------


def r3_vol_mismatch(
    fine: np.ndarray, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    daily = fine[:, :: cfg.daily_step]
    v_true = float(bs.bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma))
    rows = []
    for s_i in SIGMA_GRID:
        res = bs.delta_hedge(daily, cfg.K, cfg.T, cfg.r, s_i)
        predicted = bs.variance_pnl(daily, cfg.K, cfg.T, cfg.r, s_i, sigma_realized=cfg.sigma)
        edge = (float(bs.bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, s_i)) - v_true) * math.exp(
            cfg.r * cfg.T
        )
        rows.append(
            {
                "sigma_implied": s_i,
                "pnl_mean": res.mean,
                "pnl_mean_se": res.std_error,
                "predicted_mean": float(predicted.mean()),
                "premium_edge": edge,
                "pnl_sd": res.std,
                "prob_profit": float((res.pnl > 0).mean()),
            }
        )
    mismatch = pd.DataFrame(rows)

    s_i = cfg.sigma_rich
    res = bs.delta_hedge(daily, cfg.K, cfg.T, cfg.r, s_i)
    proxy = bs.variance_pnl(daily, cfg.K, cfg.T, cfg.r, s_i)
    realized = np.sqrt(252.0 * np.mean(np.diff(np.log(daily), axis=1) ** 2, axis=1))
    r2_raw = float(np.corrcoef(s_i**2 - realized**2, res.pnl)[0, 1] ** 2)
    r2_proxy = float(np.corrcoef(proxy, res.pnl)[0, 1] ** 2)
    slope, intercept = np.polyfit(proxy, res.pnl, 1)
    bucket = np.abs(realized - cfg.sigma) < 0.005
    near_strike = np.abs(np.log(daily[:, -1] / cfg.K)) < 0.02
    p5, p95 = np.percentile(res.pnl[bucket], [5, 95])

    choice_rows = []
    target = (float(bs.bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, s_i)) - v_true) * math.exp(
        cfg.r * cfg.T
    )
    for label, hedge_vol in (("implied", s_i), ("realised", cfg.sigma)):
        for n in (63, cfg.n_fine):
            out = bs.delta_hedge(
                fine[:, :: cfg.n_fine // n], cfg.K, cfg.T, cfg.r, hedge_vol, sigma_price=s_i
            )
            choice_rows.append(
                {
                    "hedge_vol": label,
                    "sigma_hedge": hedge_vol,
                    "n_rebalances": n,
                    "pnl_mean": out.mean,
                    "pnl_sd": out.std,
                    "prob_loss": float((out.pnl < 0).mean()),
                    "target_edge": target,
                }
            )
    choice = pd.DataFrame(choice_rows)
    scatter = pd.DataFrame({"pnl": res.pnl, "proxy": proxy, "realized_vol": realized})
    summary = {
        "sigma_rich": s_i,
        "r2_raw_variance_gap": r2_raw,
        "r2_dollar_gamma_proxy": r2_proxy,
        "proxy_slope": float(slope),
        "proxy_intercept": float(intercept),
        "bucket_n": int(bucket.sum()),
        "bucket_p5": float(p5),
        "bucket_p95": float(p95),
        "bucket_mean_near_strike": float(res.pnl[bucket & near_strike].mean()),
        "bucket_mean_far": float(res.pnl[bucket & ~near_strike].mean()),
        "max_abs_mean_minus_predicted_in_se": float(
            np.max(
                np.abs(mismatch["pnl_mean"] - mismatch["predicted_mean"]) / mismatch["pnl_mean_se"]
            )
        ),
    }
    return mismatch, choice, summary, scatter


def fig_r3(scatter: pd.DataFrame, summary: dict[str, Any], cfg: Config, path: Path) -> None:
    sample = scatter.sample(n=min(4000, len(scatter)), random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    ax = axes[0]
    ax.scatter(
        sample["realized_vol"] * 100, sample["pnl"], s=6, color=SERIES[0], alpha=0.35, linewidths=0
    )
    ax.set_xlabel("Realised volatility of the path (%)")
    ax.set_ylabel("Hedged P&L, per 100 notional")
    ax.set_title(f"Realised vol alone: R² = {summary['r2_raw_variance_gap']:.2f}")
    ax = axes[1]
    ax.scatter(sample["proxy"], sample["pnl"], s=6, color=SERIES[0], alpha=0.35, linewidths=0)
    lims = np.array([sample["proxy"].min(), sample["proxy"].max()])
    ax.plot(lims, lims, color=INK_2, linewidth=1.0, label="y = x")
    ax.set_xlabel(
        r"$\sum \frac{1}{2}\Gamma S^2(\sigma_i^2\Delta t - \delta S^2/S^2)$ along the path"
    )
    ax.set_title(f"Dollar-gamma-weighted variance gap: R² = {summary['r2_dollar_gamma_proxy']:.2f}")
    ax.legend(loc="upper left")
    fig.suptitle(
        f"R3. Sold at {cfg.sigma_rich:.0%}, realised {cfg.sigma:.0%}: the P&L depends on where the variance is realised",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    _save(
        fig,
        path,
        "R3 path dependence of the hedged P&L under a misspecified volatility",
        "Hedged P&L of a short ATM call sold at 24% on 20% paths, against realised vol and against the dollar-gamma-weighted variance gap.",
    )


# ---- R4 ---------------------------------------------------------------------------------------


def r4_spy(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market = bs.align_market_data(
        bs.load_price_series(data_dir / "SPY_2020_2024.csv"),
        bs.load_fred_series(data_dir / "VIXCLS_2020_2024.csv"),
        bs.load_fred_series(data_dir / "DTB3_2020_2024.csv"),
    )
    trades = bs.short_vol_backtest(market, horizon=21)
    pnl = trades["pnl"]
    n = len(trades)
    worst = trades.loc[pnl.idxmin()]
    best = trades.loc[pnl.idxmax()]
    robustness = []
    for label, kwargs in (
        ("baseline", {}),
        ("sold 2 vol points below VIX", {"vol_shift": -0.02}),
        ("1 bp one-way cost", {"cost": 1e-4}),
    ):
        t = trades if not kwargs else bs.short_vol_backtest(market, horizon=21, **kwargs)
        robustness.append(
            {
                "variant": label,
                "n_trades": len(t),
                "pnl_mean": float(t["pnl"].mean()),
                "pnl_mean_se": float(t["pnl"].std(ddof=1) / math.sqrt(len(t))),
                "prob_profit": float((t["pnl"] > 0).mean()),
                "pnl_worst": float(t["pnl"].min()),
            }
        )
    offsets = [bs.short_vol_backtest(market, horizon=21, offset=o)["pnl"] for o in range(21)]
    offset_means = [float(o.mean()) for o in offsets]
    summary = {
        "n_trades": n,
        "first_date": str(trades["start"].iloc[0].date()),
        "last_date": str(trades["end"].iloc[-1].date()),
        "pnl_mean": float(pnl.mean()),
        "pnl_mean_se": float(pnl.std(ddof=1) / math.sqrt(n)),
        "pnl_median": float(pnl.median()),
        "pnl_skew": float(pnl.skew()),
        "prob_profit": float((pnl > 0).mean()),
        "premium_mean": float(trades["premium"].mean()),
        "pnl_over_premium_mean": float((pnl / trades["premium"]).mean()),
        "implied_mean": float(trades["implied_vol"].mean()),
        "realized_mean": float(trades["realized_vol"].mean()),
        "share_implied_above_realized": float(
            (trades["implied_vol"] > trades["realized_vol"]).mean()
        ),
        "worst_start": str(worst["start"].date()),
        "worst_pnl": float(worst["pnl"]),
        "worst_implied": float(worst["implied_vol"]),
        "worst_realized": float(worst["realized_vol"]),
        "best_start": str(best["start"].date()),
        "best_pnl": float(best["pnl"]),
        "r2_dollar_gamma_proxy": float(np.corrcoef(trades["variance_proxy"], pnl)[0, 1] ** 2),
        "r2_raw_variance_gap": float(
            np.corrcoef(trades["implied_vol"] ** 2 - trades["realized_vol"] ** 2, pnl)[0, 1] ** 2
        ),
        "offset_mean_min": min(offset_means),
        "offset_mean_max": max(offset_means),
        "mean_gamma": float(trades["gamma"].mean()),
        "mean_theta": float(trades["theta"].mean()),
        "mean_carry": float(trades["carry"].mean()),
    }
    return trades, pd.DataFrame(robustness), summary


def fig_r4(trades: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]}
    )
    ax = axes[0]
    ax.plot(
        trades["start"],
        trades["implied_vol"] * 100,
        color=SERIES[0],
        label="VIX at inception (sold)",
    )
    ax.plot(
        trades["start"],
        trades["realized_vol"] * 100,
        color=SERIES[1],
        label="SPY realised over the next 21 days",
    )
    ax.set_ylabel("Volatility (%)")
    ax.set_title(
        f"R4. SPY 2020-2024: implied above realised in {summary['share_implied_above_realized']:.0%} of the {summary['n_trades']} months"
    )
    ax.legend(loc="upper right")
    ax = axes[1]
    colors = [POSITIVE if v > 0 else NEGATIVE for v in trades["pnl"]]
    ax.bar(trades["start"], trades["pnl"], width=18, color=colors)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_ylabel("Hedged P&L, per 100 notional")
    ax.set_title(
        f"Short 1M ATM call, delta-hedged daily: mean {summary['pnl_mean']:+.2f} per trade, "
        f"{summary['prob_profit']:.0%} of months positive"
    )
    worst = trades.loc[trades["pnl"].idxmin()]
    ax.annotate(
        f"{worst['start']:%b %Y}: sold at {worst['implied_vol']:.0%}, realised {worst['realized_vol']:.0%}",
        (worst["start"], worst["pnl"]),
        xytext=(14, 4),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout()
    _save(
        fig,
        path,
        "R4 SPY short-volatility backtest 2020-2024",
        "Monthly ATM call on SPY sold at the VIX and delta-hedged daily on observed closes: implied vs realised vol and hedged P&L per trade.",
    )


# ---- driver -----------------------------------------------------------------------------------


def run(cfg: Config, out: Path, update_readme: bool) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    _style()

    fine = bs.simulate_gbm(cfg.S0, cfg.mu, cfg.sigma, cfg.T, cfg.n_fine, cfg.n_paths, seed=cfg.seed)
    r1, s1 = r1_replication(fine, cfg)
    r2, s2 = r2_costs(fine, cfg)
    r3, r3_choice, s3, scatter = r3_vol_mismatch(fine, cfg)
    r4, r4_rob, s4 = r4_spy(DATA)

    r1.to_csv(out / "r1_replication.csv", index=False, float_format="%.10g")
    r2.to_csv(out / "r2_costs.csv", index=False, float_format="%.10g")
    pd.DataFrame(s2["optimum"]).to_csv(out / "r2_optimum.csv", index=False, float_format="%.10g")
    r3.to_csv(out / "r3_vol_mismatch.csv", index=False, float_format="%.10g")
    r3_choice.to_csv(out / "r3_hedge_vol_choice.csv", index=False, float_format="%.10g")
    r4.to_csv(out / "r4_spy_trades.csv", index=False, float_format="%.10g", date_format="%Y-%m-%d")
    r4_rob.to_csv(out / "r4_spy_robustness.csv", index=False, float_format="%.10g")

    fig_r1(r1, s1, figures / "r1_replication_error.png")
    fig_r2(r2, s2, figures / "r2_costs.png")
    fig_r3(scatter, s3, cfg, figures / "r3_path_dependence.png")
    fig_r4(r4, s4, figures / "r4_spy_backtest.png")

    summary = {
        "config": asdict(cfg),
        "bspricer_version": bs.__version__,
        "r1": s1,
        "r1_table": r1.to_dict(orient="records"),
        "r2": s2,
        "r3": s3,
        "r3_table": r3.to_dict(orient="records"),
        "r3_choice": r3_choice.to_dict(orient="records"),
        "r4": s4,
        "r4_robustness": r4_rob.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if update_readme:
        readme = ROOT / "README.md"
        readme.write_text(
            readme_block.update(readme.read_text(encoding="utf-8"), summary), encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="2,000 paths and a 252-step grid")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--no-readme", action="store_true", help="do not rewrite README.md")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    summary = run(cfg, args.out, update_readme=not args.no_readme and not args.quick)
    print(
        f"R1 slope {summary['r1']['slope']:.3f} | "
        f"R2 N* {[o['n_star'] for o in summary['r2']['optimum']]} | "
        f"R3 R2 {summary['r3']['r2_dollar_gamma_proxy']:.3f} | "
        f"R4 mean {summary['r4']['pnl_mean']:+.3f} over {summary['r4']['n_trades']} trades"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
