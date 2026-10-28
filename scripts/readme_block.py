"""Render the generated blocks of README.md from ``results/summary.json``.

Every number shown in the README results sections comes from this renderer, which
only reads the summary written by ``scripts/run_results.py``. ``tests/test_readme.py``
fails if the README blocks differ from what this module renders from the committed
summary, so a stale or hand-edited figure cannot reach the README.

Blocks are delimited by ``<!-- BEGIN GENERATED: name -->`` / ``<!-- END GENERATED: name -->``.
"""

from __future__ import annotations

import re
from typing import Any

BLOCKS = ("key-results", "details")


def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f} %"


def _signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def _num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def key_results(s: dict[str, Any]) -> str:
    r1, r2, r3, r4 = s["r1"], s["r2"], s["r3"], s["r4"]
    cfg = s["config"]
    opt = {o["cost_bp"]: o for o in r2["optimum"]}
    interior = [c for c in sorted(opt) if not opt[c]["at_grid_edge"]]
    n_star = " / ".join(str(opt[c]["n_star"]) for c in interior)
    n_theory = " / ".join(f"{opt[c]['n_star_theory']:.0f}" for c in interior)
    costs = " / ".join(str(c) for c in interior)
    lo, hi = r1["slope_ci95"]
    choice = {(row["hedge_vol"], row["n_rebalances"]): row for row in s["r3_choice"]}
    fine_n = r1["finest_n"]
    implied_fine, realised_fine = choice[("implied", fine_n)], choice[("realised", fine_n)]
    rows = [
        (
            "**R1** Replication error vs. rebalancing frequency (Monte Carlo)",
            f"log-log slope **{_num(r1['slope'], 3)}** (95 % CI {_num(lo, 3)} to {_num(hi, 3)}) "
            f"for N ≥ 63; daily hedge leaves sd = {_pct(r1['daily_sd_pct_premium'] / 100, 1)} of the premium, "
            f"{r1['daily_ratio_kd']:.2f}× the Kamal-Derman value",
            "The error decays like 1/√N: halving it takes four times as many rebalances.",
        ),
        (
            "**R2** Optimal rebalancing with proportional costs",
            f"N\\* = **{n_star}** rebalances at {costs} bp one-way cost "
            f"(model: {n_theory}); at {min(opt)} bp the RMSE still falls at the finest grid point, "
            f"N = {opt[min(opt)]['n_star']} (model: {opt[min(opt)]['n_star_theory']:.0f})",
            "Each doubling of the cost roughly halves the optimal number of rebalances "
            "(N\\* ∝ 1/c to leading order).",
        ),
        (
            f"**R3** Misspecified volatility (sold at σᵢ, realised σᵣ = {_pct(cfg['sigma'])})",
            f"mean P&L = dollar-gamma × variance gap to within "
            f"{r3['max_abs_mean_minus_predicted_in_se']:.1f} SE for σᵢ from "
            f"{_pct(s['r3_table'][0]['sigma_implied'])} to {_pct(s['r3_table'][-1]['sigma_implied'])}; "
            f"at σᵢ = {_pct(r3['sigma_rich'])}, path-wise R² **{r3['r2_dollar_gamma_proxy']:.2f}** "
            f"vs {r3['r2_raw_variance_gap']:.2f} for realised vol alone",
            f"The P&L depends on *where* variance is realised: with realised vol within ±0.5 pt of "
            f"{_pct(cfg['sigma'])}, it still ranges from {_num(r3['bucket_p5'])} to {_num(r3['bucket_p95'])} "
            f"(5th–95th pct.). Hedging at the realised instead of the implied vol cuts its sd from "
            f"{_num(implied_fine['pnl_sd'])} to {_num(realised_fine['pnl_sd'])} "
            f"(N = {fine_n}).",
        ),
        (
            "**R4** SPY 2020–2024: short 1-month ATM call sold at the VIX, hedged daily on real closes",
            f"{r4['n_trades']} monthly trades: mean **{_signed(r4['pnl_mean'])}** (SE {_num(r4['pnl_mean_se'])}) "
            f"per 100 notional ({_pct(r4['pnl_over_premium_mean'])} of the premium), "
            f"{_pct(r4['prob_profit'])} positive; VIX above realised vol in {_pct(r4['share_implied_above_realized'])} "
            "of months",
            f"Selling implied vol earned a premium, paid for by crash risk: the worst month "
            f"({r4['worst_start'][:7]}, sold at {_pct(r4['worst_implied'])}, realised {_pct(r4['worst_realized'])}) "
            f"lost {_num(-r4['worst_pnl'])}, about {(-r4['worst_pnl'] / r4['pnl_mean']):.0f} average months.",
        ),
    ]
    lines = ["| Result | Measurement | Interpretation |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += [
        "",
        "![R1: replication error versus rebalancing frequency](results/figures/r1_replication_error.png)",
        "",
        "![R4: SPY short-volatility backtest](results/figures/r4_spy_backtest.png)",
    ]
    return "\n".join(lines)


def details(s: dict[str, Any]) -> str:
    cfg, r1, r2, r3, r4 = s["config"], s["r1"], s["r2"], s["r3"], s["r4"]
    out: list[str] = []
    out += [
        "<details><summary><b>R1</b> Replication error by rebalancing frequency</summary>",
        "",
        f"{r1['n_paths']:,} paths, common random numbers (the same Brownian paths observed at every N). "
        f"Premium {_num(r1['premium'], 4)}, vega {_num(r1['vega'], 3)}.",
        "",
        "| N | per day | sd(P&L) | sd / premium | Kamal-Derman | ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in s.get("r1_table", []):
        out.append(
            f"| {row['n_rebalances']} | {row['rebalances_per_day']:.2f} | {row['pnl_sd']:.4f} | "
            f"{row['pnl_sd_pct_premium']:.2f} % | {row['kamal_derman_sd']:.4f} | {row['sd_over_kamal_derman']:.3f} |"
        )
    lo, hi = r1["slope_all_ci95"]
    out += [
        "",
        f"Slope over all N from {r1['coarsest_n']}: {_num(r1['slope_all'], 3)} (95 % CI {_num(lo, 3)} to "
        f"{_num(hi, 3)}). At N = {r1['coarsest_n']} the simulated sd is {r1['coarsest_ratio_kd']:.2f}× the "
        "Kamal-Derman value: the formula is asymptotic. Mean P&L: largest |t| across N is "
        f"{r1['max_abs_mean_t']:.1f} on these shared paths; an independent run of "
        f"{r1['independent_n_paths']:,} fresh paths hedged daily gives a mean of "
        f"{_signed(r1['independent_mean'], 4)} (t = {_num(r1['independent_mean_t'])}), consistent with an unbiased hedge.",
        "",
        "</details>",
        "",
        "<details><summary><b>R2</b> Optimal rebalancing frequency with proportional costs</summary>",
        "",
        "Criterion: N\\* minimises the RMSE of the total P&L, costs included, on the grid of divisors of "
        f"{cfg['n_fine']}. Model: sd² ≈ a/N and traded notional ≈ F + b√N, with a = {_num(r2['a'])}, "
        f"F = {_num(r2['fixed_turnover'], 1)} (entry and exit trades; the initial delta alone is "
        f"{_num(r2['initial_trade_notional'], 1)}) and b = {_num(r2['b'], 1)} (fit error ≤ "
        f"{_pct(r2['turnover_fit_max_rel_error'], 1)}).",
        "",
        "| one-way cost | N\\* (simulated) | N\\* (model) | RMSE at N\\* | mean cost at N\\* | RMSE if daily |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for o in r2["optimum"]:
        n_star = f"≥ {o['n_star']}" if o["at_grid_edge"] else str(o["n_star"])
        out.append(
            f"| {o['cost_bp']} bp | {n_star} | {o['n_star_theory']:.0f} | {o['rmse_at_n_star']:.3f} | "
            f"{o['mean_cost_at_n_star']:.3f} | {o['rmse_daily']:.3f} |"
        )
    out += [
        "",
        f"Fitted scaling of the simulated optimum: N\\* ∝ c^{_num(r2['n_star_cost_exponent'])}.",
        "",
        "![R2: RMSE versus rebalancing frequency for several cost levels](results/figures/r2_costs.png)",
        "",
        "</details>",
        "",
        "<details><summary><b>R3</b> Misspecified volatility</summary>",
        "",
        f"Paths at σᵣ = {_pct(cfg['sigma'])}, daily rebalancing; the option is sold and hedged at σᵢ. "
        "Prediction: E[Σ ½ e^{r(T−t)} Γ S² (σᵢ² − σᵣ²) Δt] along the simulated paths.",
        "",
        "| σᵢ | mean P&L | SE | predicted | premium edge e^{rT}(V(σᵢ) − V(σᵣ)) | sd | P(P&L > 0) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in s["r3_table"]:
        out.append(
            f"| {_pct(row['sigma_implied'])} | {_signed(row['pnl_mean'], 3)} | {row['pnl_mean_se']:.3f} | "
            f"{_signed(row['predicted_mean'], 3)} | {_signed(row['premium_edge'], 3)} | {row['pnl_sd']:.3f} | "
            f"{_pct(row['prob_profit'], 1)} |"
        )
    out += [
        "",
        f"Path dependence at σᵢ = {_pct(r3['sigma_rich'])}: regressing the P&L on the path-wise proxy "
        f"Σ ½ Γ S² (σᵢ²Δt − δS²/S²) gives slope {_num(r3['proxy_slope'], 3)} and R² {r3['r2_dollar_gamma_proxy']:.3f}; "
        f"on σᵢ² − σ̂ᵣ² alone, R² {r3['r2_raw_variance_gap']:.3f}. Among the {r3['bucket_n']:,} paths whose "
        f"realised vol is within ±0.5 pt of {_pct(cfg['sigma'])}, those ending within 2 % of the strike "
        f"earn {_num(r3['bucket_mean_near_strike'])} on average against {_num(r3['bucket_mean_far'])} for "
        "the others: gamma is largest there.",
        "",
        "Which volatility to hedge with (sold at σᵢ, Ahmad-Wilmott 2005):",
        "",
        "| hedge vol | N | mean P&L | sd | P(loss) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in s["r3_choice"]:
        out.append(
            f"| {row['hedge_vol']} ({_pct(row['sigma_hedge'])}) | {row['n_rebalances']} | "
            f"{_signed(row['pnl_mean'], 3)} | {row['pnl_sd']:.3f} | {_pct(row['prob_loss'], 1)} |"
        )
    out += [
        "",
        f"Locked-in value e^{{rT}}(V(σᵢ) − V(σᵣ)) = {_signed(s['r3_choice'][0]['target_edge'], 3)}: hedging at the "
        "realised vol converges to it, hedging at the implied vol keeps a path-dependent residual.",
        "",
        "![R3: path dependence of the hedged P&L](results/figures/r3_path_dependence.png)",
        "",
        "</details>",
        "",
        "<details><summary><b>R4</b> SPY backtest, robustness</summary>",
        "",
        f"{r4['n_trades']} non-overlapping 21-trading-day trades from {r4['first_date']} to {r4['last_date']}. "
        f"Mean implied (VIX) {_pct(r4['implied_mean'], 1)}, mean realised {_pct(r4['realized_mean'], 1)}; "
        f"median P&L {_signed(r4['pnl_median'])}, skewness {_num(r4['pnl_skew'])}. Mean attribution per trade: "
        f"theta {_signed(r4['mean_theta'])}, gamma {_signed(r4['mean_gamma'])}, carry {_signed(r4['mean_carry'])}. "
        f"Path-wise R² of the dollar-gamma proxy {r4['r2_dollar_gamma_proxy']:.2f}, of VIX² − σ̂² alone "
        f"{r4['r2_raw_variance_gap']:.2f}.",
        "",
        "| variant | trades | mean P&L | SE | P(P&L > 0) | worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in s["r4_robustness"]:
        out.append(
            f"| {row['variant']} | {row['n_trades']} | {_signed(row['pnl_mean'], 3)} | {row['pnl_mean_se']:.3f} | "
            f"{_pct(row['prob_profit'], 1)} | {_signed(row['pnl_worst'])} |"
        )
    out += [
        "",
        f"Moving the first trade date by 0 to 20 trading days changes the mean P&L between "
        f"{_signed(r4['offset_mean_min'], 3)} and {_signed(r4['offset_mean_max'], 3)}. "
        "The standard errors assume independent months.",
        "",
        "</details>",
    ]
    return "\n".join(out)


def render(summary: dict[str, Any]) -> dict[str, str]:
    return {"key-results": key_results(summary), "details": details(summary)}


def _pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(.*?)(<!-- END GENERATED: {re.escape(name)} -->)",
        re.DOTALL,
    )


def update(text: str, summary: dict[str, Any]) -> str:
    """Replace every generated block of ``text`` with its rendering from ``summary``."""
    for name, body in render(summary).items():
        pattern = _pattern(name)
        if not pattern.search(text):
            raise ValueError(f"README has no generated block named {name!r}")
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(3), text)
    return text


def extract(text: str) -> dict[str, str]:
    """Return the current content of each generated block of ``text`` (without the final newline)."""
    found = {}
    for name in BLOCKS:
        match = _pattern(name).search(text)
        if match:
            found[name] = match.group(2).removesuffix("\n")
    return found
