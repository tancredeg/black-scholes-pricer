"""The README results are generated, consistent with results/, and reproducible from the code.

These tests enforce the rule "every number in the README comes from an executed script":

* the generated README blocks equal what ``scripts/readme_block.py`` renders from the committed
  ``results/summary.json``;
* the summary agrees with the committed CSV tables;
* the real-data result (R4, deterministic) is recomputed from ``data/sample`` and must match;
* the Monte Carlo results are recomputed on a smaller sample (slow) and must agree within
  their sampling error.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

import readme_block  # noqa: E402
import run_results  # noqa: E402


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text())


def test_readme_blocks_are_rendered_from_the_summary(summary):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme_block.extract(readme) == readme_block.render(summary)
    assert readme_block.update(readme, summary) == readme


def test_figures_referenced_by_the_readme_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("r1_replication_error", "r2_costs", "r3_path_dependence", "r4_spy_backtest"):
        path = RESULTS / "figures" / f"{name}.png"
        assert path.exists() and path.stat().st_size > 10_000
        assert f"results/figures/{name}.png" in readme


def test_summary_matches_the_csv_tables(summary):
    r1 = pd.read_csv(RESULTS / "r1_replication.csv")
    fine = r1["n_rebalances"] >= 63
    slope = np.polyfit(np.log(r1.loc[fine, "n_rebalances"]), np.log(r1.loc[fine, "pnl_sd"]), 1)[0]
    assert slope == pytest.approx(summary["r1"]["slope"], abs=1e-9)
    assert summary["r1"]["slope_ci95"][0] < summary["r1"]["slope"] < summary["r1"]["slope_ci95"][1]

    r2 = pd.read_csv(RESULTS / "r2_costs.csv")
    for opt in summary["r2"]["optimum"]:
        sub = r2[r2["cost_bp"] == opt["cost_bp"]]
        assert sub.loc[sub["pnl_rmse"].idxmin(), "n_rebalances"] == opt["n_star"]

    r3 = pd.read_csv(RESULTS / "r3_vol_mismatch.csv")
    np.testing.assert_allclose(
        r3["pnl_mean"], [row["pnl_mean"] for row in summary["r3_table"]], rtol=1e-8
    )

    trades = pd.read_csv(RESULTS / "r4_spy_trades.csv")
    assert len(trades) == summary["r4"]["n_trades"]
    assert trades["pnl"].mean() == pytest.approx(summary["r4"]["pnl_mean"], rel=1e-8)
    assert (trades["pnl"] > 0).mean() == pytest.approx(summary["r4"]["prob_profit"])


def test_r4_is_reproduced_exactly_from_the_frozen_data():
    trades, robustness, _ = run_results.r4_spy(ROOT / "data" / "sample")
    committed = pd.read_csv(RESULTS / "r4_spy_trades.csv", parse_dates=["start", "end"])
    pd.testing.assert_frame_equal(
        trades, committed, check_exact=False, check_dtype=False, rtol=1e-8, atol=1e-10
    )
    committed_rob = pd.read_csv(RESULTS / "r4_spy_robustness.csv")
    pd.testing.assert_frame_equal(robustness, committed_rob, check_exact=False, rtol=1e-8)


@pytest.mark.slow
def test_monte_carlo_results_are_reproduced_on_a_smaller_sample(summary, tmp_path):
    """Quick mode: 2,000 paths, 252-step grid; checks agree within sampling error."""
    quick = run_results.run(run_results.QUICK, tmp_path, update_readme=False)
    full = summary
    assert abs(quick["r1"]["slope"] - full["r1"]["slope"]) < 0.05
    assert 0.9 < quick["r1"]["daily_ratio_kd"] < 1.1
    q_opt = {o["cost_bp"]: o["n_star"] for o in quick["r2"]["optimum"]}
    f_opt = {o["cost_bp"]: o["n_star"] for o in full["r2"]["optimum"]}
    for cost in (10, 20):  # optima inside the quick grid
        assert 0.5 <= q_opt[cost] / f_opt[cost] <= 2.0
    for q_row, f_row in zip(quick["r3_table"], full["r3_table"], strict=True):
        se = math.hypot(q_row["pnl_mean_se"], f_row["pnl_mean_se"])
        assert abs(q_row["pnl_mean"] - f_row["pnl_mean"]) < 5 * se
    assert quick["r4"] == full["r4"]  # R4 uses real data only: identical in both modes
