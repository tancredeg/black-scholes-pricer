"""
audit_phase0.py — Phase 0 audit of tancredeg/black-scholes-pricer (HEAD 6339479).

Independent numerical checks behind every figure quoted in AUDIT.md.
Each block prints an identifier (A01, A02, ...) that AUDIT.md references.

Usage (from anywhere):
    python audit_phase0.py --repo <path/to/black-scholes-pricer> --spy <path/to/SPY.csv> \
        [--spy-fresh data_snapshots/SPY_yahoo_dl_2026-10-12.csv]

Deterministic: every random draw uses np.random.default_rng with a fixed seed.
Runtime: ~1-2 min on a laptop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

RESULTS: dict[str, dict] = {}


def record(key: str, **values) -> None:
    RESULTS[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in values.items()}
    body = ", ".join(f"{k}={_fmt(v)}" for k, v in values.items())
    print(f"[{key}] {body}")


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--spy", required=True, type=Path)
    ap.add_argument("--spy-fresh", default=None, type=Path,
                    help="optional second SPY download, to measure adjusted-close drift")
    ap.add_argument("--json", default="audit_phase0_results.json", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import bs_pricer as bs  # noqa: E402

    warnings.simplefilter("ignore", RuntimeWarning)

    S0, K, T, r, SIG, MU = 310.0, 310.0, 30 / 252, 0.05, 0.19, 0.14

    # ------------------------------------------------------------------
    # A01 — sigma -> 0 limit: degenerate branch keys on S >= K instead of
    #       the forward S e^{-qT} >= K e^{-rT}. Wrong value, can be negative.
    # ------------------------------------------------------------------
    cases = [
        ("call S<K, fwd ITM", 305.0, 310.0, 1.0, 0.05, 0.00, "call"),
        ("call S=K, q>r",     310.0, 310.0, 1.0, 0.00, 0.03, "call"),
        ("put  S<K, fwd OTM", 305.0, 310.0, 1.0, 0.05, 0.00, "put"),
    ]
    for label, S, Kc, Tc, rc, qc, opt in cases:
        fwd_call = S * np.exp(-qc * Tc) - Kc * np.exp(-rc * Tc)
        limit = max(fwd_call, 0.0) if opt == "call" else max(-fwd_call, 0.0)
        prices = {f"sigma={s:g}": float(bs.bs_price(S, Kc, Tc, rc, s, opt, q=qc))
                  for s in (1e-6, 1e-8, 1e-10, 0.0)}
        record(f"A01 {label}", limit=limit, **prices)

    # ------------------------------------------------------------------
    # A02 — Hedging engine ignores dividends received on the share leg (q>0)
    # ------------------------------------------------------------------
    def hedge_vec(paths, Kh, Th, rh, sig_h, opt="call", q=0.0, dividends=True,
                  cost=0.0):
        """Reference self-financing hedge (short option), dt = Th/N.
        Dividends on held shares accrue to cash (trapezoidal). Proportional
        cost `cost` charged on every share traded, including t=0 and the
        final liquidation."""
        n = paths.shape[1] - 1
        dt = Th / n
        g = np.exp(rh * dt)
        s = paths[:, 0]
        c0 = bs.bs_price(s, Kh, Th, rh, sig_h, opt, q=q)
        d = bs.bs_greeks_vec(s, Kh, Th, rh, sig_h, opt, q=q)["delta"]
        cash = c0 - d * s - cost * np.abs(d) * s
        tc = cost * np.abs(d) * s
        for k in range(1, n):
            sk = paths[:, k]
            cash = cash * g
            if dividends:
                cash = cash + d * q * dt * 0.5 * (paths[:, k - 1] + sk)
            d_new = bs.bs_greeks_vec(sk, Kh, Th - k * dt, rh, sig_h, opt, q=q)["delta"]
            trade = d_new - d
            cash = cash - trade * sk - cost * np.abs(trade) * sk
            tc = tc + cost * np.abs(trade) * sk
            d = d_new
        st = paths[:, -1]
        cash = cash * g
        if dividends:
            cash = cash + d * q * dt * 0.5 * (paths[:, -2] + st)
        cash = cash - cost * np.abs(d) * st  # liquidate hedge
        tc = tc + cost * np.abs(d) * st
        payoff = np.maximum(st - Kh, 0) if opt == "call" else np.maximum(Kh - st, 0)
        return cash + d * st - payoff, tc

    # sanity: reference engine == repo engine when q=0, no costs
    p_chk = bs.simulate_gbm(S0, MU, SIG, T, 30, 500, seed=1)
    e_ref, _ = hedge_vec(p_chk, K, T, r, SIG)
    e_rep, _, _ = bs.run_hedging_experiment_vec(p_chk, K, T, r, SIG)
    record("A02 reference engine vs repo engine (q=0)",
           max_abs_diff=float(np.max(np.abs(e_ref - e_rep))))

    q = 0.013
    n_paths = 20_000
    paths_q = bs.simulate_gbm(S0, MU - q, SIG, T, 30, n_paths, seed=7)
    e_repo, _, m_repo = bs.run_hedging_experiment_vec(paths_q, K, T, r, SIG, q=q)
    e_fix, _ = hedge_vec(paths_q, K, T, r, SIG, q=q, dividends=True)
    se_repo = e_repo.std(ddof=1) / np.sqrt(n_paths)
    se_fix = e_fix.std(ddof=1) / np.sqrt(n_paths)
    record("A02 q=0.013 repo engine", mean=m_repo, se=se_repo, t_stat=m_repo / se_repo)
    record("A02 q=0.013 with dividend accrual", mean=float(e_fix.mean()), se=se_fix,
           t_stat=float(e_fix.mean() / se_fix))

    # ------------------------------------------------------------------
    # A03/A04 — Section 7 gamma/theta decomposition
    #   A03: theta_pnl = theta_per_day * DT (DT=1/252) is 252x too small.
    #   A04: even with correct units, E[1/2 G dS^2 + Theta dt] = r(C - S Delta) dt,
    #        not 0: the financing term is missing.
    # ------------------------------------------------------------------
    n_dec = 20_000
    p = bs.simulate_gbm(S0, MU, SIG, T, 30, n_dec, seed=11)
    dt = T / 30
    gam = np.zeros(n_dec)
    th_nb = np.zeros(n_dec)
    th_ok = np.zeros(n_dec)
    fin = np.zeros(n_dec)
    for k in range(30):
        sk = p[:, k]
        g = bs.bs_greeks_vec(sk, K, T - k * dt, r, SIG, "call")
        ck = bs.bs_price(sk, K, T - k * dt, r, SIG, "call")
        ds = p[:, k + 1] - sk
        gam += 0.5 * g["gamma"] * ds ** 2
        th_nb += g["theta"] * (1 / 252)          # notebook convention
        th_ok += g["theta"] * (dt * 252)         # per-day theta x number of days
        fin += -r * (ck - sk * g["delta"]) * dt  # long option, delta-hedged, financed
    record("A03 notebook convention (theta*DT), mean sum over path",
           gamma=float(gam.mean()), theta=float(th_nb.mean()),
           total=float((gam + th_nb).mean()))
    record("A03/A04 correct theta units, no financing",
           gamma=float(gam.mean()), theta=float(th_ok.mean()),
           total=float((gam + th_ok).mean()),
           se=float((gam + th_ok).std(ddof=1) / np.sqrt(n_dec)))
    tot = gam + th_ok + fin
    record("A04 with financing term -r(C - S*Delta)dt",
           financing=float(fin.mean()), total=float(tot.mean()),
           se=float(tot.std(ddof=1) / np.sqrt(n_dec)))

    # ------------------------------------------------------------------
    # A05 — Notebook cell 13 (PDE plot): plots Theta - 1/2 s^2 S^2 G labelled "~0"
    # ------------------------------------------------------------------
    g0 = bs.bs_greeks(S0, K, T, r, SIG, "call")
    c0 = bs.bs_price(S0, K, T, r, SIG, "call")
    th_a = g0["theta"] * 252
    half = 0.5 * SIG ** 2 * S0 ** 2 * g0["gamma"]
    record("A05 ATM annual terms", theta_annual=th_a,
           notebook_sum_theta_minus=th_a - half,
           correct_sum_theta_plus=th_a + half,
           r_times_C_minus_S_Delta=r * (c0 - S0 * g0["delta"]))

    # ------------------------------------------------------------------
    # A06 — Notebook Section 4 probabilities vs markdown text
    # ------------------------------------------------------------------
    _, d2q = bs._d1_d2(S0, K, T, r, SIG)
    d2p = (np.log(S0 / K) + (MU - 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    record("A06 ITM probabilities", N_d2_Q=float(norm.cdf(d2q)),
           N_d2_P=float(norm.cdf(d2p)), notebook_text_Q=0.48, notebook_text_P=0.57,
           printed_by_fstring=f"{float(norm.cdf(d2q)):.1f}%")

    # ------------------------------------------------------------------
    # A07 — Realised-vol LaTeX formula (notebook Section 1) vs implementation
    # ------------------------------------------------------------------
    spy = pd.read_csv(args.spy, index_col=0, parse_dates=True).squeeze("columns")
    lr = np.log(spy / spy.shift(1)).dropna().to_numpy()
    w = 30
    win = lr[-w:]
    impl = np.std(win, ddof=1) * np.sqrt(252)
    latex = np.sqrt(np.sum(win ** 2) / (w - 1) - (np.sum(win) / (w - 1)) ** 2) * np.sqrt(252)
    win_c = lr[45:45 + w]  # March 2020 window, larger mean
    impl_c = np.std(win_c, ddof=1) * np.sqrt(252)
    latex_c = np.sqrt(np.sum(win_c ** 2) / (w - 1) - (np.sum(win_c) / (w - 1)) ** 2) * np.sqrt(252)
    record("A07 last 30d window", implemented=impl, latex_formula=latex)
    record("A07 Mar-2020 window", implemented=impl_c, latex_formula=latex_c,
           rel_gap=float(latex_c / impl_c - 1))
    gaps = []
    for i in range(w, len(lr) + 1):
        x = lr[i - w:i]
        a_ = np.std(x, ddof=1)
        b_ = np.sqrt(np.sum(x ** 2) / (w - 1) - (np.sum(x) / (w - 1)) ** 2)
        gaps.append(abs(b_ / a_ - 1))
    record("A07 max relative gap over all 30d windows 2020-2024", value=float(max(gaps)))

    # ------------------------------------------------------------------
    # A08 — README Greeks example (q=0.013) vs actual output
    # ------------------------------------------------------------------
    gq = bs.bs_greeks(310.0, 310.0, 30 / 252, 0.05, 0.19, "call", q=0.013)
    readme = {"delta": 0.5446, "gamma": 0.01946, "theta": -0.1666, "vega": 42.30, "rho": 19.19}
    record("A08 README q=0.013", **{f"{k}_code": gq[k] for k in readme},
           **{f"{k}_readme": v for k, v in readme.items()})

    # ------------------------------------------------------------------
    # A09 — Analytical Greeks vs central finite differences (tolerance basis)
    # ------------------------------------------------------------------
    worst = {k: 0.0 for k in ("delta", "gamma", "theta", "vega", "rho")}
    worst_abs = dict(worst)
    for S in (250.0, 290.0, 310.0, 330.0, 380.0):
        for Tg in (10 / 252, 30 / 252, 90 / 252, 1.0):
            for sg in (0.10, 0.19, 0.40):
                for qg in (0.0, 0.013):
                    for opt in ("call", "put"):
                        def P(S_=S, T_=Tg, r_=r, s_=sg, opt=opt, qg=qg):
                            return float(bs.bs_price(S_, K, T_, r_, s_, opt, q=qg))
                        an = bs.bs_greeks(S, K, Tg, r, sg, opt, q=qg)
                        hS = 1e-3 * S
                        fd = {
                            "delta": (P(S_=S + hS) - P(S_=S - hS)) / (2 * hS),
                            "gamma": (P(S_=S + hS) - 2 * P() + P(S_=S - hS)) / hS ** 2,
                            "theta": -(P(T_=Tg + 1e-5) - P(T_=Tg - 1e-5)) / 2e-5 / 252,
                            "vega": (P(s_=sg + 1e-5) - P(s_=sg - 1e-5)) / 2e-5,
                            "rho": (P(r_=r + 1e-6) - P(r_=r - 1e-6)) / 2e-6,
                        }
                        for k_, v in fd.items():
                            err = abs(v - an[k_])
                            worst_abs[k_] = max(worst_abs[k_], err)
                            if abs(an[k_]) > 1e-4:
                                worst[k_] = max(worst[k_], err / abs(an[k_]))
    record("A09 FD vs analytic, max relative error (|greek|>1e-4)", **worst)
    record("A09 FD vs analytic, max absolute error", **worst_abs)

    # ------------------------------------------------------------------
    # A10 — Replication error vs rebalancing count N (common random numbers)
    # ------------------------------------------------------------------
    n_fine, n_p10 = 480, 20_000
    fine = bs.simulate_gbm(S0, MU, SIG, T, n_fine, n_p10, seed=2026)
    vega = bs.bs_greeks(S0, K, T, r, SIG)["vega"]
    Ns = [3, 5, 6, 10, 15, 30, 60, 120, 240, 480]
    stds = []
    for N in Ns:
        sub = fine[:, :: n_fine // N]
        e, _, _ = bs.run_hedging_experiment_vec(sub, K, T, r, SIG, dt=T / N)
        stds.append(float(e.std(ddof=1)))
    logN, logS = np.log(Ns), np.log(stds)
    slope_all, icpt = np.polyfit(logN, logS, 1)
    mask = np.array(Ns) >= 10
    slope_10 = np.polyfit(logN[mask], logS[mask], 1)[0]
    kd = [np.sqrt(np.pi / 4) * vega * SIG / np.sqrt(N) for N in Ns]
    for N, s_, kd_ in zip(Ns, stds, kd, strict=True):
        record(f"A10 N={N}", std=s_, kamal_derman=kd_, ratio=s_ / kd_)
    record("A10 log-log slope", slope_all_N=slope_all, slope_N_ge_10=slope_10)

    # current pytest check: ratio of two independent samples, band [0.8, 1.3]*sqrt(2)
    record("A10 test_rmse_scales_as_sqrt_dt band",
           lower=0.8 * np.sqrt(2), upper=1.3 * np.sqrt(2),
           ratio_if_slope_minus_0_25=2 ** 0.25, ratio_if_slope_minus_0_75=2 ** 0.75)

    # ------------------------------------------------------------------
    # A11 — Vol mismatch: path-wise P&L vs 1/2 sum e^{r(T-t)} G_h S^2 (s_h^2 - s_r^2) dt
    # ------------------------------------------------------------------
    p11 = bs.simulate_gbm(S0, MU, SIG, T, 30, 10_000, seed=99)
    dt = T / 30
    for sig_h in (0.15, 0.23):
        e, _, m = bs.run_hedging_experiment_vec(p11, K, T, r, sig_h)
        pred_dollar_gamma = np.zeros(p11.shape[0])
        pred_realised = np.zeros(p11.shape[0])
        for k in range(30):
            sk = p11[:, k]
            gk = bs.bs_greeks_vec(sk, K, T - k * dt, r, sig_h)["gamma"]
            disc = np.exp(r * (T - (k + 1) * dt))
            pred_dollar_gamma += 0.5 * disc * gk * sk ** 2 * (sig_h ** 2 - SIG ** 2) * dt
            ret2 = (p11[:, k + 1] / sk - 1) ** 2
            pred_realised += 0.5 * disc * gk * sk ** 2 * (sig_h ** 2 * dt - ret2)
        for name, pred in (("expected-var", pred_dollar_gamma), ("realised-var", pred_realised)):
            b, a = np.polyfit(pred, e, 1)
            r2 = np.corrcoef(pred, e)[0, 1] ** 2
            record(f"A11 sigma_h={sig_h} predictor={name}", mean_pnl=m,
                   mean_pred=float(pred.mean()), slope=b, intercept=a, R2=r2)

    # call/put engine identity holds even with sigma mismatch -> it tests parity, not gamma
    e_c, _, _ = bs.run_hedging_experiment_vec(p11[:200], K, T, r, 0.25, "call")
    e_pt, _, _ = bs.run_hedging_experiment_vec(p11[:200], K, T, r, 0.25, "put")
    record("A11 call/put hedging errors with sigma_h=0.25 != sigma_r",
           max_abs_diff=float(np.max(np.abs(e_c - e_pt))))

    # ------------------------------------------------------------------
    # A12 — IV solver behaviour at the edges (README claims "returns lower bracket")
    # ------------------------------------------------------------------
    def try_iv(label, price, S, Kc, Tc, rc, opt):
        try:
            iv = bs.bs_implied_vol(price, S, Kc, Tc, rc, opt)
            record(f"A12 {label}", price=price, iv=iv)
        except ValueError as exc:
            record(f"A12 {label}", price=price, raised=str(exc)[:70])

    c_deep = float(bs.bs_price(420.0, 310.0, 5 / 252, 0.05, 0.19, "call"))
    try_iv("deep ITM call 5d, true sigma 0.19", c_deep, 420.0, 310.0, 5 / 252, 0.05, "call")
    lb = 420.0 - 310.0 * np.exp(-0.05 * 5 / 252)
    try_iv("price = lower bound + 1e-9", lb + 1e-9, 420.0, 310.0, 5 / 252, 0.05, "call")
    try_iv("price = lower bound - 1e-11", lb - 1e-11, 420.0, 310.0, 5 / 252, 0.05, "call")

    # ------------------------------------------------------------------
    # A13 — Edge cases: S -> 0, S -> inf, invalid inputs accepted silently
    # ------------------------------------------------------------------
    record("A13 S=1e-8", call=float(bs.bs_price(1e-8, K, T, r, SIG, "call")),
           put=float(bs.bs_price(1e-8, K, T, r, SIG, "put")),
           put_expected=K * np.exp(-r * T))
    record("A13 S=0", call=float(bs.bs_price(0.0, K, T, r, SIG, "call")),
           gamma=bs.bs_greeks(0.0, K, T, r, SIG)["gamma"])
    big = 1e8
    record("A13 S=1e8 call - (S - K e^-rT)",
           diff=float(bs.bs_price(big, K, T, r, SIG, "call") - (big - K * np.exp(-r * T))))
    record("A13 negative sigma accepted", price=float(bs.bs_price(305.0, 310.0, 1.0, 0.05, -0.2)))
    record("A13 negative T accepted", price=float(bs.bs_price(320.0, 310.0, -1.0, 0.05, 0.2)))
    gz = bs.bs_greeks_vec(np.array([300.0, 310.0, 320.0]), K, T, r, np.array([0.19, 0.0, 0.19]))
    record("A13 bs_greeks_vec with a zero sigma", gamma=gz["gamma"].tolist(),
           vega=gz["vega"].tolist())

    # ------------------------------------------------------------------
    # A14 — No guard on N_steps * dt == T_total
    # ------------------------------------------------------------------
    p14 = bs.simulate_gbm(S0, MU, SIG, 60 / 252, 30, 2_000, seed=5)
    _, rmse_bad, mean_bad = bs.run_hedging_experiment_vec(p14, K, 60 / 252, r, SIG)  # dt default 1/252
    _, rmse_ok, mean_ok = bs.run_hedging_experiment_vec(p14, K, 60 / 252, r, SIG, dt=2 / 252)
    record("A14 T=60/252, 30 steps, default dt=1/252 (silent)", rmse=rmse_bad, mean=mean_bad)
    record("A14 same paths, dt=2/252 (consistent)", rmse=rmse_ok, mean=mean_ok)

    # ------------------------------------------------------------------
    # A15 — Expired Greeks at S=K (README: put delta = -1)
    # ------------------------------------------------------------------
    record("A15 T=0, S=K", call_delta=bs.bs_greeks(K, K, 0.0, r, SIG, "call")["delta"],
           put_delta=bs.bs_greeks(K, K, 0.0, r, SIG, "put")["delta"])

    # ------------------------------------------------------------------
    # A16 — SPY data: integrity and reproduction of notebook Section 1 figures
    # ------------------------------------------------------------------
    vol = bs.rolling_realized_vol(spy, windows=[30, 60, 252])
    v30, v252 = vol["vol_30d"].dropna(), vol["vol_252d"].dropna()
    div = (v30[v252.index] - v252).abs()
    record("A16 SPY.csv", rows=len(spy), first=str(spy.index[0].date()),
           last=str(spy.index[-1].date()), n_nan=int(spy.isna().sum()),
           dup_dates=int(spy.index.duplicated().sum()),
           monotonic=bool(spy.index.is_monotonic_increasing))
    record("A16 vol figures", peak_30d=float(v30.max()), peak_date=str(v30.idxmax().date()),
           trough_30d=float(v30.min()), trough_date=str(v30.idxmin().date()),
           first_252d_date=str(v252.index[0].date()),
           max_div=float(div.max()), max_div_date=str(div.idxmax().date()))
    record("A16 min 30d vol by period (notebook text: 6-8% in 2021 H1 and 2023-2024)",
           h1_2021=float(v30["2021-01-01":"2021-06-30"].min()),
           y2023=float(v30["2023"].min()), y2024=float(v30["2024"].min()))
    if args.spy_fresh is not None:
        fresh = pd.read_csv(args.spy_fresh, index_col=0, parse_dates=True).squeeze("columns")
        ratio = fresh / spy
        dlr = (np.log(fresh).diff() - np.log(spy).diff()).abs().max()
        record("A21 adjusted-close drift (fresh Yahoo download vs attached CSV)",
               same_dates=bool(fresh.index.equals(spy.index)),
               price_ratio_min=float(ratio.min()), price_ratio_max=float(ratio.max()),
               max_abs_logreturn_diff=float(dlr))

    # ------------------------------------------------------------------
    # A17 — Speed-up claim (machine-dependent; ratio only)
    # ------------------------------------------------------------------
    p17 = bs.simulate_gbm(S0, MU, SIG, T, 30, 300, seed=3)
    t0 = time.perf_counter()
    bs.run_hedging_experiment(p17, K, T, r, SIG)
    t_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(10):
        bs.run_hedging_experiment_vec(p17, K, T, r, SIG)
    t_v = (time.perf_counter() - t0) / 10
    record("A17 scalar vs vectorised, 300 paths x 30 steps", scalar_s=t_s, vec_s=t_v,
           speedup=t_s / t_v)

    # ------------------------------------------------------------------
    # A18 — Vega units in notebook Section 3 text
    # ------------------------------------------------------------------
    record("A18 vega", vega_per_unit=g0["vega"], usd_per_vol_point=g0["vega"] * 0.01,
           notebook_claims_per_point=42.3464)

    # ------------------------------------------------------------------
    # A19 — Max PDE residual over the pytest grid (README says < 1e-10, test < 1e-9)
    # ------------------------------------------------------------------
    mx = 0.0
    for S in (200.0, 280.0, 310.0, 340.0, 420.0):
        for Tg in (5 / 252, 30 / 252, 90 / 252, 1.0):
            for rg in (0.0, 0.05):
                for sg in (0.05, 0.19, 0.40, 0.80):
                    for qg in (0.0, 0.013, 0.03):
                        for opt in ("call", "put"):
                            mx = max(mx, abs(bs.bs_pde_residual(S, K, Tg, rg, sg, opt, qg)))
    record("A19 max |PDE residual| on test grid", value=mx)

    # ------------------------------------------------------------------
    # A20 — MC put (NOTEBOOK_PATCHES.md says 7.2151, executed notebook 7.1480)
    # ------------------------------------------------------------------
    pq = bs.simulate_gbm(S0, r, SIG, T, 30, 200_000, seed=42)
    put_mc = np.exp(-r * T) * np.maximum(K - pq[:, -1], 0).mean()
    record("A20 MC put seed=42", mc_put=float(put_mc),
           bs_put=float(bs.bs_price(S0, K, T, r, SIG, "put")))

    # ------------------------------------------------------------------
    # F01 — Feasibility for Phase 4 result 2: proportional costs, optimal N
    #        (criterion: RMSE of total hedge P&L, costs included). Preliminary.
    # ------------------------------------------------------------------
    fineF = fine[:5_000]
    for kc in (0.0001, 0.0005, 0.001):
        rows = []
        for N in Ns:
            sub = fineF[:, :: n_fine // N]
            e, tc = hedge_vec(sub, K, T, r, SIG, cost=kc, dividends=False)
            rows.append((N, float(np.sqrt(np.mean(e ** 2))), float(tc.mean())))
        best = min(rows, key=lambda t: t[1])
        record(f"F01 cost={kc * 1e4:g}bp", best_N=best[0], best_rmse=best[1],
               rmse_by_N={n_: round(v, 3) for n_, v, _ in rows},
               mean_cost_by_N={n_: round(c, 3) for n_, _, c in rows})

    args.json.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
