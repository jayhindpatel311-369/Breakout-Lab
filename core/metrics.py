"""
Performance statistics.

All risk metrics are computed off the *time-weighted* return series, not the raw
equity curve. With a SIP running these are very different animals: a portfolio
that gets a fresh Rs 50,000 every month has an equity curve that barely dips,
which makes the drawdown look wonderful and means nothing.

Implemented from scratch rather than pulling in quantstats, so there is no
version-drift surprise and you can read exactly what every number means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def to_returns(index_series: pd.Series) -> pd.Series:
    return index_series.pct_change().fillna(0.0)


def cagr(index_series: pd.Series) -> float:
    s = index_series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def cumulative_return(index_series: pd.Series) -> float:
    s = index_series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1)


def annual_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 else float("nan")


def drawdown_series(index_series: pd.Series) -> pd.Series:
    s = index_series.dropna()
    peak = s.cummax()
    return (s / peak - 1.0).rename("drawdown")


def max_drawdown(index_series: pd.Series) -> float:
    dd = drawdown_series(index_series)
    return float(dd.min()) if len(dd) else float("nan")


def drawdown_table(index_series: pd.Series, top: int = 10) -> pd.DataFrame:
    """Every distinct underwater episode, worst first."""
    dd = drawdown_series(index_series)
    if dd.empty:
        return pd.DataFrame()
    under = dd < -1e-12
    episodes = []
    start = None
    for d, flag in under.items():
        if flag and start is None:
            start = d
        elif not flag and start is not None:
            seg = dd.loc[start:d]
            episodes.append((start, d, seg.min(), seg.idxmin()))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        episodes.append((start, dd.index[-1], seg.min(), seg.idxmin()))

    rows = []
    for s, e, depth, trough in episodes:
        rows.append(
            {
                "Start": s.date(),
                "Trough": trough.date(),
                "End": e.date(),
                "Depth (%)": depth * 100,
                "Length (days)": (e - s).days,
                "Recovery (days)": (e - trough).days,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Depth (%)").head(top).reset_index(drop=True)


def longest_drawdown_days(index_series: pd.Series) -> float:
    t = drawdown_table(index_series, top=10_000)
    return float(t["Length (days)"].max()) if not t.empty else 0.0


def avg_drawdown(index_series: pd.Series) -> float:
    t = drawdown_table(index_series, top=10_000)
    return float(t["Depth (%)"].mean()) if not t.empty else 0.0


def avg_drawdown_days(index_series: pd.Series) -> float:
    t = drawdown_table(index_series, top=10_000)
    return float(t["Length (days)"].mean()) if not t.empty else 0.0


# --------------------------------------------------------------------------- #
# ratios
# --------------------------------------------------------------------------- #
def sharpe(returns: pd.Series, rf_annual: float = 0.065) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    excess = r - rf_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_annual: float = 0.065) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    excess = r - rf_annual / TRADING_DAYS
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).sum() / len(excess))
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def calmar(index_series: pd.Series) -> float:
    mdd = max_drawdown(index_series)
    c = cagr(index_series)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(c / abs(mdd))


def beta_alpha(returns: pd.Series, bench_returns: pd.Series, rf_annual: float = 0.065) -> tuple[float, float]:
    """CAPM beta and annualised alpha (Jensen's alpha)."""
    df = pd.concat([returns.rename("p"), bench_returns.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return float("nan"), float("nan")
    rf_d = rf_annual / TRADING_DAYS
    p = df["p"] - rf_d
    b = df["b"] - rf_d
    var_b = b.var(ddof=1)
    if var_b == 0 or np.isnan(var_b):
        return float("nan"), float("nan")
    beta = float(p.cov(b) / var_b)
    alpha_daily = float(p.mean() - beta * b.mean())
    return beta, float((1 + alpha_daily) ** TRADING_DAYS - 1)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    return float(np.percentile(r, (1 - level) * 100)) if len(r) > 20 else float("nan")


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    cutoff = np.percentile(r, (1 - level) * 100)
    tail = r[r <= cutoff]
    return float(tail.mean()) if len(tail) else float("nan")


# --------------------------------------------------------------------------- #
# period aggregation
# --------------------------------------------------------------------------- #
def _resample_returns(index_series: pd.Series, rule: str) -> pd.Series:
    s = index_series.dropna()
    if s.empty:
        return s
    try:
        agg = s.resample(rule).last()
    except ValueError:
        agg = s.resample({"ME": "M", "QE": "Q", "YE": "A"}.get(rule, rule)).last()
    first = pd.Series([s.iloc[0]], index=[s.index[0]])
    agg = pd.concat([first, agg]).sort_index()
    agg = agg[~agg.index.duplicated(keep="first")]
    return agg.pct_change().dropna()


def monthly_returns(index_series: pd.Series) -> pd.Series:
    return _resample_returns(index_series, "ME")


def quarterly_returns(index_series: pd.Series) -> pd.Series:
    return _resample_returns(index_series, "QE")


def yearly_returns(index_series: pd.Series) -> pd.Series:
    return _resample_returns(index_series, "YE")


def monthly_table(index_series: pd.Series) -> pd.DataFrame:
    """Year x month matrix of returns in %, ready for a heatmap."""
    m = monthly_returns(index_series) * 100
    if m.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    table = df.pivot_table(index="year", columns="month", values="ret", aggfunc="last")
    table = table.reindex(columns=range(1, 13))
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return table.sort_index(ascending=False)


def yearly_table(
    equity: pd.Series,
    cashflows: pd.Series,
    twr: pd.Series,
) -> pd.DataFrame:
    """Year-by-year: opening capital, money added, profit, closing capital, and
    the time-weighted return that actually describes the strategy's skill."""
    if equity.empty:
        return pd.DataFrame()
    years = sorted(set(equity.index.year))
    rows = []
    for y in years:
        mask = equity.index.year == y
        eq_y = equity[mask]
        cf_y = cashflows[mask] if cashflows is not None else pd.Series(dtype=float)
        twr_y = twr[mask]
        if eq_y.empty:
            continue
        prior = equity[equity.index < eq_y.index[0]]
        start_cap = float(prior.iloc[-1]) if len(prior) else 0.0
        added = float(cf_y.sum())
        end_cap = float(eq_y.iloc[-1])
        prior_twr = twr[twr.index < twr_y.index[0]]
        twr_start = float(prior_twr.iloc[-1]) if len(prior_twr) else float(twr_y.iloc[0])
        ret = (float(twr_y.iloc[-1]) / twr_start - 1) * 100 if twr_start > 0 else np.nan
        rows.append(
            {
                "Year": y,
                "Months": len(set(eq_y.index.month)),
                "Return (%)": ret,
                "Start Capital": start_cap,
                "Added (SIP)": added,
                "Net Profit": end_cap - start_cap - added,
                "End Capital": end_cap,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the full report
# --------------------------------------------------------------------------- #
def full_stats(
    twr: pd.Series,
    bench_twr: pd.Series | None = None,
    rf_annual: float = 0.065,
) -> pd.DataFrame:
    """Everything, side by side with the benchmark — same layout as the
    'Performance metrics' panel you screenshotted."""
    r = to_returns(twr)
    br = to_returns(bench_twr) if bench_twr is not None and len(bench_twr) else None

    def col(series_idx: pd.Series, rets: pd.Series) -> dict:
        m = monthly_returns(series_idx)
        q = quarterly_returns(series_idx)
        yy = yearly_returns(series_idx)
        up_m = m[m > 0]
        dn_m = m[m < 0]
        return {
            "Start Period": series_idx.index[0].date() if len(series_idx) else None,
            "End Period": series_idx.index[-1].date() if len(series_idx) else None,
            "Cumulative Return": cumulative_return(series_idx),
            "CAGR": cagr(series_idx),
            "Volatility (ann.)": annual_vol(rets),
            "Sharpe": sharpe(rets, rf_annual),
            "Sortino": sortino(rets, rf_annual),
            "Sortino/sqrt(2)": sortino(rets, rf_annual) / np.sqrt(2),
            "Max Drawdown": max_drawdown(series_idx),
            "Calmar (RoMaD)": calmar(series_idx),
            "Skew": float(rets.skew()) if len(rets) > 3 else np.nan,
            "Kurtosis": float(rets.kurtosis()) if len(rets) > 3 else np.nan,
            "Expected Daily": float(rets.mean()),
            "Expected Monthly": float(m.mean()) if len(m) else np.nan,
            "Expected Yearly": float(yy.mean()) if len(yy) else np.nan,
            "Daily VaR (95%)": value_at_risk(rets),
            "Expected Shortfall (cVaR)": conditional_var(rets),
            "Best Day": float(rets.max()) if len(rets) else np.nan,
            "Worst Day": float(rets.min()) if len(rets) else np.nan,
            "Best Month": float(m.max()) if len(m) else np.nan,
            "Worst Month": float(m.min()) if len(m) else np.nan,
            "Best Year": float(yy.max()) if len(yy) else np.nan,
            "Worst Year": float(yy.min()) if len(yy) else np.nan,
            "Longest DD (days)": longest_drawdown_days(series_idx),
            "Avg Drawdown": avg_drawdown(series_idx) / 100,
            "Avg Drawdown (days)": avg_drawdown_days(series_idx),
            "Avg Up Month": float(up_m.mean()) if len(up_m) else np.nan,
            "Avg Down Month": float(dn_m.mean()) if len(dn_m) else np.nan,
            "Win Days %": float((rets > 0).mean()),
            "Win Month %": float((m > 0).mean()) if len(m) else np.nan,
            "Win Quarter %": float((q > 0).mean()) if len(q) else np.nan,
            "Win Year %": float((yy > 0).mean()) if len(yy) else np.nan,
        }

    data = {"Strategy": col(twr, r)}
    if br is not None:
        data["Benchmark"] = col(bench_twr, br)
        b, a = beta_alpha(r, br, rf_annual)
        data["Strategy"]["Beta"] = b
        data["Strategy"]["Alpha (ann.)"] = a
        data["Benchmark"]["Beta"] = 1.0
        data["Benchmark"]["Alpha (ann.)"] = 0.0

    df = pd.DataFrame(data)
    order = [
        "Start Period", "End Period", "Cumulative Return", "CAGR",
        "Volatility (ann.)", "Sharpe", "Sortino", "Sortino/sqrt(2)",
        "Max Drawdown", "Calmar (RoMaD)", "Beta", "Alpha (ann.)",
        "Skew", "Kurtosis", "Expected Daily", "Expected Monthly", "Expected Yearly",
        "Daily VaR (95%)", "Expected Shortfall (cVaR)",
        "Best Day", "Worst Day", "Best Month", "Worst Month", "Best Year", "Worst Year",
        "Longest DD (days)", "Avg Drawdown", "Avg Drawdown (days)",
        "Avg Up Month", "Avg Down Month",
        "Win Days %", "Win Month %", "Win Quarter %", "Win Year %",
    ]
    df = df.reindex([o for o in order if o in df.index])
    return df


PERCENT_ROWS = {
    "Cumulative Return", "CAGR", "Volatility (ann.)", "Max Drawdown",
    "Alpha (ann.)", "Expected Daily", "Expected Monthly", "Expected Yearly",
    "Daily VaR (95%)", "Expected Shortfall (cVaR)", "Best Day", "Worst Day",
    "Best Month", "Worst Month", "Best Year", "Worst Year", "Avg Drawdown",
    "Avg Up Month", "Avg Down Month", "Win Days %", "Win Month %",
    "Win Quarter %", "Win Year %",
}


def format_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of `full_stats` for display."""
    out = df.copy().astype(object)
    for row in out.index:
        for c in out.columns:
            v = out.loc[row, c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out.loc[row, c] = "—"
            elif row in PERCENT_ROWS and isinstance(v, (int, float)):
                out.loc[row, c] = f"{v * 100:,.2f}%"
            elif isinstance(v, float):
                out.loc[row, c] = f"{v:,.2f}"
            else:
                out.loc[row, c] = str(v)
    # Uniform string dtype — mixed object columns make Arrow (and therefore
    # Streamlit's table renderer) complain.
    return out.astype(str)


def trade_summary(trades: pd.DataFrame, rebalance_log: pd.DataFrame) -> dict:
    """Headline trade stats, in the shape of the 'Trade Summary' panel."""
    out = {
        "Total trades": 0,
        "Winning periods": 0,
        "Losing periods": 0,
        "Win rate": np.nan,
        "Avg win": np.nan,
        "Avg loss": np.nan,
        "Profit factor": np.nan,
        "Total costs": 0.0,
    }
    if trades is not None and not trades.empty:
        out["Total trades"] = int(len(trades))
        out["Total costs"] = float(trades["cost"].sum())
    if rebalance_log is not None and not rebalance_log.empty and "Return (%)" in rebalance_log:
        r = rebalance_log["Return (%)"].dropna()
        wins, losses = r[r > 0], r[r <= 0]
        out["Winning periods"] = int(len(wins))
        out["Losing periods"] = int(len(losses))
        out["Win rate"] = float(len(wins) / len(r)) if len(r) else np.nan
        out["Avg win"] = float(wins.mean()) / 100 if len(wins) else np.nan
        out["Avg loss"] = float(losses.mean()) / 100 if len(losses) else np.nan
        gross_win = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        out["Profit factor"] = gross_win / gross_loss if gross_loss > 0 else np.nan
    return out
