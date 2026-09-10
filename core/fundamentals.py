"""Fundamental scoring for the names the breakout scan throws up.

Why this exists
---------------
The technical score answers "which of these breakouts has the best chart". It
says nothing about whether the business behind the chart is any good. This
module answers the second question, on the five things you asked for:

    sales growth, profit margins, ROE, ROCE, and cash from operations

plus one this app adds: **balance-sheet safety** — debt, interest cover and
share-count dilution. Over the four-to-six-month hold this system actually has, a
business does not change; what changes is whether it blows up. Leverage and quiet
dilution are what turn a 15% drawdown into a 60% one, and none of the other five
would catch them.

Cash from operations carries the heaviest weight, because it is the one number
a company cannot easily dress up. Profit is an opinion; cash is a fact.

The part that matters: context, not thresholds
----------------------------------------------
A naive screen says "ROE above 20 = good". That rule is wrong often enough to
be dangerous, in both directions, so this module scores each pillar and then
applies explicit, named adjustments:

* **Low ROE is not automatically bad.** A company that funds itself with equity
  rather than debt carries a large equity base, which pushes ROE down by
  construction. If the debt is low and the cash flow is healthy, that is a
  conservative balance sheet, not a weak business.
* **Low ROCE is not automatically bad.** Money poured into R&D, a new plant or
  distribution shows up as capital employed long before it shows up as profit.
  A company in that phase looks worse than it is.
* **High ROE is not automatically good.** Paired with a much lower ROCE it
  usually means leverage is doing the work — the return belongs to the lenders'
  money, and it disappears when rates or refinancing turn.
* **Very high ROE can be an artefact.** After a run of losses the equity base is
  eroded; the first small profit then divides by a tiny denominator and prints
  an ROE nobody actually earned.
* **Earnings without cash are a warning.** Profit rising while cash from
  operations does not follow is the single most reliable sign that the profit is
  accounting rather than economic.

Every adjustment writes a plain-English reason into the result, so the app can
show *why* a stock scored what it scored rather than just the number.

Where the data comes from, and what that costs
----------------------------------------------
Yahoo Finance, through yfinance, cached on disk like the price data. Yahoo
gives the last four annual statements and the recent quarters — enough for a
three-year growth read and a cash-versus-profit check, and not enough for
anything deeper. Two honest limitations:

1. **It is today's fundamentals.** Yahoo does not serve point-in-time
   statements, so using this inside a backtest is look-ahead bias: you would be
   ranking a 2021 breakout using numbers published in 2026. The app defaults
   the toggle OFF for the backtest and ON for the live buy list for exactly
   this reason, and says so on screen.
2. **Coverage is patchy for small caps**, which is most of your universe. A
   name with no usable statements is marked "Data unavailable" and scored
   neutral rather than dropped — a missing filing is not evidence of a bad
   business.
3. **None of this has been backtested**, because point 1 makes that impossible.
   Every other number in this app was chosen by simulation; these weights were
   chosen by reasoning about a four-month holding period. Keep the fundamentals
   slider low until live use has built enough cache to test the question
   properly.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DEFAULT_CACHE, to_yahoo

# --------------------------------------------------------------------------- #
# pillar weights — cash from operations carries the most, as asked
# --------------------------------------------------------------------------- #
PILLAR_WEIGHTS = {
    "cash": 0.40,       # cash from operations, and whether profit converts to it
    "balance": 0.20,    # debt, interest cover, and dilution
    "growth": 0.15,     # sales growth
    "margin": 0.15,     # profit margins and their direction
    "roce": 0.07,       # return on capital employed
    "roe": 0.03,        # return on equity
}
"""Weighted for a four-to-six-month hold, which is what this system actually has.

Over that window a business does not change — so the job here is not "find a
great company", it is **do not step on a landmine**. That reorders things:

* **Cash (40%)** is the strongest single tell for the accounting blow-ups that
  kill small-cap positions, so it gets the most.
* **Balance sheet (20%)** is the second: leverage and quiet equity dilution are
  what turn a 15% drawdown into a 60% one. Dilution in particular is close to
  the defining Indian small-cap risk and nothing else here would catch it.
* **ROE and ROCE together get 10%.** They are long-horizon quality measures with
  little to say about the next four months, they are the easiest numbers to
  flatter, and they need the most context-correction — three of the adjustment
  rules below exist purely to un-mislead them. A number that needs that much
  repair should not carry much weight.
* **Sales growth is cut to 15%** because it double-counts: a small cap making a
  fresh 52-week high almost always has growth already, and the market has
  already paid for it. Its marginal information here is smaller than it looks.
"""

PILLAR_LABELS = {
    "cash": "Cash from operations",
    "balance": "Balance-sheet safety",
    "growth": "Sales growth",
    "margin": "Profit margins",
    "roce": "ROCE",
    "roe": "ROE",
}


# --------------------------------------------------------------------------- #
# the raw numbers we pull out of the statements
# --------------------------------------------------------------------------- #
@dataclass
class Facts:
    """Everything the scorer needs, per symbol. NaN where Yahoo gave nothing."""

    symbol: str = ""
    revenue: list[float] = field(default_factory=list)      # newest first
    net_income: list[float] = field(default_factory=list)
    ebit: list[float] = field(default_factory=list)
    cfo: list[float] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    total_assets: list[float] = field(default_factory=list)
    current_liabilities: list[float] = field(default_factory=list)
    total_debt: float = float("nan")
    capex: list[float] = field(default_factory=list)
    interest: list[float] = field(default_factory=list)
    shares: list[float] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str = ""

    def ok(self) -> bool:
        return bool(self.revenue) and bool(self.net_income)


def _series(df: pd.DataFrame, names: list[str]) -> list[float]:
    """First matching row of a Yahoo statement, newest column first."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    idx = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        key = n.strip().lower()
        if key in idx:
            row = df.loc[idx[key]]
            vals = [float(v) for v in row.tolist() if pd.notna(v)]
            return vals
    return []


def _pick(vals: list[float], i: int = 0) -> float:
    return float(vals[i]) if len(vals) > i else float("nan")


# --------------------------------------------------------------------------- #
# fetching, with the same on-disk cache pattern as prices
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str) -> str:
    os.makedirs(os.path.join(cache_dir, "meta"), exist_ok=True)
    return os.path.join(cache_dir, "meta", "fundamentals.pkl")


def load_cache(cache_dir: str = DEFAULT_CACHE) -> dict[str, Facts]:
    p = _cache_path(cache_dir)
    if not os.path.exists(p):
        return {}
    try:
        blob = pd.read_pickle(p)
        return dict(blob.get("facts", {}))
    except Exception:
        return {}


def save_cache(cache: dict[str, Facts], cache_dir: str = DEFAULT_CACHE) -> None:
    try:
        pd.to_pickle({"facts": cache, "saved_at": time.time()}, _cache_path(cache_dir))
    except Exception:
        pass


def fetch_facts(
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    refresh_days: int = 30,
    offline: bool = False,
    progress_cb=None,
) -> dict[str, Facts]:
    """Statements for `symbols`, cached on disk for `refresh_days`.

    Statements move once a quarter, so a month-old cache is fine and saves a
    request per symbol. Anything that fails is cached as an error so the app
    does not retry a delisted ticker on every rerun.
    """
    cache = load_cache(cache_dir)
    now = time.time()
    stale = refresh_days * 86400
    todo = [s for s in symbols
            if s not in cache or (now - getattr(cache[s], "fetched_at", 0)) > stale]

    if todo and not offline:
        try:
            import yfinance as yf
        except Exception:
            todo = []
            yf = None
        if todo and yf is not None:
            for n, sym in enumerate(todo):
                cache[sym] = _fetch_one(yf, sym)
                if progress_cb and (n % 5 == 0 or n == len(todo) - 1):
                    progress_cb(n + 1, len(todo))
            save_cache(cache, cache_dir)

    return {s: cache.get(s, Facts(symbol=s, error="not fetched")) for s in symbols}


def _fetch_one(yf, sym: str) -> Facts:
    f = Facts(symbol=sym, fetched_at=time.time())
    try:
        tk = yf.Ticker(to_yahoo(sym))
        inc = getattr(tk, "income_stmt", None)
        bal = getattr(tk, "balance_sheet", None)
        cfs = getattr(tk, "cashflow", None)

        f.revenue = _series(inc, ["Total Revenue", "Operating Revenue"])
        f.net_income = _series(inc, ["Net Income", "Net Income Common Stockholders",
                                     "Net Income From Continuing Operation Net Minority Interest"])
        f.ebit = _series(inc, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
        f.cfo = _series(cfs, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                              "Total Cash From Operating Activities"])
        f.capex = _series(cfs, ["Capital Expenditure", "Purchase Of PPE"])
        f.interest = _series(inc, ["Interest Expense", "Interest Expense Non Operating",
                                   "Net Interest Income"])
        f.shares = _series(bal, ["Ordinary Shares Number", "Share Issued",
                                 "Common Stock Shares Outstanding"])
        f.equity = _series(bal, ["Stockholders Equity", "Total Stockholder Equity",
                                 "Common Stock Equity"])
        f.total_assets = _series(bal, ["Total Assets"])
        f.current_liabilities = _series(bal, ["Current Liabilities", "Total Current Liabilities"])
        td = _series(bal, ["Total Debt"])
        if td:
            f.total_debt = td[0]
        else:
            ld = _pick(_series(bal, ["Long Term Debt"]))
            sd = _pick(_series(bal, ["Current Debt", "Short Long Term Debt"]))
            parts = [v for v in (ld, sd) if np.isfinite(v)]
            f.total_debt = float(sum(parts)) if parts else float("nan")

        if not f.ok():
            f.error = "no usable statements"
    except Exception as exc:                                   # noqa: BLE001
        f.error = f"{type(exc).__name__}"
    return f


# --------------------------------------------------------------------------- #
# derived ratios
# --------------------------------------------------------------------------- #
@dataclass
class Ratios:
    sales_growth_1y: float = float("nan")     # %
    sales_cagr_3y: float = float("nan")       # %
    net_margin: float = float("nan")          # %
    margin_trend: float = float("nan")        # pts, latest vs oldest available
    roe: float = float("nan")                 # %
    roce: float = float("nan")                # %
    cfo_latest: float = float("nan")
    cfo_to_pat: float = float("nan")          # x, 3-year sum basis
    cfo_positive_years: int = 0
    cfo_years: int = 0
    debt_to_equity: float = float("nan")      # x
    interest_cover: float = float("nan")      # EBIT / interest expense, x
    dilution_pa: float = float("nan")         # share-count CAGR, % a year
    capex_intensity: float = float("nan")     # capex / revenue, %
    asset_growth: float = float("nan")        # %, newest vs oldest
    equity_eroded: bool = False               # equity shrank badly = ROE artefact risk


def _cagr(newest: float, oldest: float, years: int) -> float:
    if not (np.isfinite(newest) and np.isfinite(oldest)) or oldest <= 0 or years <= 0:
        return float("nan")
    if newest <= 0:
        return -100.0
    return ((newest / oldest) ** (1 / years) - 1) * 100


def ratios(f: Facts) -> Ratios:
    r = Ratios()
    if not f.ok():
        return r

    rev, ni = f.revenue, f.net_income
    r.sales_growth_1y = ((rev[0] / rev[1] - 1) * 100
                         if len(rev) > 1 and rev[1] > 0 else float("nan"))
    if len(rev) >= 3:
        r.sales_cagr_3y = _cagr(rev[0], rev[-1], len(rev) - 1)

    if len(rev) and rev[0] > 0 and len(ni):
        r.net_margin = ni[0] / rev[0] * 100
    if len(rev) >= 3 and len(ni) >= 3 and rev[-1] > 0 and rev[0] > 0:
        r.margin_trend = (ni[0] / rev[0] - ni[-1] / rev[-1]) * 100

    eq = f.equity
    if len(eq) and len(ni):
        base = np.mean([e for e in eq[:2] if np.isfinite(e)]) if len(eq) > 1 else eq[0]
        if np.isfinite(base) and base > 0:
            r.roe = ni[0] / base * 100
        if len(eq) >= 3 and eq[-1] > 0:
            r.equity_eroded = eq[0] < eq[-1] * 0.6

    if len(f.ebit) and len(f.total_assets) and len(f.current_liabilities):
        ce = f.total_assets[0] - f.current_liabilities[0]
        if ce > 0:
            r.roce = f.ebit[0] / ce * 100

    if f.cfo:
        r.cfo_latest = f.cfo[0]
        r.cfo_years = len(f.cfo)
        r.cfo_positive_years = int(sum(1 for v in f.cfo if v > 0))
        n = min(3, len(f.cfo), len(ni))
        pat = sum(ni[:n])
        if n and pat > 0:
            r.cfo_to_pat = sum(f.cfo[:n]) / pat

    if np.isfinite(f.total_debt) and len(eq) and eq[0] > 0:
        r.debt_to_equity = f.total_debt / eq[0]

    if f.interest and len(f.ebit):
        interest = abs(f.interest[0])
        if interest > 0:
            r.interest_cover = f.ebit[0] / interest
        elif f.ebit[0] > 0:
            r.interest_cover = 99.0            # no interest to cover at all

    if len(f.shares) >= 2:
        newest, oldest = f.shares[0], f.shares[-1]
        yrs = len(f.shares) - 1
        if oldest > 0 and yrs > 0:
            r.dilution_pa = ((newest / oldest) ** (1 / yrs) - 1) * 100

    if f.capex and len(rev) and rev[0] > 0:
        r.capex_intensity = abs(f.capex[0]) / rev[0] * 100

    if len(f.total_assets) >= 3 and f.total_assets[-1] > 0:
        r.asset_growth = (f.total_assets[0] / f.total_assets[-1] - 1) * 100

    return r


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _band(x: float, lo: float, hi: float) -> float:
    """Map x onto 0..100 across [lo, hi], flat outside."""
    if not np.isfinite(x):
        return float("nan")
    if hi == lo:
        return 50.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)


@dataclass
class Score:
    symbol: str = ""
    total: float = float("nan")               # 0..100
    pillars: dict = field(default_factory=dict)
    good: list[str] = field(default_factory=list)      # "what is good"
    watch: list[str] = field(default_factory=list)     # what to be careful about
    adjustments: list[str] = field(default_factory=list)
    available: bool = False
    note: str = ""
    ratios: Ratios = field(default_factory=Ratios)

    def summary(self) -> str:
        if not self.available:
            return self.note or "Data unavailable"
        return " · ".join(self.good) if self.good else "nothing stands out"


def score_one(f: Facts, weights: dict | None = None) -> Score:
    """Score one company 0..100 and say, in words, what carried the score."""
    w = dict(PILLAR_WEIGHTS)
    if weights:
        w.update(weights)

    s = Score(symbol=f.symbol)
    r = ratios(f)
    s.ratios = r
    if not f.ok():
        s.note = "Data unavailable" + (f" ({f.error})" if f.error else "")
        s.total = float("nan")
        return s
    s.available = True

    # ---------------- pillar: cash from operations (the heaviest) -----------
    cash = np.nan
    if r.cfo_years:
        consistency = r.cfo_positive_years / r.cfo_years * 100
        conv = _band(r.cfo_to_pat, 0.4, 1.3) if np.isfinite(r.cfo_to_pat) else np.nan
        parts = [p for p in (consistency, conv) if np.isfinite(p)]
        if parts:
            cash = float(np.average(parts, weights=[0.4, 0.6][:len(parts)]))
        if r.cfo_positive_years == r.cfo_years and r.cfo_years >= 3:
            s.good.append(f"cash from operations positive in all {r.cfo_years} years on file")
        elif r.cfo_positive_years and r.cfo_years:
            s.watch.append(f"cash from operations negative in "
                           f"{r.cfo_years - r.cfo_positive_years} of {r.cfo_years} years")
        if np.isfinite(r.cfo_to_pat):
            if r.cfo_to_pat >= 1.0:
                s.good.append(f"every rupee of profit backed by ₹{r.cfo_to_pat:.2f} of operating cash")
            elif r.cfo_to_pat < 0.6:
                s.watch.append(f"only ₹{r.cfo_to_pat:.2f} of cash per rupee of reported profit "
                               "— earnings quality is weak")

    # ---------------- pillar: sales growth ----------------------------------
    g_parts = []
    if np.isfinite(r.sales_cagr_3y):
        g_parts.append(_band(r.sales_cagr_3y, 0, 30))
    if np.isfinite(r.sales_growth_1y):
        g_parts.append(_band(r.sales_growth_1y, -5, 35))
    growth = float(np.mean(g_parts)) if g_parts else np.nan
    if np.isfinite(r.sales_cagr_3y) and r.sales_cagr_3y >= 15:
        s.good.append(f"sales compounding {r.sales_cagr_3y:.0f}% a year")
    elif np.isfinite(r.sales_cagr_3y) and r.sales_cagr_3y < 0:
        s.watch.append(f"sales shrinking {abs(r.sales_cagr_3y):.0f}% a year")

    # ---------------- pillar: margins ---------------------------------------
    m_parts = []
    if np.isfinite(r.net_margin):
        m_parts.append(_band(r.net_margin, 0, 20))
    if np.isfinite(r.margin_trend):
        m_parts.append(_band(r.margin_trend, -5, 5))
    margin = float(np.mean(m_parts)) if m_parts else np.nan
    if np.isfinite(r.net_margin) and r.net_margin >= 10:
        s.good.append(f"net margin {r.net_margin:.1f}%")
    if np.isfinite(r.margin_trend) and r.margin_trend >= 2:
        s.good.append(f"margins up {r.margin_trend:.1f} pts over the period on file")
    elif np.isfinite(r.margin_trend) and r.margin_trend <= -3:
        s.watch.append(f"margins down {abs(r.margin_trend):.1f} pts")

    # ---------------- pillar: balance-sheet safety --------------------------
    # Leverage and quiet dilution are what turn a bad quarter into a wipeout, and
    # neither shows up anywhere else in this score.
    b_parts, b_w = [], []
    if np.isfinite(r.debt_to_equity):
        b_parts.append(_band(-r.debt_to_equity, -2.0, 0.0)); b_w.append(0.35)
        if r.debt_to_equity >= 1.5:
            s.watch.append(f"debt/equity {r.debt_to_equity:.1f} — a leveraged balance sheet")
    if np.isfinite(r.interest_cover):
        b_parts.append(_band(r.interest_cover, 1.5, 8.0)); b_w.append(0.35)
        if r.interest_cover >= 6:
            s.good.append(f"interest covered {min(r.interest_cover, 99):.0f}x by operating profit")
        elif r.interest_cover < 2:
            s.watch.append(f"interest covered only {r.interest_cover:.1f}x — thin cushion")
    if np.isfinite(r.dilution_pa):
        b_parts.append(_band(-r.dilution_pa, -12.0, 0.0)); b_w.append(0.30)
        if r.dilution_pa <= 0.5:
            s.good.append("share count flat — shareholders are not being diluted")
        elif r.dilution_pa >= 5:
            s.watch.append(f"share count growing {r.dilution_pa:.0f}% a year — your slice keeps shrinking")
    balance = float(np.average(b_parts, weights=b_w)) if b_parts else np.nan

    # ---------------- pillars: ROE and ROCE ---------------------------------
    roe = _band(r.roe, 5, 25) if np.isfinite(r.roe) else np.nan
    roce = _band(r.roce, 5, 25) if np.isfinite(r.roce) else np.nan
    if np.isfinite(r.roce) and r.roce >= 20:
        s.good.append(f"ROCE {r.roce:.0f}%")
    if np.isfinite(r.roe) and 15 <= r.roe <= 45:
        s.good.append(f"ROE {r.roe:.0f}%")
    if np.isfinite(r.debt_to_equity) and r.debt_to_equity <= 0.3:
        s.good.append(f"debt/equity {r.debt_to_equity:.2f} — funded by its own money")

    # ---------------- the nuance: context adjustments -----------------------
    cash_ok = np.isfinite(cash) and cash >= 55
    low_debt = np.isfinite(r.debt_to_equity) and r.debt_to_equity <= 0.35

    # 1. low ROE because the company is equity funded, not because it is weak
    if np.isfinite(roe) and roe < 45 and low_debt and cash_ok:
        roe = min(100.0, roe + 22)
        s.adjustments.append(
            "ROE marked up: it is low because the company carries almost no debt, "
            "and the cash flow is healthy — a large equity base, not a weak return")

    # 2. low ROCE while capital is being planted
    if (np.isfinite(roce) and roce < 45 and cash_ok
            and ((np.isfinite(r.capex_intensity) and r.capex_intensity >= 8)
                 or (np.isfinite(r.asset_growth) and r.asset_growth >= 40))):
        roce = min(100.0, roce + 18)
        s.adjustments.append(
            "ROCE marked up: capital employed has grown faster than profit because the "
            "company is still building — reinvestment phase, not a broken return")

    # 3. high ROE riding on debt rather than on the business
    if (np.isfinite(r.roe) and np.isfinite(r.roce) and r.roe >= 18
            and r.roce < r.roe * 0.6
            and np.isfinite(r.debt_to_equity) and r.debt_to_equity >= 1.0):
        roe = max(0.0, roe - 30)
        s.watch.append(f"ROE {r.roe:.0f}% sits well above ROCE {r.roce:.0f}% on debt/equity "
                       f"{r.debt_to_equity:.1f} — the return is leverage, not the business")
        s.adjustments.append("ROE marked down: the gap to ROCE is being paid for with debt")

    # 4. ROE flattered by an equity base that past losses ate
    if r.equity_eroded and np.isfinite(r.roe) and r.roe >= 30:
        roe = max(0.0, roe - 35)
        s.watch.append("ROE looks high mainly because past losses shrank the equity base")
        s.adjustments.append("ROE marked down: small denominator, not a large return")

    # 5. profit that cash does not follow poisons everything above it
    if np.isfinite(r.cfo_to_pat) and r.cfo_to_pat < 0.5 and np.isfinite(r.net_margin) and r.net_margin > 0:
        margin = margin * 0.7 if np.isfinite(margin) else margin
        roe = roe * 0.7 if np.isfinite(roe) else roe
        roce = roce * 0.7 if np.isfinite(roce) else roce
        s.adjustments.append(
            "margins, ROE and ROCE all marked down: reported profit is not turning into cash")

    # 6. debt is only dangerous when it cannot be serviced
    if (np.isfinite(balance) and np.isfinite(r.debt_to_equity) and r.debt_to_equity >= 1.0
            and np.isfinite(r.interest_cover) and r.interest_cover >= 6 and cash_ok):
        balance = min(100.0, balance + 15)
        s.adjustments.append(
            "balance sheet marked up: the debt is large but comfortably serviced — "
            f"interest covered {min(r.interest_cover, 99):.0f}x with healthy operating cash")

    # 7. a turnaround worth noticing rather than punishing
    if (np.isfinite(r.sales_cagr_3y) and r.sales_cagr_3y >= 10 and cash_ok
            and np.isfinite(r.roce) and r.roce < 12 and low_debt):
        s.good.append("growing and cash-generative while returns are still depressed "
                      "— reinvestment or turnaround")

    # ---------------- combine ------------------------------------------------
    pillars = {"cash": cash, "balance": balance, "growth": growth,
               "margin": margin, "roce": roce, "roe": roe}
    s.pillars = {k: (round(float(v), 1) if np.isfinite(v) else None) for k, v in pillars.items()}

    usable = {k: v for k, v in pillars.items() if np.isfinite(v)}
    if not usable:
        s.available = False
        s.note = "Data unavailable (statements present but nothing computable)"
        s.total = float("nan")
        return s

    wt = np.array([w.get(k, 0.0) for k in usable])
    if wt.sum() <= 0:
        wt = np.ones(len(usable))
    s.total = float(np.clip(np.average(list(usable.values()), weights=wt), 0, 100))

    missing = [PILLAR_LABELS[k] for k in pillars if k not in usable]
    if missing:
        s.note = "scored on what was available; missing: " + ", ".join(missing)
    return s


def score_frame(
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    offline: bool = False,
    refresh_days: int = 30,
    weights: dict | None = None,
    progress_cb=None,
) -> pd.DataFrame:
    """One row per symbol: score, the pillars, and the words behind them."""
    cols = ["symbol", "fundamentals_score", "available", "what is good", "watch",
            "context applied", "sales CAGR 3y %", "sales growth 1y %", "net margin %",
            "ROE %", "ROCE %", "CFO / PAT (3y)", "debt / equity", "interest cover x",
            "dilution % pa", "note"]
    if not symbols:
        return pd.DataFrame(columns=cols).set_index("symbol")

    facts = fetch_facts(symbols, cache_dir=cache_dir, refresh_days=refresh_days,
                        offline=offline, progress_cb=progress_cb)
    rows = []
    for sym in symbols:
        sc = score_one(facts[sym], weights)
        r = sc.ratios
        rows.append({
            "symbol": sym,
            "fundamentals_score": sc.total,
            "available": sc.available,
            "what is good": sc.summary(),
            "watch": " · ".join(sc.watch) if sc.watch else "",
            "context applied": " · ".join(sc.adjustments) if sc.adjustments else "",
            "sales CAGR 3y %": r.sales_cagr_3y,
            "sales growth 1y %": r.sales_growth_1y,
            "net margin %": r.net_margin,
            "ROE %": r.roe,
            "ROCE %": r.roce,
            "CFO / PAT (3y)": r.cfo_to_pat,
            "debt / equity": r.debt_to_equity,
            "interest cover x": r.interest_cover,
            "dilution % pa": r.dilution_pa,
            "note": sc.note,
        })
    return pd.DataFrame(rows).set_index("symbol")
