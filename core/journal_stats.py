"""
The analytics the live journal never had.

The maths for CAGR, drawdown, Calmar, Sharpe and the monthly / yearly tables has
been in `core/metrics.py` all along — the backtest has used it since day one.
What was missing was the *input*. Every one of those numbers needs a **daily
mark-to-market equity curve**, and the journal only ever recorded fills. A list
of fills tells you realised P&L; it cannot tell you what a drawdown was, because
a drawdown happens in the weeks when nothing is filled at all.

So the first function here builds that curve — cash plus the day-by-day value of
whatever was open — and everything below it is that curve, or the round trips,
put through the existing library rather than a second implementation of it.

Three things worth being straight about:

* **A drawdown needs prices, not fills.** Feed these functions the same daily
  Close frame the rest of the app uses. Without it you get realised P&L and
  nothing else, and the functions say so rather than inventing a curve.
* **CAGR on a young book is noise.** Under three months a lucky week
  annualises to a number nobody should believe. The UI shows CAGR once the
  equity curve is at least 90 days (Zerodha / broker-style annualised return:
  `(end/start)^(365.25/days) − 1`).
* **Index buckets are today's.** NSE does not publish historical constituents,
  so a stock is bucketed where it sits now. A winner is exactly the stock most
  likely to have moved up a bucket since you bought it, which flatters the
  larger buckets. It is still worth looking at; it is not worth trusting to the
  second decimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .journal import RUNG_LABELS, Book, live_stop_price

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# the missing piece: a real equity curve
# --------------------------------------------------------------------------- #
def _as_day(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    try:
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_convert(None)
    except Exception:
        try:
            t = t.tz_localize(None)
        except Exception:
            pass
    return pd.Timestamp(t).normalize()


def equity_curve(book: Book, close: pd.DataFrame | None) -> pd.Series:
    """Daily portfolio value: cash + the marked value of everything open.

    Walks the ledger in order, holding a running cash balance and a running
    share count per symbol, and values the book on every trading day in
    `close`. That is the series a drawdown can actually be measured on.

    Dates are normalised (tz stripped) so a Monday fill matches Monday's bar.
    Cash-flows (add/withdraw) are applied on their day. Compounding steps are
    ignored here — they only change sizing, not cash.
    """
    if not book.ledger or close is None or close.empty:
        return pd.Series(dtype=float)

    px = close.copy()
    px.index = pd.DatetimeIndex([_as_day(d) for d in px.index])
    px = px[~px.index.duplicated(keep="last")].sort_index()

    led = pd.DataFrame(book.ledger).copy()
    led["date"] = led["date"].map(_as_day)
    for c in ("qty", "price", "value"):
        led[c] = pd.to_numeric(led[c], errors="coerce").fillna(0.0)
    led = led.sort_values("date")

    first = led["date"].min()
    idx = px.index[px.index >= first]
    if len(idx) == 0:
        return pd.Series(dtype=float)

    flows: dict[pd.Timestamp, float] = {}
    net_flows = 0.0
    for f in getattr(book, "cash_flows", []) or []:
        if str(f.get("kind") or "") == "compound":
            continue
        amt = float(f.get("amount") or 0.0)
        net_flows += amt
        d = _as_day(f.get("date") or first)
        flows[d] = flows.get(d, 0.0) + amt

    # capital already includes deposits; start from capital net of later flows
    # so a deposit on 12 Sep does not inflate 1 Sep's equity.
    cash = float(book.capital) - net_flows
    holdings: dict[str, int] = {}
    out = pd.Series(index=idx, dtype=float)
    fills = {d: g for d, g in led.groupby("date")}

    for day in idx:
        cash += flows.get(day, 0.0)
        for _, r in fills.get(day, pd.DataFrame()).iterrows():
            sym, qty = str(r["symbol"]), int(r["qty"])
            if str(r["side"]) == "BUY":
                cash -= float(r["value"])
                holdings[sym] = holdings.get(sym, 0) + qty
            else:
                cash += float(r["value"])
                holdings[sym] = holdings.get(sym, 0) - qty
                if holdings[sym] <= 0:
                    holdings.pop(sym, None)
        mark = 0.0
        if holdings:
            row = px.loc[day] if day in px.index else None
            for sym, qty in holdings.items():
                p = float(row.get(sym, np.nan)) if row is not None else np.nan
                if not np.isfinite(p) and sym in px.columns:
                    hist = px[sym].loc[:day].dropna()
                    p = float(hist.iloc[-1]) if len(hist) else 0.0
                if not np.isfinite(p):
                    p = 0.0
                mark += qty * p
        out.loc[day] = cash + mark

    out = out.dropna().rename("equity")
    # last point must tie to the live book: cash + today's mark
    if len(out) and px.shape[1]:
        last = px.iloc[-1]
        mtm = 0.0
        for p in book.positions:
            if not p.is_open():
                continue
            v = float(last.get(p.symbol, np.nan)) if p.symbol in last.index else np.nan
            if not np.isfinite(v):
                v = float(p.entry_price)
            mtm += p.open_qty * v
        out.iloc[-1] = float(book.cash) + mtm
    return out


# --------------------------------------------------------------------------- #
# round trips
# --------------------------------------------------------------------------- #
def round_trips(book: Book) -> pd.DataFrame:
    """One row per closed trade — every partial exit of it collapsed into one.

    A position that booked at +20%, again at +40% and finally on the 20 EMA is
    ONE trade with three fills, and reading it as three is how a win rate ends
    up saying 70% on a system that loses money.
    """
    sells = [r for r in book.ledger if r.get("side") == "SELL"]
    if not sells:
        return pd.DataFrame()
    df = pd.DataFrame(sells)
    for c in ("qty", "price", "pnl", "gain_pct"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["entry_date"] = pd.to_datetime(df.get("entry_date", df["date"]))
    df["entry_price"] = pd.to_numeric(df.get("entry_price"), errors="coerce")
    df["cost"] = df["qty"] * df["entry_price"]

    rows = []
    for (sym, ent), g in df.groupby(["symbol", "entry_date"]):
        cost = float(g["cost"].sum())
        pnl = float(g["pnl"].sum())
        rungs = [r for r in g["rung"].tolist() if r]
        exit_date = g["date"].max()
        rows.append({
            "symbol": sym,
            "entry_date": pd.Timestamp(ent).date(),
            "exit_date": pd.Timestamp(exit_date).date(),
            "days held": int((exit_date - pd.Timestamp(ent)).days),
            "qty": int(g["qty"].sum()),
            "entry_price": round(float(g["entry_price"].iloc[0]), 2),
            "capital": round(cost, 0),
            "P&L": round(pnl, 0),
            "return %": round(pnl / cost * 100, 2) if cost else np.nan,
            "exits": len(g),
            "booked a target": any(str(r).startswith("profit_") for r in rungs),
            "last exit": RUNG_LABELS.get(str(g.sort_values("date")["rung"].iloc[-1]),
                                          str(g.sort_values("date")["rung"].iloc[-1])),
            "rungs hit": ", ".join(RUNG_LABELS.get(r, r) for r in dict.fromkeys(rungs)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("exit_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# headline numbers
# --------------------------------------------------------------------------- #
def stats(book: Book, close: pd.DataFrame | None = None) -> dict:
    """Everything the header tiles and the summary table need.

    Values that cannot honestly be computed come back as NaN rather than 0 —
    a Profit Factor of 0 and a Profit Factor that does not exist yet are very
    different statements.
    """
    eq = equity_curve(book, close)
    rt = round_trips(book)
    sells = [r for r in book.ledger if r.get("side") == "SELL"]

    realised = float(sum(float(r.get("pnl", 0.0) or 0.0) for r in sells))
    last_px = close.iloc[-1] if close is not None and not close.empty else None
    unreal = 0.0
    open_val = 0.0
    for p in book.positions:
        if not p.is_open():
            continue
        px = float(last_px.get(p.symbol, np.nan)) if last_px is not None else np.nan
        if np.isfinite(px):
            open_val += p.open_qty * px
            unreal += (px - p.entry_price) * p.open_qty
        else:
            open_val += p.open_qty * p.entry_price

    divs = float(sum(float(r.get("amount", 0.0)) for r in getattr(book, "income", [])))
    # Live book is the source of truth: cash + marked opens. ROI, net P&L and
    # "final capital" all come off this so they cannot disagree.
    value = float(book.cash) + open_val
    net = value - float(book.capital)
    out = {
        "Net P&L": net,
        "Realised P&L": realised,
        "Unrealised P&L": unreal,
        "Dividends": divs,
        "Portfolio value": value,
        "Capital": book.capital,
        "ROI %": (net / book.capital) * 100 if book.capital else np.nan,
        "Trades closed": len(rt),
        "Open positions": len([p for p in book.positions if p.is_open()]),
        "Drafts waiting": len(book.drafts),
    }

    # --- trade-level ------------------------------------------------------- #
    if len(rt):
        wins, losses = rt[rt["P&L"] > 0], rt[rt["P&L"] <= 0]
        gross_win = float(wins["P&L"].sum())
        gross_loss = float(-losses["P&L"].sum())
        wr = len(wins) / len(rt)
        avg_win = float(wins["return %"].mean()) if len(wins) else np.nan
        avg_loss = float(losses["return %"].mean()) if len(losses) else np.nan
        out.update({
            "Win rate %": wr * 100,
            "Wins": len(wins),
            "Losses": len(losses),
            "Profit factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
            "Expectancy ₹": float(rt["P&L"].mean()),
            "Expectancy %": float(rt["return %"].mean()),
            "Avg win %": avg_win,
            "Avg loss %": avg_loss,
            "Risk-reward": (abs(avg_win / avg_loss)
                            if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss
                            else np.nan),
            "Best trade ₹": float(rt["P&L"].max()),
            "Worst trade ₹": float(rt["P&L"].min()),
            "Avg days held": float(rt["days held"].mean()),
        })
    else:
        for k in ("Win rate %", "Profit factor", "Expectancy ₹", "Expectancy %",
                  "Avg win %", "Avg loss %", "Risk-reward", "Best trade ₹",
                  "Worst trade ₹", "Avg days held"):
            out[k] = np.nan
        out["Wins"] = out["Losses"] = 0

    # --- curve-level ------------------------------------------------------- #
    if len(eq) > 2:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        rets = M.to_returns(eq)
        out.update({
            "Book age (years)": years,
            "CAGR %": M.cagr(eq) * 100 if years > 0 else np.nan,
            "Max drawdown %": M.max_drawdown(eq) * 100,
            "Calmar": M.calmar(eq),
            "Sharpe": M.sharpe(rets),
            "Days tracked": len(eq),
        })
    else:
        for k in ("Book age (years)", "CAGR %", "Max drawdown %", "Calmar", "Sharpe"):
            out[k] = np.nan
        out["Days tracked"] = len(eq)
    return out


# --------------------------------------------------------------------------- #
# period tables
# --------------------------------------------------------------------------- #
def _period_table(eq: pd.Series, freq: str, label: str) -> pd.DataFrame:
    """Return and worst drawdown per calendar period, measured inside it.

    The drawdown column is the deepest fall *within* that period, not the
    portfolio's all-time drawdown sliced up — which is the number you want when
    asking "how bad did 2024 feel while I was living through it".
    """
    if len(eq) < 2:
        return pd.DataFrame()
    rows = []
    for period, chunk in eq.groupby(eq.index.to_period(freq)):
        if len(chunk) < 2:
            continue
        dd = float((chunk / chunk.cummax() - 1).min() * 100)
        rows.append({
            label: str(period),
            "Start": round(float(chunk.iloc[0]), 0),
            "End": round(float(chunk.iloc[-1]), 0),
            "Return %": round((chunk.iloc[-1] / chunk.iloc[0] - 1) * 100, 2),
            "P&L": round(float(chunk.iloc[-1] - chunk.iloc[0]), 0),
            "Drawdown %": round(dd, 2),
        })
    return pd.DataFrame(rows)


def yearly(eq: pd.Series) -> pd.DataFrame:
    return _period_table(eq, "Y", "Year")


def monthly(eq: pd.Series) -> pd.DataFrame:
    return _period_table(eq, "M", "Month")


def weekly(eq: pd.Series) -> pd.DataFrame:
    """Friday-week marks — the cadence this system actually trades on."""
    return _period_table(eq, "W-FRI", "Week")


# --------------------------------------------------------------------------- #
# where the money came from
# --------------------------------------------------------------------------- #
def exit_reasons(book: Book) -> pd.DataFrame:
    """P&L by the rung that caused each exit — the ladder's own scorecard.

    Two columns that are easy to conflate are kept apart on purpose. **Avg gain
    %** counts every exit once; **P&L** is in rupees and so is weighted by size.
    A rung can average a healthy gain and still lose money if its losing exits
    carry far more shares — which is exactly what a stop-loss rung looks like.
    """
    sells = [r for r in book.ledger if r.get("side") == "SELL"]
    if not sells:
        return pd.DataFrame()
    df = pd.DataFrame(sells)
    for c in ("qty", "pnl", "gain_pct"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    g = df.groupby("rung")
    out = pd.DataFrame({
        "Exits": g.size(),
        "Shares sold": g["qty"].sum(),
        "Avg gain %": g["gain_pct"].mean().round(2),
        "P&L": g["pnl"].sum().round(0),
        "Win rate %": (g["pnl"].apply(lambda x: (x > 0).mean() * 100)).round(1),
    })
    out.index = [RUNG_LABELS.get(i, i) for i in out.index]
    out.index.name = "Exit reason"
    return out.sort_values("P&L", ascending=False)


def partial_booking(rt: pd.DataFrame) -> pd.DataFrame:
    """Did booking a target actually help?

    Splits closed trades by whether a profit target ever fired. It is not a
    controlled experiment — a trade books a target *because* it went up — so
    read it as a description of the two populations, not as proof that booking
    caused the difference.
    """
    if rt is None or rt.empty:
        return pd.DataFrame()
    rows = []
    for booked, label in ((True, "Booked a profit target"), (False, "Never reached a target")):
        d = rt[rt["booked a target"] == booked]
        if d.empty:
            continue
        rows.append({
            "Group": label,
            "Trades": len(d),
            "Share of book %": round(len(d) / len(rt) * 100, 1),
            "Win rate %": round((d["P&L"] > 0).mean() * 100, 1),
            "Total P&L": round(float(d["P&L"].sum()), 0),
            "Avg P&L": round(float(d["P&L"].mean()), 0),
            "Avg return %": round(float(d["return %"].mean()), 2),
            "Avg days held": round(float(d["days held"].mean()), 0),
        })
    return pd.DataFrame(rows)


def top_movers(rt: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(best n, worst n) closed trades."""
    if rt is None or rt.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = ["symbol", "entry_date", "exit_date", "days held", "capital",
            "P&L", "return %", "last exit", "rungs hit"]
    cols = [c for c in cols if c in rt.columns]
    s = rt.sort_values("P&L", ascending=False)
    return s.head(n)[cols].reset_index(drop=True), \
        s.tail(n)[cols].iloc[::-1].reset_index(drop=True)


def characteristics(rt: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """What the best trades had in common that the worst did not.

    Deliberately only the few things the journal actually knows — holding
    period, position size, how far up the ladder the trade got, and how it
    ended. Anything else would be invented.
    """
    if rt is None or len(rt) < 4:
        return pd.DataFrame()
    s = rt.sort_values("P&L", ascending=False)
    k = min(n, len(rt) // 2)
    best, worst = s.head(k), s.tail(k)
    rows = [
        ("Trades compared", k, k),
        ("Avg days held", round(best["days held"].mean(), 0), round(worst["days held"].mean(), 0)),
        ("Median days held", round(best["days held"].median(), 0),
         round(worst["days held"].median(), 0)),
        ("Avg capital ₹", round(best["capital"].mean(), 0), round(worst["capital"].mean(), 0)),
        ("Avg return %", round(best["return %"].mean(), 2), round(worst["return %"].mean(), 2)),
        ("Booked a target %", round(best["booked a target"].mean() * 100, 1),
         round(worst["booked a target"].mean() * 100, 1)),
        ("Avg exits per trade", round(best["exits"].mean(), 2), round(worst["exits"].mean(), 2)),
        ("Most common ending", best["last exit"].mode().iloc[0] if len(best["last exit"].mode()) else "—",
         worst["last exit"].mode().iloc[0] if len(worst["last exit"].mode()) else "—"),
    ]
    return pd.DataFrame(rows, columns=["", f"Best {k}", f"Worst {k}"])


# --------------------------------------------------------------------------- #
# by index bucket
# --------------------------------------------------------------------------- #
def by_index(rt: pd.DataFrame, bucket_map: dict[str, str],
             order: list[str] | None = None) -> pd.DataFrame:
    """Return, win rate and drawdown per NSE size bucket.

    **What "Drawdown ₹" means here.** Portfolio drawdown cannot be split across
    buckets — cash is shared and a single equity curve does not decompose. So
    this is the deepest fall in *that bucket's own cumulative realised P&L*,
    trade by trade in exit order: how far the Smallcap 250 names dug their own
    hole before climbing out of it. It answers "which end of the market keeps
    hurting me" without pretending to be a portfolio number.
    """
    if rt is None or rt.empty or not bucket_map:
        return pd.DataFrame()
    d = rt.copy()
    d["bucket"] = d["symbol"].map(lambda s: bucket_map.get(s, "—"))
    rows = []
    for b, g in d.groupby("bucket"):
        g = g.sort_values("exit_date")
        # start the curve at zero, or a bucket whose very first trade lost money
        # would report a drawdown of nothing
        curve = pd.concat([pd.Series([0.0]), g["P&L"].cumsum()], ignore_index=True)
        dd = float((curve - curve.cummax()).min())
        cost = float(g["capital"].sum())
        rows.append({
            "Index bucket": b,
            "Trades": len(g),
            "Symbols": g["symbol"].nunique(),
            "Capital deployed": round(cost, 0),
            "Total P&L": round(float(g["P&L"].sum()), 0),
            "Return on capital %": round(float(g["P&L"].sum()) / cost * 100, 2) if cost else np.nan,
            "Win rate %": round((g["P&L"] > 0).mean() * 100, 1),
            "Avg return %": round(float(g["return %"].mean()), 2),
            "Avg days held": round(float(g["days held"].mean()), 0),
            "Drawdown ₹": round(dd, 0),
        })
    out = pd.DataFrame(rows)
    if order:
        rank = {b: i for i, b in enumerate(order)}
        out = out.sort_values("Index bucket", key=lambda s: s.map(
            lambda b: rank.get(b, len(rank))))
    return out.reset_index(drop=True)


def open_by_index(book: Book, bucket_map: dict[str, str],
                  last_price: pd.Series | None = None,
                  order: list[str] | None = None) -> pd.DataFrame:
    """Where the money is sitting right now, by size bucket."""
    rows: dict[str, dict] = {}
    total = 0.0
    for p in book.positions:
        if not p.is_open():
            continue
        b = bucket_map.get(p.symbol, "—")
        px = float(last_price.get(p.symbol, np.nan)) if last_price is not None else np.nan
        if not np.isfinite(px):
            px = p.entry_price
        val = p.open_qty * px
        total += val
        r = rows.setdefault(b, {"Index bucket": b, "Positions": 0, "Value": 0.0,
                                 "Unrealised": 0.0, "Symbols": []})
        r["Positions"] += 1
        r["Value"] += val
        r["Unrealised"] += (px - p.entry_price) * p.open_qty
        r["Symbols"].append(p.symbol)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame([
        {"Index bucket": r["Index bucket"], "Positions": r["Positions"],
         "Value": round(r["Value"], 0),
         "% of open value": round(r["Value"] / total * 100, 1) if total else np.nan,
         "Unrealised": round(r["Unrealised"], 0),
         "Stocks": ", ".join(sorted(r["Symbols"])[:12])}
        for r in rows.values()
    ])
    if order:
        rank = {b: i for i, b in enumerate(order)}
        out = out.sort_values("Index bucket", key=lambda s: s.map(
            lambda b: rank.get(b, len(rank))))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the open book, right now
# --------------------------------------------------------------------------- #
def open_dashboard(book: Book, last_price: pd.Series | None = None,
                   today: object = None, ema_fast: pd.Series | None = None) -> dict:
    """Everything the running-positions header needs, in one pass.

    Deliberately about **exposure and risk**, not performance — performance is
    the journal's job and needs closed trades. What this answers is the set of
    questions you actually have with the book open in front of you: how much is
    working, how much is spare, how many are up, what is the worst one, and what
    happens if every stop is hit tomorrow.

    Open Risk is leftover **capital**: max(0, buy − live stop) × qty. A stock
    that has run 40% with the 20 EMA still under the buy still risks the same
    rupees as at entry. Once the trail sits at the buy, Open Risk is zero.
    """
    now = pd.Timestamp(today or pd.Timestamp.today())
    rows = []
    for p in book.positions:
        if not p.is_open():
            continue
        px = float(last_price.get(p.symbol, np.nan)) if last_price is not None else np.nan
        if not np.isfinite(px):
            px = p.entry_price                       # unpriced: hold at cost, not at zero
        cost = p.open_qty * p.entry_price
        value = p.open_qty * px
        stop = float(getattr(p, "initial_stop", 0.0) or 0.0)
        hard = float(getattr(p, "hard_stop", 0.0) or 0.0)
        ef = float(ema_fast.get(p.symbol, np.nan)) if ema_fast is not None else np.nan
        live_stop = live_stop_price(p.entry_price, stop, hard, ef, px)
        linked = book.demerger_linked(p.symbol)
        # Open Risk = capital that can still be lost if the live stop hits.
        # Buy 100, stop 80 → ₹20/share. Price 120, stop still 80 → still ₹20
        # (running profit is not extra risk). Price 140, stop trailed to 100 → ₹0.
        cap_risk = (np.nan if live_stop <= 0 else
                    max(0.0, (p.entry_price - live_stop) * p.open_qty))
        rows.append({
            "symbol": p.symbol,
            "days held": int((now - pd.Timestamp(p.entry_date)).days),
            "qty": p.open_qty,
            "cost": cost,
            "value": value,
            "unrealised": value - cost,
            "unrealised %": (value / cost - 1) * 100 if cost else np.nan,
            "risk to stop": (0.0 if linked else cap_risk),
            "booked so far": float(sum(f.get("pnl", 0.0) for f in p.fills)),
            "demerger": linked,
        })
    det = pd.DataFrame(rows)

    cash = float(book.cash)
    deployed = float(det["value"].sum()) if len(det) else 0.0
    cost = float(det["cost"].sum()) if len(det) else 0.0
    port = cash + deployed
    unreal = deployed - cost
    wins = int((det["unrealised"] > 0).sum()) if len(det) else 0
    losses = int((det["unrealised"] < 0).sum()) if len(det) else 0
    risk = float(det["risk to stop"].sum(skipna=True)) if len(det) else 0.0

    best = worst = None
    pool = det[~det["demerger"]] if (len(det) and "demerger" in det.columns) else det
    if len(pool):
        b = pool.loc[pool["unrealised"].idxmax()]
        w = pool.loc[pool["unrealised"].idxmin()]
        best = {"symbol": b["symbol"], "pnl": float(b["unrealised"]),
                "pct": float(b["unrealised %"])}
        worst = {"symbol": w["symbol"], "pnl": float(w["unrealised"]),
                 "pct": float(w["unrealised %"])}

    return {
        "detail": det.sort_values("unrealised", ascending=False).reset_index(drop=True)
        if len(det) else det,
        "positions": len(det),
        "cash": cash,
        "deployed": deployed,
        "cost": cost,
        "portfolio value": port,
        "deployed %": deployed / port * 100 if port else np.nan,
        "cash %": cash / port * 100 if port else np.nan,
        "unrealised": unreal,
        "unrealised %": unreal / cost * 100 if cost else np.nan,
        "winners": wins,
        "losers": losses,
        "flat": len(det) - wins - losses,
        "win share %": wins / len(det) * 100 if len(det) else np.nan,
        "best": best,
        "worst": worst,
        "largest %": float(det["value"].max()) / port * 100 if len(det) and port else np.nan,
        "largest": det.loc[det["value"].idxmax(), "symbol"] if len(det) else "",
        "avg days held": float(det["days held"].mean()) if len(det) else np.nan,
        "risk to stops": risk,
        "risk %": risk / port * 100 if port else np.nan,
        "risk % of open": risk / deployed * 100 if deployed else np.nan,
        "risk % of cost": risk / cost * 100 if cost else np.nan,
        "realised so far": book.realised_pnl(),
        "booked in open trades": float(det["booked so far"].sum()) if len(det) else 0.0,
    }


def winners_losers(det: pd.DataFrame) -> pd.DataFrame:
    """The open book split in two — because an average hides which side is which."""
    if det is None or det.empty:
        return pd.DataFrame()
    rows = []
    for label, mask in (("Up", det["unrealised"] > 0), ("Down", det["unrealised"] < 0)):
        d = det[mask]
        if d.empty:
            continue
        rows.append({
            "Side": label,
            "Positions": len(d),
            "Capital": round(float(d["cost"].sum()), 0),
            "Value now": round(float(d["value"].sum()), 0),
            "Unrealised": round(float(d["unrealised"].sum()), 0),
            "Avg %": round(float(d["unrealised %"].mean()), 2),
            # one column, not a Best and a Worst that are NaN in each other's row
            "Furthest": str(d.iloc[0]["symbol"] if label == "Up" else d.iloc[-1]["symbol"]),
            "Avg days": round(float(d["days held"].mean()), 0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# demergers
# --------------------------------------------------------------------------- #
def demerger_pairs(book: Book, last_price: pd.Series | None = None) -> pd.DataFrame:
    """Parent and child read together, which is the only honest reading.

    A demerger books no profit and no loss — it splits one cost basis across two
    listed companies. Read alone afterwards, the parent looks like a sudden loss
    (its cost basis stayed while its price dropped by the value that left) and
    the child looks like a windfall (shares that appeared for no cash). Neither
    happened. Only the pair together answers whether the holding was worth it.
    """
    kids = [p for p in list(book.positions) + list(book.closed)
            if getattr(p, "demerged_from", "")]
    if not kids:
        return pd.DataFrame()

    def _mark(p):
        px = float(last_price.get(p.symbol, np.nan)) if last_price is not None else np.nan
        if not np.isfinite(px):
            px = p.entry_price
        booked = float(sum(f.get("pnl", 0.0) for f in p.fills))
        return p.open_qty * px, p.open_qty * p.entry_price, booked

    rows = []
    for kid in kids:
        parent = None
        for p in list(book.positions) + list(book.closed):
            if p.symbol == kid.demerged_from:
                parent = p
                break
        kv, kc, kb = _mark(kid)
        pv, pc, pb = _mark(parent) if parent is not None else (0.0, 0.0, 0.0)
        cost = kc + pc
        value = kv + pv
        rows.append({
            "Parent": kid.demerged_from,
            "Demerged": kid.symbol,
            "Cost still held": round(cost, 0),
            "Value now": round(value, 0),
            "Unrealised": round(value - cost, 0),
            "Booked so far": round(kb + pb, 0),
            "Combined P&L": round(value - cost + kb + pb, 0),
            "Combined %": (round((value - cost + kb + pb) / cost * 100, 2)
                           if cost else np.nan),
            "Parent alone %": round((pv / pc - 1) * 100, 2) if pc else np.nan,
            "Demerged alone %": round((kv / kc - 1) * 100, 2) if kc else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def slippage_report(book: Book) -> pd.DataFrame:
    """How much the gap between the buy list and the actual fill has cost.

    Only entries confirmed from a draft have a decision price, so a book filled
    before drafts existed reports nothing rather than zero.
    """
    buys = [r for r in book.ledger
            if r.get("side") == "BUY" and float(r.get("decision_price") or 0) > 0]
    if not buys:
        return pd.DataFrame()
    df = pd.DataFrame(buys)
    for c in ("qty", "price", "decision_price", "slippage_%", "slippage_rs"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Recompute from prices — old books stored the opposite sign.
    df["slippage %"] = ((df["decision_price"] - df["price"]) / df["decision_price"] * 100).round(2)
    df["slippage ₹"] = ((df["decision_price"] - df["price"]) * df["qty"]).round(0)
    return df[["date", "symbol", "qty", "decision_price", "price",
               "slippage %", "slippage ₹"]].rename(columns={
        "decision_price": "decision price", "price": "fill price"})


def monthly_trade_table(rt: pd.DataFrame) -> pd.DataFrame:
    if rt is None or rt.empty:
        return pd.DataFrame()
    d = rt.copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    d["Month"] = d["exit_date"].dt.strftime("%b %Y")
    d["ym"] = d["exit_date"].dt.to_period("M")
    rows = []
    for _, g in d.groupby("ym"):
        wins = g[g["P&L"] > 0]
        loss = g[g["P&L"] <= 0]
        pnl = float(g["P&L"].sum())
        cap = float(g["capital"].sum())
        rows.append({
            "Month": g["Month"].iloc[0],
            "Trades": len(g),
            "Win %": round((g["P&L"] > 0).mean() * 100, 1),
            "P&L ₹": round(pnl, 0),
            "P&L %": round(pnl / cap * 100, 2) if cap else np.nan,
            "Capital": round(cap, 0),
            "Avg gain": round(float(wins["return %"].mean()), 2) if len(wins) else np.nan,
            "Avg loss": round(float(loss["return %"].mean()), 2) if len(loss) else np.nan,
            "Best": round(float(g["return %"].max()), 2),
            "Worst": round(float(g["return %"].min()), 2),
        })
    return pd.DataFrame(rows)


def yearly_trade_table(rt: pd.DataFrame) -> pd.DataFrame:
    if rt is None or rt.empty:
        return pd.DataFrame()
    d = rt.copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    d["Year"] = d["exit_date"].dt.year
    rows = []
    for y, g in d.groupby("Year"):
        wins = g[g["P&L"] > 0]
        loss = g[g["P&L"] <= 0]
        pnl = float(g["P&L"].sum())
        cap = float(g["capital"].sum())
        rows.append({
            "Year": int(y),
            "Trades": len(g),
            "Win %": round((g["P&L"] > 0).mean() * 100, 1),
            "P&L ₹": round(pnl, 0),
            "P&L %": round(pnl / cap * 100, 2) if cap else np.nan,
            "Capital": round(cap, 0),
            "Avg gain": round(float(wins["return %"].mean()), 2) if len(wins) else np.nan,
            "Avg loss": round(float(loss["return %"].mean()), 2) if len(loss) else np.nan,
        })
    return pd.DataFrame(rows)


def streaks(rt: pd.DataFrame) -> dict:
    out = {"win streak": 0, "loss streak": 0, "max win streak": 0, "max loss streak": 0}
    if rt is None or rt.empty:
        return out
    d = rt.sort_values("exit_date")
    w = l = mw = ml = 0
    for pnl in d["P&L"]:
        if pnl > 0:
            w += 1
            l = 0
        else:
            l += 1
            w = 0
        mw, ml = max(mw, w), max(ml, l)
    cur_w = cur_l = 0
    for pnl in reversed(list(d["P&L"])):
        if pnl > 0:
            if cur_l:
                break
            cur_w += 1
        else:
            if cur_w:
                break
            cur_l += 1
    return {"win streak": cur_w, "loss streak": cur_l,
            "max win streak": mw, "max loss streak": ml}


def best_month(rt: pd.DataFrame) -> dict | None:
    mt = monthly_trade_table(rt)
    if mt.empty:
        return None
    i = mt["P&L ₹"].idxmax()
    r = mt.loc[i]
    pct = float(r["P&L %"]) if np.isfinite(r["P&L %"]) else np.nan
    return {"month": r["Month"], "pnl": float(r["P&L ₹"]), "pct": pct}
