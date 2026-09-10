"""
The weekly simulation.

How a week works, and why it works that way
-------------------------------------------
Signals are read off the **Friday close** and filled at the **following Monday's
open**. That one-bar lag is the difference between a backtest and a fantasy:
ranking on a close and filling at that same close hands you a price nobody could
have got, and on a breakout system — where the signal *is* a sharp move — that
free lunch is large.

Exits work the same way. A weekly close below the 20 EMA on Friday is acted on
at Monday's open, not at Friday's close.

Cash is real. Entries are skipped when the money isn't there, rather than
quietly running leverage: with five new names a week and no cap on how many you
hold at once, an unconstrained book would drift into 300% invested and print a
return you could never have financed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .breakout import BreakoutConfig, BreakoutSignals, RegimeConfig, qualifying_at, regime_blocked, score_week
from .exits import (
    Position, Rung, SizingConfig, check_hard_stop, check_position, check_trail_stop,
    entry_stop, hard_stop_price, size_position,
)


@dataclass
class CostConfig:
    brokerage_pct: float = 0.05       # per side, %
    slippage_pct: float = 0.15        # per side, %
    stt_pct: float = 0.10             # on the sell side, %

    def buy_price(self, px: float) -> float:
        return px * (1 + self.slippage_pct / 100.0)

    def sell_price(self, px: float) -> float:
        return px * (1 - self.slippage_pct / 100.0)

    def buy_cost(self, notional: float) -> float:
        return notional * self.brokerage_pct / 100.0

    def sell_cost(self, notional: float) -> float:
        return notional * (self.brokerage_pct + self.stt_pct) / 100.0


@dataclass
class RunConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    entries_per_week: int = 5
    allow_repeat_entry: bool = False   # re-enter a name you already hold?
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    ladder: list[Rung] = field(default_factory=list)
    rearm_ema: bool = False       # can the EMA rungs fire again after price recovers above them?
    targets_on_daily_close: bool = True
    """Check the profit targets on every DAILY close instead of only Friday's.

    A stock can trade through +25% on a Wednesday and close the week below it.
    Checked weekly, that target never books — you saw the profit and did not take
    it. Checked daily, it books on Wednesday's close and fills at Thursday's
    open, which is what you would actually have done.

    This is a daily CLOSE, not an intraday touch: a real observable at a real
    time with a real fill the next morning, so it adds no look-ahead. The
    stop-loss rungs (20 EMA / 50 EMA) stay on the weekly close either way, which
    is asymmetric on purpose — the scan is weekly and the stop is a weekly-chart
    rule. The moved-up trail floor is always checked daily, because a floor you
    have deliberately placed under a position is not something to look at once a
    week.
    """


@dataclass
class RunResult:
    equity: pd.Series                  # daily mark-to-market portfolio value
    cash: pd.Series
    deployed: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    """Daily capital actually deployed, at cost — what you paid for the shares you
    are holding, not their current value. This is the number that answers 'how
    much of my money was actually in the market that month'."""
    n_open: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    blocked_weeks: list[pd.Timestamp] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _week_first_trading_day(daily_index: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    """W-FRI label -> the first actual trading day inside that week."""
    s = pd.Series(daily_index, index=daily_index)
    grouped = s.resample("W-FRI").first().dropna()
    return {pd.Timestamp(k): pd.Timestamp(v) for k, v in grouped.items()}


def run_backtest(
    panel: dict[str, pd.DataFrame],
    sig: BreakoutSignals,
    cfg: RunConfig,
    screen_mask: pd.DataFrame | None = None,
    bench_daily: pd.Series | None = None,
    vol_surge_weekly: pd.DataFrame | None = None,
    daily_ok: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    progress_cb=None,
) -> RunResult:
    daily_close = panel.get("Close", pd.DataFrame())
    wk = sig.weekly
    w_close, w_open = wk.get("Close", pd.DataFrame()), wk.get("Open", pd.DataFrame())
    if daily_close.empty or w_close.empty:
        empty = pd.Series(dtype=float)
        return RunResult(equity=empty, cash=empty, notes=["No price data."])

    weeks = [w for w in w_close.index if cfg.start <= w <= cfg.end]
    if len(weeks) < 2:
        empty = pd.Series(dtype=float)
        return RunResult(equity=empty, cash=empty,
                         notes=["Date range too short — need at least two weekly bars."])

    first_day = _week_first_trading_day(daily_close.index)
    ladder = cfg.ladder or []

    cash = float(cfg.sizing.capital)
    open_positions: list[Position] = []
    closed: list[Position] = []
    trades: list[dict] = []
    cash_deltas: list[tuple[pd.Timestamp, float]] = [(pd.Timestamp(cfg.start), cash)]
    qty_deltas: list[tuple[pd.Timestamp, str, int]] = []
    cost_deltas: list[tuple[pd.Timestamp, float]] = []      # capital deployed, at cost
    open_deltas: list[tuple[pd.Timestamp, int]] = []        # count of open positions
    weekly_rows: list[dict] = []
    blocked_weeks: list[pd.Timestamp] = []

    all_weeks = list(w_close.index)
    week_pos = {w: i for i, w in enumerate(all_weeks)}

    # ---- daily scaffolding for the intra-week exit pass ----
    d_close = daily_close
    d_open = panel.get("Open", pd.DataFrame())
    if d_open.empty:
        d_open = d_close
    trail_daily = {k: v for k, v in sig.trail_emas.items() if not v.empty}
    _di = list(d_close.index)
    next_day = {d: _di[i + 1] for i, d in enumerate(_di[:-1])}
    week_days: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for lbl, vals in pd.Series(_di, index=_di).groupby(pd.Grouper(freq="W-FRI")):
        week_days[pd.Timestamp(lbl)] = list(vals.index)

    for n, week in enumerate(weeks):
        i = week_pos[week]
        if i + 1 >= len(all_weeks):
            break                                   # nothing left to fill against
        nxt = all_weeks[i + 1]
        exec_date = first_day.get(nxt)
        if exec_date is None or exec_date > cfg.end:
            break

        # ---------------- exits first: they release the cash entries need -----
        # Two passes, on purpose:
        #   1. every DAILY close inside this week — the profit targets (if that
        #      mode is on) and the moved-up trail floor, filling at the NEXT
        #      day's open;
        #   2. this week's Friday close — the 20 / 50 EMA rungs, filling at the
        #      following Monday's open.
        # Days are walked in order so that a target booking on Wednesday is
        # already recorded when Friday's EMA break is judged, which is what
        # decides whether that break is a trim or a full stop-loss.
        n_exits = 0

        def _book(pos, ev, fill_date, ref_price, when, signal_on):
            """Record one exit fill. Returns True if the position is now flat.

            `signal_on` is the bar that actually fired the rung — a daily close
            inside the week for the daily pass, the week's Friday for the weekly
            one. Tagging a Wednesday exit with that week's Friday label would put
            the fill BEFORE its own signal, which is nonsense in any audit of the
            trade log.
            """
            nonlocal cash, n_exits
            raw = d_open.at[when, pos.symbol] if (
                when is not None and when in d_open.index
                and pos.symbol in d_open.columns) else np.nan
            if pd.isna(raw):
                raw = ref_price
            px = cfg.costs.sell_price(float(raw))
            gross = ev["qty"] * px
            fee = cfg.costs.sell_cost(gross)
            cash += gross - fee
            cash_deltas.append((fill_date, gross - fee))
            qty_deltas.append((fill_date, pos.symbol, -ev["qty"]))
            cost_deltas.append((fill_date, -ev["qty"] * pos.cost_basis()))
            basis = pos.cost_basis()
            rec = {
                "date": fill_date, "symbol": pos.symbol, "side": "SELL",
                "qty": ev["qty"], "price": px, "reason": ev["reason"], "rung": ev["rung"],
                "entry_date": pd.Timestamp(pos.entry_date), "entry_price": basis,
                "gain_pct": (px / basis - 1) * 100.0,
                "pnl": (px - basis) * ev["qty"] - fee, "fees": fee,
                "signal_week": signal_on,
                "held_weeks": (pd.Timestamp(fill_date) - pd.Timestamp(pos.entry_date)).days / 7.0,
            }
            trades.append(rec)
            pos.fills.append(rec)
            n_exits += 1
            return not pos.is_open()

        def _close_out(pos, fill_date):
            open_positions.remove(pos)
            closed.append(pos)
            open_deltas.append((fill_date, -1))

        # ---- pass 1: the daily bars of this week ----
        for d in week_days.get(week, []):
            if not open_positions:
                break
            nd = next_day.get(d)
            fill_on = nd if nd is not None else d
            for pos in list(open_positions):
                if pos.symbol not in d_close.columns:
                    continue
                if pd.Timestamp(pos.entry_date) > d:
                    continue                    # not bought yet on this day
                c = d_close.at[d, pos.symbol]
                if pd.isna(c):
                    continue

                # the fixed % stop first — it is the hard floor under the trade,
                # and it runs alongside the weekly 20 EMA rung: whichever level
                # is breached first ends the position
                hs = check_hard_stop(pos, float(c))
                if hs is not None:
                    pos.open_qty = 0
                    _book(pos, hs, fill_on, float(c), nd, d)
                    _close_out(pos, fill_on)
                    continue

                # then the trail floor: a floor set by today's booking is live
                # from tomorrow, not against the close that set it
                lv = {k: v.at[d, pos.symbol] for k, v in trail_daily.items()
                      if pos.symbol in v.columns and d in v.index}
                ts = check_trail_stop(pos, float(c), lv)
                if ts is not None:
                    pos.open_qty = 0
                    _book(pos, ts, fill_on, float(c), nd, d)
                    _close_out(pos, fill_on)
                    continue

                if not cfg.targets_on_daily_close:
                    continue
                for f in check_position(pos, d, float(c), np.nan, np.nan, ladder,
                                         rearm_ema=cfg.rearm_ema,
                                         only_triggers={"gain_pct"}):
                    flat = _book(pos, f, fill_on, float(c), nd, d)
                    if flat:
                        _close_out(pos, fill_on)
                        break

        # ---- pass 2: this week's close ----
        for pos in list(open_positions):
            if week not in w_close.index or pos.symbol not in w_close.columns:
                continue
            close = w_close.at[week, pos.symbol]
            if pd.isna(close):
                continue
            ef = float(sig.ema_fast.at[week, pos.symbol]) if pos.symbol in sig.ema_fast.columns else np.nan
            es = float(sig.ema_slow.at[week, pos.symbol]) if pos.symbol in sig.ema_slow.columns else np.nan

            triggers = {"below_ema_fast", "below_ema_slow"}
            if not cfg.targets_on_daily_close:
                triggers = triggers | {"gain_pct"}

            for f in check_position(pos, week, float(close), ef, es, ladder,
                                     rearm_ema=cfg.rearm_ema, only_triggers=triggers):
                raw = w_open.at[nxt, pos.symbol] if pos.symbol in w_open.columns else np.nan
                if pd.isna(raw):
                    raw = close
                px = cfg.costs.sell_price(float(raw))
                gross = f["qty"] * px
                fee = cfg.costs.sell_cost(gross)
                cash += gross - fee
                cash_deltas.append((exec_date, gross - fee))
                qty_deltas.append((exec_date, pos.symbol, -f["qty"]))
                cost_deltas.append((exec_date, -f["qty"] * pos.cost_basis()))
                basis = pos.cost_basis()
                rec = {
                    "date": exec_date, "symbol": pos.symbol, "side": "SELL",
                    "qty": f["qty"], "price": px, "reason": f["reason"], "rung": f["rung"],
                    "entry_date": pd.Timestamp(pos.entry_date), "entry_price": basis,
                    "gain_pct": (px / basis - 1) * 100.0,
                    "pnl": (px - basis) * f["qty"] - fee, "fees": fee, "signal_week": week,
                    "held_weeks": (pd.Timestamp(exec_date) - pd.Timestamp(pos.entry_date)).days / 7.0,
                }
                trades.append(rec)
                pos.fills.append(rec)
                n_exits += 1

            if not pos.is_open():
                _close_out(pos, exec_date)

        # ---------------- regime filter ---------------------------------------
        blocked, _ = regime_blocked(bench_daily, week, cfg.regime) if bench_daily is not None else (False, {})
        if blocked:
            blocked_weeks.append(week)

        # ---------------- entries ---------------------------------------------
        n_entries = 0
        picks_note = ""
        n_screened = n_breakouts = 0
        skipped_cash = 0
        cands: list[str] = []
        if True:
            held = {p.symbol for p in open_positions}
            screen_ok = None
            if screen_mask is not None and not screen_mask.empty:
                idx = screen_mask.index[screen_mask.index <= week]
                if len(idx):
                    row = screen_mask.loc[idx[-1]]
                    screen_ok = list(row.index[row.astype(bool)])
            n_screened = len(screen_ok) if screen_ok is not None else len(w_close.columns)
            raw_breaks = qualifying_at(sig, week, None, exclude=None)
            n_breakouts = len(raw_breaks)
            cands = qualifying_at(sig, week, screen_ok,
                                   exclude=None if cfg.allow_repeat_entry else held)
            if daily_ok is not None and not daily_ok.empty and week in daily_ok.index:
                drow = daily_ok.loc[week]
                cands = [c for c in cands if c in drow.index and bool(drow[c])]
            if cands and not blocked:
                # `fundamentals` is a single snapshot, so switching it on inside a
                # backtest ranks a 2021 breakout with numbers published later —
                # look-ahead. The app defaults it off here and warns; see README.
                scored = score_week(sig, week, cands, cfg.breakout, vol_surge_weekly,
                                    fundamentals=fundamentals)
                picks = scored.head(int(cfg.entries_per_week))
                picks_note = ", ".join(picks.index[:5])

                equity_now = cash + sum(
                    p.open_qty * float(w_close.at[week, p.symbol])
                    for p in open_positions
                    if p.symbol in w_close.columns and pd.notna(w_close.at[week, p.symbol])
                )

                for sym, r in picks.iterrows():
                    raw = w_open.at[nxt, sym] if sym in w_open.columns else np.nan
                    if pd.isna(raw):
                        continue
                    px = cfg.costs.buy_price(float(raw))
                    ema_stop = float(r["ema_fast"]) if pd.notna(r["ema_fast"]) else px * 0.92
                    # size off whichever stop is TIGHTER, otherwise a 30%-away
                    # EMA justifies a position the 20% cap makes a lie of
                    stop = entry_stop(ema_stop, px, cfg.sizing.max_stop_distance_pct)
                    sized = size_position(px, stop, cfg.sizing, equity_now)
                    if sized.qty <= 0:
                        continue
                    fee = cfg.costs.buy_cost(sized.cost)
                    if sized.cost + fee > cash:
                        skipped_cash += 1
                        continue                     # no money: skip, don't borrow
                    cash -= sized.cost + fee
                    cash_deltas.append((exec_date, -(sized.cost + fee)))
                    qty_deltas.append((exec_date, sym, sized.qty))
                    cost_deltas.append((exec_date, sized.cost))
                    open_deltas.append((exec_date, 1))

                    pos = Position(
                        symbol=sym, entry_date=exec_date, entry_price=px, qty=sized.qty,
                        open_qty=sized.qty, initial_stop=stop, entry_score=float(r["score"]),
                        hard_stop=hard_stop_price(px, cfg.sizing.max_stop_distance_pct),
                    )
                    open_positions.append(pos)
                    trades.append({
                        "date": exec_date, "symbol": sym, "side": "BUY", "qty": sized.qty,
                        "price": px, "reason": f"fresh {cfg.breakout.lookback_weeks}w high",
                        "rung": "entry", "entry_date": exec_date, "entry_price": px,
                        "gain_pct": 0.0, "pnl": -fee, "fees": fee, "signal_week": week,
                        "score": float(r["score"]), "stop": stop, "held_weeks": 0.0,
                    })
                    n_entries += 1

        weekly_rows.append({
            "signal_week": week, "executed": exec_date, "entries": n_entries,
            "exits": n_exits, "open_positions": len(open_positions),
            "cash": cash, "regime_blocked": blocked, "picks": picks_note,
            "passed_screen": n_screened, "breakouts": n_breakouts,
            "candidates": len(cands),
            "skipped_no_cash": skipped_cash,
        })
        if progress_cb and (n % 10 == 0 or n == len(weeks) - 1):
            progress_cb(n + 1, len(weeks))

    # ------------------------------------------------------------------ curve --
    cal = daily_close.index[(daily_close.index >= cfg.start) & (daily_close.index <= cfg.end)]
    cash_s = pd.Series(0.0, index=cal)
    for d, amt in cash_deltas:
        d = pd.Timestamp(d)
        pos_idx = cal[cal >= d]
        if len(pos_idx):
            cash_s.loc[pos_idx[0]] += amt
    cash_s = cash_s.cumsum()

    qty = pd.DataFrame(0.0, index=cal, columns=sorted({s for _, s, _ in qty_deltas}))
    for d, s, q in qty_deltas:
        d = pd.Timestamp(d)
        pos_idx = cal[cal >= d]
        if len(pos_idx) and s in qty.columns:
            qty.loc[pos_idx[0], s] += q
    qty = qty.cumsum()

    if len(qty.columns):
        px = daily_close.reindex(index=cal, columns=qty.columns).ffill()
        holdings_value = (qty * px).sum(axis=1)
    else:
        holdings_value = pd.Series(0.0, index=cal)

    equity = (cash_s + holdings_value).rename("equity")

    def _series(deltas, name):
        out = pd.Series(0.0, index=cal)
        for d, amt in deltas:
            idx = cal[cal >= pd.Timestamp(d)]
            if len(idx):
                out.loc[idx[0]] += amt
        return out.cumsum().rename(name)

    deployed = _series(cost_deltas, "deployed")
    n_open = _series(open_deltas, "n_open")

    return RunResult(
        equity=equity, cash=cash_s, deployed=deployed, n_open=n_open,
        trades=pd.DataFrame(trades),
        weekly_log=pd.DataFrame(weekly_rows), positions=open_positions,
        closed=closed, blocked_weeks=blocked_weeks,
    )


# --------------------------------------------------------------------------- #
# summary of the exit ladder's behaviour — the number a breakout trader wants
# --------------------------------------------------------------------------- #
def ladder_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """How much of the P&L came from each rung of the ladder."""
    if trades.empty or "rung" not in trades.columns:
        return pd.DataFrame()
    sells = trades[trades["side"] == "SELL"]
    if sells.empty:
        return pd.DataFrame()
    g = sells.groupby("rung")
    out = pd.DataFrame({
        "Exits": g.size(),
        "Shares": g["qty"].sum(),
        "Avg gain %": g["gain_pct"].mean().round(2),
        "Total P&L": g["pnl"].sum().round(0),
        "Win rate %": g["pnl"].apply(lambda s: (s > 0).mean() * 100).round(1),
    })
    nice = {
        "profit_1": "1. Profit target 1", "profit_2": "2. Profit target 2",
        "trail_ema_fast": "3. Below 20 EMA", "trail_ema_slow": "4. Below 50 EMA",
    }
    out.index = [nice.get(i, i) for i in out.index]
    return out.sort_index()


def monthly_breakdown(res: RunResult) -> pd.DataFrame:
    """Month by month: what it returned, and how much money was actually in play.

    'Max capital used' is the peak cost of open positions during the month — the
    real answer to "how much of my ₹50L was deployed". Its % is against that
    month's closing equity, so 100% means fully invested.
    """
    eq = res.equity
    if eq is None or eq.empty:
        return pd.DataFrame()

    dep = res.deployed if res.deployed is not None and len(res.deployed) else pd.Series(0.0, index=eq.index)
    nop = res.n_open if res.n_open is not None and len(res.n_open) else pd.Series(0.0, index=eq.index)

    g_eq = eq.resample("ME")
    end_eq = g_eq.last()
    start_eq = g_eq.first()
    ret = (end_eq / start_eq - 1) * 100

    max_dep = dep.resample("ME").max()
    avg_dep = dep.resample("ME").mean()
    max_open = nop.resample("ME").max()
    avg_open = nop.resample("ME").mean()

    t = res.trades
    if t is not None and not t.empty:
        td = t.copy()
        td["date"] = pd.to_datetime(td["date"])
        entries = td[td["side"] == "BUY"].set_index("date").resample("ME").size()
        exits = td[td["side"] == "SELL"].set_index("date").resample("ME").size()
        pnl = td[td["side"] == "SELL"].set_index("date")["pnl"].resample("ME").sum()
    else:
        entries = exits = pnl = pd.Series(dtype=float)

    out = pd.DataFrame({
        "Month": end_eq.index.strftime("%Y-%m"),
        "Return %": ret.round(2).values,
        "Equity (end)": end_eq.round(0).values,
        "Max capital used": max_dep.reindex(end_eq.index).round(0).values,
        "Max used %": (max_dep.reindex(end_eq.index) / end_eq * 100).round(1).values,
        "Avg capital used": avg_dep.reindex(end_eq.index).round(0).values,
        "Max open positions": max_open.reindex(end_eq.index).fillna(0).astype(int).values,
        "Avg open": avg_open.reindex(end_eq.index).round(1).values,
        "Entries": entries.reindex(end_eq.index).fillna(0).astype(int).values,
        "Exit fills": exits.reindex(end_eq.index).fillna(0).astype(int).values,
        "Booked P&L": pnl.reindex(end_eq.index).fillna(0).round(0).values,
    })
    return out.iloc[::-1].reset_index(drop=True)      # newest month first


def yearly_breakdown(res: RunResult) -> pd.DataFrame:
    """Year by year, with the same capital-usage columns."""
    eq = res.equity
    if eq is None or eq.empty:
        return pd.DataFrame()

    dep = res.deployed if res.deployed is not None and len(res.deployed) else pd.Series(0.0, index=eq.index)
    nop = res.n_open if res.n_open is not None and len(res.n_open) else pd.Series(0.0, index=eq.index)

    rows = []
    for year in sorted(set(eq.index.year)):
        eq_y = eq[eq.index.year == year]
        if eq_y.empty:
            continue
        prior = eq[eq.index < eq_y.index[0]]
        start = float(prior.iloc[-1]) if len(prior) else float(eq_y.iloc[0])
        end = float(eq_y.iloc[-1])
        dep_y = dep[dep.index.year == year]
        nop_y = nop[nop.index.year == year]
        run_max = eq_y.cummax()
        dd = ((eq_y - run_max) / run_max).min() * 100
        rows.append({
            "Year": year,
            "Return (%)": round((end / start - 1) * 100, 2) if start else np.nan,
            "Start Capital": round(start, 0),
            "End Capital": round(end, 0),
            "Net Profit": round(end - start, 0),
            "Max drawdown %": round(float(dd), 2),
            "Max capital used": round(float(dep_y.max()), 0) if len(dep_y) else 0.0,
            "Max used %": round(float(dep_y.max()) / end * 100, 1) if len(dep_y) and end else 0.0,
            "Avg capital used": round(float(dep_y.mean()), 0) if len(dep_y) else 0.0,
            "Max open positions": int(nop_y.max()) if len(nop_y) else 0,
        })
    return pd.DataFrame(rows)


def round_trip_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per completed position: what you put in, what came back."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (sym, ent), grp in trades.groupby(["symbol", "entry_date"]):
        buys = grp[grp["side"] == "BUY"]
        sells = grp[grp["side"] == "SELL"]
        if buys.empty:
            continue
        bought = int(buys["qty"].sum())
        sold = int(sells["qty"].sum()) if not sells.empty else 0
        invested = float((buys["qty"] * buys["price"]).sum())
        realised = float(sells["pnl"].sum()) if not sells.empty else 0.0
        rows.append({
            "symbol": sym,
            "entry_date": pd.Timestamp(ent).date(),
            "exit_date": pd.Timestamp(sells["date"].max()).date() if not sells.empty else None,
            "qty": bought,
            "entry_price": round(float(buys["price"].mean()), 2),
            "invested": round(invested, 0),
            "sold_qty": sold,
            "still_open": bought - sold,
            "realised_pnl": round(realised, 0),
            "return_%": round(realised / invested * 100, 2) if invested else np.nan,
            "rungs_hit": ", ".join(sorted(set(sells["rung"]))) if not sells.empty else "",
        })
    return pd.DataFrame(rows).sort_values("entry_date")
