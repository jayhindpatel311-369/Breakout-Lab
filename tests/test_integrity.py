"""Dates, sessions and the journal's equity curve — the "Today P&L" class of bugs."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import corpact as ca
from core import data as data_mod
from core import journal as jn
from core import journal_stats as js

IDX = pd.bdate_range("2026-09-21", "2026-10-05")       # Mon 21 Sep .. Mon 5 Oct


def _book():
    b = jn.Book(name="t", capital=1_000_000, cash=1_000_000)
    b.buy("OPEN1", IDX[0].date(), 100, 100.0, 90)
    b.buy("BALAMINES", IDX[0].date(), 500, 200.0, 180)
    return b


def _prices(**cols):
    return pd.DataFrame({k: v for k, v in cols.items()}, index=IDX)


FLAT_OPEN = [100.0] * len(IDX)
BAL = [200.0] * (len(IDX) - 2) + [210.0] * 2


# --------------------------------------------------------------------------- #
# equity curve
# --------------------------------------------------------------------------- #
def test_weekend_sell_lands_on_friday():
    b = _book()
    b.sell("BALAMINES", pd.Timestamp("2026-10-03").date(), 500, 210.0, "manual")  # Saturday
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN, BALAMINES=BAL))
    assert js.window_pnl(eq, 1)["pnl"] == pytest.approx(0.0)
    assert eq.loc["2026-10-02"] == pytest.approx(b.cash + 100 * 100)
    assert eq.attrs["reconcile_gap"] == pytest.approx(0.0)


def test_closed_symbol_missing_from_prices_is_not_marked_at_zero():
    """The reported bug: proceeds of a sold stock showing as Today P&L every day."""
    b = _book()
    b.sell("BALAMINES", pd.Timestamp("2026-10-03").date(), 500, 210.0, "manual")
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN))        # BALAMINES not priced
    assert js.window_pnl(eq, 1)["pnl"] == pytest.approx(0.0)
    assert "BALAMINES" in eq.attrs["unpriced"]
    # held at its last fill price rather than ₹0 → no 1 lakh hole before the exit
    assert eq.min() > 1_000_000 - 1


def test_fill_after_last_bar_lands_on_last_bar():
    b = _book()
    b.sell("BALAMINES", pd.Timestamp("2026-10-07").date(), 500, 210.0, "manual")
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN, BALAMINES=BAL))
    assert eq.iloc[-1] == pytest.approx(b.cash + 100 * 100)
    assert eq.attrs["reconcile_gap"] == pytest.approx(0.0)


def test_weekend_deposit_and_dividend_are_counted():
    b = _book()
    b.add_capital(50_000)
    b.cash_flows[-1]["date"] = "2026-09-26"                 # a Saturday
    ca.apply(b, ca.Action(symbol="OPEN1", kind="dividend", ex_date="2026-09-27", amount=5.0))
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN, BALAMINES=[200.0] * len(IDX)))
    assert eq.attrs["reconcile_gap"] == pytest.approx(0.0)
    # deposit and dividend both on Friday 25 Sep, not lost
    assert eq.loc["2026-09-25"] - eq.loc["2026-09-24"] == pytest.approx(50_000 + 500)
    assert js.window_pnl(eq, 1)["pnl"] == pytest.approx(0.0)


def test_split_restates_history_without_a_jump():
    b = _book()
    # 1:5 split of OPEN1 effective 28 Sep; Yahoo back-adjusts the whole series
    ca.apply(b, ca.Action(symbol="OPEN1", kind="split", ex_date="2026-09-28", a=1, b=5))
    adj = [20.0] * len(IDX)
    eq = js.equity_curve(b, _prices(OPEN1=adj, BALAMINES=[200.0] * len(IDX)))
    assert eq.attrs["reconcile_gap"] == pytest.approx(0.0)
    assert eq.max() - eq.min() == pytest.approx(0.0)


def test_split_does_not_rewrite_the_ledger():
    b = _book()
    b.sell("OPEN1", IDX[3].date(), 40, 110.0, "profit_1")
    ca.apply(b, ca.Action(symbol="OPEN1", kind="split", ex_date="2026-09-28", a=1, b=5))
    sell = [r for r in b.ledger if r["side"] == "SELL"][0]
    assert sell["qty"] == 40 and sell["price"] == 110.0      # ledger as filled
    assert b.find("OPEN1").fills[0]["qty"] == 200            # position restated


def test_rights_issue_counted():
    b = _book()
    ca.apply(b, ca.Action(symbol="OPEN1", kind="rights", ex_date="2026-09-29",
                          a=1, b=2, amount=80.0, subscribed=True))
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN, BALAMINES=[200.0] * len(IDX)))
    assert eq.attrs["reconcile_gap"] == pytest.approx(0.0)


def test_unexplained_gap_is_reported_not_hidden():
    b = _book()
    b.cash += 12_345                                     # a change with no record
    eq = js.equity_curve(b, _prices(OPEN1=FLAT_OPEN, BALAMINES=[200.0] * len(IDX)))
    assert eq.attrs["reconcile_gap"] == pytest.approx(12_345)
    assert js.window_pnl(eq, 1)["pnl"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# sessions and dates
# --------------------------------------------------------------------------- #
def test_session_settled():
    fri = pd.Timestamp("2026-10-02")
    assert not data_mod.session_settled(fri, datetime(2026, 10, 2, 14, 30))
    assert data_mod.session_settled(fri, datetime(2026, 10, 2, 16, 5))
    assert data_mod.session_settled(fri, datetime(2026, 10, 3, 9, 0))


def test_last_session_date():
    sat = datetime(2026, 10, 3, 11, 0)
    assert data_mod.last_session_date(None, None, sat) == pd.Timestamp("2026-10-02")
    mon = datetime(2026, 10, 5, 10, 0)
    # market open today → today
    assert data_mod.last_session_date(IDX[:-1], pd.Timestamp("2026-10-05"), mon) == \
        pd.Timestamp("2026-10-05")
    # weekday holiday: live bars belong to Friday
    assert data_mod.last_session_date(IDX[:-1], pd.Timestamp("2026-10-02"), mon) == \
        pd.Timestamp("2026-10-02")
    # no live, panel ends Friday
    assert data_mod.last_session_date(IDX[:-1], None, mon) == pd.Timestamp("2026-10-02")


def test_expected_last_bar():
    exp = data_mod.PriceStore.expected_last_bar
    end = pd.Timestamp("2026-10-07")                       # Wednesday
    assert exp(end, datetime(2026, 10, 7, 11, 0)) == pd.Timestamp("2026-10-06")
    assert exp(end, datetime(2026, 10, 7, 17, 0)) == pd.Timestamp("2026-10-07")
    assert exp(pd.Timestamp("2026-10-05"), datetime(2026, 10, 5, 9, 0)) == \
        pd.Timestamp("2026-10-02")                         # Monday morning → Friday


def test_needs_refresh_catches_one_day_old_cache(tmp_path):
    store = data_mod.PriceStore(cache_dir=str(tmp_path))
    cached = pd.DataFrame({"Close": [1.0, 1.0]},
                          index=pd.to_datetime(["2026-10-02", "2026-10-05"]))
    # Wednesday evening: Tuesday and Wednesday are missing — old rule said "fresh"
    import core.data as d
    real = d.now_ist
    d.now_ist = lambda: datetime(2026, 10, 7, 18, 0)
    try:
        assert store._needs_refresh(cached, pd.Timestamp("2026-10-07"))
    finally:
        d.now_ist = real


def test_drop_unsettled_bar():
    df = pd.DataFrame({"Close": [1.0, 2.0]},
                      index=pd.to_datetime(["2026-10-02", "2026-10-05"]))
    assert len(data_mod.drop_unsettled(df, datetime(2026, 10, 5, 11, 0))) == 1
    assert len(data_mod.drop_unsettled(df, datetime(2026, 10, 5, 16, 30))) == 2


def test_apply_live_mark_on_weekday_holiday_adds_no_day(monkeypatch):
    close = pd.DataFrame({"A": [100.0, 101.0]}, index=pd.to_datetime(["2026-10-01", "2026-10-02"]))
    monkeypatch.setattr(data_mod, "now_ist", lambda: datetime(2026, 10, 5, 11, 0))
    out_open = data_mod.apply_live_mark(close, {"A": 105.0}, pd.Timestamp("2026-10-05"))
    out_holiday = data_mod.apply_live_mark(close, {"A": 101.5}, pd.Timestamp("2026-10-02"))
    assert len(out_open) == 3 and out_open["A"].iloc[-1] == 105.0
    assert len(out_holiday) == 2 and out_holiday["A"].iloc[-1] == 101.5


def test_align_panel_keeps_young_holding():
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.DataFrame({"OLD": 100.0, "IPO": [float("nan")] * 260 + [50.0] * 40}, index=idx)
    panel = {"Close": close}
    _, dropped = data_mod.align_panel(panel, idx[0], idx[-1], min_history_days=250,
                                      tail_days=250)
    assert "IPO" in dropped
    out, dropped = data_mod.align_panel(panel, idx[0], idx[-1], min_history_days=250,
                                        tail_days=250, keep_always={"IPO"})
    assert "IPO" not in dropped and "IPO" in out["Close"].columns


def test_below_fast_ema_watch():
    b = jn.Book(name="t", capital=1_000_000, cash=1_000_000)
    for sym in ("A", "B", "C", "D"):
        b.buy(sym, IDX[0].date(), 10, 100.0, 90)
    cmp = pd.Series({"A": 95.0, "B": 105.0, "C": 90.0, "D": 80.0})
    ema = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0, "D": 100.0})
    w = jn.below_fast_ema(b, cmp, ema, skip={"C"})
    assert list(w["symbol"]) == ["D", "A"]          # most extended first, C skipped
    assert w.loc[w["symbol"] == "A", "gap_%"].iloc[0] == pytest.approx(-5.0)
    # an EMA on a pre-split scale (> 1.8x CMP) is ignored
    w2 = jn.below_fast_ema(b, pd.Series({"A": 50.0}), pd.Series({"A": 100.0}))
    assert w2.empty


def test_lagging_symbols_names_the_stale_ones(monkeypatch):
    monkeypatch.setattr(data_mod, "now_ist", lambda: datetime(2026, 10, 9, 13, 0))
    close = pd.DataFrame({"A": [1.0, np.nan], "B": [1.0, 1.0]},
                         index=pd.to_datetime(["2026-10-07", "2026-10-08"]))
    lag = data_mod.lagging_symbols(close, pd.Timestamp("2026-10-09"))
    assert lag == {"A": pd.Timestamp("2026-10-07")}


def test_store_tops_up_a_cache_a_session_short(tmp_path, monkeypatch):
    """A refresh that failed (rate limit) left the cache at Wednesday and the
    three-hour retry throttle kept it there; the top-up fetches Thursday."""
    import time as _time
    monkeypatch.setattr(data_mod, "now_ist", lambda: datetime(2026, 10, 9, 13, 0))
    store = data_mod.PriceStore(cache_dir=str(tmp_path))
    idx = pd.bdate_range("2024-01-01", "2026-10-07")
    old = pd.DataFrame({"Close": 100.0, "AdjFactor": 1.0}, index=idx)
    store._write_cache("VENUSPIPES.NS", old)
    # main refresh was tried a minute ago, so the throttle serves the stale cache
    store._write_attempts({"VENUSPIPES.NS": _time.time() - 60})
    calls = []

    def fake(tickers, start, end, threads=True):
        calls.append((tuple(tickers), pd.Timestamp(start)))
        return {"VENUSPIPES.NS": pd.DataFrame(
            {"Close": [2238.5, 2119.8], "AdjFactor": [1.0, 1.0]},
            index=pd.to_datetime(["2026-10-07", "2026-10-08"]))}

    monkeypatch.setattr(store, "_download_batch", fake)
    p = store.get(["VENUSPIPES"], "2026-01-01", "2026-10-09")
    assert p["RawClose"]["VENUSPIPES"].index.max() == pd.Timestamp("2026-10-08")
    assert p["RawClose"]["VENUSPIPES"].iloc[-1] == 2119.8
    assert len(calls) == 1 and calls[0][1] >= pd.Timestamp("2026-09-20")   # a short fetch
    # and not again within the top-up window
    store.get(["VENUSPIPES"], "2026-01-01", "2026-10-09")
    assert len(calls) == 1


def test_fill_from_quotes_puts_prev_close_on_the_missing_session():
    now = datetime(2026, 10, 9, 13, 0)                     # Friday, market open
    close = pd.DataFrame({"VENUSPIPES": [2238.5, np.nan], "PEER": [10.0, 11.0]},
                         index=pd.to_datetime(["2026-10-07", "2026-10-08"]))
    q = pd.DataFrame({"prev_close": [2119.8], "price": [2063.9],
                      "session": [pd.Timestamp("2026-10-09")]}, index=["VENUSPIPES"])
    out, filled = data_mod.fill_from_quotes(close, q, now)
    assert filled == ["VENUSPIPES"]
    assert out.loc["2026-10-08", "VENUSPIPES"] == 2119.8
    assert out.loc["2026-10-07", "VENUSPIPES"] == 2238.5       # what Yahoo sent stays
    assert "2026-10-09" not in out.index.strftime("%Y-%m-%d")   # today is still live


def test_fill_from_quotes_never_overwrites_and_adds_a_settled_session():
    now = datetime(2026, 10, 9, 17, 0)                     # after the close
    close = pd.DataFrame({"A": [100.0, 101.0]}, index=pd.to_datetime(["2026-10-07", "2026-10-08"]))
    q = pd.DataFrame({"prev_close": [999.0], "price": [103.0],
                      "session": [pd.Timestamp("2026-10-09")]}, index=["A"])
    out, filled = data_mod.fill_from_quotes(close, q, now)
    assert out.loc["2026-10-08", "A"] == 101.0                  # not overwritten
    assert out.loc["2026-10-09", "A"] == 103.0
    assert filled == ["A"]


def test_quote_session_reads_both_forms():
    ts = pd.Timestamp("2026-10-09 13:45", tz="Asia/Kolkata")
    assert data_mod._quote_session(ts) == pd.Timestamp("2026-10-09")       # what yfinance gives
    assert data_mod._quote_session(int(ts.timestamp())) == pd.Timestamp("2026-10-09")
    assert data_mod._quote_session(None) is None
