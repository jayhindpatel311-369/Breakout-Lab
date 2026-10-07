import json

import numpy as np
import pandas as pd
import pytest

from core import journal as jn
from core import journal_stats as js


def _book():
    b = jn.Book(name="t", capital=1_000_000, cash=1_000_000)
    b.buy("BALAMINES", pd.Timestamp("2026-09-07").date(), 100, 200.0, 180)
    b.sell("BALAMINES", pd.Timestamp("2026-10-03").date(), 100, 190.0, "manual")  # a Saturday
    return b


def test_edit_exit_date_and_price():
    b = _book()
    rec = b.edit_exit_fill("BALAMINES", "2026-09-07", "2026-10-03", 100, 190.0, "manual",
                           185.0, new_date=pd.Timestamp("2026-10-01").date())
    assert rec and rec["new_sell_date"] == "2026-10-01"
    sell = [r for r in b.ledger if r["side"] == "SELL"][0]
    assert sell["date"] == "2026-10-01" and sell["price"] == 185.0
    assert sell["pnl"] == pytest.approx(-1500.0)
    assert b.closed[0].fills[0]["date"] == "2026-10-01"
    assert b.cash == pytest.approx(1_000_000 - 20_000 + 18_500)
    rt = js.round_trips(b)
    assert str(rt["exit_date"].iloc[0]) == "2026-10-01"


def test_edit_exit_date_before_entry_is_refused():
    b = _book()
    assert b.edit_exit_fill("BALAMINES", "2026-09-07", "2026-10-03", 100, 190.0, "manual",
                            190.0, new_date=pd.Timestamp("2026-09-01").date()) is None
    assert [r for r in b.ledger if r["side"] == "SELL"][0]["date"] == "2026-10-03"


def test_price_only_edit_keeps_date():
    b = _book()
    rec = b.edit_exit_fill("BALAMINES", "2026-09-07", "2026-10-03", 100, 190.0, "manual", 191.0)
    assert rec["new_sell_date"] == "2026-10-03"


def test_charges_set_replace_remove_and_persist(tmp_path):
    b = _book()
    b.set_charge("2026-09", 432.5, "Sep notes")
    b.set_charge("2026-10-15", 231.0)
    b.set_charge("2026-09", 500.0, "corrected")
    assert [c["month"] for c in b.charges] == ["2026-09", "2026-10"]
    assert b.total_charges() == pytest.approx(731.0)
    b.set_charge("2026-10", 0)
    assert b.total_charges() == pytest.approx(500.0)
    path = jn.save_book(str(tmp_path), b)
    again = jn.load_book(path)
    assert again.charges == b.charges
    # a book saved before charges existed loads with none
    blob = json.load(open(path))
    blob.pop("charges")
    json.dump(blob, open(path, "w"))
    assert jn.load_book(path).charges == []


def test_mfe_summary_counts_trades_that_never_ran():
    b = jn.Book(name="t", capital=1_000_000, cash=1_000_000)
    b.buy("FLAT", pd.Timestamp("2026-09-01").date(), 10, 100.0, 90)
    b.sell("FLAT", pd.Timestamp("2026-09-10").date(), 10, 95.0, "manual")
    b.buy("SMALL", pd.Timestamp("2026-09-01").date(), 10, 100.0, 90)
    b.sell("SMALL", pd.Timestamp("2026-09-10").date(), 10, 104.0, "manual")
    idx = pd.bdate_range("2026-09-01", "2026-09-10")
    close = pd.DataFrame({"FLAT": [100.0, 98, 97, 96, 95, 95, 95, 95],
                          "SMALL": [100.0, 103, 108, 106, 104, 104, 104, 104]}, index=idx)
    summ = js.mfe_report(js.round_trips(b), close)["summary"].set_index("how far it ran")
    assert summ.loc["Never above entry", "stocks"] == 1
    assert summ.loc["Up, but under 20%", "stocks"] == 1
    assert summ.loc["20% – 50%", "stocks"] == 0
    assert "70% – 100%" not in summ.index and "150% – 200%" not in summ.index


def test_mfe_summary_counts_each_trade_once():
    b = jn.Book(name="t", capital=10_000_000, cash=10_000_000)
    peaks = {"A": 320.0, "B": 160.0, "C": 125.0, "D": 90.0, "E": 700.0}
    for s in peaks:
        b.buy(s, pd.Timestamp("2026-09-01").date(), 10, 100.0, 90)
        b.sell(s, pd.Timestamp("2026-09-10").date(), 10, 110.0, "manual")
    idx = pd.bdate_range("2026-09-01", "2026-09-10")
    close = pd.DataFrame({s: [100.0, p, p, 110, 110, 110, 110, 110] for s, p in peaks.items()},
                         index=idx)
    summ = js.mfe_report(js.round_trips(b), close)["summary"].set_index("how far it ran")
    assert summ.loc["200% – 300%", "stocks"] == 1          # A peaked at +220%
    assert summ.loc["100% – 200%", "stocks"] == 0
    assert summ.loc["50% – 100%", "stocks"] == 1           # B +60%
    assert summ.loc["20% – 50%", "stocks"] == 1            # C +25%
    assert summ.loc["Up, but under 20%", "stocks"] == 1    # D: later closes at 110
    assert summ.loc["500% or more", "stocks"] == 1         # E +600%
    body = summ.drop("Total closed trades")
    assert body["stocks"].sum() == summ.loc["Total closed trades", "stocks"] == 5
    assert abs(body["% of closed trades"].sum() - 100) < 0.5


def _two_month_book():
    b = jn.Book(name="t", capital=1_000_000, cash=1_000_000)
    b.buy("AAA", pd.Timestamp("2026-01-05").date(), 100, 100.0, 90)
    b.sell("AAA", pd.Timestamp("2026-01-30").date(), 100, 120.0, "manual")
    b.buy("BBB", pd.Timestamp("2026-02-02").date(), 100, 100.0, 90)
    b.sell("BBB", pd.Timestamp("2026-02-27").date(), 100, 95.0, "manual")
    return b


def test_monthly_return_grid_from_booked_pnl():
    b = _two_month_book()
    grid, basis = js.monthly_return_grid(None, b.capital, rt=js.round_trips(b))
    assert basis == "booked"
    assert grid.loc[2026, 1] == 0.2 and grid.loc[2026, 2] == -0.05
    assert abs(grid.loc[2026, "Year"] - 0.15) < 1e-9


def test_monthly_return_grid_matches_equity_months():
    idx = pd.bdate_range("2026-01-01", "2026-02-27")
    eq = pd.Series(np.linspace(1_000_000, 1_100_000, len(idx)), index=idx)
    grid, basis = js.monthly_return_grid(eq, 1_000_000)
    mt = js.monthly(eq, capital=1_000_000)
    assert basis == "equity"
    assert grid.loc[2026, 1] == mt["Return %"].iloc[0]
    expect = ((1 + mt["Return %"].iloc[0] / 100) * (1 + mt["Return %"].iloc[1] / 100) - 1) * 100
    assert abs(grid.loc[2026, "Year"] - expect) < 0.02


def test_weekly_pnl_candles_chain_and_end_on_total_pnl():
    idx = pd.bdate_range("2026-01-05", "2026-01-30")
    eq = pd.Series(1_000_000 + np.arange(len(idx)) * 1000.0, index=idx)
    c = js.weekly_pnl_candles(eq, 1_000_000)
    assert len(c) == 4
    assert c["open"].iloc[0] == 0
    assert (c["open"].iloc[1:].values == c["close"].iloc[:-1].values).all()
    assert c["close"].iloc[-1] == eq.iloc[-1] - 1_000_000
    assert (c["high"] >= c[["open", "close"]].max(axis=1)).all()


def test_weekly_pnl_candles_take_deposits_out():
    idx = pd.bdate_range("2026-01-05", "2026-01-16")
    eq = pd.Series(1_000_000.0, index=idx)
    eq.loc["2026-01-12":] += 500_000                     # a deposit, not a profit
    flows = pd.Series({pd.Timestamp("2026-01-12"): 500_000.0})
    c = js.weekly_pnl_candles(eq, 1_500_000, flows)
    assert c["close"].abs().max() == 0


def test_pnl_by_stock_sums_round_trips():
    b = _two_month_book()
    b.buy("AAA", pd.Timestamp("2026-03-02").date(), 100, 100.0, 90)
    b.sell("AAA", pd.Timestamp("2026-03-20").date(), 100, 110.0, "manual")
    t = js.pnl_by_stock(js.round_trips(b))
    assert list(t["symbol"]) == ["AAA", "BBB"]
    assert t.loc[0, "pnl"] == 3000 and t.loc[0, "trades"] == 2


def test_relative_returns_rebase_to_zero():
    idx = pd.bdate_range("2026-01-01", periods=5)
    r = js.relative_returns({"a": pd.Series([100, 110, 121, 121, 121.0], idx),
                             "b": pd.Series([50, 50, 55, 60, 60.0], idx)})
    assert (r.iloc[0] == 0).all()
    assert r["a"].iloc[2] == 21.0 and r["b"].iloc[3] == 20.0
