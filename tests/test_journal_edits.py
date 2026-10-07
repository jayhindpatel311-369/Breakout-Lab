import json

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
