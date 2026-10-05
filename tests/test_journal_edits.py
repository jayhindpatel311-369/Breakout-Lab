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
