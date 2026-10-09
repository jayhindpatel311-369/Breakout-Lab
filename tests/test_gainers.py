import numpy as np
import pandas as pd
import pytest

from core.gainers import COLUMNS, day_moves, prev_close_before, top_movers


def _close(rows, dates):
    return pd.DataFrame(rows, index=pd.to_datetime(dates))


def test_prev_close_is_chosen_by_date_not_row():
    close = _close({"A": [100.0, 110.0], "B": [50.0, 55.0]}, ["2026-10-01", "2026-10-02"])
    # market hours Monday: panel ends Friday → previous close is Friday's
    assert prev_close_before(close, pd.Timestamp("2026-10-05"))["A"] == 110.0
    # after Friday's close the panel already has Friday → previous is Thursday's
    assert prev_close_before(close, pd.Timestamp("2026-10-02"))["A"] == 100.0


def test_prev_close_skips_symbol_gap():
    close = _close({"A": [100.0, np.nan]}, ["2026-10-01", "2026-10-02"])
    assert prev_close_before(close, pd.Timestamp("2026-10-05"))["A"] == 100.0


def test_top_five_gainers_and_losers_split_and_sorted():
    syms = list("ABCDEFGH")
    prev = pd.Series(100.0, index=syms)
    cmp = pd.Series([101, 110, 104, 115, 105, 102, 90, 97], index=syms, dtype=float)
    qty = pd.Series(10, index=syms)
    moves, missing = day_moves(syms, cmp, prev, qty)
    up, down = top_movers(moves, 5)
    assert list(up["Stock"]) == ["D", "B", "E", "C", "F"]
    assert list(down["Stock"]) == ["G", "H"]          # never padded with gainers
    assert up["Day P&L"].iloc[0] == pytest.approx(150.0)
    assert missing == []
    assert list(up.columns) == COLUMNS


def test_unpriced_holding_is_listed_not_shown_flat():
    moves, missing = day_moves(["A", "B", "C"], pd.Series({"A": 105.0, "C": 0.0}),
                               pd.Series({"A": 100.0, "B": 50.0, "C": 10.0}))
    assert list(moves["Stock"]) == ["A"]
    assert missing == ["B", "C"]


def test_flat_stock_is_in_neither_list():
    moves, _ = day_moves(["A"], pd.Series({"A": 100.0}), pd.Series({"A": 100.0}))
    up, down = top_movers(moves)
    assert up.empty and down.empty


def test_ties_break_alphabetically():
    moves, _ = day_moves(["Z", "A"], pd.Series({"Z": 105.0, "A": 21.0}),
                         pd.Series({"Z": 100.0, "A": 20.0}))
    assert list(top_movers(moves)[0]["Stock"]) == ["A", "Z"]


def test_empty_inputs():
    moves, missing = day_moves([], pd.Series(dtype=float), pd.Series(dtype=float))
    up, down = top_movers(moves)
    assert up.empty and down.empty and missing == []
    assert prev_close_before(pd.DataFrame(), pd.Timestamp("2026-10-05")).empty


def test_symbol_a_session_behind_is_left_out_not_carried_forward():
    # Yahoo sent Thursday's bar for B but not for A: A's "previous close" would
    # be Wednesday's, and its two-day fall would read as today's move
    close = _close({"A": [2238.5, np.nan], "B": [100.0, 101.0]}, ["2026-10-07", "2026-10-08"])
    prev = prev_close_before(close, pd.Timestamp("2026-10-09"))
    assert "A" not in prev.index
    assert prev["B"] == 101.0
    moves, missing = day_moves(["A", "B"], pd.Series({"A": 2064.5, "B": 102.0}), prev)
    assert missing == ["A"]
