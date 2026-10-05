import numpy as np
import pandas as pd
import pytest

from core import data as data_mod
from core.gainers import COLUMNS, daily_changes, top_gainers


def _panel(rows: dict[str, list[float]], start: str = "2026-09-28") -> pd.DataFrame:
    n = len(next(iter(rows.values())))
    return pd.DataFrame(rows, index=pd.bdate_range(start, periods=n))


def test_top_five_sorted_by_percentage():
    close = _panel({
        "A": [100, 101],   # +1%
        "B": [100, 110],   # +10%
        "C": [50, 52],     # +4%
        "D": [200, 230],   # +15%
        "E": [10, 10.5],   # +5%
        "F": [100, 102],   # +2%
        "G": [100, 90],    # -10%
    })
    top = top_gainers(close, n=5)
    assert list(top["Symbol"]) == ["D", "B", "E", "C", "F"]
    assert list(top.index) == [1, 2, 3, 4, 5]
    assert list(top.columns) == COLUMNS
    assert top["Change %"].iloc[0] == pytest.approx(15.0)
    assert top["Change"].iloc[0] == pytest.approx(30.0)
    assert top["Prev close"].iloc[0] == pytest.approx(200.0)
    assert top.attrs["as_of"] == close.index[-1]
    assert top.attrs["prev_date"] == close.index[-2]


def test_losers_never_pad_the_list():
    close = _panel({"A": [100, 103], "B": [100, 95], "C": [100, 100]})
    top = top_gainers(close, n=5)
    assert list(top["Symbol"]) == ["A"]


def test_ties_break_alphabetically():
    close = _panel({"ZED": [100, 105], "ABC": [10, 10.5], "MID": [20, 21]})
    assert list(top_gainers(close, n=3)["Symbol"]) == ["ABC", "MID", "ZED"]


def test_stock_not_trading_today_is_excluded():
    close = _panel({"A": [100, 101, np.nan], "B": [100, 100, 104]})
    top = top_gainers(close, n=5)
    assert list(top["Symbol"]) == ["B"]


def test_skipped_day_compares_with_last_print():
    close = _panel({"A": [100, np.nan, 120], "B": [100, 101, 102]})
    ch = daily_changes(close).set_index("Symbol")
    assert ch.loc["A", "Change %"] == pytest.approx(20.0)


def test_split_uses_adjusted_pct_and_raw_price():
    # 1:10 split on the last day: adjusted series is smooth, raw drops 90%
    adj = _panel({"S": [40.0, 44.0], "T": [100.0, 101.0]})
    raw = _panel({"S": [400.0, 44.0], "T": [100.0, 101.0]})
    top = top_gainers(adj, n=5, raw_close=raw)
    row = top.set_index("Symbol").loc["S"]
    assert row["Change %"] == pytest.approx(10.0)
    assert row["Close"] == pytest.approx(44.0)
    assert row["Prev close"] == pytest.approx(40.0)


def test_raw_price_shown_when_adjusted_differs():
    # dividend-adjusted history: % from adjusted, price from what traded
    adj = _panel({"A": [95.0, 104.5]})
    raw = _panel({"A": [100.0, 110.0]})
    row = top_gainers(adj, raw_close=raw).iloc[0]
    assert row["Change %"] == pytest.approx(10.0)
    assert row["Close"] == pytest.approx(110.0)
    assert row["Prev close"] == pytest.approx(100.0)
    assert row["Change"] == pytest.approx(10.0)


def test_min_price_filter():
    close = _panel({"PENNY": [2, 3], "BIG": [500, 520]})
    assert list(top_gainers(close, n=5, min_price=10)["Symbol"]) == ["BIG"]


def test_bad_values_are_ignored():
    close = _panel({"Z": [0.0, 5.0], "N": [-1.0, 2.0], "OK": [10.0, 11.0]})
    assert list(top_gainers(close)["Symbol"]) == ["OK"]


@pytest.mark.parametrize("close", [
    pd.DataFrame(),
    None,
    _panel({"A": [100.0]}),
    _panel({"A": [np.nan, np.nan]}),
])
def test_empty_inputs(close):
    top = top_gainers(close)
    assert top.empty
    assert list(top.columns) == COLUMNS


def test_n_zero_returns_empty():
    assert top_gainers(_panel({"A": [1, 2]}), n=0).empty


def test_unsorted_index_and_duplicates():
    close = _panel({"A": [100, 110], "B": [100, 105]})
    messy = pd.concat([close.iloc[[1]], close.iloc[[0]], close.iloc[[1]]])
    assert list(top_gainers(messy)["Symbol"]) == ["A", "B"]


def test_synthetic_panel():
    syms = [f"S{i}" for i in range(30)]
    panel = data_mod.synthetic_panel(syms, start="2026-08-01", end="2026-10-02")
    top = top_gainers(panel["Close"], n=5, raw_close=panel["RawClose"])
    assert len(top) <= 5
    assert top["Change %"].is_monotonic_decreasing
    assert (top["Change %"] > 0).all()
    full = daily_changes(panel["Close"])
    best = full.sort_values("Change %", ascending=False).iloc[0]
    if best["Change %"] > 0:
        assert top.iloc[0]["Symbol"] == best["Symbol"]
