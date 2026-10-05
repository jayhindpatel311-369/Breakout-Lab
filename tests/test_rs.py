import numpy as np
import pandas as pd
import pytest

from core import fundamentals as F
from core.breakout import select_by_rs
from core.rs import rs_raw, rs_rating


def _close(n=300, **growth):
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({k: 100 * np.exp(np.linspace(0, g, n)) for k, g in growth.items()},
                        index=idx)


# --------------------------------------------------------------------------- #
# RS
# --------------------------------------------------------------------------- #
def test_rs_raw_formula():
    idx = pd.bdate_range("2025-01-01", periods=253)
    px = pd.Series(100.0, index=idx)
    px.iloc[-1] = 120.0                             # flat for a year, +20% on the last day
    raw = rs_raw(pd.DataFrame({"A": px}))
    assert raw.loc["A", "rs_raw"] == pytest.approx(20.0)   # every window returned 20%
    assert not raw.loc["A", "rs_short"]


def test_rs_weights_recent_quarter():
    idx = pd.bdate_range("2025-01-01", periods=253)
    early = pd.Series(np.r_[np.linspace(100, 150, 190), np.full(63, 150.0)], index=idx)
    late = pd.Series(np.r_[np.full(190, 100.0), np.linspace(100, 150, 63)], index=idx)
    raw = rs_raw(pd.DataFrame({"EARLY": early, "LATE": late}))
    assert raw.loc["LATE", "rs_raw"] > raw.loc["EARLY", "rs_raw"]


def test_short_history_renormalised_and_flagged():
    c = _close(n=100, IPO=0.2)
    raw = rs_raw(c)
    assert raw.loc["IPO", "rs_short"]
    assert raw.loc["IPO", "rs_days"] == 99
    assert rs_raw(_close(n=50, TINY=0.1)).empty            # under 63 days: not rated


def test_asof_ignores_later_bars():
    c = _close(n=300, A=0.3)
    asof = c.index[-50]
    assert rs_raw(c, asof).loc["A", "rs_raw"] == pytest.approx(rs_raw(c.loc[:asof]).loc["A", "rs_raw"])


def test_rating_against_reference_and_outsider():
    ref = pd.Series(np.arange(100, dtype=float))        # 0..99
    r = rs_rating(pd.Series({"TOP": 1000.0, "MID": 49.5, "LOW": -5.0}), ref)
    assert r["TOP"] == 99 and r["LOW"] == 1
    assert 49 <= r["MID"] <= 51
    # same order whatever the reference — only the number changes
    alone = rs_rating(pd.Series({"A": 10.0, "B": 5.0, "C": 1.0}))
    assert list(alone.sort_values(ascending=False).index) == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _scored(rows):
    return pd.DataFrame(rows).set_index("sym")


def test_rs_order_and_tie_breakers():
    sc = _scored([
        {"sym": "A", "rs_rating": 90, "delivery_pct": 40, "volume_surge": 1.0},
        {"sym": "B", "rs_rating": 95, "delivery_pct": 20, "volume_surge": 1.0},
        {"sym": "C", "rs_rating": 90, "delivery_pct": 60, "volume_surge": 1.0},
        {"sym": "D", "rs_rating": 90, "delivery_pct": 60, "volume_surge": 3.0},
    ])
    picks, _ = select_by_rs(sc, 4, diversify=False)
    assert list(picks.index) == ["B", "D", "C", "A"]
    assert list(picks["rank"]) == [1, 2, 3, 4]


def test_sector_cap_is_strict():
    sc = _scored([{"sym": s, "rs_rating": 99 - i} for i, s in enumerate("ABCDEF")])
    sectors = {s: "Capital Goods" for s in "ABCDE"} | {"F": "IT"}
    picks, rej = select_by_rs(sc, 5, sectors, max_per_sector=2, max_rank=10)
    assert list(picks.index) == ["A", "B", "F"]          # never filled over the cap
    assert rej.loc["C", "why"].startswith("sector full")


def test_rank_floor_is_hard():
    sc = _scored([{"sym": s, "rs_rating": 99 - i} for i, s in enumerate("ABCDEFG")])
    sectors = {s: "X" for s in "ABCDE"} | {"F": "Y", "G": "Z"}
    picks, rej = select_by_rs(sc, 5, sectors, max_per_sector=2, max_rank=6)
    assert list(picks.index) == ["A", "B", "F"]          # G is rank 7, below the floor
    assert "rank floor" in rej.loc["G", "why"]


def test_gate_fail_vetoes_unknown_passes():
    sc = _scored([
        {"sym": "A", "rs_rating": 99, "gate": "fail", "gate why": "interest coverage 0.93x"},
        {"sym": "B", "rs_rating": 98, "gate": "unknown"},
        {"sym": "C", "rs_rating": 97, "gate": "pass"},
    ])
    picks, rej = select_by_rs(sc, 2, diversify=False)
    assert list(picks.index) == ["B", "C"]
    assert "interest coverage" in rej.loc["A", "why"]


def test_unknown_sector_not_capped():
    sc = _scored([{"sym": s, "rs_rating": 99 - i} for i, s in enumerate("ABC")])
    picks, _ = select_by_rs(sc, 3, {}, max_per_sector=1)
    assert len(picks) == 3 and set(picks["pick"]) == {"{?}"}


# --------------------------------------------------------------------------- #
# fundamentals gate
# --------------------------------------------------------------------------- #
def _facts(**kw):
    base = dict(symbol="X", revenue=[100, 90, 80], net_income=[10, 9, 8],
                ebit=[20, 18, 16], interest=[2, 2, 2], cfo=[12, 10, 9],
                shares=[100, 100, 100], total_assets=[500, 450, 400], equity=[200, 180, 160])
    base.update(kw)
    return F.Facts(**base)


def test_gate_pass():
    assert F.gate(_facts()) == ("pass", [])


def test_gate_cfo_negative_two_of_three():
    st, why = F.gate(_facts(cfo=[-5, 3, -1]))
    assert st == "fail" and "cash from operations" in why[0]
    assert F.gate(_facts(cfo=[-5, 3, 1]))[0] == "pass"


def test_gate_interest_cover():
    st, why = F.gate(_facts(ebit=[2.8, 5, 5], interest=[3, 3, 3]))
    assert st == "fail" and "interest coverage 0.93x" in why[0]


def test_gate_dilution_without_assets():
    st, why = F.gate(_facts(shares=[130, 110, 100], total_assets=[420, 410, 400]))
    assert st == "fail" and "share count up 30%" in why[0]
    # the same dilution that bought assets is fine
    assert F.gate(_facts(shares=[130, 110, 100], total_assets=[600, 450, 400]))[0] == "pass"


def test_gate_missing_data_is_unknown():
    st, why = F.gate(F.Facts(symbol="X", revenue=[1], net_income=[1]))
    assert st == "unknown" and "not enough data" in why[0]
