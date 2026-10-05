"""
Today's top gainers — the day's biggest percentage moves across the universe.

Pure pandas, no Streamlit, so it can be tested on a hand-built panel.

The percentage is read off the split-adjusted close, the rupee figures off the
actual traded price. A stock that split 1:10 overnight is not a 90% loser, and
a ₹45 adjusted close on a day it really traded at ₹450 is not its price — the
same split the rest of the app keeps (see core/data.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Symbol", "Prev close", "Close", "Change", "Change %"]


def _empty() -> pd.DataFrame:
    out = pd.DataFrame(columns=COLUMNS)
    out.attrs["as_of"] = None
    out.attrs["prev_date"] = None
    return out


def daily_changes(close: pd.DataFrame, raw_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every symbol's move on the latest bar, against its previous close.

    Only symbols that actually printed on the latest bar are included — a stock
    that did not trade today has no "today" move, and carrying its last price
    forward would rank a stale number. The previous close is the symbol's own
    last print before that bar, so a name that skipped a day is compared with
    the day it last traded rather than dropped.
    """
    if close is None or close.empty:
        return _empty()

    px = close.apply(pd.to_numeric, errors="coerce").sort_index()
    px = px[~px.index.duplicated(keep="last")].dropna(how="all")
    if len(px) < 2:
        return _empty()

    as_of = px.index[-1]
    last = px.iloc[-1]
    prev = px.iloc[:-1].ffill().iloc[-1]
    pct = last / prev - 1.0
    ok = last.gt(0) & prev.gt(0) & np.isfinite(pct)
    if not ok.any():
        return _empty()
    pct = pct[ok]

    shown = last[ok]
    if raw_close is not None and not raw_close.empty:
        raw = raw_close.apply(pd.to_numeric, errors="coerce").sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        if as_of in raw.index:
            raw_last = raw.loc[as_of].reindex(pct.index)
            good = raw_last.gt(0)
            shown = raw_last.where(good, shown)

    # previous close in today's price units, so a split day still reads sanely
    prev_shown = shown / (1.0 + pct)
    out = pd.DataFrame({
        "Symbol": pct.index.astype(str),
        "Prev close": prev_shown.values,
        "Close": shown.values,
        "Change": (shown - prev_shown).values,
        "Change %": (pct * 100.0).values,
    })
    out.attrs["as_of"] = pd.Timestamp(as_of)
    out.attrs["prev_date"] = pd.Timestamp(px.index[-2])
    return out


def top_gainers(
    close: pd.DataFrame,
    n: int = 5,
    raw_close: pd.DataFrame | None = None,
    min_price: float = 0.0,
) -> pd.DataFrame:
    """The `n` biggest percentage gainers on the latest bar.

    Only names that are actually up make the list: on a day the whole market
    falls there may be fewer than `n`, and padding it with the smallest losers
    would call them gainers. Ties break alphabetically so the list is stable.
    `min_price` drops penny stocks by traded price.
    """
    allc = daily_changes(close, raw_close)
    as_of, prev_date = allc.attrs.get("as_of"), allc.attrs.get("prev_date")
    if allc.empty or n <= 0:
        out = _empty()
        out.attrs.update(as_of=as_of, prev_date=prev_date)
        return out

    up = allc[(allc["Change %"] > 0) & (allc["Close"] >= float(min_price or 0))]
    up = up.sort_values(["Change %", "Symbol"], ascending=[False, True]).head(int(n))
    up = up.reset_index(drop=True)
    up.index = up.index + 1
    up.index.name = "Rank"
    up.attrs.update(as_of=as_of, prev_date=prev_date)
    return up
