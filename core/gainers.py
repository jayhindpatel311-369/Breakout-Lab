"""
Today's movers inside the book — the open positions' biggest gainers and losers.

Only what you hold. The universe never enters this: a list of the market's top
movers says nothing about your book, and pricing five hundred names to rank the
twelve you own is waiting for nothing.

Pure pandas, no Streamlit, so it can be tested on hand-built inputs.

The day's move is CMP against the previous session's close. "Previous" is
chosen by date, never by row position: during market hours the panel's last
row is yesterday, after the close it may be today, and on a holiday the live
quote belongs to an earlier session altogether. Picking "the second-last row"
gets one of those three wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Stock", "Prev close", "CMP", "Day %", "Day P&L"]


def prev_close_before(close: pd.DataFrame, session: pd.Timestamp) -> pd.Series:
    """Each symbol's last close strictly before `session`."""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    px = close.apply(pd.to_numeric, errors="coerce").sort_index()
    idx = pd.DatetimeIndex(px.index).normalize()
    before = px[idx < pd.Timestamp(session).normalize()]
    if before.empty:
        return pd.Series(dtype=float)
    return before.ffill().iloc[-1].dropna()


def day_moves(
    symbols,
    cmp: pd.Series,
    prev_close: pd.Series,
    qty: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """The day's move for each held symbol, and the ones that could not be priced.

    A symbol without a CMP or a previous close is left out and listed, not shown
    as 0% — a flat line in a movers table reads as "did nothing today", which is
    a claim the data does not support.
    """
    rows, missing = [], []
    for sym in sorted(set(symbols)):
        c = float(cmp.get(sym, np.nan)) if cmp is not None else np.nan
        p = float(prev_close.get(sym, np.nan)) if prev_close is not None else np.nan
        if not (np.isfinite(c) and np.isfinite(p) and c > 0 and p > 0):
            missing.append(sym)
            continue
        q = float(qty.get(sym, np.nan)) if qty is not None else np.nan
        rows.append({
            "Stock": sym,
            "Prev close": p,
            "CMP": c,
            "Day %": (c / p - 1.0) * 100.0,
            "Day P&L": (c - p) * q if np.isfinite(q) else np.nan,
        })
    return pd.DataFrame(rows, columns=COLUMNS), missing


def top_movers(moves: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(top `n` gainers, top `n` losers). Only names actually up go in the first,
    only names actually down in the second — never padded with the other side.
    Ties break alphabetically so the order is stable between refreshes."""
    if moves is None or moves.empty or n <= 0:
        empty = pd.DataFrame(columns=COLUMNS)
        return empty, empty.copy()
    up = moves[moves["Day %"] > 0].sort_values(["Day %", "Stock"], ascending=[False, True])
    down = moves[moves["Day %"] < 0].sort_values(["Day %", "Stock"], ascending=[True, True])
    return up.head(int(n)).reset_index(drop=True), down.head(int(n)).reset_index(drop=True)
