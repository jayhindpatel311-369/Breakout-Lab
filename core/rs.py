"""
RS Rating — IBD-style relative strength, 1 to 99, from the daily closes alone.

The raw number is a weighted return over the last year, leaning on the most
recent quarter:

    0.4 x r(63)  +  0.2 x r(126)  +  0.2 x r(189)  +  0.2 x r(252)

where r(n) is the return over the last n trading days (about 3 / 6 / 9 / 12
months). The rating is where that raw number sits among a reference group —
the selected universe — as a percentile from 1 to 99.

Two things worth knowing:

* **A stock does not have to be in the reference group to be rated.** A
  Chartink name from outside the Nifty 500 gets its own raw number and is then
  placed in the Nifty 500's distribution. The universe is the ruler, not a
  membership test.
* **Inside one week the order is the same whatever the reference.** The
  rating is a monotonic function of the raw number, so ranking this week's
  candidates by rating or by raw number picks the same stocks. The reference
  only decides what "RS 90" *means* — 90% of the market, rather than 90% of a
  handful of breakouts.

A company listed too recently for the full year is rated on the periods it has
(at least 63 days), with the weights renormalised over them, and flagged —
recent IPOs are exactly what a Chartink list is full of, so dropping them would
be wrong, and pretending they have a year of history would be worse.

The anchor date is the signal week's last bar, never a live price, so the
rating is what the stock showed when it qualified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (63, 126, 189, 252)
WEIGHTS = (0.4, 0.2, 0.2, 0.2)
MIN_DAYS = WINDOWS[0]


def rs_raw(close: pd.DataFrame, asof=None, windows=WINDOWS, weights=WEIGHTS) -> pd.DataFrame:
    """Raw weighted return per symbol as of `asof` (default: the last row).

    Returns a frame with `rs_raw` (in %), `rs_days` (how many trading days of
    history it had) and `rs_short` (True when fewer than the longest window).
    Symbols with under `MIN_DAYS` of history, or no price at `asof`, are absent.
    """
    cols = ["rs_raw", "rs_days", "rs_short"]
    if close is None or close.empty:
        return pd.DataFrame(columns=cols)
    px = close.apply(pd.to_numeric, errors="coerce").sort_index()
    if asof is not None:
        px = px.loc[px.index <= pd.Timestamp(asof)]
    if px.empty:
        return pd.DataFrame(columns=cols)

    rows = {}
    longest = max(windows)
    for sym in px.columns:
        col = px[sym].dropna()
        if len(col) <= MIN_DAYS or col.index[-1] != px.index[-1]:
            continue
        last = float(col.iloc[-1])
        if last <= 0:
            continue
        num, den = 0.0, 0.0
        for n, w in zip(windows, weights):
            if len(col) <= n:
                continue
            base = float(col.iloc[-(n + 1)])
            if base > 0:
                num += w * (last / base - 1.0)
                den += w
        if den <= 0:
            continue
        rows[sym] = {"rs_raw": num / den * 100.0, "rs_days": len(col) - 1,
                     "rs_short": len(col) - 1 < longest}
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def rs_rating(raw: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """Place each raw value in the reference distribution: 1 (weakest) .. 99.

    `reference` defaults to `raw` itself. Ties count half, so a value equal to
    the reference median rates about 50 whichever side of it the ties fall.
    """
    raw = pd.to_numeric(raw, errors="coerce").dropna()
    if raw.empty:
        return pd.Series(dtype=float)
    ref = pd.to_numeric(reference if reference is not None else raw, errors="coerce").dropna()
    ref = np.sort(ref.to_numpy(dtype=float))
    if len(ref) == 0:
        return pd.Series(np.nan, index=raw.index)
    lo = np.searchsorted(ref, raw.to_numpy(dtype=float), side="left")
    hi = np.searchsorted(ref, raw.to_numpy(dtype=float), side="right")
    pct = (lo + 0.5 * (hi - lo)) / len(ref)
    return pd.Series(np.clip(np.rint(1 + 98 * pct), 1, 99), index=raw.index)
