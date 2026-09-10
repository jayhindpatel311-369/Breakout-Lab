"""
Optional NSE bhavcopy source — number of trades and delivery percentage.

Why this file exists
--------------------
The screen you want includes "Buyer initiated trades >= 200" and "Seller
initiated trades >= 200". That split — classifying each trade by which side was
the aggressor — is **not published by NSE and is not available for free
anywhere**. Screeners that show it derive it from a paid tick-level feed.

What NSE *does* publish daily, and what this module fetches, is:

* ``NO_OF_TRADES``  — the total number of trades in that scrip that day
* ``DELIV_QTY`` / ``DELIV_PER`` — how much was taken to delivery rather than
  squared off intraday

Total trades is the honest stand-in: a "buyer >= 200 AND seller >= 200" filter
is asking for a scrip with at least a few hundred trades a day, which
``NO_OF_TRADES >= 400`` expresses directly. It is a proxy, not the same number,
and the app labels it as such.

Delivery percentage is a bonus you do not get from Yahoo at all, and for Indian
equities it is a genuinely useful quality filter — high delivery means real
positional buying rather than intraday churn.

Reliability warning
-------------------
NSE actively discourages scraping. This fetcher sets browser-like headers and
picks up a session cookie first, which usually works from a home connection, but
it can and does break: rate limits, layout changes, an office firewall. When it
fails it fails *loudly* — the screen reports the input as missing and the rules
that needed it are skipped, rather than quietly passing every stock.

The first build downloads one small CSV per trading day. Roughly 250 files per
year, a few seconds each with parallel fetches. Everything is cached, so you pay
that cost once and later runs only fetch new days.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import pandas as pd

from .data import DEFAULT_CACHE

BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports",
}

# sec_bhavdata_full has leading spaces in most column names — normalise them.
_WANT = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "DATE1": "date",
    "CLOSE_PRICE": "close",
    "TTL_TRD_QNTY": "volume",
    "TURNOVER_LACS": "turnover_lacs",
    "NO_OF_TRADES": "trades",
    "DELIV_QTY": "deliv_qty",
    "DELIV_PER": "deliv_pct",
}


def _cache_dir(cache_dir: str = DEFAULT_CACHE) -> str:
    d = os.path.join(cache_dir, "bhavcopy")
    os.makedirs(d, exist_ok=True)
    return d


def _day_path(day: pd.Timestamp, cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(_cache_dir(cache_dir), f"{day:%Y%m%d}.pkl")


def _make_session():
    import requests

    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
        s.get("https://www.nseindia.com/all-reports", timeout=15)
    except Exception:
        pass
    return s


def _parse(text: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        return None
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "NO_OF_TRADES" not in df.columns:
        return None
    keep = {k: v for k, v in _WANT.items() if k in df.columns}
    df = df[list(keep)].rename(columns=keep)
    if "series" in df.columns:
        df["series"] = df["series"].astype(str).str.strip()
        df = df[df["series"].isin(["EQ", "BE", "BZ", "SM", "ST"])]
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for c in ("close", "volume", "turnover_lacs", "trades", "deliv_qty", "deliv_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    return df.drop(columns=[c for c in ("series", "date") if c in df.columns])


def _fetch_day(sess, day: pd.Timestamp, cache_dir: str) -> tuple[pd.Timestamp, pd.DataFrame | None, str]:
    path = _day_path(day, cache_dir)
    if os.path.exists(path):
        try:
            return day, pd.read_pickle(path), "cached"
        except Exception:
            pass
    url = BHAV_URL.format(ddmmyyyy=f"{day:%d%m%Y}")
    try:
        r = sess.get(url, timeout=25)
    except Exception as e:  # noqa: BLE001
        return day, None, f"{type(e).__name__}"
    if r.status_code == 404:
        # market holiday, or the file predates this report format
        try:
            pd.to_pickle(pd.DataFrame(), path)
        except Exception:
            pass
        return day, pd.DataFrame(), "holiday/404"
    if r.status_code != 200:
        return day, None, f"HTTP {r.status_code}"
    df = _parse(r.text)
    if df is None:
        return day, None, "unparseable"
    try:
        pd.to_pickle(df, path)
    except Exception:
        pass
    return day, df, "ok"


def fetch_bhavcopy(
    calendar: pd.DatetimeIndex,
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    max_workers: int = 6,
    progress_cb=None,
    stop_after_failures: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch NO_OF_TRADES and DELIV_PER for `symbols` over `calendar`.

    Returns (trades_wide, delivery_wide, report). Both frames are date x symbol.
    `report` carries counts of cached / downloaded / failed days so the UI can
    be honest about coverage instead of showing a filter that silently did
    nothing.
    """
    days = [pd.Timestamp(d).normalize() for d in calendar]
    wanted = set(s.strip().upper() for s in symbols)
    report = {"days": len(days), "cached": 0, "downloaded": 0, "holidays": 0,
              "failed": 0, "errors": [], "aborted": False}
    if not days:
        return pd.DataFrame(), pd.DataFrame(), report

    # Anything already on disk needs no session at all.
    todo = [d for d in days if not os.path.exists(_day_path(d, cache_dir))]
    frames: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in days:
        if d not in todo:
            try:
                frames[d] = pd.read_pickle(_day_path(d, cache_dir))
                report["cached"] += 1
            except Exception:
                todo.append(d)

    if todo:
        try:
            sess = _make_session()
        except ImportError:
            report["errors"].append("The `requests` package is not installed.")
            report["failed"] = len(todo)
            return _wide(frames, wanted, days) + (report,)

        consecutive_failures = 0
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_day, sess, d, cache_dir): d for d in todo}
            for fut in as_completed(futures):
                day, df, status = fut.result()
                done += 1
                if df is None:
                    report["failed"] += 1
                    consecutive_failures += 1
                    if len(report["errors"]) < 5:
                        report["errors"].append(f"{day:%Y-%m-%d}: {status}")
                else:
                    consecutive_failures = 0
                    frames[day] = df
                    if df.empty:
                        report["holidays"] += 1
                    else:
                        report["downloaded"] += 1
                if progress_cb and (done % 10 == 0 or done == len(todo)):
                    progress_cb(done, len(todo))
                if consecutive_failures >= stop_after_failures:
                    report["aborted"] = True
                    report["errors"].append(
                        f"Gave up after {consecutive_failures} failures in a row — "
                        "NSE is most likely rate-limiting or blocking this connection."
                    )
                    for f in futures:
                        f.cancel()
                    break
        time.sleep(0.05)

    trades, delivery = _wide(frames, wanted, days)
    return trades, delivery, report


def _wide(
    frames: dict[pd.Timestamp, pd.DataFrame],
    wanted: set[str],
    days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_rows: dict[pd.Timestamp, pd.Series] = {}
    dl_rows: dict[pd.Timestamp, pd.Series] = {}
    for day, df in frames.items():
        if df is None or df.empty or "symbol" not in df.columns:
            continue
        sub = df[df["symbol"].isin(wanted)]
        if sub.empty:
            continue
        sub = sub.drop_duplicates("symbol").set_index("symbol")
        if "trades" in sub.columns:
            tr_rows[day] = sub["trades"]
        if "deliv_pct" in sub.columns:
            dl_rows[day] = sub["deliv_pct"]

    cols = sorted(wanted)
    idx = pd.DatetimeIndex(sorted(days))
    trades = (pd.DataFrame(tr_rows).T.reindex(index=idx, columns=cols)
              if tr_rows else pd.DataFrame(index=idx, columns=cols, dtype=float))
    delivery = (pd.DataFrame(dl_rows).T.reindex(index=idx, columns=cols)
                if dl_rows else pd.DataFrame(index=idx, columns=cols, dtype=float))
    return trades.astype(float), delivery.astype(float)


def coverage_summary(frame: pd.DataFrame) -> str:
    """One line on how much of the requested grid actually has data."""
    if frame is None or frame.empty:
        return "no data"
    filled = float(frame.notna().to_numpy().mean()) * 100
    days_with_data = int(frame.notna().any(axis=1).sum())
    return f"{filled:.0f}% of cells filled, {days_with_data} of {len(frame)} days have data"


def synthetic_trades(calendar: pd.DatetimeIndex, symbols: list[str], seed: int = 5):
    """Demo-mode stand-in so the trades/delivery filters can be explored offline."""
    rng = np.random.default_rng(seed)
    tr = pd.DataFrame(
        rng.integers(50, 40_000, size=(len(calendar), len(symbols))).astype(float),
        index=calendar, columns=symbols,
    )
    dl = pd.DataFrame(
        rng.uniform(15, 85, size=(len(calendar), len(symbols))),
        index=calendar, columns=symbols,
    )
    return tr, dl
