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

from .data import DEFAULT_CACHE, now_ist

BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"

# NSE publishes the day's bhavcopy in the evening. Before that the URL is a 404
# exactly like a holiday's — so a 404 is only trusted as "holiday" for days old
# enough that the file would certainly exist by now.
HOLIDAY_404_AFTER_DAYS = 3
# statuses worth a retry: rate limits, NSE's bot wall, transient server errors
_RETRY_STATUS = {401, 403, 429, 500, 502, 503, 504}

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


def _get(sess, url: str, tries: int = 3, backoff: float = 1.0, timeout: float = 20,
         sleep=time.sleep, **kw):
    """GET with retry and exponential backoff on network errors and NSE's
    rate-limit / bot-wall statuses. Returns the last response, or None if every
    attempt raised. A 401/403 also re-primes the session cookies once — NSE's
    cookies expire after a few minutes."""
    resp = None
    for i in range(max(1, tries)):
        try:
            resp = sess.get(url, timeout=timeout, **kw)
        except Exception:                                      # noqa: BLE001
            resp = None
        if resp is not None and resp.status_code not in _RETRY_STATUS:
            return resp
        if i < tries - 1:
            if resp is not None and resp.status_code in (401, 403):
                _prime(sess)
            sleep(backoff * (2 ** i))
    return resp


def _prime(sess) -> None:
    for u in ("https://www.nseindia.com", "https://www.nseindia.com/all-reports"):
        try:
            sess.get(u, timeout=15)
        except Exception:                                      # noqa: BLE001
            pass


def _fetch_day(sess, day: pd.Timestamp, cache_dir: str) -> tuple[pd.Timestamp, pd.DataFrame | None, str]:
    path = _day_path(day, cache_dir)
    if os.path.exists(path):
        try:
            return day, pd.read_pickle(path), "cached"
        except Exception:
            pass
    url = BHAV_URL.format(ddmmyyyy=f"{day:%d%m%Y}")
    r = _get(sess, url, tries=3, backoff=1.0, timeout=25)
    if r is None:
        return day, None, "network error"
    if r.status_code == 404:
        # market holiday, or the file predates this report format — or simply
        # not published YET. Only an old enough day is cached as a holiday;
        # caching today's evening-404 made that trading day a permanent hole.
        age = (pd.Timestamp(now_ist().date()) - pd.Timestamp(day).normalize()).days
        if age < HOLIDAY_404_AFTER_DAYS:
            return day, None, "not published yet"
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
    frames, report = _collect(days, cache_dir, max_workers, progress_cb, stop_after_failures)
    if not days:
        return pd.DataFrame(), pd.DataFrame(), report
    w = _wide(frames, wanted, days)
    return w["trades"], w["deliv_pct"], report


def fetch_bhav_close(
    calendar: pd.DatetimeIndex,
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    max_workers: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """NSE's official closing price (CLOSE_PRICE) for `symbols` over `calendar`.

    The raw, unadjusted close the exchange publishes — the number a broker's
    "previous close" and every exchange-based EMA are built on. Yahoo's daily
    close for an NSE stock is usually the same, but not always (it can carry the
    last traded price rather than the official closing price). Same cache as
    `fetch_bhavcopy`: a day downloaded for one is free for the other.
    """
    days = [pd.Timestamp(d).normalize() for d in calendar]
    wanted = set(s.strip().upper() for s in symbols)
    # a few failures in a row means NSE is refusing this network; give up fast
    frames, report = _collect(days, cache_dir, max_workers, None, 4)
    if not days:
        return pd.DataFrame(), report
    return _wide(frames, wanted, days)["close"], report


def _collect(days, cache_dir, max_workers, progress_cb, stop_after_failures):
    """Every requested day's parsed bhavcopy, from the cache or NSE."""
    report = {"days": len(days), "cached": 0, "downloaded": 0, "holidays": 0,
              "failed": 0, "errors": [], "aborted": False}
    if not days:
        return {}, report

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
            return frames, report

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
    return frames, report


def _wide(
    frames: dict[pd.Timestamp, pd.DataFrame],
    wanted: set[str],
    days: list[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """date x symbol frames for trades, delivery % and the official close.

    The close was parsed all along and thrown away here; it is kept now. When a
    symbol appears in more than one series the EQ row wins (when the series is
    still known — cached days have it stripped, and then the first row does).
    """
    fields = ("trades", "deliv_pct", "close")
    rows: dict[str, dict[pd.Timestamp, pd.Series]] = {f: {} for f in fields}
    for day, df in frames.items():
        if df is None or df.empty or "symbol" not in df.columns:
            continue
        sub = df[df["symbol"].isin(wanted)]
        if sub.empty:
            continue
        if "series" in sub.columns:
            sub = sub.assign(_eq=(sub["series"] != "EQ")).sort_values("_eq")
        sub = sub.drop_duplicates("symbol").set_index("symbol")
        for f in fields:
            if f in sub.columns:
                rows[f][day] = sub[f]

    cols = sorted(wanted)
    idx = pd.DatetimeIndex(sorted(days))
    out = {}
    for f in fields:
        out[f] = (pd.DataFrame(rows[f]).T.reindex(index=idx, columns=cols)
                  if rows[f] else pd.DataFrame(index=idx, columns=cols, dtype=float)).astype(float)
    return out


# --------------------------------------------------------------------------- #
# live quotes — holdings only
# --------------------------------------------------------------------------- #
def _parse_quote(js: dict) -> dict | None:
    """lastPrice, previousClose and the session date out of NSE's quote JSON."""
    try:
        pi = js.get("priceInfo") or {}
        last = float(pi.get("lastPrice"))
        prev = float(pi.get("previousClose"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not (np.isfinite(last) and last > 0):
        return None
    session = None
    stamp = (js.get("metadata") or {}).get("lastUpdateTime") \
        or (js.get("preOpenMarket") or {}).get("lastUpdateTime")
    if stamp:
        try:
            session = pd.Timestamp(pd.to_datetime(stamp, format="%d-%b-%Y %H:%M:%S")).normalize()
        except Exception:                                      # noqa: BLE001
            session = None
    return {"last": last, "prev_close": prev if np.isfinite(prev) and prev > 0 else np.nan,
            "session": session}


def live_quotes(
    symbols: list[str],
    sess=None,
    per_second: float = 3.0,
    give_up_after: int = 3,
    sleep=time.sleep,
) -> tuple[pd.DataFrame, dict]:
    """NSE's own live quote for a handful of symbols — meant for the open book.

    One request per symbol, paced to `per_second`, each with retry and
    backoff. NSE throttles and sometimes blocks outright (cloud data-centre
    addresses especially), so after `give_up_after` symbols fail in a row it
    stops asking and reports `blocked` — the caller falls back to Yahoo for
    whatever is missing.

    Returns (frame indexed by symbol with `last`, `prev_close`, `session`), report).
    """
    report = {"asked": len(symbols), "ok": 0, "failed": [], "blocked": False}
    rows: dict[str, dict] = {}
    if not symbols:
        return pd.DataFrame(columns=["last", "prev_close", "session"]), report
    if sess is None:
        try:
            sess = _make_session()
        except ImportError:
            report["blocked"] = True
            return pd.DataFrame(columns=["last", "prev_close", "session"]), report
        sess.headers.update({"Accept": "application/json,text/plain,*/*",
                             "Referer": "https://www.nseindia.com/get-quotes/equity"})
    gap = 1.0 / per_second if per_second > 0 else 0.0
    streak = 0
    for sym in symbols:
        from urllib.parse import quote as _q
        r = _get(sess, QUOTE_URL.format(symbol=_q(sym)), tries=2, backoff=1.0,
                 timeout=10, sleep=sleep)
        q = None
        if r is not None and r.status_code == 200:
            try:
                q = _parse_quote(r.json())
            except Exception:                                  # noqa: BLE001
                q = None
        if q is None:
            report["failed"].append(sym)
            streak += 1
            if streak >= give_up_after and report["ok"] == 0:
                report["blocked"] = True
                report["failed"].extend(s for s in symbols if s not in rows and s != sym
                                        and s not in report["failed"])
                break
        else:
            rows[sym] = q
            report["ok"] += 1
            streak = 0
        if gap:
            sleep(gap)
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["last", "prev_close", "session"])
    return df, report


def overlay_close(panel: dict[str, pd.DataFrame], nse_close: pd.DataFrame) -> tuple[dict, int]:
    """Put NSE's official closes into a Yahoo panel, where NSE has them.

    `RawClose` takes the NSE number as-is (both are traded prices). `Close` is
    the adjusted series, so the NSE close is scaled by that day's Yahoo
    adjustment factor (Close / RawClose) to stay on the same basis — otherwise
    a past dividend or split would put a step into the EMA.

    Returns (new panel, cells replaced). Days or symbols NSE did not cover keep
    Yahoo's value.
    """
    if nse_close is None or nse_close.empty:
        return panel, 0
    raw = panel.get("RawClose", pd.DataFrame())
    adj = panel.get("Close", pd.DataFrame())
    if raw.empty or adj.empty:
        return panel, 0
    nse = nse_close.reindex(index=raw.index, columns=raw.columns)
    factor = (adj / raw.replace(0.0, np.nan)).reindex_like(nse)
    mask = nse.notna() & (nse > 0) & factor.notna()
    n = int(mask.to_numpy().sum())
    if not n:
        return panel, 0
    out = dict(panel)
    out["RawClose"] = raw.where(~mask, nse)
    out["Close"] = adj.where(~mask.reindex_like(adj).fillna(False), (nse * factor).reindex_like(adj))
    return out, n


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
