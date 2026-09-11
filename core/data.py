"""
Price data layer.

Responsibilities
----------------
* Download daily OHLCV from Yahoo Finance (yfinance) for NSE symbols.
* Cache every symbol on disk so a re-run is instant and you stop hammering Yahoo.
* Hand the rest of the app clean, aligned, wide DataFrames indexed by date.
* Tell you honestly which symbols are dead (`validate_symbols`).

Why Yahoo?  It is free, needs no API key, and covers NSE equities, ETFs and the
Nifty indices.  It is *not* perfect: corporate-action handling is occasionally
off and very illiquid scrips have gaps.  For a monthly-rebalanced momentum
backtest that is acceptable; for tick-level work it is not.  If you later get a
paid feed, the only file you need to replace is this one — everything downstream
takes plain DataFrames.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".momentum_lab_cache")
_UNIVERSE_DIR = Path(__file__).resolve().parent.parent / "universes"

# Bump this whenever the cached frame layout changes, so stale caches from an
# older version are ignored instead of silently feeding wrong columns.
CACHE_VERSION = "v2"

# What we keep per symbol on disk: unadjusted OHLCV plus the adjustment factor.
RAW_FIELDS = ["Open", "High", "Low", "Close", "Volume", "AdjFactor"]

# What the rest of the app consumes.
#   Open/High/Low/Close -> split & dividend adjusted, for returns and indicators
#   RawClose            -> the actual traded price, for price filters and market cap
#   Volume              -> actual shares traded
FIELDS = ["Open", "High", "Low", "Close", "RawClose", "Volume"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_yahoo(symbol: str) -> str:
    """NSE symbol -> Yahoo ticker.  Indices already carry '^' or a suffix."""
    s = "".join(ch for ch in symbol.strip().upper() if ch.isalnum() or ch in ".-^")
    if s.startswith("^") or s.endswith((".NS", ".BO")):
        return s
    return f"{s}.NS"


def from_yahoo(ticker: str) -> str:
    t = ticker.strip().upper()
    for suffix in (".NS", ".BO"):
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def live_last_prices(symbols: list[str]) -> pd.Series:
    """Yahoo last traded price right now — not yesterday's daily close.

    The daily cache is EOD. During market hours CMP on the positions tab has
    to come from a 5-minute bar (or fast_info), and must not be written back
    into the historical cache.
    """
    if not symbols:
        return pd.Series(dtype=float)
    import yfinance as yf

    tickers = [to_yahoo(s) for s in symbols]
    out: dict[str, float] = {}
    try:
        raw = yf.download(
            tickers=tickers if len(tickers) > 1 else tickers[0],
            period="1d",
            interval="5m",
            progress=False,
            threads=True,
            group_by="ticker",
            auto_adjust=False,
        )
        if raw is not None and len(raw):
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                for t in tickers:
                    try:
                        sub = raw[t] if t in level0 else raw.xs(t, axis=1, level=1)
                        close = pd.to_numeric(sub["Close"], errors="coerce").dropna()
                        if len(close):
                            out[from_yahoo(t)] = float(close.iloc[-1])
                    except Exception:
                        continue
            elif "Close" in raw.columns:
                close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
                if len(close):
                    out[from_yahoo(tickers[0])] = float(close.iloc[-1])
    except Exception:
        pass

    for s in symbols:
        if s in out:
            continue
        try:
            fi = yf.Ticker(to_yahoo(s)).fast_info
            px = None
            if isinstance(fi, dict):
                px = fi.get("last_price") or fi.get("lastPrice") or fi.get("regularMarketPrice")
            else:
                px = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
                if px is None:
                    try:
                        px = fi["last_price"]
                    except Exception:
                        px = None
            if px is not None and np.isfinite(float(px)) and float(px) > 0:
                out[s] = float(px)
        except Exception:
            continue
    return pd.Series(out, dtype=float)


def apply_live_mark(close: pd.DataFrame, live) -> pd.DataFrame:
    """Stamp live CMP onto the curve without erasing yesterday's close.

    P&L and drawdown must share the same last point. If we overwrite the last
    historical bar with today's live price, yesterday's peak disappears and
    max DD is understated. On a weekday, when the daily cache has not yet
    grown a bar for today, we APPEND a live mark. Weekend: leave Friday as
    the last bar and only refresh its last-traded print.
    """
    if close is None or close.empty or live is None or len(live) == 0:
        return close
    px = close.copy()
    idx = []
    for d in px.index:
        t = pd.Timestamp(d)
        try:
            if getattr(t, "tzinfo", None) is not None:
                t = t.tz_convert(None)
        except Exception:
            try:
                t = t.tz_localize(None)
            except Exception:
                pass
        idx.append(pd.Timestamp(t).normalize())
    px.index = pd.DatetimeIndex(idx)
    px = px[~px.index.duplicated(keep="last")].sort_index()

    live_map = dict(live) if not isinstance(live, dict) else live
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    last = pd.Timestamp(px.index[-1]).normalize()
    if last < now and now.weekday() < 5:
        px.loc[now] = px.iloc[-1]
        target = now
    else:
        target = last
    for sym, p in live_map.items():
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0 and sym in px.columns:
            px.loc[target, sym] = v
    return px


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "_IDX_").replace(".", "_").replace("/", "_")


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #
@dataclass
class PriceStore:
    cache_dir: str = DEFAULT_CACHE
    max_stale_days: int = 1          # re-download if cache is older than this
    offline: bool = False            # never hit the network; cache only

    def __post_init__(self) -> None:
        self.price_dir = os.path.join(self.cache_dir, CACHE_VERSION)
        os.makedirs(self.price_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "meta"), exist_ok=True)

    # ---------------------------------------------------------------- paths --
    def _path(self, ticker: str) -> str:
        return os.path.join(self.price_dir, f"{_safe_name(ticker)}.pkl")

    # ---------------------------------------------------------------- cache --
    def _read_cache(self, ticker: str) -> pd.DataFrame | None:
        p = self._path(ticker)
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_pickle(p)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            return None
        return None

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        try:
            df.to_pickle(self._path(ticker))
        except Exception:
            pass

    def clear_cache(self) -> int:
        n = 0
        for root, _dirs, files in os.walk(self.cache_dir):
            for fn in files:
                if fn.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(root, fn))
                        n += 1
                    except OSError:
                        pass
        return n

    def cache_info(self) -> pd.DataFrame:
        rows = []
        for fn in sorted(os.listdir(self.price_dir)):
            if not fn.endswith(".pkl"):
                continue
            p = os.path.join(self.price_dir, fn)
            try:
                df = pd.read_pickle(p)
                rows.append(
                    {
                        "file": fn,
                        "rows": len(df),
                        "from": df.index.min().date() if len(df) else None,
                        "to": df.index.max().date() if len(df) else None,
                        "kb": round(os.path.getsize(p) / 1024, 1),
                    }
                )
            except Exception:
                continue
        return pd.DataFrame(rows)

    # ------------------------------------------------------------- download --
    def _needs_refresh(self, cached: pd.DataFrame | None, end: pd.Timestamp) -> bool:
        if cached is None or cached.empty:
            return True
        last = cached.index.max()
        # Yahoo has no weekend bars; allow a few days of slack.
        return (end - last).days > max(self.max_stale_days, 3)

    def _download_batch(
        self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        """yfinance batch download -> {ticker: OHLCV DataFrame}."""
        import yfinance as yf

        out: dict[str, pd.DataFrame] = {}
        if not tickers:
            return out

        raw = yf.download(
            tickers=tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw is None or len(raw) == 0:
            return out

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(raw.columns.get_level_values(0))
            for t in tickers:
                try:
                    if t in level0:                       # group_by='ticker'
                        sub = raw[t]
                    else:                                  # group_by='column'
                        sub = raw.xs(t, axis=1, level=1)
                except (KeyError, IndexError):
                    continue
                sub = sub.dropna(how="all")
                if not sub.empty:
                    out[t] = _tidy(sub)
        else:
            sub = raw.dropna(how="all")
            if not sub.empty:
                out[tickers[0]] = _tidy(sub)
        return out

    # ------------------------------------------------------------------ get --
    def get(
        self,
        symbols: list[str],
        start: str | date | pd.Timestamp,
        end: str | date | pd.Timestamp,
        progress_cb=None,
        batch_size: int = 40,
    ) -> dict[str, pd.DataFrame]:
        """Return {'Close': wide_df, 'Open': ..., 'Volume': ...} for `symbols`.

        Every returned frame is indexed by date and has one column per *input*
        symbol (not the Yahoo ticker).  Symbols that failed entirely are simply
        absent — check `missing` on the result of `get_panel` if you care.
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        # Pad the start so indicators have warm-up history available.
        fetch_start = start - pd.Timedelta(days=500)

        tickers = [to_yahoo(s) for s in symbols]
        frames: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for t in tickers:
            cached = self._read_cache(t)
            if cached is not None and not self._needs_refresh(cached, end) and cached.index.min() <= fetch_start:
                frames[t] = cached
            else:
                to_fetch.append(t)

        if to_fetch and not self.offline:
            done = 0
            for i in range(0, len(to_fetch), batch_size):
                chunk = to_fetch[i : i + batch_size]
                got: dict[str, pd.DataFrame] = {}
                for attempt in range(3):
                    try:
                        got = self._download_batch(chunk, fetch_start, end)
                        break
                    except Exception:
                        time.sleep(1.5 * (attempt + 1))
                for t, df in got.items():
                    old = self._read_cache(t)
                    merged = _merge(old, df)
                    self._write_cache(t, merged)
                    frames[t] = merged
                # anything the network could not give us: fall back to stale cache
                for t in chunk:
                    if t not in frames:
                        old = self._read_cache(t)
                        if old is not None:
                            frames[t] = old
                done += len(chunk)
                if progress_cb:
                    progress_cb(min(done, len(to_fetch)), len(to_fetch))
        elif to_fetch:
            for t in to_fetch:
                old = self._read_cache(t)
                if old is not None:
                    frames[t] = old

        # -------- assemble wide frames, keyed by the ORIGINAL symbol --------- #
        panel: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
        for sym, tick in zip(symbols, tickers):
            df = frames.get(tick)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            factor = df["AdjFactor"] if "AdjFactor" in df.columns else 1.0
            # OHLC adjusted together by the same factor, so ranges stay coherent
            for f in ("Open", "High", "Low", "Close"):
                if f in df.columns:
                    panel[f][sym] = df[f] * factor
            panel["RawClose"][sym] = df["Close"]
            if "Volume" in df.columns:
                panel["Volume"][sym] = df["Volume"]

        out: dict[str, pd.DataFrame] = {}
        for f in FIELDS:
            if panel[f]:
                wide = pd.DataFrame(panel[f]).sort_index()
                wide = wide[~wide.index.duplicated(keep="last")]
                out[f] = wide.loc[wide.index <= end]
            else:
                out[f] = pd.DataFrame()
        return out

    # ------------------------------------------------------------- validate --
    def validate_symbols(
        self, symbols: list[str], start: str | pd.Timestamp = "2015-01-01"
    ) -> pd.DataFrame:
        """Try to download each symbol; report what actually came back."""
        end = pd.Timestamp.today().normalize()
        data = self.get(symbols, start, end)
        close = data.get("Close", pd.DataFrame())
        rows = []
        for s in symbols:
            if s in close.columns:
                col = close[s].dropna()
            else:
                col = pd.Series(dtype=float)
            rows.append(
                {
                    "symbol": s,
                    "yahoo": to_yahoo(s),
                    "ok": len(col) > 100,
                    "bars": len(col),
                    "first": col.index.min().date() if len(col) else None,
                    "last": col.index.max().date() if len(col) else None,
                    "last_price": round(float(col.iloc[-1]), 2) if len(col) else None,
                }
            )
        return pd.DataFrame(rows).sort_values(["ok", "bars"], ascending=[False, False])


# --------------------------------------------------------------------------- #
# frame utilities
# --------------------------------------------------------------------------- #
def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw yfinance frame and derive the adjustment factor.

    We download UNADJUSTED prices and keep the factor separately, because the
    two are needed for different jobs. A stock that did a 1:10 split shows a
    split-adjusted close of Rs 45 for a day it actually traded at Rs 450 — so a
    "price above Rs 30" screen or a market-cap calculation must use the raw
    price, while returns and moving averages must use the adjusted one.
    Conflating them is a quiet, systematic error that only bites on the names
    that had corporate actions.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.index = df.index.normalize()

    if "Adj Close" in df.columns and "Close" in df.columns:
        close = df["Close"].replace(0.0, np.nan)
        df["AdjFactor"] = (df["Adj Close"] / close).astype(float)
    else:
        df["AdjFactor"] = 1.0
    df["AdjFactor"] = df["AdjFactor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    keep = [c for c in RAW_FIELDS if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # A zero close is Yahoo noise, not a real price.
    if "Close" in df.columns:
        df = df[df["Close"].fillna(0) > 0]
    return df


def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def align_panel(
    panel: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_history_days: int = 250,
    max_missing_frac: float = 0.20,
    tail_days: int = 0,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Drop symbols with too little history, forward-fill small gaps.

    Returns (clean_panel, dropped_symbols).  Forward-filling is limited to 5
    sessions so a genuinely suspended scrip does not quietly turn into a flat
    line that the momentum ranker then loves.

    `tail_days` changes which question the gap test asks, and the difference
    matters more than it sounds.

      * **0 (the backtest)** — "is this symbol present across the whole window?"
        Right for a backtest: a stock that listed in 2024 has no 2020 to
        simulate, and pretending otherwise would put trades in years it did not
        exist.
      * **> 0 (live scanning)** — "is its most recent `tail_days` complete?"
        Right for a live scan: a company that listed eighteen months ago has a
        perfectly good chart *today*, and judging it against a six-year window
        drops it for the crime of being young. That is what was quietly removing
        real Chartink qualifiers — MUKKA, INDGN, RBZJEWEL and the rest were 58%
        to 61% NaN against a 2020 start while holding 600+ clean trading days
        of their own.

    A suspended or barely-traded scrip still fails the tail test, which is the
    thing this rule was actually protecting you from.
    """
    close = panel.get("Close", pd.DataFrame())
    if close.empty:
        return panel, []

    window = close.loc[(close.index >= start - pd.Timedelta(days=400)) & (close.index <= end)]
    dropped: list[str] = []
    keep: list[str] = []
    for c in window.columns:
        col = window[c]
        if col.dropna().shape[0] < min_history_days:
            dropped.append(c)
            continue
        in_range = (col.iloc[-int(tail_days):] if tail_days
                    else col.loc[col.index >= start])
        if len(in_range) and in_range.isna().mean() > max_missing_frac:
            dropped.append(c)
            continue
        keep.append(c)

    out: dict[str, pd.DataFrame] = {}
    for f, df in panel.items():
        if df.empty:
            out[f] = df
            continue
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub = sub.ffill(limit=5)
        out[f] = sub
    return out, dropped


def column_coverage(close: pd.DataFrame) -> pd.DataFrame:
    """First bar, last bar and how many trading days each symbol actually has.

    Taken BEFORE `align_panel` runs, so it describes what the download really
    returned rather than what survived the filters. That distinction is the
    whole point of `explain_gaps` below.
    """
    if close is None or close.empty:
        return pd.DataFrame(columns=["first", "last", "days"])
    rows = {}
    for c in close.columns:
        col = close[c].dropna()
        rows[c] = {
            "first": col.index[0].date() if len(col) else None,
            "last": col.index[-1].date() if len(col) else None,
            "days": int(len(col)),
        }
    return pd.DataFrame(rows).T


def explain_gaps(
    symbols: list[str],
    priced: list[str],
    coverage: pd.DataFrame,
    need_days: int,
) -> pd.DataFrame:
    """Why each missing symbol is missing — three different answers, not one.

    "No price history" was being said about all of them, and it was true of
    almost none. A stock that listed last year has plenty of history; it simply
    has less than the window asked for. Telling you which of the three it is
    turns a dead end into a decision.
    """
    have = set(priced)
    rows = []
    for s in symbols:
        if s in have:
            continue
        info = coverage.loc[s].to_dict() if (coverage is not None and s in coverage.index) else {}
        days = int(info.get("days") or 0)
        if days == 0:
            status, means = ("Nothing came back", "Yahoo has no data under this ticker — "
                                                  "check the symbol, or it may be renamed")
        elif days < need_days:
            status, means = ("Listed too recently",
                             f"has {days} trading days, the scan needs about {need_days}")
        else:
            status, means = ("Gaps in the recent data",
                             "suspended, or too thinly traded to price every day")
        rows.append({"symbol": s, "why": status, "first bar": info.get("first"),
                     "last bar": info.get("last"), "trading days": days,
                     "what it means": means})
    return pd.DataFrame(rows)


def trading_calendar(close: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    idx = close.index
    return idx[(idx >= start) & (idx <= end)]


def rebalance_dates(
    calendar: pd.DatetimeIndex,
    freq: str = "M",
    on_first_of_period: bool = True,
) -> list[pd.Timestamp]:
    """Pick the rebalance sessions out of the trading calendar.

    freq: 'W' weekly, 'M' monthly, 'Q' quarterly, '2W' fortnightly, 'Y' yearly.
    on_first_of_period=True  -> first trading day of each period (what most
    people actually do: "rebalance on the 1st").
    on_first_of_period=False -> last trading day of each period.
    """
    if len(calendar) == 0:
        return []
    s = pd.Series(calendar, index=calendar)
    rule = {"W": "W", "2W": "2W", "M": "ME", "Q": "QE", "Y": "YE"}.get(freq, "ME")
    try:
        grouped = s.resample(rule)
    except ValueError:  # older pandas wants 'M'/'Q'/'A'
        rule = {"ME": "M", "QE": "Q", "YE": "A"}.get(rule, rule)
        grouped = s.resample(rule)
    picked = grouped.first() if on_first_of_period else grouped.last()
    dates = [pd.Timestamp(d) for d in picked.dropna().tolist()]
    return sorted(set(dates))


def synthetic_panel(
    symbols: list[str],
    start: str = "2018-01-01",
    end: str = "2026-08-01",
    seed: int = 7,
    annual_drift: float = 0.12,
    annual_vol: float = 0.24,
) -> dict[str, pd.DataFrame]:
    """Deterministic fake OHLCV — used by the test suite and by the app's
    "Demo mode" when you have no internet.  Never used for real results."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    out_close, out_open, out_high, out_low, out_vol = {}, {}, {}, {}, {}
    for i, s in enumerate(symbols):
        drift = annual_drift * (0.4 + 0.35 * ((i % 7) - 3))
        vol = annual_vol * (0.7 + 0.12 * (i % 5))
        # a slow regime wave so cross-sectional momentum has something to find
        wave = np.sin(np.linspace(0, 2 * np.pi * (1 + i % 4), n)) * 0.0006
        rets = rng.normal(drift / 252, vol / np.sqrt(252), n) + wave
        px = 100 * np.exp(np.cumsum(rets))
        close = pd.Series(px, index=idx)
        out_close[s] = close
        out_open[s] = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.001, n))
        out_high[s] = close * (1 + np.abs(rng.normal(0, 0.004, n)))
        out_low[s] = close * (1 - np.abs(rng.normal(0, 0.004, n)))
        out_vol[s] = pd.Series(rng.integers(50_000, 5_000_000, n), index=idx).astype(float)
    return {
        "Close": pd.DataFrame(out_close),
        "RawClose": pd.DataFrame(out_close),      # no corporate actions in fake data
        "Open": pd.DataFrame(out_open),
        "High": pd.DataFrame(out_high),
        "Low": pd.DataFrame(out_low),
        "Volume": pd.DataFrame(out_vol),
    }


def synthetic_shares(symbols: list[str], seed: int = 21) -> pd.Series:
    """Fake share counts for demo mode, spread so the market-cap filter has
    something to bite on (roughly Rs 200 cr to Rs 90,000 cr at a price of 100)."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        {s: float(rng.integers(2_000_000, 900_000_000)) for s in symbols},
        name="shares_outstanding",
    )


# --------------------------------------------------------------------------- #
# shares outstanding -> historical market cap
# --------------------------------------------------------------------------- #
def fetch_shares_outstanding(
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    progress_cb=None,
    refresh_days: int = 14,
) -> pd.Series:
    """Shares outstanding per symbol, cached on disk.

    Yahoo only reliably exposes the *current* count. That means historical
    market cap here is `today's share count x that day's raw price` — it does
    not know about issuance, buybacks or QIPs that happened in between. For a
    company that doubled its share count in 2021, the 2020 market cap comes out
    roughly double what it really was.

    In practice this matters at the boundaries of a wide band (your 500 to
    50,000 crore filter) rather than in the middle of it, and Indian mid-caps
    dilute slowly enough that most names land on the right side. But it IS an
    approximation, and if a paid feed with point-in-time share counts ever
    becomes available, this is the function to replace.
    """
    os.makedirs(os.path.join(cache_dir, "meta"), exist_ok=True)
    path = os.path.join(cache_dir, "meta", "shares_outstanding.pkl")

    cached: dict[str, float] = {}
    stamped: dict[str, float] = {}
    if os.path.exists(path):
        try:
            blob = pd.read_pickle(path)
            cached = dict(blob.get("shares", {}))
            stamped = dict(blob.get("fetched_at", {}))
        except Exception:
            cached, stamped = {}, {}

    now = time.time()
    stale_after = refresh_days * 86400
    todo = [
        s for s in symbols
        if s not in cached or not np.isfinite(cached.get(s, np.nan))
        or (now - stamped.get(s, 0)) > stale_after
    ]

    if todo:
        import yfinance as yf

        for i, sym in enumerate(todo):
            shares = np.nan
            try:
                tk = yf.Ticker(to_yahoo(sym))
                try:
                    shares = float(tk.fast_info["shares"])
                except Exception:
                    info = tk.get_info() or {}
                    shares = float(
                        info.get("sharesOutstanding")
                        or info.get("impliedSharesOutstanding")
                        or np.nan
                    )
            except Exception:
                shares = np.nan
            if np.isfinite(shares) and shares > 0:
                cached[sym] = shares
                stamped[sym] = now
            if progress_cb and (i % 10 == 0 or i == len(todo) - 1):
                progress_cb(i + 1, len(todo))
        try:
            pd.to_pickle({"shares": cached, "fetched_at": stamped}, path)
        except Exception:
            pass

    return pd.Series({s: cached.get(s, np.nan) for s in symbols}, name="shares_outstanding")


def market_cap_frame(raw_close: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Market cap in Rs crore: raw price x shares outstanding / 1e7."""
    if raw_close.empty or shares.empty:
        return pd.DataFrame(index=raw_close.index, columns=raw_close.columns, dtype=float)
    sh = shares.reindex(raw_close.columns)
    return raw_close.mul(sh, axis=1) / 1e7


# --------------------------------------------------------------------------- #
# NSE index constituents
# --------------------------------------------------------------------------- #
NSE_INDEX_FILES: dict[str, str] = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty Next 50": "ind_niftynext50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "Nifty Midcap 150": "ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "ind_niftysmallcap250list.csv",
    "Nifty Microcap 250": "ind_niftymicrocap250_list.csv",
    "Nifty Total Market (750)": "ind_niftytotalmarket_list.csv",
}

ALL_NSE_LABEL = "All NSE mainboard (~2000)"
"""The whole listed board, not an index. Handled separately — it comes from a
different NSE file and is not a constituent list."""

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_index_constituents(index_name: str, timeout: int = 20) -> tuple[list[str], str]:
    """Download the current constituent list for an NSE index.

    Returns (symbols, note). On any failure it returns an empty list and a note
    explaining what went wrong, so the caller can fall back to the bundled list
    and *say so* rather than pretending the download worked.

    Important: this is the list as it stands TODAY. NSE does not publish
    historical constituents for free, so backtesting a 2020 period against
    today's index carries survivorship bias — the companies that fell out of
    the index because they did badly are simply not in your universe.
    """
    fn = NSE_INDEX_FILES.get(index_name)
    if not fn:
        return [], f"Unknown index '{index_name}'."

    url = f"https://nsearchives.nseindia.com/content/indices/{fn}"
    try:
        import requests

        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        # NSE hands out a cookie on the homepage and rejects requests without it
        try:
            sess.get("https://www.nseindia.com", timeout=timeout)
        except Exception:
            pass
        resp = sess.get(url, timeout=timeout)
        if resp.status_code != 200:
            return [], f"NSE returned HTTP {resp.status_code} for {fn}."
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))
        col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
        if col is None:
            return [], f"No 'Symbol' column in {fn}."
        syms = [str(x).strip().upper() for x in df[col].dropna()]
        syms = [s for s in syms if s and s.isascii()]
        if len(syms) < 10:
            return [], f"Only {len(syms)} symbols parsed from {fn} — looks wrong."
        return syms, f"Downloaded {len(syms)} symbols from NSE ({fn})."
    except ImportError:
        return [], "The `requests` package is not installed."
    except Exception as e:  # noqa: BLE001
        return [], f"Could not reach NSE ({type(e).__name__}). Using the bundled list."


# --------------------------------------------------------------------------- #
# the whole board, and what sector each name is in
# --------------------------------------------------------------------------- #
def fetch_all_nse_equities(timeout: int = 20) -> tuple[list[str], str]:
    """Every equity listed on the NSE mainboard, from NSE's own EQUITY_L.csv.

    Roughly 2,000 names against the 750 in the widest index. Two things to know
    before you switch to it:

    * **It is not survivorship-free either.** It is today's list. Companies
      delisted in 2022 are not in it, same as with an index.
    * **The first download is long.** Two thousand symbols at Yahoo's pace is
      tens of minutes and a few hundred megabytes of cache. After that it is
      incremental like everything else.

    Series EQ and BE only: EQ is normal rolling settlement, BE is the trade-for-
    trade segment (surveillance, usually illiquid). Everything else on that file
    — debentures, warrants, government stock — is not equity you would trade
    this system in. NSE's SME board is a separate file and is not included.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        import requests

        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        try:
            sess.get("https://www.nseindia.com", timeout=timeout)
        except Exception:
            pass
        resp = sess.get(url, timeout=timeout)
        if resp.status_code != 200:
            return [], f"NSE returned HTTP {resp.status_code} for EQUITY_L.csv."
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))
        df.columns = [str(c).strip().upper() for c in df.columns]
        if "SYMBOL" not in df.columns:
            return [], "No SYMBOL column in EQUITY_L.csv."
        keep = df
        if "SERIES" in df.columns:
            ser = df["SERIES"].astype(str).str.strip().str.upper()
            keep = df[ser.isin(["EQ", "BE"])]
        syms = [str(x).strip().upper() for x in keep["SYMBOL"].dropna()]
        syms = sorted({s for s in syms if s and s.isascii()})
        if len(syms) < 500:
            return [], f"Only {len(syms)} symbols parsed from EQUITY_L.csv — looks wrong."
        return syms, (f"Downloaded {len(syms)} mainboard equities from NSE "
                      "(series EQ and BE). The first price download will take a while.")
    except ImportError:
        return [], "The `requests` package is not installed."
    except Exception as e:                                     # noqa: BLE001
        return [], f"Could not reach NSE ({type(e).__name__})."


def index_industries(index_name: str, timeout: int = 20) -> dict[str, str]:
    """symbol -> industry, from an NSE index constituent file.

    The index CSVs carry an Industry column that EQUITY_L.csv does not, so this
    is the cheap way to sector-map most of the liquid board in one request.
    Anything it misses falls back to Yahoo, one symbol at a time.
    """
    fn = NSE_INDEX_FILES.get(index_name)
    if not fn:
        return {}
    try:
        import requests
        from io import StringIO

        sess = requests.Session()
        sess.headers.update(_NSE_HEADERS)
        try:
            sess.get("https://www.nseindia.com", timeout=timeout)
        except Exception:
            pass
        resp = None
        for base in (
            "https://nsearchives.nseindia.com/content/indices/",
            "https://archives.nseindia.com/content/indices/",
        ):
            try:
                r = sess.get(base + fn, timeout=timeout)
                if r.status_code == 200 and "Symbol" in r.text:
                    resp = r
                    break
            except Exception:
                continue
        if resp is None:
            return {}
        df = pd.read_csv(StringIO(resp.text))
        cols = {str(c).strip().lower(): c for c in df.columns}
        sym_c = cols.get("symbol")
        ind_c = cols.get("industry")
        if not sym_c or not ind_c:
            return {}
        out = {}
        for s, i in zip(df[sym_c], df[ind_c]):
            s, i = str(s).strip().upper(), str(i).strip()
            if s and i and i.lower() != "nan":
                out[s] = i
        return out
    except Exception:                                          # noqa: BLE001
        return {}


def _sector_cache_path(cache_dir: str) -> str:
    os.makedirs(os.path.join(cache_dir, "meta"), exist_ok=True)
    return os.path.join(cache_dir, "meta", "sectors.pkl")


def bundled_sectors() -> dict[str, str]:
    """Offline NSE industry map shipped with the app (Nifty Total Market + extras)."""
    out: dict[str, str] = {}
    for name in ("sectors.json", "sectors_extra.json"):
        p = _UNIVERSE_DIR / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out.update({str(k).strip().upper(): str(v).strip()
                            for k, v in data.items() if k and v})
        except Exception:
            continue
    return out


def fetch_sectors(
    symbols: list[str],
    cache_dir: str = DEFAULT_CACHE,
    seed: dict[str, str] | None = None,
    offline: bool = False,
    progress_cb=None,
) -> dict[str, str]:
    """symbol -> sector, cached on disk forever (sectors do not move).

    `seed` is anything already known — typically the Industry column off an NSE
    index file — and is written straight into the cache. Whatever is left over
    is looked up on Yahoo one symbol at a time, which is why this is only ever
    called on the handful of names that qualified in a week rather than on the
    whole universe.

    A symbol Yahoo has no sector for is cached as "" so the lookup is not
    retried on every rerun, and is reported as *Unknown* rather than guessed.
    """
    cache: dict[str, str] = {}
    p = _sector_cache_path(cache_dir)
    if os.path.exists(p):
        try:
            cache = dict(pd.read_pickle(p).get("sectors", {}))
        except Exception:
            cache = {}

    dirty = False
    # bundled NSE map first, then the live seed, then disk — never let an empty
    # failed Yahoo lookup hide a name we already know
    for s, sec in {**bundled_sectors(), **(seed or {})}.items():
        s = str(s).strip().upper()
        if sec and cache.get(s) != sec:
            cache[s] = sec
            dirty = True

    todo = [s for s in symbols if not str(cache.get(s) or "").strip()]
    if todo and not offline:
        try:
            import yfinance as yf
        except Exception:
            todo = []
            yf = None
        if todo and yf is not None:
            for n, sym in enumerate(todo):
                sec = ""
                try:
                    tk = yf.Ticker(to_yahoo(sym))
                    info = {}
                    try:
                        info = tk.get_info() or {}
                    except Exception:
                        info = getattr(tk, "info", None) or {}
                    sec = str(info.get("sector") or info.get("industry") or "").strip()
                except Exception:
                    sec = ""
                if sec:
                    cache[sym] = sec
                    dirty = True
                if progress_cb and (n % 5 == 0 or n == len(todo) - 1):
                    progress_cb(n + 1, len(todo))

    if dirty:
        try:
            pd.to_pickle({"sectors": cache, "saved_at": time.time()}, p)
        except Exception:
            pass

    return {s: cache.get(s, "") for s in symbols}
