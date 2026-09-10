"""
Reading a Chartink export.

Chartink runs your scan against the whole NSE board and hands you a CSV. This
module takes ONE thing out of that file: the list of symbols. Nothing else.

That restraint is the point. The CSV also carries a price, a % change and a
volume, and every one of them is a snapshot of whenever you pressed Download —
possibly mid-session, possibly from a different day than the weekly close the
app reasons about. Ranking, position sizing and stop levels all have to come
from the app's own priced data or the buy list stops matching the backtest.
So the file answers exactly one question — *which stocks qualified* — and the
rest of the pipeline is untouched.

The parser is deliberately loose about shape, because a Chartink export is not
a stable format: the column set changes with whatever you added to your scan,
there is usually a trailing empty "Add Column", numbers arrive as text with
Indian digit grouping ("18,82,75,533"), and a copy-paste out of the browser is
neither of those. So:

  * a column named like a symbol column wins ("Symbol", "NSE Code", "Ticker"…);
  * failing that, the column whose values most look like NSE tickers wins;
  * failing even that — a plain list pasted one-per-line — the whole text is
    read as symbols.

A file that yields nothing says so rather than returning an empty list that
looks like "no stocks qualified this week".
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

# Header names that mean "this column holds the symbol", lower-cased and
# stripped of anything that is not a letter.
SYMBOL_HEADERS: tuple[str, ...] = (
    "symbol", "symbols", "nsecode", "nsesymbol", "nsccode", "code", "ticker",
    "tickers", "scrip", "scripcode", "scripname", "stocksymbol", "sym",
    "tradingsymbol", "instrument", "security", "securityid",
)

# Header names that are definitely NOT the symbol, even if their values happen
# to look like tickers. "Stock Name" is the trap: "SBC Exports Ltd" is close
# enough to a ticker for a heuristic to fall for it.
NON_SYMBOL_HEADERS: tuple[str, ...] = (
    "sr", "srno", "sno", "no", "stockname", "companyname", "name", "company",
    "links", "link", "url", "close", "price", "ltp", "open", "high", "low",
    "volume", "change", "chg", "pchange", "percentchange", "addcolumn",
    "marketcap", "mcap", "sector", "industry", "date",
)

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:[&\-][A-Z0-9]+)*$")
"""What an NSE symbol looks like: capitals and digits, & and - allowed inside.
M&M, BAJAJ-AUTO, NIFTY50 all pass; "SBC Exports Ltd" does not, because of the
spaces, which is exactly the distinction the heuristic needs."""

MAX_SYMBOL_LEN = 25


@dataclass
class ImportResult:
    """What came out of the file, and everything that was thrown away."""

    symbols: list[str] = field(default_factory=list)
    column: str = ""                       # which column the symbols came from
    rows: int = 0                          # rows in the file
    duplicates: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)   # values that were not symbols
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def ok(self) -> bool:
        return bool(self.symbols) and not self.error


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm_header(h: object) -> str:
    return re.sub(r"[^a-z]", "", str(h).lower())


def clean_symbol(value: object) -> str:
    """One cell to one NSE symbol, or "" if it plainly is not one.

    Handles the three prefixes/suffixes that turn up in exports and pastes:
    an exchange prefix (``NSE:TITAN``), a Yahoo suffix (``TITAN.NS``) and stray
    quoting or whitespace. Everything else is left alone — this normalises, it
    does not guess.
    """
    s = str(value).strip().strip('"\'').strip()
    if not s or s.lower() in {"nan", "none", "-", "--"}:
        return ""
    if ":" in s:                                   # NSE:TITAN / NSE : TITAN
        s = s.split(":")[-1].strip()
    s = s.upper()
    for suffix in (".NS", ".NSE", ".BO", "-EQ", "-BE"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = s.strip()
    if not s or len(s) > MAX_SYMBOL_LEN:
        return ""
    return s if TICKER_RE.match(s) else ""


def _looks_like_symbols(values: pd.Series) -> float:
    """Fraction of non-empty cells in a column that parse as NSE symbols."""
    vals = [str(v) for v in values if str(v).strip() and str(v).strip().lower() != "nan"]
    if not vals:
        return 0.0
    return sum(1 for v in vals if clean_symbol(v)) / len(vals)


def _pick_column(df: pd.DataFrame) -> tuple[str, str]:
    """(column, why). Named columns win; otherwise the most ticker-like one."""
    for c in df.columns:
        if _norm_header(c) in SYMBOL_HEADERS:
            return c, f"used the “{c}” column"

    best, best_score = "", 0.0
    for c in df.columns:
        if _norm_header(c) in NON_SYMBOL_HEADERS:
            continue
        score = _looks_like_symbols(df[c])
        if score > best_score:
            best, best_score = c, score
    if best and best_score >= 0.6:
        return best, (f"no column is named “Symbol”, so “{best}” was used — "
                      f"{best_score:.0%} of its values look like NSE symbols")
    return "", ""


# --------------------------------------------------------------------------- #
# the two entry points
# --------------------------------------------------------------------------- #
def read_csv(data: bytes | str, filename: str = "") -> ImportResult:
    """Parse a Chartink CSV (or anything close enough to one)."""
    res = ImportResult()
    if isinstance(data, bytes):
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            res.error = "That file is not text this app can read."
            return res
    else:
        text = str(data)

    if not text.strip():
        res.error = "The file is empty."
        return res

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                         skip_blank_lines=True)
    except Exception:                                          # noqa: BLE001
        # not a table at all — treat it as a pasted list
        return read_text(text)

    # drop the columns Chartink leaves behind with nothing in them
    df = df.loc[:, [c for c in df.columns if str(df[c].astype(str).str.strip().str.len().sum())]]
    df = df.dropna(how="all")
    res.rows = len(df)
    if df.empty:
        res.error = "The file has headers but no rows."
        return res

    col, why = _pick_column(df)
    if not col:
        if df.shape[1] == 1:
            # a one-column file is a list; pandas will have eaten its first row
            # as the header, so put that back if it is itself a symbol
            head = df.columns[0]
            vals = list(df[head])
            if clean_symbol(head):
                vals.insert(0, head)
            res.column = "single column"
            return _finish(res, vals, filename)
        # A real table with no symbol column anywhere. Saying so beats scraping
        # words out of the company names, which is what a text fallback would do.
        res.error = ("No **Symbol** column found in that file — the columns are: "
                     + ", ".join(str(c) for c in df.columns)
                     + ". Export the Chartink scan results, which carry one.")
        return res
    res.column = col
    if why:
        res.notes.append(why)
    return _finish(res, list(df[col]), filename)


def read_text(text: str) -> ImportResult:
    """Parse a pasted list — one per line, or comma / space separated."""
    res = ImportResult()
    parts = [p for p in re.split(r"[\s,;|]+", str(text)) if p.strip()]
    res.rows = len(parts)
    res.column = "pasted list"
    return _finish(res, parts, "")


def _finish(res: ImportResult, raw: list, filename: str) -> ImportResult:
    seen: set[str] = set()
    for value in raw:
        sym = clean_symbol(value)
        if not sym:
            v = str(value).strip()
            if v and v.lower() != "nan":
                res.rejected.append(v)
            continue
        if sym in seen:
            res.duplicates.append(sym)
            continue
        seen.add(sym)
        res.symbols.append(sym)

    if not res.symbols:
        res.error = ("No NSE symbols found in that file. Chartink's export has a "
                     "**Symbol** column — check you exported the scan results and "
                     "not something else.")
        return res
    if filename:
        res.notes.insert(0, f"{filename}: {len(res.symbols)} symbols")
    if res.duplicates:
        res.notes.append(f"{len(res.duplicates)} duplicate row(s) collapsed: "
                         + ", ".join(sorted(set(res.duplicates))[:8]))
    if res.rejected:
        shown = ", ".join(res.rejected[:5])
        more = f" and {len(res.rejected) - 5} more" if len(res.rejected) > 5 else ""
        res.notes.append(f"{len(res.rejected)} value(s) skipped — not symbols: {shown}{more}")
    return res


# --------------------------------------------------------------------------- #
def split_by_coverage(symbols: list[str], priced: list[str]) -> tuple[list[str], list[str]]:
    """(have prices, no prices).

    A symbol with no price history cannot be ranked, sized or given a stop, so
    it is reported rather than silently dropped: a name missing from Yahoo is
    usually a recent listing or a ticker that was renamed, and you want to know
    which it was.
    """
    have = set(priced)
    return [s for s in symbols if s in have], [s for s in symbols if s not in have]
