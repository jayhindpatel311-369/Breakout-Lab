"""
Which index a stock belongs to.

Two different questions get asked of this file.

*Where am I concentrated right now?* — you hold fourteen positions and want to
know how many are large caps you already own through a mutual fund and how many
are microcaps nobody has heard of. That is a **size bucket**, and NSE's own
buckets are a clean partition of the top 750: Nifty 50, Next 50 (51-100),
Midcap 150 (101-250), Smallcap 250 (251-500), Microcap 250 (501-750). A stock
outside all five is outside the Total Market index entirely — which, on a scan
that runs over the whole ~2,000-name board, is where a lot of them live.

*Which end of the market actually pays me?* — the same bucket applied to closed
trades in the journal.

**The honest limit, and it is a real one.** NSE publishes only TODAY's
constituent lists; historical membership is not free. So a stock bought in 2023
is bucketed by where it sits now, not where it sat then — and a stock that ran
hard is exactly the one most likely to have moved up a bucket since. That
biases the journal's per-bucket read towards the larger buckets. Nothing here
can fix that; the app says so wherever these numbers are shown.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import date

from . import data as data_mod
from . import universe as uni_mod

# The five mutually exclusive size buckets, narrowest first. Order matters:
# a symbol is assigned to the FIRST one it appears in.
SIZE_BUCKETS: tuple[str, ...] = (
    "Nifty 50",
    "Nifty Next 50",
    "Nifty Midcap 150",
    "Nifty Smallcap 250",
    "Nifty Microcap 250",
)

# The umbrella indices. Not buckets — they overlap the ones above — but worth
# knowing ("is this in the Nifty 500?" is a question people actually ask).
UMBRELLA_INDICES: tuple[str, ...] = (
    "Nifty 100", "Nifty 200", "Nifty 500", "Nifty Total Market (750)",
)

ALL_INDICES: tuple[str, ...] = SIZE_BUCKETS + UMBRELLA_INDICES

OUTSIDE = "Outside the top 750"
"""The bucket for everything the Total Market index does not reach. On the full
NSE board that is roughly 1,250 of the 2,000 names, so it is a normal answer,
not an error."""

REFRESH_DAYS = 7
"""Index constituents are reviewed twice a year, so a week-old list is never
meaningfully wrong — and a stale list is far better than no list."""


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str) -> str:
    os.makedirs(os.path.join(cache_dir, "meta"), exist_ok=True)
    return os.path.join(cache_dir, "meta", "index_members.pkl")


def load_cache(cache_dir: str = data_mod.DEFAULT_CACHE) -> dict:
    path = _cache_path(cache_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return blob if isinstance(blob, dict) else {}
    except Exception:                                          # noqa: BLE001
        return {}


def save_cache(blob: dict, cache_dir: str = data_mod.DEFAULT_CACHE) -> None:
    try:
        with open(_cache_path(cache_dir), "wb") as f:
            pickle.dump(blob, f)
    except Exception:                                          # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# the lists
# --------------------------------------------------------------------------- #
def index_sets(
    cache_dir: str = data_mod.DEFAULT_CACHE,
    offline: bool = False,
    refresh_days: int = REFRESH_DAYS,
    progress_cb=None,
) -> tuple[dict[str, set[str]], list[str]]:
    """{index name -> set of symbols}, plus a note per index about where it came from.

    Downloads what is stale, keeps what is fresh, and falls back to the bundled
    lists for anything NSE will not serve — so this never returns nothing, and
    always says which of the three happened.
    """
    blob = load_cache(cache_dir)
    fetched: dict[str, float] = dict(blob.get("_fetched") or {})
    sets: dict[str, set[str]] = {k: set(v) for k, v in (blob.get("sets") or {}).items()}
    notes: list[str] = []
    now = time.time()
    changed = False

    for name in ALL_INDICES:
        age_days = (now - fetched.get(name, 0.0)) / 86_400
        if name in sets and sets[name] and age_days < refresh_days:
            continue
        if offline:
            if name not in sets or not sets[name]:
                sets[name] = set(uni_mod.clean_symbols(
                    list(uni_mod.LIVE_INDEX_FALLBACKS.get(name, []))))
                notes.append(f"{name}: bundled list ({len(sets[name])})")
            continue
        if progress_cb:
            progress_cb(name)
        syms, _note = data_mod.fetch_index_constituents(name)
        if syms:
            sets[name] = set(uni_mod.clean_symbols(list(syms)))
            fetched[name] = now
            changed = True
        elif name not in sets or not sets[name]:
            sets[name] = set(uni_mod.clean_symbols(
                list(uni_mod.LIVE_INDEX_FALLBACKS.get(name, []))))
            notes.append(f"{name}: NSE did not answer, using the bundled list "
                         f"({len(sets[name])})")
            changed = True

    if changed:
        save_cache({"sets": {k: sorted(v) for k, v in sets.items()},
                    "_fetched": fetched, "_saved": str(date.today())}, cache_dir)
    return sets, notes


# --------------------------------------------------------------------------- #
# the two questions
# --------------------------------------------------------------------------- #
def bucket_of(symbol: str, sets: dict[str, set[str]]) -> str:
    """The one size bucket this stock sits in, or OUTSIDE."""
    for name in SIZE_BUCKETS:
        if symbol in sets.get(name, ()):
            return name
    return OUTSIDE


def buckets(symbols, sets: dict[str, set[str]]) -> dict[str, str]:
    return {s: bucket_of(s, sets) for s in symbols}


def memberships(symbols, sets: dict[str, set[str]]) -> dict[str, list[str]]:
    """Every index a stock is in, size buckets first — for a tooltip or a detail row."""
    out = {}
    for s in symbols:
        out[s] = [n for n in ALL_INDICES if s in sets.get(n, ())]
    return out


def bucket_order(values) -> list[str]:
    """Buckets in market-cap order for a table, with OUTSIDE last.

    Sorting these alphabetically puts Microcap above Nifty 50, which reads as
    nonsense in a summary, so the order is fixed here once.
    """
    order = list(SIZE_BUCKETS) + [OUTSIDE]
    seen = set(values)
    return [b for b in order if b in seen] + sorted(b for b in seen if b not in order)
