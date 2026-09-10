"""
The breakout screen itself: a fresh N-week high, generalised from your Chartink scan.

Your scan is six stacked conditions:

    weekly close        >  max(52 weekly closes) as of 1 week ago
    close 1 week ago    <  max(52 weekly closes) as of 2 weeks ago
    close 2 weeks ago   <  max(52 weekly closes) as of 3 weeks ago
    ... and so on for 3, 4, 5 weeks ago

The first line is "new 52-week high". The other five say "and it was NOT already
making new highs for the previous five weeks" — i.e. this is the *first* break
in at least six weeks, not week nine of an extended run. `lookback_weeks`
replaces every 52 with 100 / 150 / 200; `fresh_weeks` replaces the five.

Everything here works on wide DataFrames (date x symbol) so the whole universe
is evaluated in one pass, and every statistic is causal — a value on week W uses
only bars up to W.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# weekly resampling
# --------------------------------------------------------------------------- #
def to_weekly(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Daily wide panel -> weekly wide panel, bars labelled by their Friday.

    Open  = first trading day's open of that week (the Monday you can act on)
    Close = last trading day's close (the Friday that confirms the signal)
    """
    out: dict[str, pd.DataFrame] = {}
    close = panel.get("Close", pd.DataFrame())
    if close.empty:
        return {k: pd.DataFrame() for k in ("Open", "High", "Low", "Close", "RawClose", "Volume")}

    rules = {
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "RawClose": "last", "Volume": "sum",
    }
    for field_name, how in rules.items():
        df = panel.get(field_name)
        if df is None or df.empty:
            out[field_name] = pd.DataFrame()
            continue
        out[field_name] = df.resample("W-FRI").agg(how)
    return out


def weekly_ema(weekly_close: pd.DataFrame, span: int) -> pd.DataFrame:
    """EMA of the weekly close, per symbol."""
    return weekly_close.ewm(span=span, adjust=False, ignore_na=True).mean()


# --------------------------------------------------------------------------- #
# the scan
# --------------------------------------------------------------------------- #
@dataclass
class BreakoutConfig:
    lookback_weeks: int = 52          # 52 / 100 / 150 / 200
    fresh_weeks: int = 5              # the "wasn't already at a high" stack
    ema_fast: int = 20                # weekly EMA used as the trailing stop
    ema_slow: int = 50                # weekly EMA used as the final stop
    require_above_fast_ema: bool = False   # optional extra confirmation

    # technical score weights (auto-normalised)
    w_freshness: float = 0.0          # smaller break above the old high scores higher
    w_volume: float = 0.0             # bigger volume surge scores higher
    w_momentum: float = 1.0           # stronger N-week return scores higher

    # how the technical and fundamental scores are blended into the final rank.
    w_technical: float = 0.5
    w_fundamental: float = 0.5

    rank_fundamentals: bool = True
    """Percentile-rank the fundamentals score inside the week's pool before blending.

    Without this the slider lies. The technical score is a percentile, so in a
    six-name pool it is spread 0/20/40/60/80/100 by construction — a standard
    deviation of about 34. Real fundamental scores clump: on a representative
    set they run 42 to 77, a standard deviation of about 11. Blending an
    always-wide number with an always-narrow one at "50/50" gives the technical
    side roughly three quarters of the say, whatever the slider reads.

    Ranking both sides the same way fixes that: 50 then means 50. The cost is
    that the fundamental component becomes relative — in a week where every
    candidate is mediocre, one of them still ranks top on fundamentals. That is
    why the absolute score is kept and shown alongside, so "ranked first, and
    only 45 out of 100" stays visible.
    """

    def label(self) -> str:
        return f"Fresh {self.lookback_weeks}-week high (first break in {self.fresh_weeks + 1} weeks)"


@dataclass
class BreakoutSignals:
    fresh: pd.DataFrame               # week x symbol, True = fresh N-week-high breakout
    prior_high: pd.DataFrame          # the N-week high it broke (excluding this week)
    extension_pct: pd.DataFrame       # how far above that high the close is, %
    momentum_pct: pd.DataFrame        # return over lookback_weeks, %
    ema_fast: pd.DataFrame
    ema_slow: pd.DataFrame
    weekly: dict[str, pd.DataFrame] = field(default_factory=dict)
    trail_emas: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Averages a booked position can trail on, all on the DAILY grid.

    Keyed by the name the exit ladder uses:
      "wema10" — 10 EMA of the WEEKLY close, forward-filled onto daily dates
      "dema20" / "dema50" — 20 / 50 EMA of the daily close

    Daily, because the moved-up stop is checked on the daily close. The weekly
    one is forward-filled rather than interpolated: on a Wednesday you know last
    Friday's weekly EMA and nothing newer, which is exactly what the fill gives
    you. Every value on day D uses only bars up to D."""
    daily_ok: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Week x symbol: does the stock pass the daily-timeframe filters?"""


def compute_signals(panel: dict[str, pd.DataFrame], cfg: BreakoutConfig) -> BreakoutSignals:
    """Evaluate the whole breakout stack across the universe, week by week."""
    weekly = to_weekly(panel)
    wc = weekly.get("Close", pd.DataFrame())
    if wc.empty:
        empty = pd.DataFrame()
        return BreakoutSignals(empty, empty, empty, empty, empty, empty, weekly)

    n = max(2, int(cfg.lookback_weeks))
    k = max(0, int(cfg.fresh_weeks))

    # highest weekly close over the trailing n weeks, EXCLUDING the current week
    prior_high = wc.shift(1).rolling(n, min_periods=n).max()

    fresh = wc > prior_high
    # ...and it was not already breaking out in any of the previous k weeks
    for j in range(1, k + 1):
        fresh &= wc.shift(j) < prior_high.shift(j)
    fresh = fresh.fillna(False)

    ema_f = weekly_ema(wc, cfg.ema_fast)
    ema_s = weekly_ema(wc, cfg.ema_slow)

    dc = panel.get("Close", pd.DataFrame())
    trail_emas = {
        # the weekly 10 EMA is known only once a week closes, so it is carried
        # forward across the following days rather than recomputed intraweek
        "wema10": weekly_ema(wc, 10).reindex(dc.index, method="ffill"),
        "dema20": dc.ewm(span=20, adjust=False, ignore_na=True).mean(),
        "dema50": dc.ewm(span=50, adjust=False, ignore_na=True).mean(),
    }

    if cfg.require_above_fast_ema:
        fresh &= wc > ema_f

    extension = (wc - prior_high) / prior_high * 100.0
    momentum = (wc / wc.shift(n) - 1.0) * 100.0

    return BreakoutSignals(
        fresh=fresh, prior_high=prior_high, extension_pct=extension,
        momentum_pct=momentum, ema_fast=ema_f, ema_slow=ema_s, weekly=weekly,
        trail_emas=trail_emas,
    )


# --------------------------------------------------------------------------- #
# scoring: which five, when more than five qualify
# --------------------------------------------------------------------------- #
def volume_surge_daily(panel: dict[str, pd.DataFrame], period: int = 50) -> pd.DataFrame:
    """Daily volume / its own SMA(period), as a daily wide frame."""
    vol = panel.get("Volume", pd.DataFrame())
    if vol.empty:
        return pd.DataFrame()
    avg = vol.rolling(period, min_periods=max(2, period // 2)).mean()
    return vol / avg.replace(0.0, np.nan)


def score_week(
    sig: BreakoutSignals,
    week: pd.Timestamp,
    candidates: list[str],
    cfg: BreakoutConfig,
    vol_surge_weekly: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank one week's qualifying names — every one of them, best to worst.

    There is no minimum pool size: two candidates get ranked 1st and 2nd exactly
    the way two hundred do.

    Three scores come back, all on 0..100:

    * **technical_score** — the chart. Freshness, volume surge and momentum, each
      percentile-ranked *within that week's candidate pool* so the weights mean
      what they say whatever the absolute numbers are. Freshness is inverted: a
      stock 1% above its old high scores higher than one 9% above, because the 9%
      one has already made most of the move you were trying to catch.
    * **fundamentals_score** — the business, from `core.fundamentals`. Absolute,
      not relative: 70 means the same thing in a thin week as in a crowded one.
      Pass `fundamentals` as a frame indexed by symbol with a
      `fundamentals_score` column. Leave it out and this is NaN.
    * **combined_score** — `w_technical x technical + w_fundamental x fundamentals`,
      renormalised over whichever of the two is present. With no fundamentals it
      is just the technical score, so the ranking is unchanged from before.

    A name whose fundamentals are missing keeps its technical score rather than
    being pushed to the bottom: an unfiled small cap is not a bad business, it is
    an unknown one. `fundamentals_available` says which is which.
    """
    if not candidates:
        return pd.DataFrame()

    def row(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or week not in df.index:
            return pd.Series(np.nan, index=candidates, dtype=float)
        return df.loc[week].reindex(candidates).astype(float)

    ext = row(sig.extension_pct)
    mom = row(sig.momentum_pct)
    vol = row(vol_surge_weekly) if vol_surge_weekly is not None else pd.Series(np.nan, index=candidates)

    out = pd.DataFrame({
        "extension_%": ext,
        "volume_surge": vol,
        "momentum_%": mom,
        "close": row(sig.weekly.get("Close")),
        "ema_fast": row(sig.ema_fast),
        "ema_slow": row(sig.ema_slow),
        "prior_high": row(sig.prior_high),
    })

    # percentile ranks, 0..1
    r_fresh = (-out["extension_%"]).rank(pct=True)          # smaller extension = better
    r_vol = out["volume_surge"].rank(pct=True)
    r_mom = out["momentum_%"].rank(pct=True)

    # a missing component scores neutral rather than knocking the name out
    r_fresh = r_fresh.fillna(0.5)
    r_vol = r_vol.fillna(0.5)
    r_mom = r_mom.fillna(0.5)

    w = np.array([cfg.w_freshness, cfg.w_volume, cfg.w_momentum], dtype=float)
    if w.sum() <= 0:
        w = np.array([1 / 3, 1 / 3, 1 / 3])
    w = w / w.sum()

    tech = r_fresh * w[0] + r_vol * w[1] + r_mom * w[2]

    out["technical_score"] = tech * 100.0
    out["rank_freshness"] = r_fresh
    out["rank_volume"] = r_vol
    out["rank_momentum"] = r_mom

    # ---- the business, if we were given it ---------------------------------
    fund = pd.Series(np.nan, index=out.index, dtype=float)
    good = pd.Series("", index=out.index, dtype=object)
    if fundamentals is not None and not fundamentals.empty:
        f = fundamentals.reindex(out.index)
        if "fundamentals_score" in f.columns:
            fund = pd.to_numeric(f["fundamentals_score"], errors="coerce")
        if "what is good" in f.columns:
            good = f["what is good"].fillna("").astype(str)
    out["fundamentals_score"] = fund
    out["fundamentals_available"] = fund.notna()
    out["what is good"] = good

    have_f = fund.notna()
    # What actually goes into the blend. Ranked, the two sides have the same
    # spread and the weight means what it says; unranked, the absolute score
    # goes in as-is and the technical side dominates. See BreakoutConfig.
    if cfg.rank_fundamentals and int(have_f.sum()) > 1:
        used = fund.rank(pct=True) * 100.0
    else:
        used = fund
    out["fundamentals_used"] = used

    wt, wf = float(cfg.w_technical), float(cfg.w_fundamental)
    if wt + wf <= 0:
        wt, wf = 1.0, 0.0
    combined = out["technical_score"].copy()
    if wf > 0 and have_f.any():
        blend = (out["technical_score"] * wt + used * wf) / (wt + wf)
        combined = combined.where(~have_f, blend)
    out["combined_score"] = combined

    # kept so older callers and saved journals do not break
    out["score"] = out["combined_score"] / 100.0
    return out.sort_values("combined_score", ascending=False)


def qualifying_at(
    sig: BreakoutSignals,
    week: pd.Timestamp,
    screen_ok: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    """Names whose breakout fired in `week`, intersected with the screen."""
    if sig.fresh.empty or week not in sig.fresh.index:
        return []
    row = sig.fresh.loc[week]
    names = [s for s in row.index[row.astype(bool)]]
    if screen_ok is not None:
        ok = set(screen_ok)
        names = [s for s in names if s in ok]
    if exclude:
        names = [s for s in names if s not in exclude]
    return names


# --------------------------------------------------------------------------- #
# regime filter
# --------------------------------------------------------------------------- #
@dataclass
class RegimeConfig:
    enabled: bool = True
    use_ema: bool = True
    ema_period: int = 50              # on the weekly close
    use_sma: bool = True
    sma_period: int = 200             # on the daily close
    mode: str = "any"                 # "any" = block if any check fails; "all" = block only if all fail


def regime_blocked(
    bench_daily: pd.Series,
    week: pd.Timestamp,
    cfg: RegimeConfig,
) -> tuple[bool, dict]:
    """Should new entries be blocked in `week`?

    Checks the benchmark's weekly close against its EMA and its daily close
    against its SMA, using only bars up to that week.
    """
    detail: dict[str, object] = {}
    if not cfg.enabled or bench_daily is None or bench_daily.empty:
        return False, detail

    hist = bench_daily.loc[bench_daily.index <= week].dropna()
    if hist.empty:
        return False, detail

    checks: list[bool] = []

    if cfg.use_ema:
        wk = hist.resample("W-FRI").last().dropna()
        if len(wk) >= cfg.ema_period:
            ema = wk.ewm(span=cfg.ema_period, adjust=False).mean()
            below = bool(wk.iloc[-1] < ema.iloc[-1])
            checks.append(below)
            detail[f"below weekly EMA{cfg.ema_period}"] = below
            detail["close"] = float(wk.iloc[-1])
            detail[f"EMA{cfg.ema_period}"] = float(ema.iloc[-1])

    if cfg.use_sma:
        if len(hist) >= cfg.sma_period:
            sma = hist.rolling(cfg.sma_period).mean()
            below = bool(hist.iloc[-1] < sma.iloc[-1])
            checks.append(below)
            detail[f"below daily SMA{cfg.sma_period}"] = below
            detail[f"SMA{cfg.sma_period}"] = float(sma.iloc[-1])

    if not checks:
        return False, detail

    blocked = any(checks) if cfg.mode == "any" else all(checks)
    return blocked, detail


# --------------------------------------------------------------------------- #
# daily-timeframe filters
# --------------------------------------------------------------------------- #
@dataclass
class DailyFilterConfig:
    """Higher-timeframe confirmation, on DAILY bars.

    Two independent switches:

      trend — the daily close against its 20 EMA, 50 EMA and 200 SMA. Each
              average has its OWN tick, so you can require price above just the
              20 EMA, or any combination:

                above_ema_fast : close > 20 EMA
                above_ema_mid  : close > 50 EMA
                above_sma_long : close > 200 SMA

              plus one further condition of a different kind:

                full_stack     : close > 20 EMA > 50 EMA > 200 SMA

              The stack is not the same as ticking all three. Those say price is
              above every average; the stack additionally says the averages
              themselves are in order, which is the difference between "price
              popped above a tangle of flat averages" and "the whole structure
              points up". The stack implies all three, so ticking everything is
              the stack.

              Nothing is implied: an unticked average is not tested at all.

      rsi   — daily RSI, with an *above* level and a *below* level as separate
              toggles. Turn on just one for a floor or a ceiling; turn on both
              for a band (e.g. above 50 and below 80 = strong but not blown off).

    Everything is read on the same day the weekly signal is read — the last
    trading day of the signal week — and acted on at the following Monday's
    open, exactly like every other rule here. No look-ahead, no part-formed bar.
    """

    use_trend: bool = False
    ema_fast: int = 20
    ema_slow: int = 50
    sma_long: int = 200
    above_ema_fast: bool = True
    above_ema_mid: bool = False
    above_sma_long: bool = False
    require_full_stack: bool = False

    use_rsi_above: bool = False
    rsi_above: float = 60.0
    use_rsi_below: bool = False
    rsi_below: float = 80.0
    rsi_period: int = 14

    def label(self) -> str:
        bits = []
        if self.use_trend:
            names = []
            if self.above_ema_fast:
                names.append(f"{self.ema_fast}EMA")
            if self.above_ema_mid:
                names.append(f"{self.ema_slow}EMA")
            if self.above_sma_long:
                names.append(f"{self.sma_long}SMA")
            if names:
                bits.append("daily close > " + "/".join(names))
            if self.require_full_stack:
                bits.append(f"close > {self.ema_fast}EMA > {self.ema_slow}EMA > "
                            f"{self.sma_long}SMA")
        if self.use_rsi_above:
            bits.append(f"daily RSI > {self.rsi_above:g}")
        if self.use_rsi_below:
            bits.append(f"daily RSI < {self.rsi_below:g}")
        return " and ".join(bits) if bits else "off"


def rsi(series: pd.Series | pd.DataFrame, period: int = 14):
    """Wilder's RSI. Works column-wise on a DataFrame."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # all-gains window: RSI is 100, not NaN
    return out.where(roll_down.ne(0) | roll_up.eq(0), 100.0)


def daily_filter_mask(
    panel: dict[str, pd.DataFrame],
    weekly_index: pd.Index,
    cfg: DailyFilterConfig,
) -> tuple[pd.DataFrame, dict]:
    """Week x symbol boolean: does each stock pass the daily filters that week?

    Returns (mask, info). `info` carries the coverage numbers so the caller can
    show them: a 200-period SMA on DAILY bars needs 200 trading days (about ten
    months), and a stock without that much history would otherwise be rejected
    for missing data rather than for its trend.

    The daily frame is reduced to the last trading day of each week and put on
    the weekly grid directly — so the week whose Friday close fires the signal
    reads that same Friday's daily values. There is no forward fill across
    weeks: a week with no data is False, not "whatever last week said".
    """
    close = panel.get("Close", pd.DataFrame())
    info: dict = {"days_available": 0, "sma_long_needs": cfg.sma_long,
                  "symbols": 0, "with_sma_long": 0, "coverage_pct": 100.0, "warnings": []}
    if close.empty or not (cfg.use_trend or cfg.use_rsi_above or cfg.use_rsi_below):
        return pd.DataFrame(True, index=weekly_index, columns=close.columns), info

    info["days_available"] = int(len(close))
    info["symbols"] = int(close.shape[1])
    ok = pd.DataFrame(True, index=close.index, columns=close.columns)

    if cfg.use_trend:
        e_f = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        e_s = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        s_l = close.rolling(cfg.sma_long, min_periods=cfg.sma_long).mean()

        uses_long = cfg.above_sma_long or cfg.require_full_stack
        have = int(s_l.notna().any().sum())
        info["with_sma_long"] = have
        info["coverage_pct"] = round(have / max(close.shape[1], 1) * 100, 1)
        if not uses_long:
            pass
        elif len(close) < cfg.sma_long:
            info["warnings"].append(
                f"The daily {cfg.sma_long} SMA needs {cfg.sma_long} trading days "
                f"(~{cfg.sma_long / 252:.1f} years) and your data has only {len(close)}. "
                "No stock can pass this filter — extend the date range, or lower the "
                "SMA period."
            )
        elif have < close.shape[1]:
            info["warnings"].append(
                f"Only {have} of {close.shape[1]} stocks have {cfg.sma_long} days of history. "
                f"The other {close.shape[1] - have} are excluded for missing data, not for "
                "their trend."
            )

        if cfg.above_ema_fast:
            ok &= close > e_f
        if cfg.above_ema_mid:
            ok &= close > e_s
        if cfg.above_sma_long:
            ok &= close > s_l
        if cfg.require_full_stack:
            ok &= (close > e_f) & (e_f > e_s) & (e_s > s_l)

    if cfg.use_rsi_above or cfg.use_rsi_below:
        r = rsi(close, cfg.rsi_period)
        if len(close) < cfg.rsi_period + 1:
            info["warnings"].append(
                f"Daily RSI({cfg.rsi_period}) needs {cfg.rsi_period + 1} daily bars; "
                f"your data has {len(close)}."
            )
        if cfg.use_rsi_above:
            ok &= r > cfg.rsi_above
        if cfg.use_rsi_below:
            ok &= r < cfg.rsi_below

    ok = ok.fillna(False)
    weekly = ok.resample("W-FRI").last()
    mask = weekly.reindex(weekly_index).fillna(False).astype(bool)
    info["pass_rate_pct"] = round(float(mask.to_numpy().mean()) * 100, 1)
    return mask, info


# --------------------------------------------------------------------------- #
# sector diversification
# --------------------------------------------------------------------------- #
def diversify_picks(
    scored: pd.DataFrame,
    n: int,
    sectors: dict[str, str] | None = None,
    max_per_sector: int = 2,
    enabled: bool = True,
    max_promote_rank: int | None = None,
) -> pd.DataFrame:
    """Take `n` names off a ranked frame without letting one sector own the week.

    A fresh-breakout scan is not sector-neutral by nature — when capital goods
    run, twelve capital goods names break out in the same week and a pure score
    ranking buys eight of them. That is one bet in eight positions, and it is the
    difference between a bad month and a bad quarter.

    The rule is deliberately simple and visible. Walk the ranking from the top;
    take a name if its sector is not already full. If the cap leaves you short of
    `n`, do a second pass and fill the rest by score, cap ignored, rather than
    handing back an under-filled list — a diversification rule that stops you
    deploying capital has cost you more than the concentration would have.

    Two columns come back so nothing is hidden:

    * ``rank`` — where the name sat on pure score. A 12 in a list of 10 means it
      was promoted past names you did not buy.
    * ``pick`` — ``{div}`` when it only got in because higher-ranked names were
      already sector-full, ``{cap}`` when the cap had to be broken to fill the
      list, blank when it would have been picked anyway.

    A name with no sector on file is exempt from the cap rather than punished by
    it — missing data is not evidence of concentration — and is marked ``{?}``.

    ``max_promote_rank`` is the floor under all of this, and it matters more than
    the cap does. Diversification may only reach down to that rank; below it the
    stock is simply not good enough to own, whatever sector it is in. Thirty names
    qualify, the 28th has a poor chart and poor numbers — buying it to balance a
    sector is a worse decision than holding a third capital-goods name. Defaults
    to ``2 * n``, so a list of ten never reaches past rank twenty.
    """
    if scored is None or scored.empty or n <= 0:
        return scored.head(0) if scored is not None else pd.DataFrame()

    out = scored.copy()
    out["rank"] = np.arange(1, len(out) + 1)
    sec = {s: str((sectors or {}).get(s, "") or "").strip() for s in out.index}
    out["sector"] = [sec[s] or "Unknown" for s in out.index]

    if not enabled or max_per_sector <= 0:
        picks = out.head(n).copy()
        picks["pick"] = ""
        return picks

    depth = int(max_promote_rank) if max_promote_rank else n * 2
    depth = max(depth, n)                # never shallower than the list itself

    counts: dict[str, int] = {}
    chosen: list[str] = []
    notes: dict[str, str] = {}

    for pos, sym in enumerate(out.index, start=1):
        if len(chosen) >= n:
            break
        if pos > depth:
            break                        # past here the name is not worth owning
        s = sec[sym]
        if not s:                                   # unknown sector: never capped
            chosen.append(sym)
            notes[sym] = "{?}"
            continue
        if counts.get(s, 0) < max_per_sector:
            counts[s] = counts.get(s, 0) + 1
            chosen.append(sym)
            notes[sym] = ""

    # the cap left the list short — fill by score and say so
    if len(chosen) < n:
        for sym in out.index:
            if len(chosen) >= n:
                break
            if sym in notes:
                continue
            chosen.append(sym)
            notes[sym] = "{cap}"

    picks = out.loc[chosen].copy()
    # anything that got in above a name it did not outrank was promoted by the cap
    cutoff = int(picks["rank"].max()) if len(picks) else 0
    skipped = [s for s in out.index[:cutoff] if s not in notes]
    if skipped:
        first_skipped = int(out.at[skipped[0], "rank"])
        for sym in picks.index:
            if not notes[sym] and int(picks.at[sym, "rank"]) > first_skipped:
                notes[sym] = "{div}"
    picks["pick"] = [notes[s] for s in picks.index]
    return picks.sort_values("rank")
