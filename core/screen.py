"""
Point-in-time universe screening.

This is the piece that answers "March 2020 me is condition ke hisab se jo stocks
the, unse; May 2021 me us waqt jo the, unse."

The idea, and why it matters
----------------------------
Most retail backtests pick a universe once — usually today's index, or a list of
names the author already knows did well — and run history through it. That is
two separate biases stacked on top of each other:

* **Survivorship bias.** Companies that collapsed, got delisted, or fell out of
  the index are missing. Your 2020 universe is quietly pre-filtered for
  "survived until 2026", which no strategy could have known in 2020.
* **Look-ahead universe bias.** A stock with a Rs 40,000 crore market cap today
  might have been a Rs 300 crore microcap in 2020 — below the floor your screen
  says you would have required. Including it means you backtested a rule you
  would not actually have followed.

The fix implemented here: evaluate the screen **on every rebalance date, using
only data available up to that date**, and let the universe be whatever passed.
The universe therefore breathes — it might be 180 names in March 2020 and 340 in
May 2021 — which is exactly what would have happened in real life.

What this does NOT fix
----------------------
The candidate pool still comes from a list of symbols you supply, and if that
list is today's Nifty Total Market then genuinely delisted companies were never
in it to be screened out. That residual bias needs a paid point-in-time
constituent feed. The app says so on screen rather than pretending otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScreenConfig:
    """A Chartink-style screen, evaluated point-in-time.

    Every threshold is optional; switch on only what you actually want.
    """

    enabled: bool = False

    # ---- market cap (Rs crore) ----
    use_market_cap: bool = False
    mcap_min_cr: float = 500.0
    mcap_max_cr: float = 50_000.0

    # ---- market cap RANK (the AMFI-style definition of large/mid/small) ----
    # Rank 1 is the biggest company in your universe on that date. AMFI calls
    # 1-100 large cap, 101-250 mid cap and 251 onwards small cap. Ranking beats a
    # fixed rupee band because the band goes stale: Rs 20,000 crore was a mid cap
    # in 2019 and is a small cap now, so a fixed threshold quietly changes what
    # it selects as the years pass. The rank is recomputed on every bar, so a
    # stock that grew out of the small-cap bucket leaves the universe by itself.
    use_mcap_rank: bool = False
    mcap_rank_min: int = 251
    mcap_rank_max: int = 10_000

    # ---- market cap PERCENTILE ----
    # Same idea as the rank, but relative to however many stocks you loaded, so
    # it cannot silently select nothing. 0% = the biggest company, 100% = the
    # smallest. "Bottom half by market cap" is 50-100.
    use_mcap_pct: bool = False
    mcap_pct_min: float = 50.0
    mcap_pct_max: float = 100.0

    # ---- price (raw, unadjusted) ----
    use_price: bool = False
    price_min: float = 30.0
    price_max: float = 1_000_000.0

    # ---- volume ----
    use_volume: bool = False
    volume_min: float = 25_000.0

    use_avg_volume: bool = False
    avg_volume_period: int = 50
    avg_volume_min: float = 50_000.0

    # ---- rupee turnover (Rs crore, median over the window) ----
    use_turnover: bool = False
    turnover_period: int = 20
    turnover_min_cr: float = 1.0

    # ---- number of trades (needs the NSE bhavcopy source) ----
    use_trades: bool = False
    trades_period: int = 1            # 1 = that day; >1 = rolling median
    trades_min: float = 400.0

    # ---- delivery percentage (needs the NSE bhavcopy source) ----
    use_delivery: bool = False
    delivery_period: int = 20
    delivery_min_pct: float = 30.0

    # ---- listing age ----
    use_min_history: bool = True
    min_history_days: int = 252       # a year of bars before it can be picked

    def active_rules(self) -> list[str]:
        out = []
        if self.use_market_cap:
            out.append(f"Market cap {self.mcap_min_cr:,.0f}–{self.mcap_max_cr:,.0f} cr")
        if self.use_mcap_rank:
            out.append(f"Market-cap rank {self.mcap_rank_min}–{self.mcap_rank_max}")
        if self.use_mcap_pct:
            out.append(f"Market-cap percentile {self.mcap_pct_min:g}–{self.mcap_pct_max:g}%")
        if self.use_price:
            out.append(f"Price {self.price_min:,.0f}–{self.price_max:,.0f}")
        if self.use_volume:
            out.append(f"Volume ≥ {self.volume_min:,.0f}")
        if self.use_avg_volume:
            out.append(f"SMA({self.avg_volume_period}) volume ≥ {self.avg_volume_min:,.0f}")
        if self.use_turnover:
            out.append(f"Turnover ≥ {self.turnover_min_cr:,.1f} cr")
        if self.use_trades:
            out.append(f"Trades ≥ {self.trades_min:,.0f}")
        if self.use_delivery:
            out.append(f"Delivery ≥ {self.delivery_min_pct:,.0f}%")
        if self.use_min_history:
            out.append(f"≥ {self.min_history_days} bars of history")
        return out


@dataclass
class ScreenResult:
    mask: pd.DataFrame                       # date x symbol, True = qualifies
    counts: pd.Series                        # how many qualified each day
    rule_counts: pd.DataFrame                # date x rule, how many each rule passed
    missing_inputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def at(self, asof: pd.Timestamp) -> list[str]:
        """Symbols that qualified on (or on the last bar before) `asof`."""
        if self.mask.empty:
            return []
        idx = self.mask.index[self.mask.index <= asof]
        if len(idx) == 0:
            return []
        row = self.mask.loc[idx[-1]]
        return list(row.index[row.astype(bool)])


def build_screen(
    panel: dict[str, pd.DataFrame],
    cfg: ScreenConfig,
    market_cap: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    delivery_pct: pd.DataFrame | None = None,
) -> ScreenResult:
    """Evaluate the screen on every bar.

    Every input is a causal rolling statistic — nothing here can see forward.
    A rule whose data is missing is reported in `missing_inputs` and skipped;
    it never silently passes everything, because a filter you think is running
    and isn't is worse than no filter at all.
    """
    close = panel.get("Close", pd.DataFrame())
    if close.empty:
        return ScreenResult(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(),
                            ["no price data"], [])

    raw_close = panel.get("RawClose")
    if raw_close is None or raw_close.empty:
        raw_close = close
    else:
        raw_close = raw_close.reindex_like(close)
    volume = panel.get("Volume")
    if volume is None or volume.empty:
        volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    else:
        volume = volume.reindex_like(close)

    mask = pd.DataFrame(True, index=close.index, columns=close.columns)
    rule_counts: dict[str, pd.Series] = {}
    missing: list[str] = []
    notes: list[str] = []

    def apply(rule_mask: pd.DataFrame, label: str) -> None:
        nonlocal mask
        rm = rule_mask.reindex_like(mask).fillna(False)
        rule_counts[label] = rm.sum(axis=1)
        mask = mask & rm

    # -- a bar must exist at all ------------------------------------------ #
    has_bar = close.notna()
    apply(has_bar, "Has a price")

    # -- listing age -------------------------------------------------------- #
    if cfg.use_min_history:
        bars_so_far = close.notna().cumsum()
        apply(bars_so_far >= cfg.min_history_days, f"≥{cfg.min_history_days} bars")

    # -- market cap ---------------------------------------------------------- #
    if cfg.use_market_cap:
        if market_cap is None or market_cap.empty or market_cap.isna().all().all():
            missing.append("market cap (no shares-outstanding data)")
        else:
            mc = market_cap.reindex_like(close)
            known = mc.notna()
            band = (mc >= cfg.mcap_min_cr) & (mc <= cfg.mcap_max_cr)
            covered = float(known.any().mean()) if known.size else 0.0
            if covered < 0.999:
                notes.append(
                    f"Shares outstanding missing for {(1 - covered) * 100:.0f}% of symbols — "
                    "those are excluded by the market-cap rule rather than waved through."
                )
            apply(band & known, "Market cap band")

    # -- market-cap rank (large / mid / small, recomputed every bar) --------- #
    if cfg.use_mcap_rank:
        if market_cap is None or market_cap.empty or market_cap.isna().all().all():
            missing.append("market-cap rank (no shares-outstanding data)")
        else:
            mc = market_cap.reindex_like(close)
            # rank 1 = largest company that day, among the symbols you supplied
            rank = mc.rank(axis=1, ascending=False, method="first")
            in_band = (rank >= cfg.mcap_rank_min) & (rank <= cfg.mcap_rank_max)
            apply(in_band & mc.notna(), f"Rank {cfg.mcap_rank_min}-{cfg.mcap_rank_max}")
            notes.append(
                f"Market-cap rank is computed within the {mc.shape[1]} symbols you loaded, not "
                "the whole market. Load a broad list (Nifty Total Market 750) if you want the "
                "ranks to mean what AMFI means by large/mid/small."
            )

    # -- market-cap percentile (robust to any universe size) ----------------- #
    if cfg.use_mcap_pct:
        if market_cap is None or market_cap.empty or market_cap.isna().all().all():
            missing.append("market-cap percentile (no shares-outstanding data)")
        else:
            mc = market_cap.reindex_like(close)
            # 0 = biggest company that day, 100 = smallest
            pct = mc.rank(axis=1, ascending=False, method="first", pct=True) * 100.0
            apply((pct >= cfg.mcap_pct_min) & (pct <= cfg.mcap_pct_max) & mc.notna(),
                  f"Cap pct {cfg.mcap_pct_min:g}-{cfg.mcap_pct_max:g}")

    # -- raw price ----------------------------------------------------------- #
    if cfg.use_price:
        apply((raw_close >= cfg.price_min) & (raw_close <= cfg.price_max), "Price band")

    # -- volume -------------------------------------------------------------- #
    if cfg.use_volume:
        apply(volume >= cfg.volume_min, "Volume")

    if cfg.use_avg_volume:
        n = max(2, int(cfg.avg_volume_period))
        sma_vol = volume.rolling(n, min_periods=max(2, n // 2)).mean()
        apply(sma_vol >= cfg.avg_volume_min, f"SMA({n}) volume")

    # -- rupee turnover ------------------------------------------------------ #
    if cfg.use_turnover:
        n = max(2, int(cfg.turnover_period))
        turn = (raw_close * volume / 1e7).rolling(n, min_periods=max(2, n // 2)).median()
        apply(turn >= cfg.turnover_min_cr, f"Turnover ≥{cfg.turnover_min_cr}cr")

    # -- number of trades ---------------------------------------------------- #
    if cfg.use_trades:
        if trades is None or trades.empty:
            missing.append("number of trades (NSE bhavcopy source is off or failed)")
        else:
            tr = trades.reindex_like(close)
            if cfg.trades_period > 1:
                tr = tr.rolling(int(cfg.trades_period),
                                min_periods=max(2, int(cfg.trades_period) // 2)).median()
            apply(tr >= cfg.trades_min, f"Trades ≥{cfg.trades_min:,.0f}")

    # -- delivery percentage -------------------------------------------------- #
    if cfg.use_delivery:
        if delivery_pct is None or delivery_pct.empty:
            missing.append("delivery % (NSE bhavcopy source is off or failed)")
        else:
            dl = delivery_pct.reindex_like(close)
            n = max(1, int(cfg.delivery_period))
            if n > 1:
                dl = dl.rolling(n, min_periods=max(2, n // 2)).mean()
            apply(dl >= cfg.delivery_min_pct, f"Delivery ≥{cfg.delivery_min_pct:.0f}%")

    counts = mask.sum(axis=1).rename("qualifying")
    return ScreenResult(
        mask=mask,
        counts=counts,
        rule_counts=pd.DataFrame(rule_counts),
        missing_inputs=missing,
        notes=notes,
    )


def universe_timeline(result: ScreenResult, rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """How the qualifying universe changed rebalance by rebalance.

    Entries and exits are the interesting part: a screen that churns 40% of its
    universe every month is describing noise, not a stable investable set.
    """
    rows = []
    prev: set[str] = set()
    for d in rebalance_dates:
        names = set(result.at(d))
        rows.append(
            {
                "date": d,
                "Qualifying": len(names),
                "Entered": len(names - prev),
                "Left": len(prev - names),
                "Churn %": (len(names ^ prev) / max(len(prev), 1) * 100) if prev else np.nan,
            }
        )
        prev = names
    return pd.DataFrame(rows)


def explain_at(
    panel: dict[str, pd.DataFrame],
    cfg: ScreenConfig,
    asof: pd.Timestamp,
    market_cap: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    delivery_pct: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-symbol values of every screened quantity on one date, so you can see
    exactly why something did or didn't make the cut."""
    close = panel.get("Close", pd.DataFrame())
    if close.empty:
        return pd.DataFrame()
    idx = close.index[close.index <= asof]
    if len(idx) == 0:
        return pd.DataFrame()
    day = idx[-1]

    raw_close = panel.get("RawClose")
    raw_close = close if raw_close is None or raw_close.empty else raw_close.reindex_like(close)
    volume = panel.get("Volume")
    volume = (pd.DataFrame(np.nan, index=close.index, columns=close.columns)
              if volume is None or volume.empty else volume.reindex_like(close))

    n_av = max(2, int(cfg.avg_volume_period))
    n_to = max(2, int(cfg.turnover_period))
    out = pd.DataFrame(index=close.columns)
    out["Price (raw)"] = raw_close.loc[day]
    if market_cap is not None and not market_cap.empty:
        mc_all = market_cap.reindex_like(close)
        out["Market cap (cr)"] = mc_all.loc[day]
        out["Cap rank"] = mc_all.rank(axis=1, ascending=False, method="first").loc[day]
        out["Cap pct"] = (mc_all.rank(axis=1, ascending=False, method="first",
                                       pct=True) * 100).round(1).loc[day]
    out["Volume"] = volume.loc[day]
    out[f"SMA{n_av} volume"] = volume.rolling(n_av, min_periods=2).mean().loc[day]
    out["Turnover (cr)"] = (raw_close * volume / 1e7).rolling(n_to, min_periods=2).median().loc[day]
    if trades is not None and not trades.empty:
        out["Trades"] = trades.reindex_like(close).loc[day]
    if delivery_pct is not None and not delivery_pct.empty:
        out["Delivery %"] = delivery_pct.reindex_like(close).loc[day]
    out["Bars of history"] = close.notna().cumsum().loc[day]

    res = build_screen(panel, cfg, market_cap, trades, delivery_pct)
    out["Qualifies"] = res.mask.loc[day].reindex(out.index).fillna(False)
    return out.sort_values("Qualifies", ascending=False).round(2)
