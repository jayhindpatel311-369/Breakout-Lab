"""
The exit ladder and position sizing.

Your rule, as given:

    book 20% of the position at +25% gain
    book 20% more at +50% gain
    book 45% when the weekly close breaks below the 20 EMA
    book the last 15% when the weekly close breaks below the 50 EMA

Those add to 100%, which works cleanly when the trade goes up first and rolls
over later. The case your rule doesn't cover is a breakout that fails
immediately: if price falls under the 50 EMA having never hit +25%, selling
"15%" would leave 85% of a losing position open below its own final stop.

So the last rung is treated as terminal — it exits **whatever is still open**.
On a winner that walked up the ladder that is exactly your 15%; on a failed
breakout it is the whole remaining position, which is what a stop-loss is for.
Every percentage, trigger and the order of the rungs is editable in the app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


# --------------------------------------------------------------------------- #
# ladder definition
# --------------------------------------------------------------------------- #
TRIGGER_LABELS = {
    "gain_pct": "Profit target",
    "below_ema_fast": "Closed below 20 EMA",
    "below_ema_slow": "Closed below 50 EMA",
}


TRAIL_MODES = {
    "": "leave the stop alone",
    "cost": "cost (breakeven)",
    "wema10": "weekly 10 EMA",
    "dema20": "daily 20 EMA",
    "dema50": "daily 50 EMA",
    "pct": "buy price + a fixed %",
}


@dataclass
class Rung:
    key: str                    # stable id, used in the journal
    trigger: str                # "gain_pct" | "below_ema_fast" | "below_ema_slow"
    value: float = 0.0          # gain % for "gain_pct"; ignored otherwise
    book_pct: float = 20.0      # % of the ORIGINAL quantity
    terminal: bool = False      # True = sell everything still open
    full_exit_if_no_profit: bool = False
    """Sell the whole position if no profit rung has fired yet.

    This is what makes the 20 EMA behave as a real stop-loss on a breakout that
    failed. Once the trade has actually paid something — the first profit target
    has booked — the same rung goes back to being a partial trail (`book_pct`).
    """

    trail_to: str = ""
    """Where to move the stop for the REMAINING quantity once this rung books.

    One of TRAIL_MODES. "" leaves the stop alone; "pct" uses `trail_pct` as a
    percentage ABOVE the buy price, so +10 means the rest of the trade is now
    protected at entry x 1.10.

    A rung set to book 0% with a trail set is a pure stop-move instruction:
    price reaches the level, nothing is sold, the stop moves.

    Worth knowing before switching these on. A stop moved to cost is not a free
    option — you trade a smaller average loss for a higher chance of being shaken
    out of a winner before it works, and on a breakout system that shake-out rate
    is high. And the floor is checked on the WEEKLY close like every other rung,
    so an intraweek dip through it that recovers by Friday does not take you out.
    Run it both ways rather than assuming it must help.
    """
    trail_pct: float = 0.0

    def describe(self, as_stop: bool = False) -> str:
        if self.trigger == "gain_pct":
            what = f"at +{self.value:g}% gain"
        elif self.trigger == "below_ema_fast":
            what = "on a weekly close below the 20 EMA"
        else:
            what = "on a weekly close below the 50 EMA"
        if as_stop:
            return f"stop-loss: exit 100% {what}"
        how = "exit everything still open" if self.terminal else f"book {self.book_pct:g}%"
        base = f"{how} {what}"
        if self.trail_to:
            base += f", then stop at {self.trail_label()}"
        return base

    def trail_label(self) -> str:
        if self.trail_to == "pct":
            return f"buy price +{self.trail_pct:g}%"
        return TRAIL_MODES.get(self.trail_to, self.trail_to)


def default_ladder() -> list[Rung]:
    """Four profit rungs plus the two EMA rungs.

    Rungs 3 and 4 book 0% by default, which makes them inert — so out of the box
    this is exactly the two-rung ladder this app has always had. Give them a
    quantity (or a stop move) to switch them on.
    """
    return [
        Rung("profit_1", "gain_pct", 25.0, 20.0),
        Rung("profit_2", "gain_pct", 50.0, 20.0),
        Rung("profit_3", "gain_pct", 75.0, 0.0),
        Rung("profit_4", "gain_pct", 100.0, 0.0),
        Rung("trail_ema_fast", "below_ema_fast", 0.0, 45.0, full_exit_if_no_profit=True),
        Rung("trail_ema_slow", "below_ema_slow", 0.0, 15.0, terminal=True),
    ]


def hard_stop_price(entry_price: float, pct: float) -> float:
    """The price implied by "no stop further than pct% below the buy price"."""
    if not pct or pct <= 0 or entry_price <= 0:
        return 0.0
    return float(entry_price) * (1 - float(pct) / 100.0)


def entry_stop(ema_stop: float, entry_price: float, max_distance_pct: float) -> float:
    """The stop that actually governs a new position — the TIGHTER of the two.

    Higher price = closer to the entry = tighter. This is also what position
    sizing must use: sizing off a 20 EMA that is 30% away while a 20% cap is
    armed would compute a position far larger than the risk really being taken.
    """
    hs = hard_stop_price(entry_price, max_distance_pct)
    if hs <= 0:
        return ema_stop
    if not ema_stop or ema_stop <= 0 or pd.isna(ema_stop):
        return hs
    return max(float(ema_stop), hs)


def check_hard_stop(pos: "Position", close: float) -> dict | None:
    """Has the fixed % stop been broken on this close? Exits everything open.

    Runs alongside the weekly 20 EMA rung rather than instead of it — whichever
    level is breached first ends the trade. Checked on the DAILY close, because a
    stop whose whole job is to cap the loss is not something to look at once a
    week; the 20 EMA stays weekly because it is a weekly-chart rule.
    """
    if not pos.is_open() or getattr(pos, "hard_stop", 0.0) <= 0:
        return None
    if pd.isna(close) or float(close) >= pos.hard_stop:
        return None
    return {
        "qty": pos.open_qty,
        "rung": "hard_stop",
        "reason": f"closed below the fixed stop at Rs {pos.hard_stop:,.2f}",
    }


def _trail_rank(mode: str, pct: float) -> float:
    """A comparable height for a stop floor, so a later rung can only tighten it.

    The moving-average floors have no fixed price at the moment a rung fires, so
    they are ranked just above cost. On a trade that has already booked a profit
    target they normally sit above the entry, and treating them as the loosest of
    the "real" floors is the conservative reading — it will not silently displace
    a locked-in +15%.
    """
    if not mode:
        return float("-inf")
    if mode == "cost":
        return 0.0
    if mode in ("wema10", "dema20", "dema50"):
        return 0.5
    if mode == "pct":
        return 1.0 + float(pct)
    return float("-inf")


def trail_floor_price(pos: "Position", levels: dict | None) -> float:
    """The price floor a booked rung has moved the stop to, or 0 if none.

    "cost" and "pct" are fixed the moment the rung fires; the moving averages
    move every week, which is why this is recomputed rather than stored.
    """
    mode = getattr(pos, "trail_mode", "") or ""
    if not mode:
        return 0.0
    if mode == "cost":
        return float(pos.cost_basis())
    if mode == "pct":
        return float(pos.entry_price) * (1 + getattr(pos, "trail_pct", 0.0) / 100.0)
    v = (levels or {}).get(mode)
    if v is None or pd.isna(v):
        return 0.0
    return float(v)


def check_trail_stop(pos: "Position", close: float, levels: dict | None) -> dict | None:
    """Has the moved-up stop been broken on this weekly close?

    Exits everything still open. Checked BEFORE the ladder each week, because
    once you have said "the rest of this trade rides at breakeven", that
    instruction outranks a profit target the stock is no longer near.

    A floor whose average is missing that week (not enough history yet) returns
    0 and simply does not fire — a missing reading must not be read as a break.
    """
    if not pos.is_open() or not getattr(pos, "trail_mode", ""):
        return None
    floor = trail_floor_price(pos, levels)
    if floor <= 0 or pd.isna(close) or float(close) >= floor:
        return None
    return {
        "qty": pos.open_qty,
        "rung": f"trail_{pos.trail_mode}",
        "reason": f"stop had been moved to {pos.trail_label()} (Rs {floor:,.2f})",
    }


def _any_profit_booked(pos: "Position", ladder: list[Rung]) -> bool:
    return any(r.key in pos.done for r in ladder if r.trigger == "gain_pct")


# --------------------------------------------------------------------------- #
# a live position
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    symbol: str
    entry_date: object                  # pd.Timestamp / date
    entry_price: float
    qty: int                            # original quantity
    open_qty: int                       # what is still held
    initial_stop: float                 # weekly 20 EMA at entry — used for risk sizing
    done: list[str] = field(default_factory=list)   # rung keys already fired
    fills: list[dict] = field(default_factory=list)  # every exit, for the journal
    entry_score: float = float("nan")
    avg_cost: float = 0.0        # price paid; kept as its own field for P&L
    trail_mode: str = ""         # set by a rung's trail_to once that rung books
    trail_pct: float = 0.0
    hard_stop: float = 0.0       # entry x (1 - max_stop_distance_pct/100); 0 = off
    demerged_from: str = ""
    """Set on a position that arrived through a demerger, naming the parent.

    A demerger does not make or lose money — it splits one cost basis across two
    listed companies. Keeping the link means the journal can show what the pair
    did together, which is the only reading that answers "was that holding worth
    it": the parent alone will look like a loss the day the child lists, and the
    child alone will look like a windfall, and neither is true.
    """

    decision_price: float = 0.0
    """The price the buy DECISION was made at — Friday's close on the buy list.

    `entry_price` is what you actually filled at on Monday. Keeping both is the
    only way to measure slippage honestly: the gap between the two is the cost
    of the weekend, and it is a real cost that the backtest cannot see. 0 means
    the position was added by hand and there was no decision price.
    """

    def slippage_pct(self) -> float:
        """Fill vs Friday decision, in %. Positive = you paid LESS than the list (good for a buy)."""
        if self.decision_price <= 0:
            return float("nan")
        return (self.decision_price - self.entry_price) / self.decision_price * 100

    def is_open(self) -> bool:
        return self.open_qty > 0

    def cost(self) -> float:
        return self.qty * self.cost_basis()

    def cost_basis(self) -> float:
        """Price paid. Falls back to the entry price for books written earlier."""
        return self.avg_cost if self.avg_cost > 0 else self.entry_price

    def trail_label(self) -> str:
        if self.trail_mode == "pct":
            return f"buy price +{self.trail_pct:g}%"
        return TRAIL_MODES.get(self.trail_mode, self.trail_mode or "—")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["entry_date"] = str(pd.Timestamp(self.entry_date).date())
        return d

    @staticmethod
    def from_dict(d: dict) -> "Position":
        d = dict(d)
        d["entry_date"] = pd.Timestamp(d["entry_date"])
        d.setdefault("entry_score", float("nan"))
        d.setdefault("avg_cost", 0.0)
        d.setdefault("trail_mode", "")
        d.setdefault("trail_pct", 0.0)
        d.setdefault("hard_stop", 0.0)
        d.setdefault("decision_price", 0.0)
        d.setdefault("demerged_from", "")
        # a book written when this app still had pyramids carries adds_done /
        # stop_mode / stop_pct; the filter below drops them rather than failing,
        # because a journal you cannot open is worse than one that loses a
        # feature that no longer exists.
        known = {f for f in Position.__dataclass_fields__}
        return Position(**{k: v for k, v in d.items() if k in known})


def next_trigger(pos: Position, ladder: list[Rung], ema_fast: float, ema_slow: float) -> str:
    """What this position is waiting for: the next profit target and the live stop.

    Both matter at once — a position is always simultaneously working towards its
    next target and sitting above its stop — so both are shown.
    """
    booked = _any_profit_booked(pos, ladder)
    parts: list[str] = []

    for r in ladder:
        if r.key in pos.done or r.trigger != "gain_pct":
            continue
        if float(r.book_pct or 0) <= 0:
            continue
        target = pos.entry_price * (1 + r.value / 100.0)
        parts.append(f"book {r.book_pct:g}% at ₹{target:,.2f} (+{r.value:g}%)")
        break

    for r in ladder:
        if r.key in pos.done or r.trigger != "below_ema_fast":
            continue
        if pd.isna(ema_fast):
            break
        if r.full_exit_if_no_profit and not booked:
            parts.append(f"SL: exit 100% below ₹{ema_fast:,.2f} (20 EMA)")
        else:
            parts.append(f"trim {r.book_pct:g}% below ₹{ema_fast:,.2f} (20 EMA)")
        break

    if not parts:
        for r in ladder:
            if r.key in pos.done or r.trigger != "below_ema_slow":
                continue
            if pd.notna(ema_slow):
                parts.append(f"exit rest below ₹{ema_slow:,.2f} (50 EMA)")
            break

    return " · ".join(parts) if parts else "—"


def check_position(
    pos: Position,
    week: pd.Timestamp,
    close: float,
    ema_fast: float,
    ema_slow: float,
    ladder: list[Rung],
    rearm_ema: bool = False,
    only_triggers: set[str] | None = None,
) -> list[dict]:
    """Run one bar past one position. Mutates `pos`; returns the exits fired.

    `only_triggers` restricts which KIND of rung is checked on this bar, so the
    profit targets can run on daily closes while the EMA rungs run on the weekly
    close. The FULL ladder is still passed in either case, because
    `full_exit_if_no_profit` has to know whether any profit rung has booked — and
    filtering the list instead would make the 20 EMA look like a full stop-loss
    on a trade that had in fact already taken profit.

    Rungs are checked in order and each fires at most once, but more than one can
    fire in the same week — a stock that gaps from +10% to +60% books both profit
    rungs on that bar rather than pretending it stopped to breathe at +25%.

    `rearm_ema` decides what a *trailing* stop means after it has fired once.
    Off (the default, and the literal reading of your rule): the 20 EMA rung
    books its 45% once and is spent — if the stock breaks the EMA, recovers, runs
    up and rolls over again, nothing further happens until the 50 EMA gives way.
    On: closing back above the EMA re-arms that rung, so a later break books
    again. Which you want depends on whether you read "45% on a close below the
    20 EMA" as one scheduled sale or as a stop that follows the trade.
    """
    if not pos.is_open():
        return []

    if rearm_ema and (only_triggers is None
                      or {"below_ema_fast", "below_ema_slow"} & only_triggers):
        if pd.notna(ema_fast) and close > ema_fast and "trail_ema_fast" in pos.done:
            pos.done.remove("trail_ema_fast")
        if pd.notna(ema_slow) and close > ema_slow and "trail_ema_slow" in pos.done:
            pos.done.remove("trail_ema_slow")

    gain_pct = (close / pos.entry_price - 1.0) * 100.0
    fired: list[dict] = []

    for r in ladder:
        if r.key in pos.done or pos.open_qty <= 0:
            continue
        if only_triggers is not None and r.trigger not in only_triggers:
            continue

        if r.trigger == "gain_pct":
            hit = gain_pct >= r.value
        elif r.trigger == "below_ema_fast":
            hit = pd.notna(ema_fast) and close < ema_fast
        elif r.trigger == "below_ema_slow":
            hit = pd.notna(ema_slow) and close < ema_slow
        else:
            hit = False

        if not hit:
            continue

        as_stop = False
        if r.terminal:
            qty = pos.open_qty
        elif r.full_exit_if_no_profit and not _any_profit_booked(pos, ladder):
            # nothing has been booked yet, so this rung IS the stop-loss
            qty = pos.open_qty
            as_stop = True
        else:
            qty = min(int(math.floor(pos.qty * r.book_pct / 100.0)), pos.open_qty)

        pos.done.append(r.key)

        # The stop moves whether or not this rung had shares left to sell: the
        # instruction is "once price reaches this level, protect the rest", and a
        # rung set to book 0% is a pure stop-move. Only ever tighten — a later
        # rung must not loosen a floor an earlier one already set.
        if r.trail_to:
            if (not pos.trail_mode
                    or _trail_rank(r.trail_to, r.trail_pct)
                    > _trail_rank(pos.trail_mode, pos.trail_pct)):
                pos.trail_mode = r.trail_to
                pos.trail_pct = float(r.trail_pct)

        if qty <= 0:
            continue

        pos.open_qty -= qty
        fired.append({
            "week": week, "symbol": pos.symbol, "rung": r.key,
            "reason": r.describe(as_stop), "trigger": r.trigger,
            "qty": qty, "gain_pct": gain_pct, "signal_close": close,
            "ema_fast": ema_fast, "ema_slow": ema_slow, "as_stop": as_stop,
        })
        if pos.open_qty <= 0:
            break

    return fired


# --------------------------------------------------------------------------- #
# sizing
# --------------------------------------------------------------------------- #
@dataclass
class SizingConfig:
    """How many shares to buy.

    Three modes:
      "fixed" — a rupee amount per stock. ₹1,00,000 each on ₹50,00,000 of capital
                means up to 50 positions at once, and every open trade is the
                same size. With `compound` on (the default) that amount is really
                a *ratio*: ₹1L of ₹50L is 2%, so when equity reaches ₹60L a new
                entry is ₹1.2L — the slot count stays 50 and the book keeps
                growing with itself. Turn compounding off to keep it at a literal
                ₹1L forever. No caps are applied either way.
      "equal" — a percentage of capital in every stock. The rupee amount then
                moves with your equity (if compounding is on), which is why two
                trades years apart can differ a lot in size.
      "risk"  — a flat percentage of capital risked between entry and the stop,
                so a tight stop gets a bigger position than a wide one.
    """

    mode: str = "fixed"               # "fixed" | "equal" | "risk"
    capital: float = 1_000_000.0
    fixed_amount: float = 100_000.0   # used when mode == "fixed"
    pct_per_stock: float = 5.0        # used when mode == "equal"
    risk_pct: float = 1.0             # used when mode == "risk"
    max_capital_pct: float = 10.0     # cap on capital in one name  ("equal"/"risk" only)
    max_stop_distance_pct: float = 0.0
    """Max SL on one stock — how far below the BUY PRICE the stop may sit. 0 = off.

    A percentage of the stock's own price, which is what the name says: 20 means
    a stock bought at Rs 100 can never have a stop below Rs 80.

    It does two things, and both matter:

      * it is a real stop. The position exits on a close below entry x (1 - pct),
        alongside the weekly 20 EMA rung — whichever level is breached FIRST ends
        the trade.
      * it feeds sizing. The stop used to size a new entry is the TIGHTER of the
        20 EMA and this cap, so a 20 EMA sitting 30% away no longer justifies a
        position that a 20% stop makes a lie of.

    (This replaced a box labelled the same way that in fact capped RUPEES as a
    percentage of capital. At 20% on Rs 50L of capital that cap was Rs 10 lakh —
    it never came close to binding, so the setting did nothing at all while
    reading as though it limited the stop. The label was right and the code was
    wrong, so the code changed.)
    """
    compound: bool = True             # size off current equity rather than the starting figure

    def max_concurrent(self) -> int | None:
        """How many positions the capital allows at once, in fixed mode.

        Unchanged by compounding: the per-stock amount and the capital grow by
        the same factor, so the number of slots stays put.
        """
        if self.mode != "fixed" or self.fixed_amount <= 0:
            return None
        return int(self.capital // self.fixed_amount)

    def slice_pct(self) -> float:
        """The per-stock amount as a % of starting capital."""
        if self.capital <= 0:
            return 0.0
        return self.fixed_amount / self.capital * 100.0

    def amount_for(self, equity: float | None) -> float:
        """The rupee slice to use for a new entry, given current equity."""
        if self.compound and equity and self.capital > 0:
            return self.fixed_amount * (float(equity) / self.capital)
        return self.fixed_amount


@dataclass
class Sized:
    qty: int
    cost: float
    risk_amount: float
    risk_pct: float
    capped_by: str | None


def size_position(entry_price: float, stop_price: float, cfg: SizingConfig,
                   equity: float | None = None) -> Sized:
    """How many shares, given the sizing mode and both caps.

    "Equal capital" gives every name the same rupee slice regardless of how far
    its stop sits. "SL-wise" gives every name the same *risk*: a stock whose 20
    EMA is 4% away gets twice the size of one whose EMA is 8% away, so a stop-out
    costs the same either way. The two caps then apply on top, whichever binds first.
    """
    base = float(equity if (cfg.compound and equity) else cfg.capital)
    if base <= 0 or entry_price <= 0:
        return Sized(0, 0.0, 0.0, 0.0, "no capital")

    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0:
        # The stop sits at or above the entry — it happens when Monday gaps down
        # through the 20 EMA. Fall back to the stop-distance cap, which is the
        # only other stop the position actually has; 10% if none is set. (This
        # used to fall back to the rupee cap, which meant one field silently
        # meant "% of capital" in one branch and "% of price" in the other.)
        fallback = cfg.max_stop_distance_pct if cfg.max_stop_distance_pct > 0 else 10.0
        per_share_risk = entry_price * (fallback / 100.0)

    # ---- fixed rupees per stock: no caps, every open trade the same size ------
    if cfg.mode == "fixed":
        amount = cfg.amount_for(equity)
        qty = max(int(math.floor(amount / entry_price)), 0)
        risk = qty * per_share_risk
        return Sized(qty, qty * entry_price, risk,
                     (risk / base * 100.0) if base else 0.0, None)

    if cfg.mode == "risk":
        qty = math.floor((base * cfg.risk_pct / 100.0) / per_share_risk)
    else:
        qty = math.floor((base * cfg.pct_per_stock / 100.0) / entry_price)

    capped: str | None = None

    max_cost = base * cfg.max_capital_pct / 100.0
    if qty * entry_price > max_cost:
        qty = math.floor(max_cost / entry_price)
        capped = f"capital cap {cfg.max_capital_pct:g}%"

    qty = max(int(qty), 0)
    return Sized(qty, qty * entry_price, qty * per_share_risk,
                 (qty * per_share_risk) / base * 100.0 if base else 0.0, capped)
