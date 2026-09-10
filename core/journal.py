"""
The trading journal — the live book you actually trade.

This is deliberately separate from the backtest. The backtest tells you what the
rules would have done; the journal records what *you* did: how many shares you
bought, at what price, and what each partial exit came out at — the profit
targets, the 20 EMA break, the 50 EMA break — with the P&L of each tranche
recorded against the rung that caused it.

Everything is plain JSON in `journal/`, one file per book: readable, diff-able,
and fixable by hand if you fat-finger a price.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .exits import (
    Position, Rung, check_hard_stop, check_position, check_trail_stop, default_ladder,
    next_trigger,
)

SCHEMA = 4


# --------------------------------------------------------------------------- #
# intentions, before they are trades
# --------------------------------------------------------------------------- #
@dataclass
class Draft:
    """A buy you have decided on but not yet filled.

    A draft is an intention and nothing else. It holds no cash, has no P&L, and
    the exit ladder never looks at it — which is the point: on Friday evening
    the list is a plan, and until Monday's order actually fills at a real price
    there is no position to mark, stop or book. Confirming a draft is what turns
    it into a `Position`, at the quantity and price you really got.

    `decision_price` is Friday's close, the number the buy list was built from.
    It is kept so the fill can be measured against it.
    """

    symbol: str
    decided_on: object                       # pd.Timestamp / date — when the list was made
    qty: int
    decision_price: float
    stop: float
    hard_stop: float = 0.0
    score: float = float("nan")
    sector: str = ""
    note: str = ""

    def cost(self) -> float:
        return self.qty * self.decision_price

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["decided_on"] = str(pd.Timestamp(self.decided_on).date())
        return d

    @staticmethod
    def from_dict(d: dict) -> "Draft":
        d = dict(d)
        d["decided_on"] = pd.Timestamp(d.get("decided_on") or date.today())
        known = {f for f in Draft.__dataclass_fields__}
        return Draft(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# the book
# --------------------------------------------------------------------------- #
@dataclass
class Book:
    name: str
    capital: float = 1_000_000.0
    cash: float = 1_000_000.0
    created: str = field(default_factory=lambda: str(date.today()))
    positions: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)      # every fill, in order
    notes: list[dict] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)     # decided, not yet filled
    corporate_actions: list[dict] = field(default_factory=list)
    """Every split, bonus, dividend, rights and demerger ever applied, with the
    before and after. Append-only: an adjustment you can read is fixable, one
    you cannot is a number that stops adding up."""
    income: list[dict] = field(default_factory=list)
    """Dividends received. New money, so it is kept out of trade P&L entirely
    and reported on its own line."""
    corrections: list[dict] = field(default_factory=list)
    cash_flows: list[dict] = field(default_factory=list)
    sizing_capital: float = 0.0
    compound_step: float = 120_000.0
    compound_buffer_pct: float = 20.0
    compounded_steps: int = 0
    schema: int = SCHEMA

    # ---------------------------------------------------------------- state --
    def open_symbols(self) -> set[str]:
        return {p.symbol for p in self.positions if p.is_open()}

    def find(self, symbol: str) -> Position | None:
        for p in self.positions:
            if p.symbol == symbol and p.is_open():
                return p
        return None

    def invested(self) -> float:
        return sum(p.open_qty * p.entry_price for p in self.positions if p.is_open())

    def realised_pnl(self) -> float:
        return float(sum(r.get("pnl", 0.0) for r in self.ledger if r.get("side") == "SELL"))

    def effective_sizing_capital(self) -> float:
        self.refresh_compounding()
        return float(self.sizing_capital or self.capital)

    def refresh_compounding(self) -> None:
        """Every ₹1.20 L of realised P&L: add 80% (₹1 L) to sizing capital, keep 20% as DD buffer."""
        if not self.sizing_capital:
            self.sizing_capital = float(self.capital)
        step = float(self.compound_step or 120_000)
        if step <= 0:
            return
        realised = max(0.0, self.realised_pnl())
        steps = int(realised // step)
        if steps <= self.compounded_steps:
            return
        keep = 1.0 - float(self.compound_buffer_pct or 0) / 100.0
        extra = (steps - self.compounded_steps) * step * keep
        self.sizing_capital = float(self.sizing_capital) + extra
        self.cash_flows.append({
            "date": str(date.today()), "kind": "compound",
            "amount": round(extra, 0),
                    "note": f"compounding step {steps} (20% DD buffer held back)",
        })
        self.compounded_steps = steps

    def add_capital(self, amount: float, note: str = "") -> None:
        amount = float(amount)
        self.cash += amount
        self.capital += amount
        self.sizing_capital = float(self.sizing_capital or 0) + amount
        if self.sizing_capital <= 0:
            self.sizing_capital = float(self.capital)
        self.cash_flows.append({
            "date": str(date.today()),
            "kind": "add" if amount >= 0 else "withdraw",
            "amount": round(amount, 0),
            "note": note,
        })

    # ---------------------------------------------------------------- drafts --
    def draft_symbols(self) -> set[str]:
        return {d.symbol for d in self.drafts}

    def add_draft(self, symbol: str, decided_on: date, qty: int, decision_price: float,
                  stop: float, hard_stop: float = 0.0, score: float = float("nan"),
                  sector: str = "", note: str = "") -> Draft | None:
        """Queue a buy. Nothing is spent and nothing is marked until it is confirmed."""
        if symbol in self.draft_symbols() or symbol in self.open_symbols():
            return None
        d = Draft(symbol=symbol, decided_on=pd.Timestamp(decided_on), qty=int(qty),
                  decision_price=float(decision_price), stop=float(stop),
                  hard_stop=float(hard_stop), score=float(score), sector=str(sector),
                  note=str(note))
        self.drafts.append(d)
        return d

    def find_draft(self, symbol: str) -> Draft | None:
        for d in self.drafts:
            if d.symbol == symbol:
                return d
        return None

    def drop_draft(self, symbol: str) -> bool:
        d = self.find_draft(symbol)
        if d is None:
            return False
        self.drafts.remove(d)
        return True

    def confirm_draft(self, symbol: str, on: date, qty: int, price: float,
                      stop: float | None = None) -> Position | None:
        """Turn an intention into a position at the price you actually got.

        This is the moment cash moves and P&L starts. The draft's decision price
        rides along so the fill can be measured against it.
        """
        d = self.find_draft(symbol)
        if d is None or int(qty) <= 0:
            return None
        pos = self.buy(symbol, on, int(qty), float(price),
                       float(d.stop if stop is None else stop),
                       score=d.score, hard_stop=d.hard_stop,
                       decision_price=d.decision_price,
                       note=d.note or "breakout entry")
        self.drafts.remove(d)
        return pos

    # ----------------------------------------------------------------- fills --
    def buy(self, symbol: str, on: date, qty: int, price: float, stop: float,
            score: float = float("nan"), note: str = "",
            hard_stop: float = 0.0, decision_price: float = 0.0) -> Position:
        qty = int(qty)
        cost = qty * float(price)
        pos = Position(
            symbol=symbol, entry_date=pd.Timestamp(on), entry_price=float(price),
            qty=qty, open_qty=qty, initial_stop=float(stop), entry_score=float(score),
            hard_stop=float(hard_stop), decision_price=float(decision_price or 0.0),
        )
        self.positions.append(pos)
        self.cash -= cost
        slip = pos.slippage_pct()
        self.ledger.append({
            "date": str(on), "symbol": symbol, "side": "BUY", "qty": qty,
            "price": float(price), "value": round(cost, 2), "reason": note or "breakout entry",
            "rung": "entry", "entry_price": float(price), "pnl": 0.0, "gain_pct": 0.0,
            "stop": float(stop),
            "decision_price": float(decision_price or 0.0),
            "slippage_%": None if pd.isna(slip) else round(slip, 3),
            "slippage_rs": (None if pd.isna(slip)
                            else round((float(decision_price) - float(price)) * qty, 2)),
        })
        return pos

    def sell(self, symbol: str, on: date, qty: int, price: float, rung: str,
             reason: str = "") -> dict | None:
        pos = self.find(symbol)
        if pos is None:
            return None
        qty = min(int(qty), pos.open_qty)
        if qty <= 0:
            return None
        price = float(price)
        proceeds = qty * price
        pnl = (price - pos.entry_price) * qty
        pos.open_qty -= qty
        if rung and rung not in pos.done:
            pos.done.append(rung)
        self.cash += proceeds
        rec = {
            "date": str(on), "symbol": symbol, "side": "SELL", "qty": qty,
            "price": price, "value": round(proceeds, 2),
            "reason": reason or RUNG_LABELS.get(rung, rung), "rung": rung,
            "entry_price": pos.entry_price, "pnl": round(pnl, 2),
            "gain_pct": round((price / pos.entry_price - 1) * 100, 2),
            "entry_date": str(pd.Timestamp(pos.entry_date).date()),
            "held_days": (pd.Timestamp(on) - pd.Timestamp(pos.entry_date)).days,
        }
        self.ledger.append(rec)
        pos.fills.append(rec)
        if not pos.is_open():
            self.positions.remove(pos)
            self.closed.append(pos)
        return rec

    def restate_entry(self, symbol: str, qty: int | None = None,
                      price: float | None = None, note: str = "") -> dict | None:
        pos = self.find(symbol)
        if pos is None:
            return None
        before = {"qty": pos.qty, "open_qty": pos.open_qty, "entry_price": pos.entry_price}
        if qty is not None and int(qty) > 0:
            qty = int(qty)
            if pos.open_qty == pos.qty:
                pos.qty = qty
                pos.open_qty = qty
            else:
                pos.open_qty = min(qty, pos.open_qty)
                pos.qty = max(pos.qty, pos.open_qty)
        if price is not None and float(price) > 0:
            pos.entry_price = float(price)
        rec = {
            "when": str(date.today()), "ex_date": str(date.today()),
            "symbol": symbol, "kind": "restate", "what": "Manual restatement of qty/price",
            "note": note or "edited after corporate action", "ok": True, "problem": "",
            "before": before,
            "after": {"qty": pos.qty, "open_qty": pos.open_qty, "entry_price": pos.entry_price},
            "cash_change": 0.0, "applies_to": "open",
        }
        self.corporate_actions.append(rec)
        return rec

    def demerger_linked(self, symbol: str) -> bool:
        pos = self.find(symbol)
        if pos is not None and getattr(pos, "demerged_from", ""):
            return True
        return any(getattr(p, "demerged_from", "") == symbol and p.is_open()
                   for p in self.positions)

    # ------------------------------------------------------- undo a mistake --
    def remove_position(self, symbol: str, entry_date=None, note: str = "") -> dict | None:
        """Erase a position and every fill of it, as if it had never been entered.

        This is NOT a sale. Selling records a real exit at a real price and books
        P&L; this removes an entry that should not exist — a test row, a wrong
        symbol, a fat-fingered quantity. Cash is put back exactly as it was and
        the ledger rows go with it, so nothing is left behind to distort a win
        rate or an equity curve.

        The removal itself is written to `corrections`, which is append-only. A
        journal you can quietly edit is not a record; one that says "these three
        rows were removed on 5 Sep, and here is what they were" still is.
        """
        pos = None
        for p in list(self.positions) + list(self.closed):
            if p.symbol != symbol:
                continue
            if entry_date is not None and \
                    pd.Timestamp(p.entry_date).date() != pd.Timestamp(entry_date).date():
                continue
            pos = p
            break
        if pos is None:
            return None

        ent = pd.Timestamp(pos.entry_date).date()
        rows, cash_back = [], 0.0
        for r in list(self.ledger):
            if r.get("symbol") != symbol:
                continue
            if r.get("side") == "BUY" and pd.Timestamp(r.get("date")).date() == ent:
                cash_back += float(r.get("value", 0.0))          # the money goes back in
            elif r.get("side") == "SELL" and r.get("entry_date") and \
                    pd.Timestamp(r["entry_date"]).date() == ent:
                cash_back -= float(r.get("value", 0.0))          # and the proceeds come out
            else:
                continue
            rows.append(r)
            self.ledger.remove(r)

        self.cash += cash_back
        (self.positions if pos in self.positions else self.closed).remove(pos)
        rec = {
            "when": str(date.today()), "what": "position removed", "symbol": symbol,
            "entry_date": str(ent), "qty": pos.qty, "entry_price": round(pos.entry_price, 4),
            "fills_removed": len(rows), "cash_restored": round(cash_back, 2),
            "note": note, "rows": rows,
        }
        self.corrections.append(rec)
        return rec


RUNG_LABELS = {
    "entry": "Entry",
    "profit_1": "Profit target 1",
    "profit_2": "Profit target 2",
    "profit_3": "Profit target 3",
    "profit_4": "Profit target 4",
    "trail_cost": "Stop moved to cost",
    "trail_wema10": "Stop moved to the weekly 10 EMA",
    "trail_dema20": "Stop moved to the daily 20 EMA",
    "trail_dema50": "Stop moved to the daily 50 EMA",
    "trail_pct": "Stop moved to a locked-in gain",
    "hard_stop": "Fixed % stop hit",
    "trail_ema_fast": "Closed below 20 EMA",
    "trail_ema_slow": "Closed below 50 EMA",
    "manual": "Manual exit",
}


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def book_path(directory: str, name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_")
    return os.path.join(directory, f"{safe or 'book'}.json")


def save_book(directory: str, book: Book) -> str:
    os.makedirs(directory, exist_ok=True)
    path = book_path(directory, book.name)
    blob = {
        "name": book.name, "capital": book.capital, "cash": book.cash,
        "created": book.created, "schema": SCHEMA,
        "positions": [p.to_dict() for p in book.positions],
        "closed": [p.to_dict() for p in book.closed],
        "ledger": book.ledger, "notes": book.notes,
        "drafts": [d.to_dict() for d in book.drafts],
        "corporate_actions": book.corporate_actions,
        "income": book.income,
        "corrections": book.corrections,
        "cash_flows": book.cash_flows,
        "sizing_capital": book.sizing_capital,
        "compound_step": book.compound_step,
        "compound_buffer_pct": book.compound_buffer_pct,
        "compounded_steps": book.compounded_steps,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)
    return path


def load_book(path: str) -> Book:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    return Book(
        name=blob.get("name", os.path.basename(path)),
        capital=float(blob.get("capital", 1_000_000)),
        cash=float(blob.get("cash", blob.get("capital", 1_000_000))),
        created=blob.get("created", str(date.today())),
        positions=[Position.from_dict(d) for d in blob.get("positions", [])],
        closed=[Position.from_dict(d) for d in blob.get("closed", [])],
        ledger=list(blob.get("ledger", [])),
        notes=list(blob.get("notes", [])),
        # books written before drafts existed simply have none
        drafts=[Draft.from_dict(d) for d in blob.get("drafts", [])],
        corporate_actions=list(blob.get("corporate_actions", [])),
        income=list(blob.get("income", [])),
        corrections=list(blob.get("corrections", [])),
        cash_flows=list(blob.get("cash_flows", [])),
        sizing_capital=float(blob.get("sizing_capital") or blob.get("capital") or 0),
        compound_step=float(blob.get("compound_step") or 120_000),
        compound_buffer_pct=float(blob.get("compound_buffer_pct") or 20),
        compounded_steps=int(blob.get("compounded_steps") or 0),
    )


def list_books(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith(".json")
    )


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def ledger_frame(book: Book) -> pd.DataFrame:
    """The full journal: every buy and every partial exit, newest last."""
    if not book.ledger:
        return pd.DataFrame(columns=[
            "date", "symbol", "side", "qty", "price", "value", "reason", "gain_pct", "pnl"
        ])
    df = pd.DataFrame(book.ledger)
    for c in ("qty", "price", "value", "pnl", "gain_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    cols = [c for c in ["date", "symbol", "side", "qty", "price", "value", "reason",
                         "entry_price", "gain_pct", "pnl", "held_days"] if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)


def drafts_frame(book: Book, last_price: pd.Series | None = None) -> pd.DataFrame:
    """The buys you have decided on but not confirmed.

    `drift %` is how far the stock has moved since the list was made — the
    number that tells you whether Monday's order is still the trade you meant
    to place, or whether the move already happened without you.
    """
    rows = []
    for d in book.drafts:
        px = float(last_price.get(d.symbol, np.nan)) if last_price is not None else np.nan
        rows.append({
            "symbol": d.symbol,
            "decided_on": pd.Timestamp(d.decided_on).date(),
            "qty": d.qty,
            "decision price": round(d.decision_price, 2),
            "last price": round(px, 2) if pd.notna(px) else None,
            "drift %": (round((px / d.decision_price - 1) * 100, 2)
                        if pd.notna(px) and d.decision_price > 0 else None),
            "capital if filled": round(d.cost(), 0),
            "stop": round(d.stop, 2),
            "sector": d.sector or "—",
        })
    return pd.DataFrame(rows)


def live_stop_price(entry: float, initial_stop: float, hard_stop: float,
                    ema_fast: float, last_price: float | None = None) -> float:
    """Trail floor for Open Risk: max(entry stop, weekly 20 EMA, hard stop).

    After a demerger Yahoo's EMA can stay on the old price scale (922 vs CMP 301).
    That would fake a stop above the buy. Ignore an EMA that is far above CMP.
    """
    live = float(initial_stop or 0.0)
    ef = float(ema_fast) if ema_fast is not None else float("nan")
    px = float(last_price) if last_price is not None else float("nan")
    if ef == ef and ef > 0:  # not NaN
        split_junk = (px == px and px > 0 and ef > px * 1.8)
        if not split_junk:
            live = max(live, ef)
    if hard_stop and hard_stop > 0:
        live = max(live, float(hard_stop))
    return live


def open_positions_frame(
    book: Book,
    last_price: pd.Series | None = None,
    ema_fast: pd.Series | None = None,
    ema_slow: pd.Series | None = None,
    ladder: list[Rung] | None = None,
    sectors: dict[str, str] | None = None,
    index_buckets: dict[str, str] | None = None,
    equity: float | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Live view of what you hold, with each position's next ladder step.

    `sectors`, `index_buckets` and `equity` are optional: leave them out and the
    frame is exactly what it always was, which is what the Excel export and the
    unpriced first render still rely on.
    """
    ladder = ladder or default_ladder()
    now = pd.Timestamp(today or date.today())
    rows = []
    for p in book.positions:
        if not p.is_open():
            continue
        px = float(last_price.get(p.symbol, np.nan)) if last_price is not None else np.nan
        ef = float(ema_fast.get(p.symbol, np.nan)) if ema_fast is not None else np.nan
        es = float(ema_slow.get(p.symbol, np.nan)) if ema_slow is not None else np.nan
        value = p.open_qty * px if pd.notna(px) else np.nan
        slip = p.slippage_pct()
        row = {
            "symbol": p.symbol,
            "entry_date": pd.Timestamp(p.entry_date).date(),
            "days held": int((now - pd.Timestamp(p.entry_date)).days),
            "bought_qty": p.qty,
            "open_qty": p.open_qty,
            "booked_qty": p.qty - p.open_qty,
            "decision price": round(p.decision_price, 2) if p.decision_price > 0 else None,
            "entry_price": round(p.entry_price, 2),
            "slippage %": None if pd.isna(slip) else round(slip, 2),
            "last_price": round(px, 2) if pd.notna(px) else None,
            "gain_%": round((px / p.entry_price - 1) * 100, 2) if pd.notna(px) else None,
            "open_value": round(value, 0) if pd.notna(value) else None,
            "capital %": (round(value / equity * 100, 2)
                          if pd.notna(value) and equity else None),
            "unrealised": round((px - p.entry_price) * p.open_qty, 0) if pd.notna(px) else None,
            "realised_so_far": round(sum(f.get("pnl", 0.0) for f in p.fills), 0),
            "20 EMA": round(ef, 2) if pd.notna(ef) else None,
            "50 EMA": round(es, 2) if pd.notna(es) else None,
            "done": ", ".join(RUNG_LABELS.get(k, k) for k in p.done) or "—",
            "next step": next_trigger(p, ladder, ef, es),
        }
        init_sl = float(getattr(p, "initial_stop", 0.0) or 0.0)
        hard_sl = float(getattr(p, "hard_stop", 0.0) or 0.0)
        live_stop = live_stop_price(p.entry_price, init_sl, hard_sl, ef, px)
        row["live_stop"] = round(live_stop, 2) if live_stop > 0 else None
        if book.demerger_linked(p.symbol):
            row["open_risk"] = 0.0
        else:
            row["open_risk"] = (round(max(0.0, (p.entry_price - live_stop) * p.open_qty), 0)
                                if live_stop > 0 else None)
        if sectors is not None:
            row["sector"] = sectors.get(p.symbol) or "Unknown"
        if index_buckets is not None:
            row["index"] = index_buckets.get(p.symbol) or "—"
        rows.append(row)
    return pd.DataFrame(rows)


def pending_actions(
    book: Book,
    week: pd.Timestamp,
    weekly_close: pd.Series,
    ema_fast: pd.Series,
    ema_slow: pd.Series,
    ladder: list[Rung],
    rearm_ema: bool = False,
    trail_levels: dict[str, pd.Series] | None = None,
    daily_close: pd.Series | None = None,
    targets_on_daily_close: bool = True,
) -> pd.DataFrame:
    """Which rungs have fired — i.e. what to sell next session.

    Mirrors the backtest exactly, which is the whole point of this function:

      * the moved-up trail floor and (in daily mode) the profit targets are read
        off the LATEST DAILY close;
      * the 20 / 50 EMA rungs are read off the latest completed WEEKLY close.

    This does NOT touch the book. It runs the ladder against a *copy* of each
    position so you can look at the list, change your mind, and only then record
    the fills you actually got.
    """
    import copy

    rows = []
    for p in book.positions:
        if not p.is_open():
            continue
        wclose = float(weekly_close.get(p.symbol, np.nan))
        dclose = (float(daily_close.get(p.symbol, np.nan))
                  if daily_close is not None else wclose)
        if pd.isna(wclose) and pd.isna(dclose):
            continue
        ef = float(ema_fast.get(p.symbol, np.nan))
        es = float(ema_slow.get(p.symbol, np.nan))
        ghost = copy.deepcopy(p)

        def add(ev, ref):
            rows.append({
                "symbol": p.symbol, "qty": ev["qty"], "rung": ev["rung"],
                "why": ev["reason"], "entry_price": round(p.entry_price, 2),
                "last_close": round(ref, 2),
                "gain_%": round((ref / p.entry_price - 1) * 100, 2),
                "est_proceeds": round(ev["qty"] * ref, 0),
                "est_pnl": round((ref - p.entry_price) * ev["qty"], 0),
            })

        # 1. the fixed % stop — the hard floor under the trade, checked daily
        if pd.notna(dclose):
            hs = check_hard_stop(ghost, dclose)
            if hs is not None:
                add(hs, dclose)
                continue

        # 2. the floor an earlier rung moved up — daily, and it outranks the rest
        lv = {k: float(v.get(p.symbol, np.nan)) for k, v in (trail_levels or {}).items()}
        if pd.notna(dclose):
            ts = check_trail_stop(ghost, dclose, lv)
            if ts is not None:
                add(ts, dclose)
                continue

        # 3. profit targets — daily close if that mode is on, else the weekly one
        if targets_on_daily_close and pd.notna(dclose):
            for f in check_position(ghost, week, dclose, np.nan, np.nan, ladder,
                                     rearm_ema=rearm_ema, only_triggers={"gain_pct"}):
                add(f, dclose)

        # 4. the EMA rungs — always the weekly close
        if pd.notna(wclose):
            triggers = {"below_ema_fast", "below_ema_slow"}
            if not targets_on_daily_close:
                triggers = triggers | {"gain_pct"}
            if book.demerger_linked(p.symbol):
                triggers = triggers - {"below_ema_fast", "below_ema_slow"}
            for f in check_position(ghost, week, wclose, ef, es, ladder,
                                     rearm_ema=rearm_ema, only_triggers=triggers):
                add(f, wclose)

    return pd.DataFrame(rows)


def summary(book: Book, last_price: pd.Series | None = None) -> dict:
    open_val = 0.0
    unreal = 0.0
    for p in book.positions:
        if not p.is_open():
            continue
        px = float(last_price.get(p.symbol, np.nan)) if last_price is not None else np.nan
        if pd.notna(px):
            open_val += p.open_qty * px
            unreal += (px - p.entry_price) * p.open_qty
        else:
            open_val += p.open_qty * p.entry_price

    sells = [r for r in book.ledger if r.get("side") == "SELL"]
    wins = [r for r in sells if r.get("pnl", 0) > 0]
    realised = float(sum(r.get("pnl", 0.0) for r in sells))
    return {
        "Capital": book.capital,
        "Cash": book.cash,
        "Open positions": len([p for p in book.positions if p.is_open()]),
        "Open value": open_val,
        "Portfolio value": book.cash + open_val,
        "Realised P&L": realised,
        "Unrealised P&L": unreal,
        "Total P&L": realised + unreal,
        "Return %": (book.cash + open_val) / book.capital * 100 - 100 if book.capital else 0.0,
        "Exits booked": len(sells),
        "Win rate %": (len(wins) / len(sells) * 100) if sells else float("nan"),
        # drafts are intentions, so they sit outside every number above and are
        # reported on their own line rather than folded into cash or exposure
        "Drafts waiting": len(book.drafts),
        "Cash needed for drafts": float(sum(d.cost() for d in book.drafts)),
        # dividends are new money, never folded into a trade's P&L
        "Dividends received": float(sum(float(r.get("amount", 0.0)) for r in book.income)),
    }


def rung_breakdown(book: Book) -> pd.DataFrame:
    """P&L split by *why* you exited — the number this whole ladder exists for."""
    sells = [r for r in book.ledger if r.get("side") == "SELL"]
    if not sells:
        return pd.DataFrame()
    df = pd.DataFrame(sells)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["gain_pct"] = pd.to_numeric(df["gain_pct"], errors="coerce")
    g = df.groupby("rung")
    out = pd.DataFrame({
        "Exits": g.size(),
        "Shares sold": g["qty"].sum(),
        "Avg gain %": g["gain_pct"].mean().round(2),
        "P&L": g["pnl"].sum().round(0),
    })
    out.index = [RUNG_LABELS.get(i, i) for i in out.index]
    return out.sort_values("P&L", ascending=False)


def equity_points(book: Book) -> pd.Series:
    """Rough realised-cash curve from the ledger — enough to chart progress."""
    if not book.ledger:
        return pd.Series(dtype=float)
    df = pd.DataFrame(book.ledger)
    df["date"] = pd.to_datetime(df["date"])
    df["flow"] = np.where(df["side"] == "BUY", -df["value"], df["value"])
    daily = df.groupby("date")["flow"].sum().sort_index()
    return (book.capital + daily.cumsum()).rename("cash_after_fills")
