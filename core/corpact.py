"""
Corporate actions — splits, bonuses, dividends, rights and demergers.

The company changes your share count or your cost without you buying or selling
anything, and a journal that does not know about it goes quietly wrong.

**The specific trap, which has bitten this project before.** Yahoo *back-adjusts*
its price history: after a 1:5 split, the series shows ₹400 for a day the stock
really traded at ₹2,000, all the way back to the beginning. Your journal holds
the broker's real number, ₹2,000. So the app computes

    gain% = 400 / 2000 - 1 = -80%

on a position that has not lost a rupee. When an earlier study replayed a real
Zerodha tradebook against Yahoo prices, 48 of 370 symbols showed this, one of
them at -92%. It is not a rounding problem; it silently rewrites your results.

So the rule here is: **adjust the position to match the adjusted series.** After
a 1:5 split the journal says 500 shares at ₹400. The stop, the ladder and the
gain% then all line up with the prices the app actually reads.

Two invariants make that safe to do:

* **Rupees never move.** 100 x ₹2,000 and 500 x ₹400 are the same ₹200,000, and
  a closed trade's recorded P&L is never touched — only the per-share numbers
  are restated. An adjustment that changes how much money you made is a bug.
* **Nothing is thrown away.** Every adjustment appends to an audit trail holding
  the before and after, so a wrong entry can be read, understood and reversed
  rather than discovered as a number that no longer adds up.

Dividends are the exception to the first rule: they *are* new money. They do not
touch the quantity or the cost — they land in cash and are reported on their own
line, never mixed into a trade's P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .exits import Position

KINDS = {
    "split": "Stock split",
    "bonus": "Bonus issue",
    "dividend": "Dividend",
    "rights": "Rights issue",
    "demerger": "Demerger",
}


# --------------------------------------------------------------------------- #
# ratios
# --------------------------------------------------------------------------- #
def split_factor(before: float, after: float) -> float:
    """Shares before : shares after. A 1:5 split is (1, 5) — factor 5.

    NSE announces splits by face value ("₹10 to ₹2") and brokers by ratio, but
    both reduce to the same question: how many shares do you hold now for each
    one you held before? A ₹10 face value going to ₹2 is five, so it is (1, 5).
    """
    before, after = float(before), float(after)
    if before <= 0 or after <= 0:
        raise ValueError("A split ratio needs two positive numbers.")
    return after / before


def bonus_factor(get: float, per: float) -> float:
    """A 1:1 bonus gives one free share per share held — you hold twice as many.

    The trap is reading 1:1 as "factor 1". It is 1 + 1/1 = 2. A 1:2 bonus (one
    free for every two held) is 1.5, not 0.5.
    """
    get, per = float(get), float(per)
    if get < 0 or per <= 0:
        raise ValueError("A bonus ratio needs a positive 'per' and a non-negative 'get'.")
    return 1.0 + get / per


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    kind: str                       # one of KINDS
    symbol: str
    ex_date: object                 # pd.Timestamp / date
    a: float = 1.0                  # split: old face / bonus: shares received / rights: offered
    b: float = 1.0                  # split: new face / bonus: per held    / rights: per held
    amount: float = 0.0             # dividend per share, or rights issue price
    keep_pct: float = 100.0         # demerger: % of cost that stays with the parent
    subscribed: bool = True         # rights: did you actually take them up
    child_symbol: str = ""          # demerger: the company that was spun off
    child_ratio: float = 0.0        # demerger: child shares received per parent share
    note: str = ""

    def factor(self) -> float:
        if self.kind == "split":
            return split_factor(self.a, self.b)
        if self.kind == "bonus":
            return bonus_factor(self.a, self.b)
        return 1.0

    def label(self) -> str:
        k = KINDS.get(self.kind, self.kind)
        if self.kind == "split":
            return f"{k} {self.a:g}:{self.b:g}"
        if self.kind == "bonus":
            return f"{k} {self.a:g}:{self.b:g}"
        if self.kind == "dividend":
            return f"{k} ₹{self.amount:g}/share"
        if self.kind == "rights":
            return (f"{k} {self.a:g}:{self.b:g} @ ₹{self.amount:g}"
                    + ("" if self.subscribed else " (not taken up)"))
        if self.kind == "demerger":
            base = f"{k} — {self.keep_pct:g}% of cost stays with the parent"
            if self.child_symbol:
                base += f", {self.child_ratio:g} x {self.child_symbol} per share"
            return base
        return k

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ex_date"] = str(pd.Timestamp(self.ex_date).date())
        return d

    @staticmethod
    def from_dict(d: dict) -> "Action":
        d = dict(d)
        d["ex_date"] = pd.Timestamp(d.get("ex_date") or date.today())
        known = {f for f in Action.__dataclass_fields__}
        return Action(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# applying one
# --------------------------------------------------------------------------- #
def _snapshot(pos) -> dict:
    return {"qty": pos.qty, "open_qty": pos.open_qty,
            "entry_price": round(pos.entry_price, 4),
            "initial_stop": round(pos.initial_stop, 4),
            "hard_stop": round(pos.hard_stop, 4),
            "decision_price": round(pos.decision_price, 4)}


def scale_position(pos, factor: float) -> None:
    """Multiply the share count and divide every price by the same factor.

    Quantity is rounded down and the entry price is then back-solved from the
    unchanged rupee cost, so `qty x entry_price` still equals what you actually
    paid. Rounding the price instead would leak a few rupees on every action.
    """
    if factor <= 0:
        raise ValueError("A corporate-action factor must be positive.")
    cost = pos.qty * pos.entry_price
    open_cost_share = pos.open_qty / pos.qty if pos.qty else 0.0

    pos.qty = int(pos.qty * factor)
    pos.open_qty = int(round(pos.qty * open_cost_share))
    pos.entry_price = cost / pos.qty if pos.qty else pos.entry_price
    if pos.avg_cost:
        pos.avg_cost = pos.avg_cost / factor
    pos.initial_stop = pos.initial_stop / factor
    pos.hard_stop = pos.hard_stop / factor if pos.hard_stop else 0.0
    pos.decision_price = pos.decision_price / factor if pos.decision_price else 0.0
    # restate the per-share numbers of exits already booked, but NEVER their
    # rupee P&L — that money has already happened
    for f in pos.fills:
        if f.get("price"):
            f["price"] = float(f["price"]) / factor
        if f.get("entry_price"):
            f["entry_price"] = float(f["entry_price"]) / factor
        if f.get("qty"):
            f["qty"] = int(round(float(f["qty"]) * factor))


def apply(book, action: Action, on_closed: bool = False) -> dict:
    """Apply one action to a book and return the audit record.

    Returns a record with `ok` False and a `problem` when there is nothing to
    apply it to, rather than raising — a typo in a symbol should be a message,
    not a stack trace on top of your journal.
    """
    sym = str(action.symbol).upper().strip()
    rec: dict = {
        "when": str(date.today()), "ex_date": str(pd.Timestamp(action.ex_date).date()),
        "symbol": sym, "kind": action.kind, "what": action.label(),
        "note": action.note, "ok": False, "problem": "", "before": None, "after": None,
        "cash_change": 0.0, "applies_to": "open",
    }

    pos = book.find(sym)
    if pos is None and on_closed:
        for p in reversed(book.closed):
            if p.symbol == sym:
                pos, rec["applies_to"] = p, "closed"
                break
    if pos is None:
        rec["problem"] = (f"No {'open or closed' if on_closed else 'open'} position in {sym}.")
        book.corporate_actions.append(rec)
        return rec

    rec["before"] = _snapshot(pos)

    if action.kind == "dividend":
        # new money, not a restatement: quantity and cost are untouched
        qty = pos.open_qty if rec["applies_to"] == "open" else pos.qty
        cash = float(action.amount) * qty
        book.cash += cash
        book.income.append({
            "date": str(pd.Timestamp(action.ex_date).date()), "symbol": sym,
            "kind": "dividend", "qty": qty, "per_share": float(action.amount),
            "amount": round(cash, 2), "note": action.note,
        })
        rec.update(ok=True, cash_change=round(cash, 2), after=_snapshot(pos))

    elif action.kind in ("split", "bonus"):
        scale_position(pos, action.factor())
        rec.update(ok=True, after=_snapshot(pos))

    elif action.kind == "rights":
        if not action.subscribed:
            rec.update(ok=True, after=_snapshot(pos),
                       problem="Not taken up — nothing to adjust.")
        else:
            extra = int(pos.open_qty * float(action.a) / float(action.b))
            if extra <= 0:
                rec["problem"] = "That ratio adds no shares to this holding."
            else:
                spend = extra * float(action.amount)
                cost = pos.qty * pos.entry_price + spend
                pos.qty += extra
                pos.open_qty += extra
                pos.entry_price = cost / pos.qty
                book.cash -= spend
                rec.update(ok=True, cash_change=round(-spend, 2), after=_snapshot(pos))

    elif action.kind == "demerger":
        keep = float(action.keep_pct) / 100.0
        if not 0 < keep <= 1:
            rec["problem"] = "The share of cost kept must be between 0 and 100%."
        else:
            # NO P&L IS BOOKED HERE. A demerger neither makes nor loses money: it
            # splits one cost basis across two listed companies. The gain or loss
            # happens later, when you sell either of them. Booking something now
            # would invent a profit out of an accounting entry.
            old_cost = pos.qty * pos.entry_price
            child_cost = old_cost * (1 - keep)
            pos.entry_price = pos.entry_price * keep
            pos.initial_stop = pos.initial_stop * keep
            pos.hard_stop = pos.hard_stop * keep if pos.hard_stop else 0.0
            rec.update(ok=True, after=_snapshot(pos))
            rec["child_cost"] = round(child_cost, 2)

            child_sym = str(action.child_symbol or "").upper().strip()
            if child_sym and action.child_ratio > 0 and rec["applies_to"] == "open":
                child_qty = int(pos.qty * float(action.child_ratio))
                if child_qty <= 0:
                    rec["problem"] = ("That ratio gives you no whole shares of "
                                      f"{child_sym}; record it by hand instead.")
                else:
                    child = Position(
                        symbol=child_sym, entry_date=pd.Timestamp(action.ex_date),
                        entry_price=child_cost / child_qty, qty=child_qty,
                        open_qty=child_qty,
                        initial_stop=0.0, demerged_from=pos.symbol,
                    )
                    book.positions.append(child)
                    # no cash entry: nothing was bought. The ledger row exists so
                    # the shares can be sold later and the cost is on the record.
                    book.ledger.append({
                        "date": str(pd.Timestamp(action.ex_date).date()),
                        "symbol": child_sym, "side": "BUY", "qty": child_qty,
                        "price": round(child.entry_price, 4), "value": 0.0,
                        "reason": f"demerged from {pos.symbol} — cost carried over, no cash paid",
                        "rung": "demerger", "entry_price": round(child.entry_price, 4),
                        "pnl": 0.0, "gain_pct": 0.0, "stop": 0.0,
                        "decision_price": 0.0, "slippage_%": None, "slippage_rs": None,
                    })
                    rec["child"] = {"symbol": child_sym, "qty": child_qty,
                                    "entry_price": round(child.entry_price, 4)}
                    rec["note"] = (rec["note"] + " · " if rec["note"] else "") + (
                        f"{child_qty} x {child_sym} created at Rs "
                        f"{child.entry_price:,.2f}, carrying {100 - action.keep_pct:g}% "
                        "of the original cost. No cash moved and no P&L was booked.")
            else:
                rec["note"] = (rec["note"] + " · " if rec["note"] else "") + (
                    f"{100 - action.keep_pct:g}% of the cost (Rs {child_cost:,.0f}) left "
                    "with the demerged company — add it as a position of its own once "
                    "it lists.")
    else:
        rec["problem"] = f"Unknown action '{action.kind}'."

    book.corporate_actions.append(rec)
    return rec


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def audit_frame(book) -> pd.DataFrame:
    """The trail, newest first. Nothing here is ever deleted."""
    if not getattr(book, "corporate_actions", None):
        return pd.DataFrame()
    rows = []
    for r in book.corporate_actions:
        b, a = r.get("before") or {}, r.get("after") or {}
        rows.append({
            "applied": r.get("when"), "ex-date": r.get("ex_date"),
            "symbol": r.get("symbol"), "what": r.get("what"),
            "on": r.get("applies_to"),
            "qty before": b.get("qty"), "qty after": a.get("qty"),
            "price before": (round(b["entry_price"], 2) if b.get("entry_price") else None),
            "price after": (round(a["entry_price"], 2) if a.get("entry_price") else None),
            "cash": r.get("cash_change") or None,
            "result": "applied" if r.get("ok") else "not applied",
            "note": r.get("problem") or r.get("note") or "",
        })
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def income_frame(book) -> pd.DataFrame:
    """Dividends received — kept out of trade P&L on purpose."""
    if not getattr(book, "income", None):
        return pd.DataFrame()
    df = pd.DataFrame(book.income)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def total_income(book) -> float:
    return float(sum(float(r.get("amount", 0.0)) for r in getattr(book, "income", [])))
