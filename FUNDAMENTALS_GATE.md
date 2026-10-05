# Fundamentals as a gate — design

Status: **phase 1 is built** (three rules, from Yahoo statements). Phase 2 —
pledge, auditor qualification, and statements from the exchanges' own filings —
is specified here and not built yet.

## The change in one line

Fundamentals stop being a *ranker* and become a *gate*.

| Before | Now |
|---|---|
| 50% weight, blended into the score | Pass / fail veto |
| Good CFO = higher rank | Bad CFO = rejected; good CFO = no bonus |

## Why

Over a breakout's four-to-six-month hold the business does not change — but it
can blow up. A stock that gaps down 40% overnight goes straight through a 20 EMA
stop; the only protection is not owning it, and that is what cash flow, pledge
and dilution checks are for. That is a gate's job, not a ranker's.

As a ranker, fundamentals push the wrong way for this system. The multibagger
shape is *change*: loss to profit, deleveraging, capacity turning into
utilisation — and those score badly on static quality metrics. A company already
at 1.4x CFO/PAT, 20% ROCE and zero debt is already recognised and already in the
price.

Examples from one week's list (at the time of writing):

- **SOLARA** scored 44 — 37% pledge, 0.93x interest cover — yet went from a
  −₹567 cr loss in FY24 to +₹17 cr PAT in Q1 FY27: the classic turnaround shape,
  which the weighting kept near the bottom. Under the gate it is rejected — for
  its interest cover today, and its pledge once phase 2 lands — a decision about
  blow-up risk, not about rank.
- **CARTRADE 86, MIDHANI 81** — top scores, but P/E 62.9 / 62.4 and ROE
  9.7% / 8.9%. Where does a 4x come from? The weighting rewarded them anyway.

## The gate — reject on any

| Rule | Phase | Source |
|---|---|---|
| Cash from operations negative in 2 of the last 3 years | 1 ✅ | Yahoo cash-flow statement (phase 2: exchange XBRL) |
| Interest coverage (EBIT / interest) < 1.5 | 1 ✅ | Yahoo income statement (phase 2: exchange XBRL) |
| Share count up ≥ 25% without assets growing as fast | 1 ✅ | Yahoo balance sheet (phase 2: exchange XBRL) |
| Promoter pledge > 25% | 2 | NSE/BSE quarterly Shareholding Pattern |
| Auditor qualification (modified opinion) | 2 | NSE/BSE financial-results filing (auditor's report) |

Missing data **passes**, marked "⚠ data nahi" — a missing filing is not evidence
of a bad business. The rejection reason is always shown next to the stock.

## Rank (after the gate) — future

The live list is ordered by RS Rating today. The intended order once the gate is
complete:

1. **Market cap** — smaller first
2. **Momentum** — RS Rating
3. **Change metrics** — sales-growth acceleration, margin expansion

## Phase 2 data source — free, official only

- **Quarterly financial results**, XBRL, from NSE/BSE corporate filings → EPS
  growth, sales growth, CFO, EBIT, interest, share count, total assets.
- **Quarterly shareholding pattern**, from NSE/BSE → promoter holding % and
  pledged % (both in the same filing).
- No paid APIs (Tickertape, Trendlyne), no scraping of Screener.in or similar.
- The cost is XBRL parsing: tags differ between Ind-AS taxonomies and between
  standalone and consolidated filings; prefer consolidated, fall back to
  standalone, and say which was used.
- Point-in-time by filing date, so the gate can later run inside the backtest
  without look-ahead (Yahoo's statements cannot — they are today's numbers).

## Where it lives

- `core/fundamentals.py` — `gate(facts) -> (status, reasons)`; `score_frame`
  carries `gate` and `gate why`.
- `core/breakout.py` — `select_by_rs` applies it as a veto after the RS order.
- Phase 2 adds a filings fetcher next to `core/nse.py` and feeds the same
  `gate()`.
