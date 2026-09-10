# Breakout Lab

Your Chartink scan — fresh N-week-high breakouts — turned into a tool you can
backtest, trade from every Monday, and keep a journal in.

Runs in the browser (Streamlit). Same app on your PC, this preview, or
Streamlit Cloud — phone / any PC.

**Phone / kisi aur PC se kholna:** see [DEPLOY.md](DEPLOY.md) (GitHub + Streamlit Cloud, ~5 minutes).

---

## Where the data comes from

**Yahoo Finance, via `yfinance`, downloaded by this app on your computer.**
Free, no API key, covers NSE equities and indices. The stock *lists* come from
NSE's own index constituent files (Nifty 50 / 100 / 200 / 500 / Midcap 150 /
Smallcap 250 / Microcap 250 / Total Market 750) — press **Fetch list from NSE**
in the sidebar.

Every symbol is cached in `~/.momentum_lab_cache`, **the same cache Momentum Lab
uses** — so anything you have already downloaded there is reused here instantly,
and anything new this app downloads is available to Momentum Lab too.

Prices are stored unadjusted with the adjustment factor kept separately: returns
and EMAs use the split-adjusted series, while price filters and market cap use
the actual traded price. A stock that split 1:10 shows an adjusted close of ₹45
on a day it really traded at ₹450 — conflating those is a quiet error that only
bites the names which had corporate actions.

If you ever get a paid feed, `core/data.py` is the only file to replace.

---

## Getting it running

**Windows** — double-click `run.bat`.
**Mac / Linux** — `chmod +x run.sh && ./run.sh`

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+. No internet? Tick **Demo mode** — synthetic prices, real interface,
meaningless numbers.

---

## Saving the sidebar

The sidebar *is* the system. Forty-odd controls decide what qualifies, how it is
ranked, how much you buy and where you get out — and re-typing them from memory
every Monday is how a live system quietly drifts away from the one you tested.

**Parameters** at the top of the sidebar saves the whole thing under a name.
Set it up once, call it *Live weekly*, and it is there next time. Save a second
one called *Research* with a wider universe and looser filters, and switch
between them from the dropdown.

* **Save current settings** writes every control to `params/<name>.json`.
* **Load** puts a saved set back and reruns the app with it.
* The set you saved or loaded **last** is the one the app opens with, so the
  normal case needs no clicking at all.

Two details worth knowing. A set stores the *name* of the index you picked, not
the two thousand symbols behind it — the list is still fetched from NSE. And a
set saved today keeps working after a new control is added later: the control
that did not exist simply falls back to its own default rather than refusing to
load. The files are plain JSON, one per set, editable by hand.

---

## Importing a Chartink list

Chartink runs your scan against the whole board. If you want the app's buy list
to be exactly the stocks Chartink showed you on Friday evening, export the scan
results and drop the CSV into **Import from Chartink CSV** at the top of *This
week's buys*.

**Only the symbols are taken.** The export also carries a price, a % change and
a volume, and all three are ignored on purpose: they are a snapshot of whenever
you pressed Download, while the rank, the quantity and the stop have to be
computed off the app's own weekly close or the buy list stops matching the
backtest. So the file answers exactly one question — *which stocks qualified* —
and everything after that is unchanged: technical score, fundamentals, combined
score, sector diversification, position size, stop level, all from your current
sidebar settings.

With the list switched on, the app's own freshness test and screen filters are
**skipped** for that week. Chartink already applied them; running them again
would quietly drop names the source list says qualified.

**Only the imported names are downloaded.** Nothing outside the list can affect
a single number on screen — the freshness test and the screen are skipped, and
the technical score is percentile-ranked *within the week's candidates* — so
loading the rest of your universe would be pure waiting. A 26-name import
downloads 26 symbols, not the 542 the sidebar universe happens to hold.

Names that cannot be ranked are reported with the actual reason, which is three
different reasons and not one:

| Why | What it means |
|---|---|
| **Nothing came back** | Yahoo has no data under this ticker — check it, or it was renamed |
| **Listed too recently** | it has real history, just less than a 52-week-high rule needs — the row shows its first bar and its day count |
| **Gaps in the recent data** | suspended, or too thinly traded to price every day |
| **Already held or drafted** | it is in this book already |

That distinction was worth fixing. A live scan used to be judged against the
**backtest's** six-year window, so a company that listed in 2024 came back 58%
to 61% empty against a 2020 start and was thrown out as having "no price
history" — while holding 600+ perfectly clean trading days of its own. Five real
Chartink qualifiers were disappearing that way. A live scan now reads about
three years and asks whether the **recent** stretch is complete, so a two-year-old
listing is kept and a suspended scrip is still dropped.

The parser is deliberately loose, because a Chartink export is not a stable
format — the columns change with your scan and there is normally a trailing
empty *Add Column*. A column named like a symbol column wins; failing that, the
column whose values most look like NSE tickers; failing that, a plain list
pasted one-per-line. `NSE:TITAN`, `TITAN.NS` and stray quoting all normalise.
A file it cannot read says so, rather than returning an empty list that looks
like "nothing qualified this week".

### A company that listed last year

It qualifies, and it is kept — but its 50-week EMA, the average that becomes its
final stop, is a young average of a short life. So the buy list marks it in a
**history** column, `{new} 78w`, rather than hiding it or dropping it. Nothing is
excluded; the number is how many weeks of history the stock actually has, and
what you do about it is your call.

Below about 58 weeks it *is* dropped, because a 52-week-high rule cannot say
anything at all about a stock that has not lived a year. Chartink will still show
such a name — its scan does not need the same history — which is why the app
explains the difference instead of going quiet.

**This is the recommended path for the full ~2,000-name board.** A sample of one
real export had 26 stocks, of which **5 were in the Nifty 500, 6 in Nifty Total
Market 750, and 20 were in no bundled list at all** — which is the whole reason
the import exists.

---

## Drafts: a buy list is not a position

Press *Queue as drafts* on the buy list and nothing is bought. A **draft** is an
intention: it holds no cash, has no P&L, and the exit ladder never looks at it.
On Friday evening the list is a plan — until Monday's order actually fills at a
real price there is no position to mark, stop or book, and pretending otherwise
puts a made-up entry price into your own performance record.

Confirming a draft — in the panel at the top of the same tab — is the moment it
becomes real, at the
quantity and price **you actually got**. That is also the only moment slippage
can be measured, so both prices are kept:

| | |
|---|---|
| **Decision price** | Friday's close — the number the buy list was built from |
| **Fill price** | what your broker gave you on Monday |
| **Slippage** | the gap, per stock and in total |

Slippage is the cost of the weekend, and the backtest cannot see it. The journal
reports it as a running average so you know what your own execution is worth.

While a buy is still a draft the table shows **drift %** — how far the stock has
moved since the list was made. That is the number that tells you whether
Monday's order is still the trade you meant to place, or whether the move
already happened without you.

Drafts survive a restart, can be edited or **dropped**, and a stock that is
already drafted will not appear on next week's list again. They live beside the
list that creates them; *Positions & exits* is for what you actually hold.

You can also **add a position you already hold** at the broker but that this
journal never saw. That goes straight in as a confirmed position with no
decision price, so it simply reports no slippage rather than a fake zero.

---

## Which index each stock is in

Every open position and every closed trade is tagged with the NSE size band it
sits in. The five bands are a clean partition of the top 750:

**Nifty 50** · **Nifty Next 50** (51-100) · **Nifty Midcap 150** (101-250) ·
**Nifty Smallcap 250** (251-500) · **Nifty Microcap 250** (501-750) ·
**Outside the top 750**

That last one is not an error. On a scan that runs over the whole ~2,000-name
board it is where most names live — one real Chartink export of 26 stocks had
five in the Nifty 500 and twenty in no bundled list at all.

The buy list carries it too, in a column beside the sector, so what you buy and
what the journal later measures are read in the same units — you can see on
Friday evening that four of this week's five are outside the top 750 before you
place a single order, not six months later.

*Positions & exits* shows how the open book is spread across the bands.
*Trading journal* shows what each band actually paid: return on capital, win
rate, average holding period, and the deepest fall in that band's own
cumulative P&L.

**The honest limit.** NSE publishes only TODAY's constituent lists; historical
membership is not free. So a stock bought in 2023 is bucketed by where it sits
now — and a stock that ran hard is exactly the one most likely to have moved up
a band since. That flatters the larger bands. Nothing can fix it, and the app
says so wherever the numbers appear.

---

## What the journal can tell you

The maths for CAGR, drawdown, Calmar and Sharpe was always in `core/metrics.py`
— the backtest has used it since day one. What was missing was the **input**.
Every one of those numbers needs a daily mark-to-market equity curve, and the
journal only ever recorded fills. A list of fills gives you realised P&L; it
cannot give you a drawdown, because a drawdown happens in the weeks when nothing
is filled at all.

`core/journal_stats.py` builds that curve — cash plus the day-by-day value of
everything open — and everything else is that curve, or the round trips, put
through the existing library rather than a second copy of it.

**Without prices** (the toggle off): net and realised P&L, ROI, win rate, profit
factor, expectancy, average win / average loss, risk-reward, P&L by exit reason,
partial-booking effectiveness, best and worst trades, index buckets, slippage.

**With prices**: all of the above plus the equity curve, max drawdown, Calmar,
CAGR, and year-by-year / month-by-month tables whose drawdown column is the
deepest fall **inside** that period — what it felt like at the time, not a slice
of the all-time figure.

Three things it deliberately refuses to do:

- **Win rate is per trade, not per exit.** A position that booked at +20%, again
  at +40% and finally on the 20 EMA is one trade with three fills. Counting it
  as three is how a win rate says 70% on a system that loses money.
- **CAGR on a young book is not shown.** Six months of a good run annualises to
  a number nobody should believe, so under a year it shows a dash and says why.
- **Missing values are NaN, not zero.** A profit factor of 0 and a profit factor
  that does not exist yet are very different statements.

---

## Where your data lives

By default the journal and the saved parameter sets sit inside the app folder,
which is fine until the day you download a new version of the app. Then your
whole trading history is one careless folder-replace away from being gone.

So it is a setting. **Data folder** in the sidebar moves it:

```
<your folder>/
    journal/   your books, one JSON each
    params/    your saved sidebar presets
    backups/   dated snapshots
```

Point it at a Google Drive or OneDrive folder and three things happen at once:
every fill is backed up as you record it, the same book opens on another
machine, and upgrading the app becomes "replace the app folder" with nothing of
yours inside it.

Moving **copies**, it never moves or deletes. A file that already exists at the
destination is reported and left alone — relocating your journal is not the
moment to discover that "move" meant "replace". The old folder stays exactly as
it was until you delete it yourself.

If the folder is not there when the app starts — an unplugged drive, a renamed
Drive folder, a sync still catching up — the app falls back to its own folder
and says so in red, rather than silently writing your next fill somewhere you
will not find it.

**Back up this book now** in the Trading journal tab writes a dated folder
holding the raw JSON, the journal as CSV and the full workbook. A backup you
have to remember to take is a backup you will not have, so it is one click.

---

## Corporate actions

The company changes your share count or your cost without you buying or selling
anything, and a journal that does not know goes quietly wrong.

**The specific trap, which has bitten this project before.** Yahoo *back-adjusts*
its price history: after a 1:5 split it shows ₹400 for a day the stock really
traded at ₹2,000, all the way back to the beginning. Your journal holds the
broker's real ₹2,000. So the app computes

    gain % = 400 / 2000 - 1 = -80 %

on a position that has not lost a rupee. When an earlier study replayed a real
Zerodha tradebook against Yahoo prices, **48 of 370 symbols showed exactly this**,
one of them at −92%. It is not a rounding error; it silently rewrites your
results, your stop and your ladder.

So **Adjust for a corporate action** on the *Positions & exits* tab restates the
position to match the adjusted series — after a 1:5 split the journal says 500
shares at ₹400 — and the gain %, the stop and the rungs all line up again.

| Action | What it does to the position |
|---|---|
| **Split** 1:5 | 100 shares at ₹2,000 → **500 at ₹400** |
| **Bonus** 1:1 | 100 at ₹2,000 → **200 at ₹1,000** (1:1 is 2x, not 1x — the usual slip) |
| **Dividend** ₹5 | quantity and cost unchanged; **₹5 x shares into cash** |
| **Rights** 1:4 @ ₹150 | 1,000 → 1,250 shares, cash out, average cost re-blended |
| **Demerger** 70/30 | share count unchanged, parent's cost basis cut to 70% |

Two invariants make this safe:

- **Rupees never move.** 100 x ₹2,000 and 500 x ₹400 are the same ₹2,00,000, and
  a closed trade's recorded P&L is never touched — only the per-share numbers
  are restated. An adjustment that changes how much money you made is a bug, and
  the tests assert it does not happen.
- **Nothing is thrown away.** Every adjustment — including the ones that failed —
  appends to an append-only audit trail holding the before and the after, so a
  wrong entry can be read and understood rather than found later as a number
  that stops adding up.

Dividends are the exception to the first rule, because they *are* new money.
They land in cash, are listed on their own line, and never enter a trade's P&L —
though they do count in the net, or the net would disagree with the equity curve.

Closed trades can be adjusted too, for an action you only noticed afterwards.
It is off by default.

---

## One list for the week

*This week's buys* opens with everything Monday needs, in one place:

```
This week — 5 thing(s) to do            Week ending 4 Sep 2026

🟢 Buy — 3
   KOPRAN   40 sh @ ~₹250   stop ₹230   rank 2   ₹10,000
   ...
🟡 Book profit — 1
   CAPLIPOINT   35 of 100 sh   profit target 1 hit   +20.4%
🔴 Exit on the stop — 1
   TNPETRO      all 60 sh      closed below the 20 EMA, no profit booked
⏳ 2 draft buy(s) still waiting to be confirmed
```

or, when there is nothing to do:

```
✅ No action required. 14 position(s) holding, and nothing hit a rung this week.
```

None of this is new information — it was always there, split across two tabs and
two buttons. But having to assemble Monday's instructions yourself, on a Sunday
night, from two screens is exactly how a rung gets missed. The fills are still
recorded on *Positions & exits*; this panel only tells you what fired.

---

## The running book, at a glance

*Positions & exits* opens with a dashboard, and it is deliberately about
**exposure and risk** rather than performance — performance needs closed trades
and is the journal's job. This answers the questions you have with the book open
in front of you:

| | |
|---|---|
| **Capital deployed** | in rupees, and what % is working versus sitting in cash |
| **Unrealised P&L** | in rupees *and* as a % of the money actually put in |
| **Up / down** | how many positions are in profit, out of how many |
| **Best / worst open** | named, with rupees and % |
| **Biggest position** | and its share of the book — concentration, before it bites |
| **If every stop hit** | today's price down to each live stop, added up |
| **Avg days held** | across what is open |

**If every stop hit** is the one nobody computes and everybody should. It is not
a forecast; it is the cost of being wrong about all of them at once, which is
the number that decides whether the book is sized sensibly. A position already
trading under its stop contributes zero rather than a negative — it is past the
point this measures.

Below that: the up half and the down half summarised separately (an average
hides which side is which), every holding with days held, capital %, slippage,
sector and index band, and the sector / index spread of the open book.

The housekeeping — adding a position the journal never saw, adjusting for a
corporate action, erasing a mistake — sits under **Manage** at the bottom, out
of the way of the numbers you read every week.

### Erasing an entry

**This is not a sale.** Selling records a real exit at a real price and books
P&L. *Remove an entry* erases it as if it had never happened: the cash goes
back, every fill of it leaves the ledger, and nothing is left to distort a win
rate or an equity curve. A partly-sold position gives back the net, not the
gross.

It exists for a test row or a wrong symbol — never for tidying away a losing
trade, which is the one use that would quietly make your record a lie. The
removal is logged with the rows it took.

---

## Demergers

Every other corporate action restates one holding. A demerger makes a second
one, and the accounting is the part people get wrong.

**No profit and no loss is booked.** Nothing was bought and nothing was sold —
one cost basis is split across two listed companies. 100 shares of a Rs 2,00,000
holding at a 70/30 split becomes Rs 1,40,000 in the parent and Rs 60,000 in the
new company. **The P&L happens when you sell either of them.**

Give the app the new symbol and the ratio and it creates that position for you,
at the carried-over cost, with no cash movement and a link back to the parent.
Leave the symbol blank and only the parent's cost is cut.

That link matters more than it sounds. Read alone afterwards:

- the **parent** looks like a sudden loss — its price dropped by the value that
  left, while its cost basis did not
- the **new company** looks like a windfall — shares that appeared for no cash

Neither happened. So *Positions & exits* shows the pair together, and
**Combined P&L** is the only number that answers whether the holding was worth
it. Both legs are also shown alone, precisely so you can see how misleading each
one is on its own.

Worked through: buy 100 RELIANCE at Rs 2,000. Demerge 70/30 into JIOFIN 1:1.
Parent 100 at Rs 1,400, JIOFIN 100 at Rs 600, cash unchanged, realised P&L zero.
Sell both at Rs 1,500 and Rs 700 and you realise **+Rs 20,000**, which is exactly
Rs 2,20,000 received against Rs 2,00,000 paid. The parent alone would have read
+7.1% and the child +16.7%; the truth is +10%.

---

## The five tabs

| Tab | What it's for |
|---|---|
| **Backtest** | Run the rules over history. If it takes no trades it shows you a funnel — screen, breakouts, candidates, regime blocks, cash — so an empty result explains itself instead of drawing a flat line. Equity curve, drawdown, how much P&L each rung of the exit ladder produced, and month-by-month / year-by-year tables including the peak capital you actually had deployed. |
| **This week's buys** | Drafts waiting to be confirmed, then one list for the week — what to buy, what to book, what to stop out, or nothing. Run it after Friday's close, or import a Chartink CSV and rank that list instead. The stocks that broke out, ranked, with quantity, stop and capital for each. One button records them into your journal. |
| **Positions & exits** | A dashboard of what is working, what is at risk and what is up or down, then every holding — with days held, capital %, sector, index band and slippage. what each position owes the ladder next, and which rungs fired this week — i.e. what to sell on Monday. Record the fills you actually got. |
| **Trading journal** | The full record and the full analysis: equity curve, ROI, CAGR, max drawdown, year and month tables, profit factor, expectancy, risk-reward, P&L by exit reason, partial-booking effectiveness, best and worst trades, index-band breakdown and slippage. Exports to CSV and Excel. |
| **Universe & data** | How many stocks pass the screen over time, and why any given stock does or doesn't qualify right now. |

---

## The scan

Your Chartink scan is six stacked conditions:

```
weekly close      >  max(52 weekly closes) as of 1 week ago      ← new 52w high
close 1 week ago  <  max(52 weekly closes) as of 2 weeks ago     ← wasn't already
close 2 weeks ago <  max(52 weekly closes) as of 3 weeks ago       making new highs
... and so on for 3, 4, 5 weeks ago
```

The first line is the breakout. The other five make it a *fresh* one — the first
break in at least six weeks, not week nine of a run that already moved.

The **New high over** slider replaces every 52 with 100 / 150 / 200. The
**previous … weeks** slider replaces the five. Nothing else changes.

Plus the daily filters from your scan: price above ₹30, volume above 25,000,
SMA(50) volume above 50,000 — all switchable, all evaluated **point-in-time**.

### Small caps only

The **Market-cap segment** selector defaults to **Small cap — bottom 50% of your
list**. Every week the app ranks your loaded stocks by market cap and keeps that
slice, so a stock that grows out of it leaves the universe on its own.

Percentages rather than absolute ranks, because absolute ranks have a trap: ask
for "rank 251+" while your list holds 230 stocks and *nothing* can ever qualify —
the backtest runs, takes no trades, and shows you a flat line. (That is exactly
what happened in an earlier build of this app.) A percentage is always relative
to the list you actually loaded, so it cannot silently select nothing. The AMFI
absolute ranks (1–100 large, 101–250 mid, 251+ small) are still available as an
option, with a hard error in the sidebar if your list is too short for them.

Ranking rather than a rupee band matters more than it sounds. ₹20,000 crore was
a mid cap in 2019 and is a small cap now, so a fixed threshold quietly selects a
different kind of company as the years pass, and a backtest run on it is
measuring two different strategies stitched together. The rank re-computes every
week, so a stock that grows out of the small-cap bucket leaves the universe on
its own — and it was correctly *in* the universe for the years when it really
was a small cap.

Two things to know:

- The band is relative to **the list you load**, not the whole market. For AMFI
  ranks to mean what AMFI means, load **Nifty Total Market (750)**.
- A **rupee band** is still available (alone or on top), and there are presets
  for micro / mid / large caps too.

### The whole board, not just an index

The Index dropdown has one entry that is not an index: **All NSE mainboard
(~2000)**. It comes from NSE's own `EQUITY_L.csv` — every listed equity in series
**EQ** (normal rolling settlement) and **BE** (trade-for-trade, usually
surveillance and illiquid). Debentures, warrants and government stock are
excluded because they are not equity you would trade this system in, and NSE's
SME board is a separate file that is not included.

That is roughly 2,000 names against 750 in the widest index, which matters for
two different reasons:

- Your Chartink scan runs on the whole board, so anything narrower is a
  different scan. If you want the app's buy list to match what Chartink shows
  you on a Friday evening, this is the setting.
- The earlier capacity work found this system is **capital-constrained, not
  idea-constrained** — and going wider only helps if you also cut the position
  size, otherwise you simply skip more breakouts than before.

**The first download is long.** Two thousand symbols at Yahoo's pace is tens of
minutes and a few hundred megabytes in `~/.momentum_lab_cache`. After that it is
incremental like any other list. Note the same caveat as the index files: it is
*today's* listed set, so a company delisted in 2022 is still missing from a 2022
backtest.

### Daily timeframe filter

Two independent switches, both off by default.

**Daily trend stack** — each average has its own tick, so you can require price
above just the 20 EMA, or any combination:

- Price > 20 EMA
- Price > 50 EMA
- Price > 200 SMA
- Price > 20 EMA > 50 EMA > 200 SMA — a different kind of test: additionally the
  averages themselves are in order. That is the difference between "price popped
  above a tangle of flat averages" and "the whole structure points up".

Nothing is implied — an unticked average is not tested at all. The stack implies
all three, so ticking everything is the stack. Periods are adjustable, and the
200-SMA history warning only appears when a 200 SMA is actually in use.

**Daily RSI** — an *above* toggle and a *below* toggle, independently. Turn on
one for a floor or a ceiling; turn on both for a band (above 50 and below 80 =
strong but not blown off). Setting an impossible band is flagged in the sidebar
rather than quietly passing nothing.

Everything is read on the same day the weekly signal is read — the last trading
day of the signal week — and acted on at the following Monday's open, exactly
like every other rule here. No look-ahead, no part-formed bar, and no forward
fill across weeks: a week with no data is a fail, not "whatever last week said".

A 200-period SMA on daily bars needs 200 trading days, a little under a year.
The app still counts how many of your stocks actually have a value and says so on
screen, so you can tell rejection-for-trend from rejection-for-missing-data.

---

## How the qualifiers are ranked

**Every stock that qualifies is ranked, first to last.** There is no minimum pool
size — two candidates get a 1st and a 2nd exactly the way two hundred do. Three
numbers come out of the ranker, all on a 0–100 scale.

### Technical score

The chart. Three components, each percentile-ranked *within that week's
candidates*, so the weights mean what they say whatever the absolute numbers
look like. Weights are yours to change in the sidebar; the defaults are:

- **Momentum (100%)** — return over the lookback window.
- **Freshness (0%)** — how far above the old high the close is. *Smaller scores
  higher*: a stock 1% through its breakout has more of the move left than one
  already 9% through it.
- **Volume surge (0%)** — the day's volume against its own 50-day average.

### Fundamentals score

The business, from `core/fundamentals.py`. Unlike the technical score this one is
**absolute, not relative** — 70 means the same thing in a thin week as in a
crowded one. Five pillars, weighted:

| Pillar | Weight | What it reads |
|---|---|---|
| **Cash from operations** | **40%** | consistency across the years on file, and CFO ÷ PAT over three years |
| **Balance-sheet safety** | **20%** | debt/equity, interest cover (EBIT ÷ interest), and share-count dilution |
| Sales growth | 15% | 3-year CAGR and the latest year |
| Profit margins | 15% | net margin, and whether it is rising or falling |
| ROCE | 7% | EBIT ÷ (total assets − current liabilities) |
| ROE | 3% | net income ÷ average shareholders' equity |

The weighting is built for **the hold this system actually has — four to six
months.** A business does not change in that window, so the job here is not
"find a great company", it is *do not step on a landmine*. That reorders things:

- **Cash gets the most** because it is the one number a company cannot easily
  dress up, and it is the strongest single tell for the accounting blow-ups that
  destroy small-cap positions. Profit is an opinion; cash is a fact.
- **The balance sheet is second.** Leverage and quiet equity dilution are what
  turn a 15% drawdown into a 60% one, and nothing else in the score catches
  them. Dilution in particular is close to *the* defining Indian small-cap risk.
- **ROE and ROCE get 10% between them.** They are long-horizon quality measures
  with little to say about the next four months, they are the easiest numbers to
  flatter, and they need the most repair — three of the context rules below exist
  purely to un-mislead them. A number that needs that much correction should not
  carry much weight.
- **Sales growth is only 15%** because it double-counts. A small cap making a
  fresh 52-week high almost always has growth already and the market has paid
  for it, so its marginal information here is smaller than it looks.

#### The context rules — why this is not a threshold screen

"ROE above 20 = good" is wrong often enough to be dangerous, in both directions.
So each pillar is scored and then adjusted, and **every adjustment writes its
reason into the output** so the app can show you why:

- **Low ROE marked UP** when debt is low and cash flow is healthy. A company that
  funds itself with equity carries a large equity base, which pushes ROE down by
  construction. That is a conservative balance sheet, not a weak business.
- **Low ROCE marked UP** when capex intensity or asset growth is high and the cash
  is good. Money put into R&D, a plant or distribution shows up as capital
  employed long before it shows up as profit.
- **High ROE marked DOWN** when it sits far above ROCE on heavy debt. The return
  belongs to the lenders' money, and it disappears when rates or refinancing turn.
- **High ROE marked DOWN** when past losses have eroded the equity base. The first
  small profit then divides by a tiny denominator and prints a return nobody
  earned.
- **Margins, ROE and ROCE all marked DOWN** when profit is not turning into cash
  (CFO ÷ PAT below 0.5). This is the most reliable sign that the profit is
  accounting rather than economic.
- **Growth plus cash on a depressed ROCE** is flagged as a reinvestment or
  turnaround rather than punished.
- **Heavy debt marked UP** when interest is comfortably covered and the cash is
  there. Debt is only dangerous when it cannot be serviced; a 2× debt/equity with
  8× interest cover is a financing choice, not a solvency problem.

Each stock also gets a plain-English **"what is good"** line — *"cash from
operations positive in all 4 years · sales compounding 19% a year · ROCE 23% ·
debt/equity 0.12 — funded by its own money"* — and a **"watch"** line for the
opposite.

### Sector diversification

A fresh-breakout scan is not sector-neutral by nature. When capital goods run,
a dozen capital-goods names break out in the same week, and a pure score ranking
buys eight of them — one bet held in eight positions, which is the difference
between a bad month and a bad quarter.

**Spread the week's buys across sectors** (on by default) walks the ranking from
the top and takes a name only if its sector is not already full. **Max stocks
from one sector** defaults to 2.

Nothing is hidden. Two columns appear in the buy list:

| Column | Meaning |
|---|---|
| **rank** | where the name sat on pure combined score. A 12 in a list of 10 means it was promoted past names you did not buy. |
| **pick** | `{div}` — only got in because higher-ranked names were sector-full · `{cap}` — the cap had to be broken to fill the list · `{?}` — no sector on file, so never capped · blank — would have been picked anyway |

**Never promote a stock ranked below** is the floor under all of this, and it
matters more than the cap does. It defaults to twice the number of entries — a
list of ten never reaches past rank twenty. Thirty names qualify, the 28th has a
poor chart and poor numbers: buying it to balance a sector is a worse decision
than holding a third capital-goods name. Set it equal to the entry count to turn
promotion off entirely and keep the cap as a pure warning.

Three deliberate choices:

- **The quality floor beats the sector rule.** Diversification never buys
  something you would not otherwise own.
- **If the cap leaves the list short, the rest is filled by score anyway** and
  marked `{cap}`. A diversification rule that stops you deploying capital has
  cost you more than the concentration would have.
- **A name with no sector on file is exempt from the cap**, not punished by it —
  missing data is not evidence of concentration, the same way a missing filing is
  not evidence of a bad business.

**Sizing is untouched by any of this.** Every pick gets the same rupees per stock
you set in *Money* — there is no sector-wise capital allocation, and a sector
holding three names does not get three times the money or a third of it. The
sector rule decides *which* stocks, never *how much*.

Sectors come from the Industry column of NSE's index constituent files where the
name is in one, and from Yahoo for everything else, cached on disk permanently.
Only the names that actually qualified in a week are ever looked up.

### Combined score

```
combined = w_technical x technical + w_fundamental x fundamentals
```

renormalised over whichever of the two is present. The sidebar slider sets the
split; **50/50 is the default, which ranks identically to simply adding the two
scores**. The final buy list is sorted by combined score, highest first.

#### Why the fundamentals side is ranked too

**Rank fundamentals within the week** is on by default, and without it the slider
lies. The technical score is a percentile, so in a six-name pool it comes out
spread 0 / 20 / 40 / 60 / 80 / 100 — a standard deviation around 34. Real
fundamental scores clump: on a representative set they run 42 to 77, a standard
deviation around 11. Blend an always-wide number with an always-narrow one at
"50/50" and the wide one wins:

| Nominal weight | Share of the actual spread |
|---|---|
| Technical 50% | **~75%** |
| Fundamentals 50% | **~25%** |

That holds at 22–28% across pool sizes from 3 to 40. Percentile-ranking the
fundamentals inside the same pool gives both sides the same spread, so 50 finally
means 50.

The cost is that the fundamental component becomes *relative* — in a week where
every candidate is mediocre, one of them still ranks top on fundamentals. That is
why the **absolute score is kept and shown next to it**, so "ranked first, and
only 45 out of 100" stays visible. Turn the toggle off to blend the raw scores
instead, and read the table knowing the chart is doing three quarters of the work.

#### A word on how much to trust this

Everything else in this app was chosen by backtest. **This was not, and cannot be
yet** — Yahoo has no point-in-time statement archive, so there is no honest way to
measure whether the fundamentals score improves anything. The weights above are
reasoning, not evidence.

So start the slider low — **25–30, not 50** — until you have watched it work. After
a month or two of live use the cache holds real statements for the names you
actually traded, and the question becomes testable: split your entries by
fundamentals score and compare the forward returns of the two halves. Until then
treat this as a sanity check on the chart, not a second opinion of equal standing.

**Fundamentals analysis toggle** — off, `w_fundamental` is 0 and the combined
score *is* the technical score, so the ranking is exactly what it was before this
feature existed.

### Missing data

Yahoo's coverage of Indian small caps is patchy. A name with no usable statements
is marked **"Data unavailable"**, scored NaN, and **keeps its technical score for
the combined rank** — it is not pushed to the bottom. An unfiled small cap is an
unknown business, not a bad one. Nothing in the pipeline raises on missing data.

### Fundamentals in the backtest: look-ahead

Yahoo serves **today's** statements. It has no point-in-time archive. Switching
fundamentals on inside the backtest therefore ranks a 2021 breakout using numbers
published in 2026 — the simulation knows things you could not have known, and the
result is optimistic by an unknown amount.

For that reason the backtest toggle (*Fundamentals → Also use it in the backtest*)
is **off by default** and prints a warning when you turn it on. Fundamentals are
meant for **This week's buys**, where the statements really are the latest ones
available. Statements are cached on disk for 30 days, and only the names that
actually qualified in a week are fetched.

---

## The exit ladder

**Four profit targets**, then the two EMA rungs. Rungs 3 and 4 book 0% by
default, which makes them inert — so out of the box this is exactly the two-rung
ladder the app has always had, and a test pins that equivalence.

```
book 20% at +25%
book 20% at +50%
book  0% at +75%     ← off until you give it a quantity
book  0% at +100%    ← off until you give it a quantity
book 45% on a weekly close below the 20 EMA
exit whatever is left on a weekly close below the 50 EMA
```

The 50 EMA rung is **terminal**: it sells whatever is still open. On a winner
that walked up the whole ladder that is the remainder; on a breakout that failed
immediately it is the whole position, which is what a stop-loss is for. The
sidebar computes the remainder live rather than printing a fixed number.

### Moving the stop after a rung books

Each of the four profit rungs has a tick: **after this exit, move the stop on
what is left** to

- **cost** — breakeven, the price you paid
- **the weekly 10 EMA**
- **the daily 20 EMA**
- **the daily 50 EMA**
- **buy price + a fixed %** — lock in a specific gain

The floor applies to the remaining quantity, and a later rung can **tighten** it
but never loosen it: book at +20% locking +15%, then book at +40% moving to cost,
and the +15% floor survives. A rung set to book **0%** with a stop-move ticked is
a pure stop-move: price reaches the level, nothing is sold, the stop moves.

An average with no value yet gives no floor at all rather than a floor at zero,
so a missing reading is never mistaken for a break.

### Which close each rule is read on

Three different bars, on purpose:

| Rule | Read on | Filled at |
|---|---|---|
| Entry (the scan) | Friday's weekly close | the following Monday's open |
| **Profit targets** | **daily close** (switchable to weekly) | the **next day's** open |
| Moved-up trail floor | daily close | the next day's open |
| 20 EMA rung | Friday's weekly close | the following Monday's open |
| 50 EMA rung | Friday's weekly close | the following Monday's open |

**Why the targets are daily.** A stock can trade through +25% on a Wednesday and
close the week below it. Checked weekly that target never books — you saw the
profit and did not take it. Checked daily it books on Wednesday's close and fills
at Thursday's open, which is what you would actually have done.

This is a daily **close**, not an intraday touch. That distinction matters: a
close is a real observable at a real time with a real fill the next morning, so
it adds no look-ahead. An intraday touch would let the backtest claim fills at a
price it only saw for a second, and on a gap it would claim one that never
existed at all. If that is ever added it will be a separate, clearly-labelled
setting.

**Why the stops stay weekly.** The scan is a weekly-chart rule and the 20/50 EMA
stops are weekly-chart stops; you look at them after Friday's close and act on
Monday. Asymmetric — targets daily, stops weekly — but deliberately so, and it
matches how the system is actually traded. The switch exists if you want both on
the weekly close; there is a test pinning that turning it off reproduces the
old weekly-only behaviour exactly.

**One consequence worth knowing.** The 20 EMA rung is a full 100% stop-loss until
the first profit books. With daily targets a profit can now book mid-week, so a
Friday EMA break that would once have been a full stop-loss can instead be a 45%
trim. That is correct — the trade really did take profit first — but it changes
results, and it is pinned by a test.

Days inside a week are walked in order, so a Wednesday booking is already
recorded when Friday's EMA break is judged. Each fill is tagged with the bar that
actually fired it, not with the week it happened to fall in — a Wednesday exit
tagged with that week's Friday would put the fill before its own signal.


## Position sizing

Three modes:

**Fixed ₹ per stock (the default).** You type the rupee amount — ₹1,00,000 — and
every stock gets that. With ₹50,00,000 of capital that is **up to 50 positions
open at once**; the sidebar shows the number as you type. When positions exit,
that money returns to cash and funds the next week's entries. No caps are applied.
If a stock's price is above the slice (a ₹1.5L share on a ₹1L slice) it sizes to
zero and is skipped.

**Grow the slice with equity (compound)** — on by default. The amount you typed
is really a *ratio*: ₹1L of ₹50L is 2%. So as the book grows the slice grows with
it, and the slot count never moves:

| Equity | Slice per stock | Slots |
|---|---|---|
| ₹50,00,000 | ₹1,00,000 | 50 |
| ₹60,00,000 | ₹1,20,000 | 50 |
| ₹75,00,000 | ₹1,50,000 | 50 |
| ₹40,00,000 (drawdown) | ₹80,000 | 50 |

It cuts both ways — a drawdown shrinks the slice too, which is what stops a
losing streak from compounding downward. Turn the toggle off and every entry
stays a literal ₹1,00,000 forever, whatever the book does.

**Equal capital per stock (%).** A percentage of capital instead of a fixed
amount. Note that with compounding on, the rupee amount grows with your equity —
this is why two trades years apart can be ₹3L and ₹15L in the same backtest.
Use fixed mode if you want them identical.

**SL-wise (equal risk).** Every name risks the same rupees between entry and its
weekly 20 EMA. A stock whose EMA is 4% away gets twice the size of one whose EMA
is 8% away, so a stop-out costs the same either way.

The last two also take **max capital in one stock** (default 10%) and **max SL on
one stock** (default 2% — the most one position may lose if stopped out).
Whichever binds first wins, and the buy list names the one that did. Fixed mode
ignores both, on purpose.

---

### Max SL on one stock

**"Max SL on one stock (% below buy price)" means a percentage of the stock's own
price.** Set 20 and a stock bought at ₹100 can never have a stop below ₹80.

It does two things:

- **It is a real stop.** The position exits on a close below that level, running
  alongside the weekly 20 EMA rung — **whichever level is breached first ends the
  trade.** Checked on the daily close, because a stop whose whole job is to cap
  the loss is not something to look at once a week; the 20 EMA stays weekly
  because it is a weekly-chart rule.
- **It sizes the position.** The stop used at entry is the *tighter* of the 20 EMA
  and this cap, so a 20 EMA sitting 30% away no longer justifies a position that a
  20% stop makes a lie of. In SL-wise mode that means a capped stop buys *more*
  shares for the same rupee risk.

A stop already inside the cap is left alone — it is a ceiling, not a setting. It
applies in every sizing mode, because a 30%-away stop is a problem however the
position was sized.

**What this replaced, and why.** A box with the same name used to cap **rupees**
as a percentage of capital. At 20% on ₹50L of capital that cap was ₹10,00,000 —
on a 5%-per-stock position the largest risk was about ₹2.25 lakh, so the setting
never came close to binding and did nothing at all, while reading as though it
limited the stop. Worse, the same number meant something different in one branch:
when the 20 EMA came out at or above the entry (a Monday gap-down), the code used
it as a percentage of *price*. One field, two meanings, neither matching the
label. The label was right and the code was wrong, so the code changed.

That fallback now uses this cap instead — the only other stop the position
actually has — or 10% when no cap is set.

On the synthetic universe, tightening it does what it should:

| Max SL | Final | Max DD | Worst exit | Fixed-stop exits |
|---|---|---|---|---|
| off | ₹74,41,989 | −7.44% | −15.6% | 0 |
| 20% | ₹74,41,989 | −7.44% | −15.6% | 0 |
| 15% | ₹74,35,420 | −7.44% | −16.6% | 4 |
| 10% | ₹74,80,366 | −7.30% | **−12.5%** | 24 |

At 20% nothing fires on this data — the 20 EMA is already tighter than that
almost everywhere. That is the cap behaving correctly, and it is also why the old
rupee version went unnoticed for so long: a setting that never binds looks exactly
like a setting that works.

**There is no separate rupee cap any more.** In SL-wise mode there never needed to
be one — the position is sized to risk exactly `risk_pct` of capital, so a cap
above that could not bind and a cap below it would contradict the box next to it.
If you want one back for Equal-capital mode, say so.


## Monthly and yearly breakup

Under the backtest you get the same year-wise and month-wise view Momentum Lab
has, plus the capital-usage columns this strategy needs:

- **Max capital used** — the peak *cost* of open positions in that month or
  year. Not their mark-to-market value: the actual money that left your account
  and was sitting in the market.
- **Max used %** — that peak against equity, so 100% means fully invested and
  40% means more than half your money sat idle.
- **Max open positions** — the most names held at once.
- Plus return, entries, exit fills and booked P&L per month; and start/end
  capital, net profit and max drawdown per year.

Both tables download as CSV. This is the pair of numbers that tells you whether
your per-stock slice is the right size: if the peak never gets near your capital,
the slice is too small (or too few stocks qualify), and if it pins at 100% for
months the strategy is turning away entries it wanted to take.

---

## The market regime filter

Blocks new entries when the index is below its weekly 50 EMA and/or its daily
200 SMA — either or both, your choice. Exits keep running normally when it is
blocking; it only stops new money going out.

Indexes available: Nifty 50, **Nifty Smallcap 250**, Nifty Midcap 150, Nifty
500, Nifty Bank, Sensex, NIFTYBEES. Yahoo's Indian index tickers are inconsistent
and change, so each one is a list of candidate symbols tried in order — the app
uses the first that returns data and tells you which. If none work it says so and
runs unfiltered rather than silently applying nothing.

---

## How a week actually works

Signals come off **Friday's close** and fill at **the following Monday's open** —
in the backtest and in the live tabs alike. Ranking on a close and filling at
that same close manufactures a price nobody could have got, and on a breakout
system, where the signal *is* a sharp move, that free lunch is large enough to
carry a whole strategy on paper and none of it in life.

Cash is real in the backtest. With five new names a week and no cap on how many
you hold, an unconstrained book drifts into 300% invested; here, entries are
skipped when the money isn't there.

---

## Two things this cannot do honestly

**1. Buyer / seller initiated trades.** Your scan has
`Daily Buyer initiated trades >= 200` and the same for sellers. That split needs
every trade classified by which side was the aggressor, from tick data. NSE does
not publish it and no free source carries it. What NSE *does* publish is the
total **number of trades**, which this app can fetch (sidebar → *Number of
trades*). `Trades >= 400` is the closest honest equivalent to "200 buyer AND 200
seller". It will not select exactly the same stocks, and the app says so rather
than letting you assume otherwise.

**2. Market cap is an estimate.** Yahoo exposes today's share count, so history
is computed as *today's shares × that day's raw price*. A company that doubled
its share count in 2021 shows a 2020 market cap roughly double the real one.
Your 500–50,000 cr band is wide enough that most names land on the right side of
it; a tight band would not survive this.

Also: today's index lists are pre-filtered for "survived until now". The
point-in-time screen fixes the market-cap and liquidity half of survivorship
bias, but companies that were delisted were never in the list to begin with.
Fixing that properly needs a paid point-in-time constituent feed.

---

## Layout

```
breakout_lab/
├── app.py                  the five tabs
├── core/
│   ├── breakout.py         the fresh N-week-high scan, scoring, regime filter
│   ├── fundamentals.py     statements -> a business score, with context rules
│   ├── exits.py            the ladder and position sizing
│   ├── engine.py           the weekly simulation
│   ├── journal.py          the live book: fills, P&L per rung, persistence
│   ├── params.py           named sidebar presets, saved to params/
│   ├── chartink.py         reads a Chartink export down to its symbols
│   ├── indices.py          which NSE size band a stock sits in
│   ├── journal_stats.py    the live book's equity curve and analytics
│   ├── corpact.py          splits, bonuses, dividends, rights, demergers
│   ├── storage.py          where your data lives, and backups
│   ├── data.py             download, cache, calendars   (shared with Momentum Lab)
│   ├── screen.py           point-in-time universe screening        (shared)
│   ├── nse.py              optional NSE bhavcopy source            (shared)
│   ├── universe.py         index lists and symbol cleaning         (shared)
│   ├── metrics.py          CAGR, Sharpe, drawdowns                 (shared)
│   └── charts.py           plotly theming                          (shared)
├── tests/test_breakout.py  553 tests — the maths
├── tests/test_app_flow.py  31 tests  — the click paths, through the real app
├── journal/                your books, as plain JSON      (movable)
├── params/                 your saved sidebar presets       (movable)
└── backups/                dated snapshots                  (movable)
```

Two suites. `python tests/test_breakout.py` — 553 tests — pins the things that
would otherwise break silently: the freshness stack fires on the first break and
not on the run after it, weekly signals on week W don't change when later bars
arrive, the ladder conserves quantity exactly (20/20/45/15 on a winner, a single
100% stop-loss on a failed breakout), fixed sizing puts the same rupees in every
stock and ignores the caps, the compounded slice tracks equity while the slot
count stays fixed, the small-cap rank band drops a stock as soon as it grows out
of it and the backtest never buys outside the band, an impossible screen is
reported rather than returning a silent flat line, the daily filter's two trend
ticks compose without implying each other and its RSI band is the intersection of
its two sides, cash never goes negative and the open-position count never
exceeds what the slice allows, risk sizing equalises
rupees at risk, the caps bind in the
right order, the backtest never sells more than it bought and never runs
negative cash, a saved journal reloads identically, and a parameter set
round-trips — including one saved before a control existed, and one whose file
has been corrupted, neither of which may take the app down with it — and a
Chartink export yields its symbols and nothing else, never mistaking the company
name or the price column for a ticker, a draft moves no cash and creates no
position until it is confirmed, an old book with no drafts key still opens, the
journal's win rate counts trades rather than fills, its drawdown is NaN rather
than zero when there are no prices to compute one from, a split changes the share
count and the price but never the rupees — not even on a trade that has already
closed — a 1:1 bonus doubles the holding rather than leaving it alone, a dividend
never enters trade P&L, moving the data folder copies rather than replaces, a company listed two years
ago survives a live scan while one suspended six months ago does not, erasing an
entry puts back the net cash and leaves no ledger row behind, and a demerger
preserves the total cost exactly while booking no P&L at all.

`python tests/test_app_flow.py` — 31 tests — drives the real app through
`streamlit.testing` in demo mode and then reads the JSON on disk, because one
whole class of bug lives outside the maths entirely. A `st.button` is True on
exactly one rerun, so gating a tab on `if not st.button(...): return` and then
putting a second button below it means the second button can never be read: its
click reruns the script, the gate is False again, and the function returns before
reaching it. That is what stopped *Queue as drafts* and *Save these exits* from
recording anything — silently, with no error, the list simply vanishing. These
tests click the whole week through and check the book afterwards: queue, confirm,
refresh, record an exit, erase a test row.

---

There is no auto-trading here and there will not be. The buy list and the exit
list are things you read and then place yourself. Backtests describe the past
under assumptions you chose; they are good for rejecting bad ideas and poor at
certifying good ones. None of this is investment advice.
