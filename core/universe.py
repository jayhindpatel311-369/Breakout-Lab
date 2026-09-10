"""
Universe definitions for the Momentum Strategy Lab.

Everything here is EDITABLE.  The built-in lists are sensible defaults, but the
app also lets you paste your own symbols or upload a CSV/XLSX, and there is a
"Validate symbols" button that actually tries to download each one and tells you
which tickers are dead.  Treat these constants as a starting point, not gospel.

Symbol convention: NSE cash-market symbol WITHOUT the ".NS" suffix.  The data
layer appends ".NS" for Yahoo Finance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_DIR = os.path.join(os.path.dirname(HERE), "universes")


# --------------------------------------------------------------------------- #
# ETFs
# --------------------------------------------------------------------------- #
# name -> pretty label, used in the rebalance log so it reads like
# "Pharma 24% | Auto 18% | Nifty Next 50 22%" instead of raw tickers.
ETF_LABELS: dict[str, str] = {
    # ---- broad market ----
    "NIFTYBEES": "Nifty 50",
    "JUNIORBEES": "Nifty Next 50",
    "MID150BEES": "Midcap 150",
    "MOSMALL250": "Smallcap 250",
    "ALPHA": "Alpha 50",
    "ALPHAETF": "Alpha Low Vol 30",
    "MOMOMENTUM": "Nifty 200 Momentum 30",
    "LOWVOL1": "Low Volatility 30",
    "MOVALUE": "Nifty 500 Value 50",
    "MOQUALITY": "Nifty 200 Quality 30",
    # ---- sector / thematic ----
    "BANKBEES": "Nifty Bank",
    "PSUBNKBEES": "PSU Bank",
    "PVTBANIETF": "Private Bank",
    "FINIETF": "Fin Services",
    "ITBEES": "IT",
    "PHARMABEES": "Pharma",
    "HEALTHY": "Healthcare",
    "AUTOBEES": "Auto",
    "CONSUMBEES": "Consumption",
    "FMCGIETF": "FMCG",
    "METALIETF": "Metal",
    "INFRAIETF": "Infrastructure",
    "MAKEINDIA": "Manufacturing",
    "CPSEETF": "CPSE",
    "PSUBANK": "PSU Bank (alt)",
    "DIVOPPBEES": "Dividend Opportunities",
    "MOREALTY": "Realty",
    "COMMOIETF": "Commodities",
    "OILIETF": "Oil & Gas",
    "TNIDETF": "Defence",
    # ---- commodities / international / cash ----
    "GOLDBEES": "Gold",
    "SILVERBEES": "Silver",
    "MAFANG": "MAFANG (US Tech)",
    "MON100": "Nasdaq 100",
    "MASPTOP50": "S&P 500 Top 50",
    "HNGSNGBEES": "Hang Seng",
    "LIQUIDBEES": "Liquid (cash proxy)",
}

# The default rotation basket — the ones that actually have decent volume and a
# long enough history to backtest from 2018-2020 onward.
ETF_ROTATOR_DEFAULT: list[str] = [
    "NIFTYBEES",
    "JUNIORBEES",
    "BANKBEES",
    "PSUBNKBEES",
    "FINIETF",
    "ITBEES",
    "PHARMABEES",
    "AUTOBEES",
    "CONSUMBEES",
    "CPSEETF",
    "GOLDBEES",
    "SILVERBEES",
    "MAFANG",
    "MON100",
    "INFRAIETF",
    "DIVOPPBEES",
]

# A short, conservative basket: broad + defensive + hard assets only.
ETF_CONSERVATIVE: list[str] = [
    "NIFTYBEES",
    "JUNIORBEES",
    "GOLDBEES",
    "SILVERBEES",
    "PHARMABEES",
    "CONSUMBEES",
    "LIQUIDBEES",
]

# Everything we know about, for the "kitchen sink" option.
ETF_ALL: list[str] = list(ETF_LABELS.keys())


# --------------------------------------------------------------------------- #
# Stocks
# --------------------------------------------------------------------------- #
NIFTY_50: list[str] = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

# Nifty Next 50 — the second tier.  Combined with NIFTY_50 this gives you the
# Nifty 100.
NIFTY_NEXT_50: list[str] = [
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "BAJAJHLDNG", "BAJAJHFL", "BANKBARODA", "BPCL", "BRITANNIA",
    "BOSCHLTD", "CANBK", "CGPOWER", "CHOLAFIN", "DABUR",
    "DIVISLAB", "DLF", "DMART", "GAIL", "GODREJCP",
    "HAVELLS", "HAL", "HYUNDAI", "ICICIGI", "ICICIPRULI",
    "INDHOTEL", "INDIGO", "IOC", "IRFC", "JINDALSTEL",
    "JSWENERGY", "LICI", "LODHA", "LTIM", "MOTHERSON",
    "NAUKRI", "PFC", "PIDILITIND", "PNB", "RECLTD",
    "SHREECEM", "SIEMENS", "SWIGGY", "TATAPOWER", "TORNTPHARM",
    "TVSMOTOR", "UNITDSPR", "VBL", "VEDL", "ZYDUSLIFE",
]

# A liquid midcap/smallcap pool — this is where momentum strategies actually
# earn their keep.  Curated for liquidity, not for any index membership.
MIDSMALL_POOL: list[str] = [
    "AARTIIND", "ABCAPITAL", "ABFRL", "ACC", "AEGISLOG",
    "ALKEM", "APLAPOLLO", "ASTRAL", "ASHOKLEY", "ASTRAMICRO",
    "AUBANK", "AUROPHARMA", "BALKRISIND", "BANDHANBNK", "BATAINDIA",
    "BHARATFORG", "BHEL", "BIOCON", "BLUESTARCO", "BSE",
    "CAMS", "CDSL", "CESC", "COFORGE", "CONCOR",
    "COROMANDEL", "CROMPTON", "CUMMINSIND", "CYIENT", "DALBHARAT",
    "DEEPAKNTR", "DELHIVERY", "DIXON", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "FORTIS", "GESHIP", "GLENMARK", "GMRAIRPORT",
    "GODREJPROP", "GRANULES", "GUJGASLTD", "HFCL", "HINDCOPPER",
    "HINDPETRO", "HUDCO", "IDEA", "IDFCFIRSTB", "IEX",
    "IGL", "INDIAMART", "INDIANB", "INDUSTOWER", "IPCALAB",
    "IRB", "IRCTC", "JKCEMENT", "JSL", "JUBLFOOD",
    "KALYANKJIL", "KAYNES", "KEI", "KPITTECH", "LAURUSLABS",
    "LICHSGFIN", "LTF", "LTTS", "M&MFIN", "MANAPPURAM",
    "MARICO", "MAXHEALTH", "MAZDOCK", "MCX", "MFSL",
    "MGL", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM",
    "NBCC", "NCC", "NHPC", "NMDC", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "PAGEIND", "PATANJALI",
    "PAYTM", "PEL", "PERSISTENT", "PETRONET", "PHOENIXLTD",
    "PIIND", "POLICYBZR", "POLYCAB", "POONAWALLA", "PRESTIGE",
    "RBLBANK", "RVNL", "SAIL", "SJVN", "SOLARINDS",
    "SONACOMS", "SRF", "STARHEALTH", "SUNTV", "SUPREMEIND",
    "SUZLON", "SYNGENE", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TATATECH", "TIINDIA", "TITAGARH", "TORNTPOWER", "TRIDENT",
    "TTML", "UBL", "UNIONBANK", "UNOMINDA", "UPL",
    "VOLTAS", "WELCORP", "YESBANK", "ZENSARTECH", "ZFCVINDIA",
]

# A broader small/micro-cap tail. Together with NIFTY_50 + NIFTY_NEXT_50 +
# MIDSMALL_POOL this gets you to roughly the Nifty Total Market 750 footprint.
#
# This is a BUNDLED FALLBACK, used only when the live NSE download fails. It is
# assembled by hand and will drift as companies list, delist and rename — run
# "Validate symbols" once and drop whatever comes back dead. The app always
# tells you which source the list came from.
SMALLCAP_TAIL: list[str] = [
    "AAVAS", "ABREL", "ABSLAMC", "ACE", "AFFLE", "AJANTPHARM", "AKZOINDIA",
    "ALIVUS", "ALOKINDS", "AMBER", "ANANDRATHI", "ANANTRAJ", "ANGELONE", "APARINDS",
    "APOLLOTYRE", "APTUS", "ARE&M", "ASAHIINDIA", "ASTERDM", "ATUL", "AVANTIFEED",
    "BAJAJELEC", "BALAMINES", "BALRAMCHIN", "BASF", "BAYERCROP", "BBTC", "BDL",
    "BEML", "BIKAJI", "BIRLACORPN", "BLS", "BLUEDART", "BLUEJET", "BOROSIL",
    "BRIGADE", "BSOFT", "CAMPUS", "CANFINHOME", "CAPLIPOINT", "CARBORUNIV",
    "CASTROLIND", "CCL", "CEATLTD", "CELLO", "CENTRALBK", "CENTURYPLY", "CERA",
    "CHALET", "CHAMBLFERT", "CHENNPETRO", "CHOLAHLDNG", "CIEINDIA", "CLEAN",
    "COCHINSHIP", "CONCORDBIO", "CRAFTSMAN", "CREDITACC", "CRISIL", "CSBBANK",
    "CUB", "CUPID", "DATAPATTNS", "DBREALTY", "DCMSHRIRAM", "DEEPAKFERT",
    "DEVYANI", "DHANUKA", "DOMS", "DRREDDY", "EIDPARRY", "EIHOTEL", "ELECON",
    "ELGIEQUIP", "EMAMILTD", "EMCURE", "ENDURANCE", "ENGINERSIN", "EPIGRAL",
    "ERIS", "FACT", "FINCABLES", "FINEORG", "FIVESTAR", "FSL", "GABRIEL",
    "GALAXYSURF", "GARFIBRES", "GILLETTE", "GLAND", "GLAXO", "GNFC", "GODAWARI",
    "GODFRYPHLP", "GODREJAGRO", "GRANULES", "GRAPHITE", "GRAVITA", "GRINDWELL",
    "GRSE", "GSFC", "GSPL", "GUJALKALI", "HAPPSTMNDS", "HATSUN", "HBLENGINE",
    "HEG", "HEIDELBERG", "HEMIPROP", "HGINFRA", "HIKAL", "HIMATSEIDE", "HOMEFIRST",
    "HONAUT", "IFCI", "IIFL", "INDIACEM", "INDIAGLYCO", "INDIANHUME", "INFIBEAM",
    "INOXINDIA", "INOXWIND", "INTELLECT", "IOB", "ISGEC", "ITDCEM", "J&KBANK",
    "JBCHEPHARM", "JBMA", "JINDALSAW", "JKLAKSHMI", "JKPAPER", "JKTYRE",
    "JMFINANCIL", "JPPOWER", "JSWHL", "JUBLINGREA", "JUBLPHARMA", "JUSTDIAL",
    "JWL", "JYOTHYLAB", "JYOTICNC", "KAJARIACER", "KANSAINER", "KARURVYSYA",
    "KEC", "KFINTECH", "KIMS", "KIRLOSBROS", "KIRLOSENG", "KNRCON", "KPIL",
    "KPRMILL", "KSB", "LATENTVIEW", "LEMONTREE", "LINDEINDIA", "LLOYDSME",
    "LTFOODS", "LUMAXTECH", "LUPIN", "MAHABANK", "MAHSEAMLES", "MANINFRA",
    "MANKIND", "MAPMYINDIA", "MASTEK", "MEDANTA", "MEDPLUS", "METROBRAND",
    "MINDACORP", "MOTILALOFS", "MSUMI", "NAM-INDIA", "NATCOPHARM", "NAVA",
    "NAVINFLUOR", "NESCO", "NETWEB", "NEULANDLAB", "NEWGEN", "NH", "NIACL",
    "NILKAMAL", "NLCINDIA", "NUVAMA", "NUVOCO", "OLECTRA", "ORCHPHARM",
    "PARADEEP", "PCBL", "PFIZER", "PGEL", "PGHH", "PIXTRANS", "PNBHOUSING",
    "PNCINFRA", "POLYMED", "POWERINDIA", "PPLPHARMA", "PRAJIND", "PRINCEPIPE",
    "PRSMJOHNSN", "PTCIL", "RADICO", "RAILTEL", "RAINBOW", "RAJESHEXPO",
    "RALLIS", "RAMCOCEM", "RATNAMANI", "RAYMOND", "RBA", "RCF", "REDINGTON",
    "RELINFRA", "RENUKA", "RHIM", "RITES", "ROUTE", "RPOWER", "RRKABEL",
    "RTNINDIA", "SAFARI", "SAGCEM", "SAMMAANCAP", "SANOFI", "SAPPHIRE",
    "SARDAEN", "SCHAEFFLER", "SCHNEIDER", "SENCO", "SEQUENT", "SHARDACROP",
    "SHOPERSTOP", "SHRIPISTON", "SHYAMMETL", "SIGNATURE", "SKFINDIA", "SOBHA",
    "SONATSOFTW", "SOUTHBANK", "SPARC", "STARCEMENT", "STYLAMIND", "SUDARSCHEM",
    "SULA", "SUMICHEM", "SUNCLAYLTD", "SUNDARMFIN", "SUNDRMFAST", "SUNFLAG",
    "SUPRAJIT", "SURYAROSNI", "SWANENERGY", "SWSOLAR", "SYRMA", "TANLA",
    "TARIL", "TASTYBITE", "TCI", "TEAMLEASE", "TECHNOE", "TEGA", "TEXRAIL",
    "THERMAX", "THOMASCOOK", "TIMKEN", "TIPSMUSIC", "TRITURBINE", "TTKPRESTIG",
    "TVSHLTD", "UCOBANK", "UJJIVANSFB", "ULTRAMAR", "UNICHEMLAB", "UNIPARTS",
    "USHAMART", "UTIAMC", "VAIBHAVGBL", "VARROC", "VESUVIUS", "VGUARD",
    "VIJAYA", "VINATIORGA", "VIPIND", "VTL", "WABAG", "WELSPUNLIV", "WESTLIFE",
    "WHIRLPOOL", "WOCKPHARMA", "ZENTEC", "ZYDUSWELL",
]

# Roughly the Nifty Total Market footprint, assembled from the pieces above.
TOTAL_MARKET_FALLBACK: list[str] = list(
    dict.fromkeys(NIFTY_50 + NIFTY_NEXT_50 + MIDSMALL_POOL + SMALLCAP_TAIL)
)

# Yahoo's coverage of Indian index tickers is patchy and its symbols change, so
# each benchmark is a LIST of candidates tried in order — the app uses the first
# one that actually returns data and tells you which. Guessing a single ticker
# and silently charting nothing is the failure mode this avoids.
BENCHMARK_CANDIDATES: dict[str, list[str]] = {
    "Nifty 50": ["^NSEI", "NIFTYBEES.NS"],
    "Nifty Smallcap 250": [
        "^CNXSC",              # Nifty Smallcap (Yahoo's long-standing small-cap series)
        "NIFTYSMLCAP250.NS",
        "^NSEMDCP50",          # last-resort mid/small proxy
    ],
    "Nifty Midcap 150": ["NIFTY_MIDCAP_100.NS", "^NSEMDCP50"],
    "Nifty 500": ["^CRSLDX", "^NSEI"],
    "Nifty Bank": ["^NSEBANK", "BANKBEES.NS"],
    "Sensex": ["^BSESN"],
    "NIFTYBEES (ETF)": ["NIFTYBEES.NS"],
}

# kept for anything that still wants a single symbol
BENCHMARKS: dict[str, str] = {k: v[0] for k, v in BENCHMARK_CANDIDATES.items()}


@dataclass
class Universe:
    """A named list of symbols plus optional human labels."""

    name: str
    symbols: list[str]
    labels: dict[str, str] = field(default_factory=dict)

    def label(self, symbol: str) -> str:
        return self.labels.get(symbol, symbol)

    def pretty(self, symbols: list[str]) -> list[str]:
        return [self.label(s) for s in symbols]


BUILTIN_UNIVERSES: dict[str, Universe] = {
    "ETF — Rotator basket (16)": Universe("ETF — Rotator basket (16)", ETF_ROTATOR_DEFAULT, ETF_LABELS),
    "ETF — Conservative (7)": Universe("ETF — Conservative (7)", ETF_CONSERVATIVE, ETF_LABELS),
    "ETF — Everything": Universe("ETF — Everything", ETF_ALL, ETF_LABELS),
    "Stocks — Nifty 50": Universe("Stocks — Nifty 50", NIFTY_50),
    "Stocks — Nifty Next 50": Universe("Stocks — Nifty Next 50", NIFTY_NEXT_50),
    "Stocks — Nifty 100": Universe("Stocks — Nifty 100", NIFTY_50 + NIFTY_NEXT_50),
    "Stocks — Mid/Small pool": Universe("Stocks — Mid/Small pool", MIDSMALL_POOL),
    "Stocks — Wide (Nifty100 + Mid/Small)": Universe(
        "Stocks — Wide (Nifty100 + Mid/Small)", NIFTY_50 + NIFTY_NEXT_50 + MIDSMALL_POOL
    ),
    "Stocks — Broad market (bundled ~520)": Universe(
        "Stocks — Broad market (bundled ~520)", TOTAL_MARKET_FALLBACK
    ),
}

# Index names the app can pull live from NSE. The bundled list is the fallback
# for each, used only when the download fails.
LIVE_INDEX_FALLBACKS: dict[str, list[str]] = {
    "Nifty 50": NIFTY_50,
    "Nifty Next 50": NIFTY_NEXT_50,
    "Nifty 100": NIFTY_50 + NIFTY_NEXT_50,
    "Nifty 200": NIFTY_50 + NIFTY_NEXT_50 + MIDSMALL_POOL[:100],
    "Nifty 500": NIFTY_50 + NIFTY_NEXT_50 + MIDSMALL_POOL + SMALLCAP_TAIL[:250],
    "Nifty Midcap 150": MIDSMALL_POOL[:150],
    "Nifty Smallcap 250": SMALLCAP_TAIL[:250],
    "Nifty Microcap 250": SMALLCAP_TAIL[-250:],
    "Nifty Total Market (750)": TOTAL_MARKET_FALLBACK,
}


def load_custom_universes() -> dict[str, Universe]:
    """Pick up any CSV in universes/ as an extra universe.

    Expected columns: `symbol` (required), `label` (optional).
    """
    out: dict[str, Universe] = {}
    if not os.path.isdir(UNIVERSE_DIR):
        return out
    for fn in sorted(os.listdir(UNIVERSE_DIR)):
        if not fn.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        path = os.path.join(UNIVERSE_DIR, fn)
        try:
            df = pd.read_csv(path) if fn.lower().endswith(".csv") else pd.read_excel(path)
        except Exception:
            continue
        sym_col = _guess_symbol_column(df)
        if sym_col is None:
            continue
        symbols = clean_symbols(df[sym_col].astype(str).tolist())
        labels = {}
        for cand in ("label", "name", "Label", "Name"):
            if cand in df.columns:
                labels = dict(zip(clean_symbols(df[sym_col].astype(str).tolist()), df[cand].astype(str)))
                break
        key = f"File — {os.path.splitext(fn)[0]}"
        out[key] = Universe(key, symbols, labels)
    return out


def _guess_symbol_column(df: pd.DataFrame) -> str | None:
    candidates = ["symbol", "symbols", "ticker", "tickers", "scrip", "stock", "nse_symbol", "instrument"]
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    # fall back to the first column that looks like text
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def parse_uploaded_symbols(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Turn an uploaded CSV/XLSX into (symbols, labels)."""
    col = _guess_symbol_column(df)
    if col is None:
        return [], {}
    symbols = clean_symbols(df[col].astype(str).tolist())
    labels: dict[str, str] = {}
    for cand in ("label", "name", "company", "Label", "Name", "Company"):
        if cand in df.columns:
            labels = {
                s: str(v)
                for s, v in zip(clean_symbols(df[col].astype(str).tolist()), df[cand].astype(str))
            }
            break
    return symbols, labels


def clean_symbols(raw: list[str]) -> list[str]:
    """Normalise user input: strip whitespace, drop .NS/NSE:, uppercase, dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for s in raw:
        if s is None:
            continue
        t = str(s).strip().upper()
        if not t or t in {"NAN", "NONE", "SYMBOL", "TICKER"}:
            continue
        for prefix in ("NSE:", "NSE-", "BSE:"):
            if t.startswith(prefix):
                t = t[len(prefix):]
        for suffix in (".NS", ".BO", ".NSE"):
            if t.endswith(suffix):
                t = t[: -len(suffix)]
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def label_for(symbol: str, extra: dict[str, str] | None = None) -> str:
    if extra and symbol in extra:
        return extra[symbol]
    return ETF_LABELS.get(symbol, symbol)
