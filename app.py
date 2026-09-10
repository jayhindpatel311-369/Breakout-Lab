"""
Breakout Lab — weekly fresh-high breakout system for NSE stocks.

Backtest the rules, get Monday's buy list, track what each position owes the
exit ladder, and keep a journal of every fill.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html as html_mod
import io
import json
import os
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import calendar as calmod
try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:
    _IST = None

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts as ch
from core import data as data_mod
from core import fundamentals as fund_mod
from core import metrics as M
from core import nse as nse_mod
from core import screen as screen_mod
from core import universe as uni_mod
from core.breakout import (
    BreakoutConfig,
    DailyFilterConfig,
    RegimeConfig,
    daily_filter_mask,
    compute_signals,
    diversify_picks,
    qualifying_at,
    regime_blocked,
    score_week,
    to_weekly,
    volume_surge_daily,
)
from core.engine import (
    CostConfig,
    RunConfig,
    ladder_summary,
    monthly_breakdown,
    round_trip_trades,
    run_backtest,
    yearly_breakdown,
)
from core.exits import (
    TRAIL_MODES, Rung, SizingConfig, default_ladder, entry_stop, hard_stop_price,
    size_position,
)
from core import chartink as ck
from core import indices as ix_mod
from core import journal as jn
from core import journal_stats as js
from core import corpact as ca
from core import params as pm
from core import storage as sg
import core.gvault as gv

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _now_ist() -> datetime:
    if _IST is not None:
        return datetime.now(_IST)
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


# Where the journal and the parameter sets live is a setting, not a constant —
# point it at a synced folder and your data survives replacing the app. These
# are functions rather than module-level paths precisely so that changing the
# setting takes effect on the next rerun without restarting anything.
def JOURNAL_DIR() -> str:                                      # noqa: N802
    return sg.journal_dir(APP_DIR)


def PARAMS_DIR() -> str:                                       # noqa: N802
    return sg.params_dir(APP_DIR)

st.set_page_config(page_title="Breakout Lab", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# --- Streamlit version compat ---------------------------------------------- #
try:
    _ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ST_VER = (1, 0)
_WIDE = {"width": "stretch"} if _ST_VER >= (1, 49) else {"use_container_width": True}


def show_df(df, **kw):
    return st.dataframe(df, **_WIDE, hide_index=True, **kw)


def show_money_df(df, money_cols=(), pct_cols=(), height=None):
    if df is None or df.empty:
        st.caption("No rows yet.")
        return
    view = df.copy()
    cfg = {}
    for c in money_cols:
        if c in view.columns:
            view[c] = view[c].map(lambda x: rupees(float(x)) if pd.notna(x) and np.isfinite(x) else "—")
    for c in pct_cols:
        if c in view.columns:
            view[c] = view[c].map(lambda x: f"{float(x):+.2f}%" if pd.notna(x) and np.isfinite(x) else "—")
    auto = 46 + 34 * max(len(view), 1)
    h = min(auto, 420) if height is None else min(max(auto, 80), int(height))
    return st.dataframe(view, **_WIDE, hide_index=True, column_config=cfg, height=h)


def period_table_html(df: pd.DataFrame, kind: str) -> str:
    """Full-width compact table — no horizontal scroll, Indian rupee shorthand."""
    if df is None or df.empty:
        return '<div class="ptable-empty">No rows yet.</div>'
    label = {"year": "Year", "month": "Month", "week": "Week"}[kind]
    head = "".join(f"<th>{c}</th>" for c in [label, "Start", "End", "Return %", "P&L", "Drawdown %"])
    rows = []
    for _, r in df.iterrows():
        period = _pretty_period(r.get(label), kind)
        ret = float(r.get("Return %", 0) or 0)
        pnl = float(r.get("P&L", 0) or 0)
        dd = float(r.get("Drawdown %", 0) or 0)
        rc = "pos" if ret > 0 else ("neg" if ret < 0 else "")
        pc = "pos" if pnl > 0 else ("neg" if pnl < 0 else "")
        rows.append(
            "<tr>"
            f"{_td(label, period)}"
            f"{_td('Start', rupees(float(r['Start'])), 'num')}"
            f"{_td('End', rupees(float(r['End'])), 'num')}"
            f"{_td('Return %', f'{ret:+.2f}%', f'num {rc}')}"
            f"{_td('P&L', rupees(pnl), f'num {pc}')}"
            f"{_td('Drawdown %', f'{dd:.2f}%', 'num neg')}"
            "</tr>"
        )
    return (f'<div class="ptable-wrap"><table class="ptable"><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _pretty_period(val, kind: str) -> str:
    s = str(val)
    try:
        if kind == "week" and "/" in s:
            a, b = s.split("/", 1)
            return f"{pd.Timestamp(a).strftime('%d %b')} – {pd.Timestamp(b).strftime('%d %b %Y')}"
        if kind == "month":
            return pd.Period(s, freq="M").strftime("%b %Y")
        if kind == "year":
            return str(int(float(s))) if str(s).replace(".", "", 1).isdigit() else s
    except Exception:
        return s
    return s


def show_chart(fig, **kw):
    try:
        fig.update_layout(autosize=True)
    except Exception:
        pass
    cfg = dict(kw.pop("config", {}) or {})
    cfg.setdefault("responsive", True)
    return st.plotly_chart(fig, **_WIDE, config=cfg, **kw)


# --------------------------------------------------------------------------- #
# styling
# --------------------------------------------------------------------------- #
def inject_css(dark: bool) -> None:
    """Mix: Option-1 cards, Option-3 calendar/KPI strip, Option-2 chart density."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
        .stApp { background: #f3f5f8; color: #111827; }
        .block-container { padding-top: 4.8rem !important; padding-bottom: 3.2rem; max-width: 1440px; }
        header[data-testid="stHeader"] { background: transparent !important; }

        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e8eaee;
        }
        section[data-testid="stSidebar"] * { color: #111827 !important; }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span { color: #4b5563 !important; }
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 { color: #111827 !important; letter-spacing: -.02em; }
        section[data-testid="stSidebar"] h4 {
            font-size: 12px !important; text-transform: uppercase; letter-spacing: .08em !important;
            color: #9ca3af !important; margin-top: 18px !important;
        }

        .sb-brand { display:flex; gap:10px; align-items:center; padding: 4px 2px 16px;
                    border-bottom: 1px solid #f3f4f6; margin-bottom: 10px; }
        .sb-mark { width:36px; height:36px; border-radius:10px; background:#2563eb; color:#fff;
                   font-weight:800; font-size:13px; display:flex; align-items:center; justify-content:center; }
        .sb-name { font-weight:800; font-size:15px; letter-spacing:-.03em; color:#111827; }
        .sb-sub { font-size:11px; color:#9ca3af; margin-top:1px; }

        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stNumberInput input, .stTextInput input, .stDateInput input, textarea {
            background: #ffffff !important; color: #111827 !important;
            border: 1px solid #e5e7eb !important; border-radius: 10px !important;
        }
        div[data-baseweb="select"] { background: #ffffff !important; }
        div[data-baseweb="popover"] { background: #ffffff !important; color: #111827 !important; }

        .stButton > button {
            background: #ffffff !important; color: #111827 !important;
            border: 1px solid #e5e7eb !important; border-radius: 10px !important;
            font-weight: 650 !important; box-shadow: 0 1px 1px rgba(17,24,39,.04);
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: #2563eb !important; color: #ffffff !important;
            border: 1px solid #2563eb !important;
        }
        .stDownloadButton > button {
            background: #eff6ff !important; color: #1d4ed8 !important;
            border: 1px solid #bfdbfe !important; border-radius: 10px !important; font-weight: 650 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px; border-bottom: 1px solid #e8eaee; background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            color: #6b7280 !important; font-weight: 650; padding: 10px 16px;
            border-radius: 10px 10px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb !important; background: #eff6ff !important;
        }

        h1, h2, h3, h4 { color: #111827 !important; letter-spacing: -0.03em; }
        .stCaption, .stCaption p { color: #6b7280 !important; }
        .stRadio label, .stCheckbox label, .stToggle label { color: #111827 !important; }

        [data-testid="stExpander"] {
            background: #fff; border: 1px solid #e8eaee; border-radius: 14px;
            margin-bottom: 8px;
        }
        div[data-testid="stAlert"] { border-radius: 12px; }

        .app-top { display:flex; justify-content:space-between; align-items:flex-end;
                   margin: 0 0 6px; }
        .app-name { font-size: 26px; font-weight: 800; letter-spacing: -.04em; color:#111827; }
        .app-sub { font-size: 13px; color:#6b7280; margin-top: 2px; }
        .page-head { margin: 4px 0 14px; }
        .page-title { font-size: 22px; font-weight: 800; letter-spacing: -.03em; color:#111827; }
        .page-sub { font-size: 13px; color:#6b7280; margin-top: 3px; }
        .sec-h { font-size: 14px; font-weight: 700; color:#111827; margin: 18px 0 8px; }
        .card-h { font-size: 15px; font-weight: 800; letter-spacing: -.02em; color:#111827;
                  margin: 2px 0 10px; }
        .card-sub { font-size: 12px; font-weight: 500; color:#9ca3af; margin: -6px 0 10px; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid #e8eaee !important;
            border-radius: 16px !important;
            padding: 10px 14px 14px !important;
            box-shadow: 0 1px 2px rgba(17,24,39,.04);
            margin-bottom: 14px;
        }
        [data-testid="stFileUploader"] {
            background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 8px;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: #2563eb !important; border-color: #2563eb !important;
        }
        [data-testid="stDataEditor"] {
            border: 1px solid #e8eaee; border-radius: 14px; overflow: hidden; background:#fff;
        }
        .sb-sec { font-size: 11px; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
                  color: #6b7280; background: #f8fafc; border: 1px solid #eef0f3;
                  border-radius: 10px; padding: 8px 12px; margin: 14px 0 8px; }

        .tile {
            background:
              radial-gradient(130px 90px at 100% 0%, rgba(191,219,254,.55), transparent 62%),
              radial-gradient(100px 80px at 0% 100%, rgba(167,243,208,.22), transparent 58%),
              #ffffff;
            border: 1px solid #e8eaee;
            border-radius: 16px;
            padding: 16px 18px 14px;
            height: 100%;
            box-shadow: 0 1px 2px rgba(17,24,39,.04);
        }
        .tile.tile-good {
            background: radial-gradient(130px 90px at 100% 0%, rgba(167,243,208,.55), transparent 62%), #fff;
        }
        .tile.tile-bad {
            background: radial-gradient(130px 90px at 100% 0%, rgba(254,202,202,.5), transparent 62%), #fff;
        }
        .tile-label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
                      color: #9ca3af; margin-bottom: 8px; font-weight: 650; }
        .tile.tile-good .tile-label { color: #059669; }
        .tile.tile-bad .tile-label { color: #dc2626; }
        .tile-value { font-size: 26px; font-weight: 800; color: #1f2937; line-height: 1.15;
                      letter-spacing: -.03em; }
        .tile.tile-good .tile-value { color: #059669; }
        .tile.tile-bad .tile-value { color: #dc2626; }
        .tile-sub { font-size: 12px; color: #9ca3af; margin-top: 6px; }
        .pos { color: #059669 !important; }
        .neg { color: #dc2626 !important; }
        .note {
            background: #eff6ff; border-left: 3px solid #2563eb;
            padding: 10px 14px; border-radius: 8px; font-size: 13px; color: #1e3a8a;
        }
        .book-tools { background: transparent; border: none; padding: 0; margin: 0; }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e8eaee; border-radius: 14px; overflow: hidden;
            background: #ffffff; box-shadow: 0 1px 2px rgba(17,24,39,.03);
        }
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { background:#fff; }

        .cal-wrap { background:#fff; border:1px solid #e8eaee; border-radius:18px;
                    padding:18px 20px 14px; margin: 8px 0 18px;
                    box-shadow: 0 1px 2px rgba(17,24,39,.04); }
        .cal-kpis { display:flex; gap:0; border-bottom:1px solid #f3f4f6; margin:0 -20px 14px;
                    padding:0 8px 14px; overflow-x:auto; }
        .cal-kpi { flex:1; min-width:118px; padding:4px 14px; border-right:1px solid #f3f4f6; }
        .cal-kpi:last-child { border-right:none; }
        .cal-kpi .k { font-size:10px; letter-spacing:.08em; text-transform:uppercase;
                      color:#9ca3af; font-weight:650; margin-bottom:6px; }
        .cal-kpi .v { font-size:20px; font-weight:800; color:#111827; line-height:1.15; letter-spacing:-.03em; }
        .cal-kpi .s { font-size:11px; color:#9ca3af; margin-top:3px; }
        .cal-kpi .v.pos { color:#059669; }
        .cal-kpi .v.neg { color:#dc2626; }
        .cal-title { font-size:11px; font-weight:800; letter-spacing:.1em;
                     text-transform:uppercase; color:#6b7280; margin: 4px 0 10px; }
        .cal-leg { display:flex; gap:14px; flex-wrap:wrap; align-items:center;
                   font-size:12px; color:#6b7280; margin-bottom:12px; }
        .cal-leg .cal-day { margin-right:4px; }
        .cal-grid { display:flex; gap:14px; overflow-x:auto; padding-bottom:6px; }
        .cal-month { min-width: 108px; }
        .cal-mh { text-align:center; font-size:11px; font-weight:800; color:#374151;
                  letter-spacing:.08em; text-transform:uppercase; margin-bottom:8px; }
        .cal-row { display:flex; align-items:center; gap:3px; margin:2px 0; }
        .cal-wd { width:28px; font-size:10px; color:#9ca3af; }
        .cal-day { display:inline-flex; align-items:center; justify-content:center;
                   width:22px; height:20px; border-radius:999px; font-size:11px;
                   color:#9ca3af; background:transparent; }
        .cal-empty { display:inline-block; width:22px; height:20px; }
        .cal-lg { background:#047857; color:#fff; font-weight:700; }
        .cal-sg { background:#6ee7b7; color:#065f46; font-weight:600; }
        .cal-be { background:#fbbf24; color:#78350f; }
        .cal-sl { background:#fecaca; color:#7f1d1d; }
        .cal-ll { background:#b91c1c; color:#fff; font-weight:700; }
        .cal-en { background:#bfdbfe; color:#1e3a8a; font-weight:600; }
        .cal-note { font-size:11px; color:#9ca3af; margin-top:10px; }
        .ptable-wrap { overflow-x: auto; overflow-y: auto; max-height: 440px; width: 100%;
                       -webkit-overflow-scrolling: touch; }
        .ptable { width: 100%; border-collapse: collapse; table-layout: auto; }
        .ptable th { text-align: left; font-size: 11px; letter-spacing: .06em;
                     text-transform: uppercase; color: #9ca3af; font-weight: 650;
                     padding: 8px 10px; border-bottom: 1px solid #eef0f3; }
        .ptable td { padding: 9px 10px; border-bottom: 1px solid #f3f4f6; color: #111827;
                     font-size: 13px; }
        .ptable td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .ptable tr:last-child td { border-bottom: none; }
        .ptable-empty { color: #9ca3af; font-size: 13px; padding: 8px 0; }

        .saas-wrap { width:100%; overflow-x:auto; }
        .saas-table { width:100%; border-collapse:collapse; }
        .saas-table th {
            text-align:left; font-size:11px; letter-spacing:.08em; text-transform:uppercase;
            color:#9ca3af; font-weight:650; padding:10px 12px; border-bottom:1px solid #eef0f3;
            white-space:nowrap;
        }
        .saas-table th.num { text-align:right; }
        .saas-table td {
            padding:14px 12px; border-bottom:1px solid #f3f4f6; color:#111827;
            font-size:14px; vertical-align:middle;
        }
        .saas-table tr:last-child td { border-bottom:none; }
        .saas-table td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
        .saas-stock { display:flex; gap:10px; align-items:center; }
        .saas-av { width:36px; height:36px; min-width:36px; border-radius:50%;
                   display:inline-flex; align-items:center; justify-content:center;
                   font-weight:800; color:#fff; font-size:13px; letter-spacing:-.02em; }
        .saas-sym { font-weight:700; letter-spacing:-.02em; line-height:1.2; }
        .saas-sub { font-size:12px; color:#9ca3af; margin-top:2px; }
        .saas-next { font-size:12px; color:#4b5563; max-width:160px; line-height:1.35; }
        .m-lab { display: none; }
        .m-val { display: inline; }
        .hold-list { display: grid; grid-template-columns: 1fr; gap: 10px; }
        .hold-desktop { display: block; }
        .hold-mobile { display: none; }
        @media (max-width: 768px) {
            .hold-desktop { display: none !important; }
            .hold-mobile { display: block !important; }
        }
        .hold-compact { width: 100%; border-collapse: collapse; font-size: 13px; }
        .hold-compact th {
            text-align: left; font-size: 10px; letter-spacing: .06em; text-transform: uppercase;
            color: #9ca3af; font-weight: 700; padding: 8px 6px; border-bottom: 1px solid #eef0f3;
            white-space: nowrap;
        }
        .hold-compact th.num, .hold-compact td.num { text-align: right; }
        .hold-compact td {
            padding: 10px 6px; border-bottom: 1px solid #f3f4f6; vertical-align: middle;
        }
        .hold-compact td.next { font-size: 11px; color: #4b5563; white-space: normal; max-width: 120px; }
        .hold-compact .saas-av { width: 24px; height: 24px; min-width: 24px; font-size: 11px; }
        .hold-compact tfoot td { font-size: 12px; color: #4b5563; border-bottom: none; padding-top: 10px; }
        @media (max-width: 768px) {
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            #MainMenu, footer, .stDeployButton { display: none !important; }
            [data-testid="stAppViewContainer"] { display: block !important; }
            section[data-testid="stSidebar"] {
                position: fixed !important;
                top: 0 !important; left: 0 !important;
                height: 100% !important;
                width: min(86vw, 340px) !important;
                min-width: 0 !important;
                z-index: 1000001 !important;
                transform: none;
            }
            [data-testid="stMain"], section.main,
            [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
                width: 100% !important;
                max-width: 100vw !important;
            }
            .hold-compact .saas-sub { display: none; }
            .hold-compact .saas-sym { font-size: 12px; }
            .block-container { padding-top: 0.8rem !important; }
        }
        .hold-card {
            background: #fff; border: 1px solid #e8eaee; border-radius: 16px;
            padding: 14px 14px 8px; box-shadow: 0 1px 2px rgba(17,24,39,.04);
        }
        .hold-head { margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #f3f4f6; }
        .hold-kvs { display: flex; flex-direction: column; }
        .kv { display: flex; justify-content: space-between; align-items: flex-start;
              gap: 10px; padding: 7px 0; border-bottom: 1px solid #f3f4f6; }
        .kv:last-child { border-bottom: none; }
        .kv span { font-size: 11px; font-weight: 700; letter-spacing: .06em;
                   text-transform: uppercase; color: #9ca3af; flex: 0 0 42%; padding-top: 2px; }
        .kv b { font-weight: 650; text-align: right; flex: 1; font-size: 14px; color: #111827; }
        .kv.pos b { color: #047857; }
        .kv.neg b { color: #b91c1c; }
        .hold-foot { grid-column: 1 / -1; font-size: 13px; color: #4b5563;
                     padding: 8px 4px 0; }
        .badge { display:inline-block; padding:3px 9px; border-radius:999px;
                 font-size:11px; font-weight:700; letter-spacing:.04em; }
        .badge-buy { background:#ecfdf5; color:#047857; }
        .badge-sell { background:#fef2f2; color:#b91c1c; }
        .act-banner {
            border-radius: 18px; padding: 18px 20px 14px; margin: 4px 0 16px;
            border: 1px solid #e8eaee; background: #fff;
            box-shadow: 0 1px 2px rgba(17,24,39,.04);
        }
        .act-banner.hot {
            background: radial-gradient(420px 160px at 100% 0%, rgba(254,202,202,.45), transparent 70%), #fff;
            border-color: #fecaca;
        }
        .act-banner.ok {
            background: radial-gradient(420px 160px at 100% 0%, rgba(167,243,208,.4), transparent 70%), #fff;
            border-color: #a7f3d0;
        }
        .act-kicker { font-size:11px; letter-spacing:.1em; text-transform:uppercase;
                      font-weight:800; color:#9ca3af; margin-bottom:4px; }
        .act-banner.hot .act-kicker { color:#b91c1c; }
        .act-banner.ok .act-kicker { color:#047857; }
        .act-headline { font-size:26px; font-weight:800; letter-spacing:-.03em;
                        color:#111827; line-height:1.2; }
        .act-sub { font-size:14px; color:#4b5563; margin-top:4px; margin-bottom:12px; }
        .act-row { display:flex; align-items:center; gap:12px; padding:10px 0;
                   border-top:1px solid #f3f4f6; }
        .act-row .sym { font-weight:800; min-width:110px; }
        .act-why { flex:1; color:#374151; font-size:14px; }
        .act-meta { font-variant-numeric:tabular-nums; font-size:13px; color:#6b7280;
                    white-space:nowrap; text-align:right; }
        .tag { display:inline-block; padding:3px 9px; border-radius:999px;
               font-size:11px; font-weight:800; letter-spacing:.04em; }
        .tag-book { background:#ecfdf5; color:#047857; }
        .tag-sl { background:#fef2f2; color:#b91c1c; }

        .kpi-grid {
            display: grid;
            gap: 10px;
            margin: 0 0 14px;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        @media (min-width: 1100px) {
            .kpi-grid.n-5 { grid-template-columns: repeat(5, 1fr); }
            .kpi-grid.n-4 { grid-template-columns: repeat(4, 1fr); }
        }

        html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
        .stApp, section.main, .block-container { overflow-x: clip; }
        footer { visibility: hidden; height: 0; }

        /* —— tablet / landscape phone —— */
        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.2rem !important;
                padding-left: max(0.8rem, env(safe-area-inset-left)) !important;
                padding-right: max(0.8rem, env(safe-area-inset-right)) !important;
                padding-bottom: max(3.5rem, env(safe-area-inset-bottom)) !important;
                max-width: 100% !important;
            }
            .app-name { font-size: 22px; }
            .page-title { font-size: 20px; }
            .tile-value { font-size: 22px; }
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .stTabs [data-baseweb="tab-list"] {
                overflow-x: auto !important; flex-wrap: nowrap !important;
                -webkit-overflow-scrolling: touch;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 12px; white-space: nowrap; font-size: 13px;
            }
            .saas-wrap, .ptable-wrap, .cal-grid, .cal-kpis {
                overflow-x: auto; -webkit-overflow-scrolling: touch;
            }
            .ptable th:first-child, .ptable td:first-child,
            .saas-table th:first-child, .saas-table td:first-child {
                position: sticky; left: 0; background: #fff; z-index: 1;
            }
            [data-testid="stAppViewContainer"] { display: block !important; }
            section[data-testid="stSidebar"] {
                position: fixed !important; z-index: 1000001 !important;
                height: 100% !important;
            }
            section.main, [data-testid="stMain"],
            [data-testid="stAppViewContainer"] > .main {
                width: 100% !important; max-width: 100% !important;
                margin-left: 0 !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                flex-wrap: wrap !important;
            }
            div[data-testid="stHorizontalBlock"] > div,
            [data-testid="stColumn"] {
                width: 100% !important; max-width: 100% !important;
                flex: 1 1 auto !important;
            }
            .js-plotly-plot, .plotly-graph-div { max-width: 100% !important; }
            [data-testid="stDataFrame"] { max-width: 100%; overflow-x: auto; }
            [data-testid="stSlider"] [role="slider"] {
                width: 22px !important; height: 22px !important;
            }
        }

        /* —— phone portrait —— */
        @media (max-width: 640px) {
            .block-container {
                padding: 4.2rem 0.65rem 5.5rem !important;
                padding-bottom: max(5.5rem, env(safe-area-inset-bottom)) !important;
                max-width: 100% !important;
            }
            header[data-testid="stHeader"] { height: 3.2rem; }
            [data-testid="collapsedControl"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stBaseButton-headerNoPadding"] {
                width: 44px !important; height: 44px !important;
                min-width: 44px !important; min-height: 44px !important;
            }
            .app-name { font-size: 20px; }
            .app-sub { font-size: 12px; }
            .page-title { font-size: 18px; }
            .page-sub { font-size: 12px; }
            .tile { padding: 12px 12px 10px; border-radius: 14px; }
            .tile-value { font-size: 20px; word-break: break-word; }
            .tile-label { font-size: 10px; margin-bottom: 4px; }
            .tile-sub { font-size: 11px; }
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
            .act-headline { font-size: 20px; }
            .act-sub { font-size: 13px; }
            .act-banner { padding: 14px 14px 10px; }
            .act-row { flex-wrap: wrap; gap: 6px 10px; }
            .act-row .sym { min-width: 0; }
            .act-meta { white-space: normal; text-align: left; width: 100%; }
            .act-why { flex: 1 1 100%; }
            .cal-wrap { padding: 12px 12px 10px; }
            .cal-kpis { margin: 0 -12px 12px; padding: 0 8px 12px; }
            .cal-month { min-width: 96px; }
            .cal-day, .cal-empty { width: 20px; height: 18px; font-size: 10px; }
            .stTabs [data-baseweb="tab"] { padding: 12px 14px; font-size: 13px; }
            .stButton > button, .stDownloadButton > button,
            .stFormSubmitButton > button,
            .stPopover > button, [data-testid="stPopover"] button {
                min-height: 44px !important;
                width: 100% !important;
                font-size: 15px !important;
                white-space: normal !important;
                height: auto !important;
                line-height: 1.3 !important;
            }
            .stNumberInput input, .stTextInput input, .stDateInput input,
            .stSelectbox [data-baseweb="select"] > div, textarea {
                font-size: 16px !important;
                min-height: 44px;
            }
            .stRadio label, .stCheckbox label {
                min-height: 40px; padding: 8px 0 !important;
                display: flex !important; align-items: center;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                flex-wrap: wrap !important;
                gap: 8px !important;
            }
            div[data-testid="stHorizontalBlock"] > div,
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
            div[data-testid="column"],
            [data-testid="stColumn"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: 1 1 auto !important;
            }
            .app-sub { display: none; }
            section[data-testid="stSidebar"] {
                position: fixed !important;
                z-index: 1000001 !important;
                min-width: min(88vw, 380px) !important;
                max-width: min(88vw, 380px) !important;
            }
            section.main, [data-testid="stAppViewContainer"] > .main,
            .stApp [data-testid="stMain"] {
                margin-left: 0 !important;
                width: 100% !important;
            }
            div[data-testid="stVerticalBlockBorderWrapper"] {
                padding: 8px 10px 12px !important;
            }
            [data-testid="stFileUploader"] { padding: 12px; }
            section[data-testid="stSidebar"] {
                min-width: min(88vw, 380px) !important;
            }
            section[data-testid="stSidebar"] .stButton > button {
                min-height: 44px !important;
            }

            /* tables become stacked cards — no sideways drag to read a row */
            .saas-table thead, .ptable thead { display: none; }
            .saas-table, .saas-table tbody, .saas-table tr, .saas-table td,
            .ptable, .ptable tbody, .ptable tr, .ptable td,
            .saas-table tfoot, .saas-table tfoot tr, .saas-table tfoot td {
                display: block; width: 100%;
            }
            .saas-table tr, .ptable tr {
                background: #fff;
                border: 1px solid #e8eaee;
                border-radius: 14px;
                margin: 0 0 10px;
                padding: 8px 10px 6px;
                box-shadow: 0 1px 2px rgba(17,24,39,.04);
            }
            .saas-table td, .ptable td {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
                text-align: right !important;
                padding: 7px 4px !important;
                border-bottom: 1px solid #f3f4f6 !important;
                font-size: 14px;
                white-space: normal !important;
            }
            .saas-table td:last-child, .ptable td:last-child { border-bottom: none !important; }
            .saas-table td::before, .ptable td::before { display: none !important; content: none !important; }
            .m-lab {
                display: inline !important;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: .06em;
                text-transform: uppercase;
                color: #9ca3af;
                text-align: left;
                flex: 0 0 42%;
                padding-top: 2px;
            }
            .m-val { display: block; text-align: right; flex: 1; min-width: 0; }
            .saas-table td:first-child, .ptable td:first-child {
                position: static; background: transparent;
            }
            .saas-table td:first-child .m-lab { display: none !important; }
            .saas-table td:first-child .m-val { text-align: left; }
            .saas-table tfoot td {
                justify-content: flex-start;
                text-align: left !important;
                font-size: 13px;
            }
            .saas-table tfoot td::before { display: none; }
            .saas-next { max-width: none; }
            .saas-av { width: 28px; height: 28px; min-width: 28px; font-size: 12px; }
        }
        @media (max-width: 640px) and (orientation: landscape) {
            .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .block-container { padding-bottom: 3rem !important; }
        }
        @media (max-width: 400px) {
            .kpi-grid { grid-template-columns: 1fr 1fr; }
            .app-name { font-size: 18px; }
            .tile-value { font-size: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    kind = " tile-good" if tone == "pos" else (" tile-bad" if tone == "neg" else "")
    valcls = f" {tone}" if tone else ""
    sub_html = f'<div class="tile-sub">{sub}</div>' if sub else ""
    return (f'<div class="tile{kind}"><div class="tile-label">{label}</div>'
            f'<div class="tile-value{valcls}">{value}</div>{sub_html}</div>')


def page_head(title: str, sub: str = "") -> None:
    extra = f'<div class="page-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="page-head"><div class="page-title">{title}</div>{extra}</div>',
                unsafe_allow_html=True)


@contextmanager
def card(title: str = "", sub: str = ""):
    """White rounded panel. Widgets inside inherit the mix design system."""
    with st.container(border=True):
        if title:
            extra = f'<div class="card-sub">{sub}</div>' if sub else ""
            st.markdown(f'<div class="card-h">{title}</div>{extra}', unsafe_allow_html=True)
        yield


def tiles_row(items) -> None:
    n = max(1, len(items))
    html = "".join(tile(lab, val, sub, tone) for lab, val, sub, tone in items)
    st.markdown(f'<div class="kpi-grid n-{n}">{html}</div>', unsafe_allow_html=True)


def rupees(v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a/1e7:,.2f} cr"
    if a >= 1e5:
        return f"{sign}₹{a/1e5:,.2f} L"
    return f"{sign}₹{a:,.0f}"


def _esc(s) -> str:
    return html_mod.escape("" if s is None else str(s), quote=True)


def _td(label: str, inner: str, cls: str = "") -> str:
    extra = f' class="{cls}"' if cls else ""
    return (f'<td{extra} data-label="{_esc(label)}">'
            f'<span class="m-lab">{_esc(label)}</span>'
            f'<span class="m-val">{inner}</span></td>')


_AVATAR = ["#2563eb", "#7c3aed", "#0891b2", "#059669", "#d97706",
           "#dc2626", "#db2777", "#4f46e5", "#0f766e", "#b45309"]


def _avatar(sym: str) -> str:
    s = str(sym or "?")
    col = _AVATAR[sum(ord(c) for c in s) % len(_AVATAR)]
    return f'<span class="saas-av" style="background:{col}">{_esc(s[:1])}</span>'


def _px(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"₹{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _signed_rupees(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    v = float(v)
    core = rupees(abs(v)).lstrip("₹")
    if v > 0:
        return f"+₹{core}"
    if v < 0:
        return f"-₹{core}"
    return "₹0"


def _tone_cls(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v) or v == 0:
        return ""
    return "pos" if v > 0 else "neg"


def saas_hold_html(df: pd.DataFrame, capital: float | None = None,
                   risk_total: float | None = None) -> str:
    """Option-1 SaaS positions table — not a Streamlit dataframe."""
    if df is None or df.empty:
        return '<div class="ptable-empty">No open positions.</div>'
    heads = ["Stock", "Entry", "Qty", "Avg price", "CMP", "Stop", "Value",
             "Unrealised", "P&L %", "Open risk", "Next"]
    th = "".join(f'<th{" class=num" if h not in ("Stock","Entry","Next") else ""}>{h}</th>'
                 for h in heads)
    table_rows = []
    compact_rows = []
    for _, r in df.iterrows():
        pnl = r.get("unrealised")
        pct = r.get("gain_%")
        sub = r.get("sector") or r.get("index") or ""
        nxt = r.get("next step") or "—"
        cmp_ = r.get("last_price")
        stop = r.get("live_stop")
        if stop is None or (isinstance(stop, float) and not np.isfinite(stop)):
            stop = r.get("20 EMA")
        qty = float(r.get("open_qty") or 0)
        entry = float(r.get("entry_price") or 0)
        risk_rs = r.get("open_risk")
        risk_note = ""
        try:
            if risk_rs is None and entry > 0 and pd.notna(stop) and float(stop) > 0:
                risk_rs = max(0.0, (entry - float(stop)) * qty)
            if risk_rs is not None:
                risk_rs = float(risk_rs)
                cost = entry * qty
                if risk_rs <= 0.5:
                    risk_note = "capital protected"
                elif cost > 0:
                    risk_note = f"{risk_rs / cost * 100:.1f}% of cost"
        except (TypeError, ValueError):
            risk_rs = None
            risk_note = ""
        stock = (
            f'<div class="saas-stock">{_avatar(r.get("symbol"))}'
            f'<div><div class="saas-sym">{_esc(r.get("symbol"))}</div>'
            f'<div class="saas-sub">{_esc(sub)}</div></div></div>'
        )
        when = _esc(pd.Timestamp(r.get("entry_date")).strftime("%d %b %Y")
                    if pd.notna(r.get("entry_date")) else "—")
        pct_s = "" if pct is None or not np.isfinite(pct) else f"{float(pct):+.2f}%"
        risk_s = (rupees(risk_rs) if risk_rs is not None else "—")
        if risk_note:
            risk_s += f'<div class="saas-sub">{risk_note}</div>'
        table_rows.append(
            "<tr>"
            + _td("Stock", stock)
            + _td("Entry", when)
            + _td("Qty", str(int(qty)), "num")
            + _td("Avg price", _px(r.get("entry_price")), "num")
            + _td("CMP", _px(cmp_), "num")
            + _td("Stop", _px(stop), "num")
            + _td("Value", rupees(r.get("open_value")), "num")
            + _td("Unrealised", _signed_rupees(pnl), f"num {_tone_cls(pnl)}")
            + _td("P&L %", pct_s, f"num {_tone_cls(pct)}")
            + _td("Open risk", risk_s, "num")
            + _td("Next", f'<div class="saas-next">{_esc(nxt)}</div>')
            + "</tr>"
        )
        compact_rows.append(
            "<tr>"
            f'<td>{stock}</td>'
            f'<td class="num">{_px(r.get("entry_price"))}</td>'
            f'<td class="num">{_px(cmp_)}</td>'
            f'<td class="num {_tone_cls(pct)}">{pct_s or "—"}</td>'
            f'<td class="num">{_px(stop)}</td>'
            "</tr>"
        )
    tot = (float(risk_total) if risk_total is not None
           else float(pd.to_numeric(df["open_risk"], errors="coerce").fillna(0).sum())
           if "open_risk" in df.columns else 0.0)
    cap_bit = ""
    if capital and float(capital) > 0:
        cap_bit = f" · {tot / float(capital) * 100:,.2f}% of capital ({rupees(capital)})"
    tfoot = (f'<tfoot><tr><td colspan="{len(heads)}" class="num">'
             f'<b>Open Risk total {rupees(tot)}</b>{cap_bit}</td></tr></tfoot>')
    table = (f'<div class="hold-desktop saas-wrap"><table class="saas-table">'
             f'<thead><tr>{th}</tr></thead><tbody>{"".join(table_rows)}</tbody>'
             f'{tfoot}</table></div>')
    cards = (
        '<div class="hold-mobile">'
        '<table class="hold-compact"><thead><tr>'
        '<th>Stock</th><th class="num">Avg</th><th class="num">CMP</th>'
        '<th class="num">P&L %</th><th class="num">Stop</th>'
        '</tr></thead>'
        f'<tbody>{"".join(compact_rows)}</tbody>'
        f'<tfoot><tr><td colspan="5"><b>Open Risk total {rupees(tot)}</b>{cap_bit}</td></tr></tfoot>'
        '</table></div>'
    )
    return table + cards


def saas_fills_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<div class="ptable-empty">No fills yet.</div>'
    heads = ["Date", "Stock", "Side", "Qty", "Price", "Value", "Reason", "P&L"]
    th = "".join(f'<th{" class=num" if h in ("Qty","Price","Value","P&L") else ""}>{h}</th>'
                 for h in heads)
    rows = []
    for _, r in df.iterrows():
        side = str(r.get("side") or "").upper()
        badge = "badge-buy" if side == "BUY" else "badge-sell"
        pnl = r.get("pnl")
        rows.append(
            "<tr>"
            + _td("Date", _esc(pd.Timestamp(r["date"]).strftime("%d %b %Y") if pd.notna(r.get("date")) else "—"))
            + _td("Stock", f'<div class="saas-stock">{_avatar(r.get("symbol"))}<div class="saas-sym">{_esc(r.get("symbol"))}</div></div>')
            + _td("Side", f'<span class="badge {badge}">{_esc(side)}</span>')
            + _td("Qty", str(int(r.get("qty") or 0)), "num")
            + _td("Price", _px(r.get("price")), "num")
            + _td("Value", rupees(r.get("value")), "num")
            + _td("Reason", _esc(r.get("reason") or "—"))
            + _td("P&L", _signed_rupees(pnl) if side == "SELL" else "—", f"num {_tone_cls(pnl)}")
            + "</tr>"
        )
    return (f'<div class="saas-wrap"><table class="saas-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def saas_simple_html(df: pd.DataFrame, money=(), pct=()) -> str:
    if df is None or df.empty:
        return '<div class="ptable-empty">No rows yet.</div>'
    cols = list(df.columns)
    th = "".join(f'<th{" class=num" if c in money or c in pct else ""}>{_esc(c)}</th>' for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if c in money:
                signed = str(c).lower() in ("unrealised", "unrealized", "p&l", "pnl", "net")
                shown = _signed_rupees(v) if signed else rupees(v)
                tds.append(_td(str(c), shown, f'num {_tone_cls(v) if signed else ""}'))
            elif c in pct:
                try:
                    fv = float(v)
                    tds.append(_td(str(c), f"{fv:+.1f}%" if np.isfinite(fv) else "—",
                                   f"num {_tone_cls(fv)}"))
                except (TypeError, ValueError):
                    tds.append(_td(str(c), _esc(v), "num"))
            else:
                tds.append(_td(str(c), _esc(v)))
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="saas-wrap"><table class="saas-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _safe_ratio(v: float) -> str:
    """Sharpe/Calmar on a flat or near-flat curve divide by ~zero and blow up.
    Show a dash rather than a number with sixteen digits in it."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v) or abs(v) > 100:
        return "—"
    return f"{v:,.2f}"


def tone_of(v: float) -> str:
    return "pos" if (v or 0) > 0 else ("neg" if (v or 0) < 0 else "")


def split_pending(pending: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profit-booking rungs vs stop / trail exits."""
    empty = pending.iloc[0:0] if pending is not None else pd.DataFrame()
    if pending is None or pending.empty:
        return empty, empty
    r = pending["rung"].astype(str)
    book = pending[r.str.startswith("profit_")]
    stop = pending[~r.str.startswith("profit_")]
    return book, stop


def take_action_html(pending: pd.DataFrame, week) -> str:
    """First-glance banner: N actions, X profit booking, Y SL."""
    book_rows, stop_rows = split_pending(pending)
    n_book = int(book_rows["symbol"].nunique()) if len(book_rows) else 0
    n_sl = int(stop_rows["symbol"].nunique()) if len(stop_rows) else 0
    n_act = len(pending) if pending is not None and not pending.empty else 0
    week_s = pd.Timestamp(week).strftime("%d %b %Y")
    if n_act == 0:
        return (
            '<div class="act-banner ok">'
            '<div class="act-kicker">Take action</div>'
            '<div class="act-headline">No action this week</div>'
            f'<div class="act-sub">Hold. Nothing booked a target or hit a stop · week ending {week_s}.</div>'
            "</div>"
        )
    bits = []
    if n_book:
        bits.append(f"<b>{n_book}</b> stock{'s' if n_book != 1 else ''} profit booking")
    if n_sl:
        bits.append(f"<b>{n_sl}</b> stock{'s' if n_sl != 1 else ''} SL")
    rows = []
    for _, r in pending.iterrows():
        is_book = str(r.get("rung") or "").startswith("profit_")
        tag = "tag-book" if is_book else "tag-sl"
        tag_txt = "BOOK" if is_book else "SL"
        pnl = r.get("est_pnl")
        rows.append(
            '<div class="act-row">'
            f'<div class="saas-stock">{_avatar(r.get("symbol"))}'
            f'<div class="sym">{_esc(r.get("symbol"))}</div></div>'
            f'<span class="tag {tag}">{tag_txt}</span>'
            f'<div class="act-why">{_esc(r.get("why") or "")}</div>'
            f'<div class="act-meta">qty {int(r.get("qty") or 0)} · '
            f'<span class="{_tone_cls(pnl)}">{_signed_rupees(pnl)}</span></div>'
            "</div>"
        )
    return (
        f'<div class="act-banner hot">'
        f'<div class="act-kicker">Take action</div>'
        f'<div class="act-headline">{n_act} action{"s" if n_act != 1 else ""} this week</div>'
        f'<div class="act-sub">{" · ".join(bits)} · week ending {week_s}</div>'
        + "".join(rows) + "</div>"
    )


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_panel(symbols: tuple[str, ...], start: str, end: str, demo: bool):
    if demo:
        panel = data_mod.synthetic_panel(list(symbols), start="2016-01-01", end=end)
        shares = data_mod.synthetic_shares(list(symbols))
        return panel, shares
    store = data_mod.PriceStore()
    panel = store.get(list(symbols), start, end)
    shares = data_mod.fetch_shares_outstanding(list(symbols))
    return panel, shares


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_fundamentals(symbols: tuple[str, ...], demo: bool):
    """Statements for a handful of symbols, cached on disk for a month.

    Only ever called on the names that actually qualified in a week — pulling
    five hundred companies' statements to rank six of them would be slow and
    rude to Yahoo. Anything it cannot fetch comes back marked unavailable.
    """
    if demo or not symbols:
        return pd.DataFrame()
    try:
        return fund_mod.score_frame(list(symbols))
    except Exception as exc:                                   # noqa: BLE001
        st.warning(f"Fundamentals could not be loaded ({type(exc).__name__}). "
                   "Ranking is on the technical score alone.")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_benchmark(candidates: tuple[str, ...], start: str, end: str, demo: bool):
    """Try each candidate ticker; return (series, ticker_used, tried).

    Yahoo's Indian index symbols are inconsistent, so the app finds one that
    works rather than assuming, and reports which it used.
    """
    if demo:
        sym = candidates[0]
        p = data_mod.synthetic_panel([sym], start="2016-01-01", end=end)
        return p["Close"][sym], sym, list(candidates)

    store = data_mod.PriceStore()
    for sym in candidates:
        try:
            p = store.get([sym], start, end)
            close = p.get("Close", pd.DataFrame())
            if not close.empty and sym in close.columns:
                series = close[sym].dropna()
                if len(series) > 200:
                    return series, sym, list(candidates)
        except Exception:
            continue
    return pd.Series(dtype=float), None, list(candidates)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_index_list(index_name: str):
    return data_mod.fetch_index_constituents(index_name)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_all_nse():
    return data_mod.fetch_all_nse_equities()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 30)
def load_sectors(symbols: tuple[str, ...], demo: bool):
    """Sector per symbol, seeded from NSE's index files and topped up from Yahoo.

    Only ever asked for the names that qualified this week, so the Yahoo leg is a
    handful of lookups, and every answer is cached on disk permanently.
    """
    if demo or not symbols:
        b = data_mod.bundled_sectors()
        return {s: b.get(s, "") for s in symbols}
    seed = st.session_state.get("industry_seed")
    if seed is None:
        seed = {}
        for idx in ("Nifty Total Market (750)", "Nifty 500"):
            seed.update(data_mod.index_industries(idx))
        st.session_state["industry_seed"] = seed
    try:
        return data_mod.fetch_sectors(list(symbols), seed=seed)
    except Exception:
        b = data_mod.bundled_sectors()
        return {s: b.get(s, "") for s in symbols}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_index_buckets(symbols: tuple[str, ...], demo: bool):
    """(symbol -> NSE size bucket, notes). Nine small CSVs, cached for a week.

    Index constituents are reviewed twice a year, so this is about as static as
    market data gets — and being one review behind never changes a number by
    enough to matter.
    """
    if not symbols:
        return {}, []
    try:
        sets, notes = ix_mod.index_sets(offline=bool(demo))
        return ix_mod.buckets(list(symbols), sets), notes
    except Exception:                                          # noqa: BLE001
        return {s: "—" for s in symbols}, ["Index lists could not be read."]


# --------------------------------------------------------------------------- #
# parameter sets
# --------------------------------------------------------------------------- #
# Every sidebar control that is part of "the system" carries a key from this
# list. Saving a set is `{k: st.session_state[k] for k in PARAM_KEYS}`; loading
# one deletes those keys so the widgets rebuild from the loaded defaults.
# A key that is missing from a saved set falls back to the widget's own default,
# which is what lets an old set survive a new control being added.
PARAM_KEYS: tuple[str, ...] = (
    "p_dark", "p_uni_src", "p_index", "p_bundled", "p_demo",
    "p_lookback", "p_fresh", "p_ema_fast", "p_ema_slow",
    "p_w_fresh", "p_w_vol", "p_w_mom",
    "p_fund_on", "p_fund_weight", "p_fund_rank", "p_fund_bt",
    "p_entries", "p_div_on", "p_div_max_sector", "p_div_max_rank",
    "p_mcap_seg", "p_mcap_pct_min", "p_mcap_pct_max",
    "p_mcap_rank_min", "p_mcap_rank_max",
    "p_mcap_rupee_on", "p_mcap_min_cr", "p_mcap_max_cr",
    "p_price_on", "p_price_min", "p_vol_on", "p_vol_min",
    "p_avgvol_on", "p_avgvol_min", "p_trades_on", "p_trades_min",
    "p_capital", "p_size_mode", "p_fixed_amount", "p_compound_fixed",
    "p_pct_per_stock", "p_risk_pct", "p_max_cap_pct", "p_compound_other",
    "cap_sl_on", "cap_sl_pct",
    "g1", "p1", "tr1", "trm1", "trp1",
    "g2", "p2", "tr2", "trm2", "trp2",
    "g3", "p3", "tr3", "trm3", "trp3",
    "g4", "p4", "tr4", "trm4", "trp4",
    "p_ema_book", "p_sl_full", "p_targets_on", "p_rearm",
    "df_trend_on", "df_ef", "df_es", "df_sl", "df_a20", "df_a50", "df_a200",
    "df_stack", "df_rsia_on", "df_rsib_on", "df_rsia", "df_rsib", "df_rsip",
    "p_regime_on", "p_bench", "p_regime_ema", "p_regime_sma", "p_regime_mode",
    "p_cost_brok", "p_cost_slip", "p_cost_stt",
    "p_start", "p_end",
)


def active_params() -> dict:
    """The parameter values in force this run.

    Loaded once per session from the last-used set, then held in session state.
    After that the widgets themselves are the source of truth — this dict only
    supplies their *defaults*, which is why it is never rewritten as you click.
    """
    if "_param_values" not in st.session_state:
        name, vals = pm.load_last(PARAMS_DIR())
        st.session_state["_param_values"] = vals
        st.session_state["_param_name"] = name
    return st.session_state.get("_param_values") or {}


def pv(key: str, default):
    """A widget's default: the saved value if there is one, coerced to the
    default's own type so Streamlit doesn't flip a float box into an int box."""
    P = st.session_state.get("_param_values") or {}
    if key not in P:
        return default
    v = P[key]
    try:
        if isinstance(default, bool):
            return bool(v)
        if isinstance(default, float):
            return float(v)
        if isinstance(default, int):
            return int(v)
        if isinstance(default, date):
            return date.fromisoformat(str(v)[:10])
        if isinstance(default, str):
            return str(v)
    except Exception:                                          # noqa: BLE001
        return default
    return v


def pvkw(key: str, default) -> dict:
    """`value=` for a widget whose state the sidebar itself may have rewritten.

    Streamlit warns when a widget is handed both a default and a session-state
    value it did not set itself. Two boxes below are written directly — the
    clamp on the promote-rank floor and the reset when the market-cap segment
    changes — so once the key is in state the default is simply left out. State
    wins in that case anyway; this only keeps the log quiet.
    """
    return {} if key in st.session_state else {"value": pv(key, default)}


def pi(key: str, options: list, default_index: int = 0) -> int:
    """Same idea for selectbox / radio, which want an index, not a value."""
    P = st.session_state.get("_param_values") or {}
    v = P.get(key)
    try:
        return options.index(v)
    except ValueError:
        return default_index


def collect_params() -> dict:
    """Whatever the sidebar is showing right now, as a plain dict."""
    out = {}
    for k in PARAM_KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            out[k] = v.isoformat() if isinstance(v, date) else v
    return out


def apply_params(name: str, values: dict) -> None:
    """Make a saved set the live one: drop the widget state, then rerun.

    Deleting the keys is the point. Streamlit gives session state precedence
    over a widget's `value=`, so leaving the old keys in place would make the
    loaded set do nothing at all.
    """
    st.session_state["_param_values"] = dict(values)
    st.session_state["_param_name"] = name
    for k in PARAM_KEYS:
        st.session_state.pop(k, None)
    for k in ("buys_ran", "pos_ran"):
        st.session_state.pop(k, None)
    st.rerun()


def params_ui() -> None:
    """The save / load box at the top of the sidebar."""
    saved = pm.list_sets(PARAMS_DIR())
    current = st.session_state.get("_param_name") or ""
    label = f"Parameters — {current}" if current else "Parameters"
    with st.expander(label, expanded=False):
        if saved:
            idx = saved.index(current) if current in saved else 0
            pick = st.selectbox("Saved sets", saved, index=idx, key="_param_pick")
            st.caption(pm.describe(PARAMS_DIR(), pick))
            c1, c2 = st.columns(2)
            if c1.button("Load", key="_param_load", **_WIDE):
                apply_params(pick, pm.load_set(PARAMS_DIR(), pick))
            if c2.button("Delete", key="_param_del", **_WIDE):
                pm.delete_set(PARAMS_DIR(), pick)
                if pick == current:
                    st.session_state["_param_name"] = ""
                st.rerun()
        else:
            st.caption("No saved sets yet. Set the sidebar the way you want it, "
                       "give it a name below and save.")

        new_name = st.text_input("Name", value=current, key="_param_name_box",
                                  placeholder="e.g. Live weekly")
        if st.button("Save current settings", type="primary", key="_param_save", **_WIDE):
            if not new_name.strip():
                st.warning("Give the set a name first.")
            else:
                pm.save_set(PARAMS_DIR(), new_name.strip(), collect_params())
                st.session_state["_param_name"] = new_name.strip()
                st.success(f"Saved “{new_name.strip()}”. It will load on next start.")
        st.caption("The set you save or load last is the one the app opens with.")


def data_folder_ui() -> None:
    """Move the journal and the presets somewhere that survives an app upgrade."""
    here = sg.data_dir(APP_DIR)
    warn = sg.data_note(APP_DIR)
    custom = here != APP_DIR
    label = "Data folder — synced" if custom else "Data folder"
    with st.expander(label, expanded=bool(warn)):
        if warn:
            st.error(warn)
        st.caption("Your books and presets live here. Point it at a Google Drive or "
                   "OneDrive folder and they are backed up as you work, open on any "
                   "machine, and are **not** inside the app folder — so a new version "
                   "of the app can never overwrite them.")
        st.code(here, language=None)

        new = st.text_input("Move to", value="" if not custom else here,
                             key="_data_dir_box",
                             placeholder=r"C:\Users\you\Google Drive\BreakoutLab")
        c1, c2 = st.columns(2)
        if c1.button("Use this folder", key="_data_dir_go", **_WIDE):
            ok, msg = sg.check_target(new)
            if not ok:
                st.error(msg)
            else:
                target = os.path.abspath(os.path.expanduser(new.strip()))
                # copy rather than move: nothing of yours is deleted, and an
                # existing file at the destination is reported, never replaced
                copied, skipped = sg.copy_data(here, target)
                sg.set_data_dir(APP_DIR, target)
                st.session_state.pop("book_path", None)
                st.session_state.pop("_book", None)
                st.session_state.pop("_book_cache_path", None)
                st.session_state.pop("_param_values", None)
                st.success(f"Now using {target}. Copied {copied} file(s)."
                           + (f" Left alone (already there): {', '.join(skipped)}."
                              if skipped else ""))
                st.rerun()
        if custom and c2.button("Back to the app folder", key="_data_dir_reset", **_WIDE):
            sg.use_app_folder(APP_DIR)
            for k in ("book_path", "_book", "_book_cache_path", "_param_values"):
                st.session_state.pop(k, None)
            st.rerun()
        st.caption("Copying never deletes or overwrites. The old folder is left exactly "
                   "as it was until you delete it yourself.")

        st.markdown("**Google Drive vault** — phone / kisi PC pe same journal")
        st.caption("Streamlit Cloud ka disk tumhara nahi hai. Drive tumhara hai. "
                   "`google_sync/Code.gs` ko Apps Script me Deploy → Web app, URL yahan paste karo. "
                   "Har save Drive folder **Breakout Lab** me JSON likhega.")
        cur_vault = gv.vault_url(APP_DIR)
        vault_in = st.text_input(
            "Google vault URL",
            value=cur_vault,
            key="_vault_url_box",
            placeholder="https://script.google.com/macros/s/…/exec",
        )
        vc1, vc2 = st.columns(2)
        if vc1.button("Push book to Drive", type="primary", key="_vault_push"):
            gv.set_vault_url(APP_DIR, vault_in)
            book = current_book()
            if not vault_in.strip():
                st.error("Pehle vault URL paste karo.")
            elif book is None:
                st.error("Koi book open nahi — pehle Import karo.")
            else:
                try:
                    blob = json.loads(_book_json_bytes(book).decode("utf-8"))
                    gv.push_book(vault_in.strip(), blob)
                    st.session_state["_vault_ok"] = True
                    st.session_state["_vault_err"] = ""
                    st.success(
                        f"Drive pe likh diya: {book.name} · "
                        f"{sum(1 for p in book.positions if p.is_open())} open · "
                        f"{len(book.ledger)} fills. Ab tab band kar sakte ho."
                    )
                except Exception as exc:
                    st.session_state["_vault_ok"] = False
                    st.error(f"Push fail: {exc}")
        if vc2.button("Pull from Drive", key="_vault_save"):
            gv.set_vault_url(APP_DIR, vault_in)
            st.session_state.pop("_ls_hydrated", None)
            try:
                books = gv.pull_all(vault_in.strip()) if vault_in.strip() else {}
                n = gv.write_pulled(JOURNAL_DIR(), books) if books else 0
                st.session_state["_vault_ok"] = True
                st.session_state["_vault_err"] = ""
                st.success(f"Vault saved. {n} book(s) Drive se aaye."
                           if vault_in.strip() else "Vault URL cleared.")
                if n:
                    st.session_state.pop("book_path", None)
                    st.session_state.pop("_book", None)
                    st.session_state.pop("_book_cache_path", None)
                st.rerun()
            except Exception as exc:
                st.session_state["_vault_ok"] = False
                st.error(f"Drive se pull nahi hua: {exc}")
        if st.session_state.get("_vault_ok"):
            st.caption("Google vault connected.")
        elif st.session_state.get("_vault_err"):
            st.caption(f"Vault last error: {st.session_state['_vault_err']}")


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    s: dict = {}
    active_params()
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand"><div class="sb-mark">BL</div>'
            '<div><div class="sb-name">Breakout Lab</div>'
            '<div class="sb-sub">Weekly NSE breakouts</div></div></div>',
            unsafe_allow_html=True,
        )
        _on_cloud = os.path.isdir("/mount/src")
        if not _on_cloud:
            try:
                with open(os.path.join(APP_DIR, "app.py"), "rb") as _af:
                    st.download_button(
                        "Download app.py",
                        _af.read(),
                        file_name="app.py",
                        mime="text/plain",
                        key="dl_app_py",
                    )
            except OSError:
                pass
            _pc_zip = os.path.join(APP_DIR, "BreakoutLab-PC.zip")
            if os.path.isfile(_pc_zip):
                with open(_pc_zip, "rb") as _zf:
                    st.download_button(
                        "Download PC zip",
                        _zf.read(),
                        file_name="breakout_lab_ui_update.zip",
                        mime="application/zip",
                        key="pc_zip_sidebar",
                        help="Extract → run.bat",
                    )
        params_ui()
        data_folder_ui()
        s["dark"] = False


        # ---------------- universe ---------------- #
        with card("Universe"):
            _srcs = ["NSE index (live)", "Bundled list"]
            src = st.radio("Where the stock list comes from", _srcs,
                            index=pi("p_uni_src", _srcs, 0), horizontal=False, key="p_uni_src")
            if src == "NSE index (live)":
                _choices = list(data_mod.NSE_INDEX_FILES.keys()) + [data_mod.ALL_NSE_LABEL]
                idx_name = st.selectbox("Index", _choices,
                                         index=pi("p_index", _choices, _choices.index("Nifty 500")),
                                         key="p_index")
                if idx_name == data_mod.ALL_NSE_LABEL:
                    st.caption("Every mainboard equity NSE lists (series EQ and BE) — about "
                               "2,000 names against 750 in the widest index. **The first price "
                               "download takes tens of minutes and a few hundred MB of cache.** "
                               "After that it is incremental like any other list.")
                if st.button("Fetch list from NSE", **_WIDE):
                    if idx_name == data_mod.ALL_NSE_LABEL:
                        with st.spinner("Downloading the full NSE equity list…"):
                            syms, note = fetch_all_nse()
                    else:
                        syms, note = fetch_index_list(idx_name)
                    st.session_state["universe_symbols"] = syms
                    st.session_state["universe_note"] = note
                syms = st.session_state.get("universe_symbols") or []
                if not syms:
                    syms = uni_mod.LIVE_INDEX_FALLBACKS.get(idx_name, uni_mod.TOTAL_MARKET_FALLBACK)
                    st.caption(f"Using the bundled fallback list ({len(syms)} symbols). "
                               "Press *Fetch list from NSE* for the current one.")
                else:
                    st.caption(st.session_state.get("universe_note", ""))
            else:
                _bundled = [k for k in uni_mod.BUILTIN_UNIVERSES if k.startswith("Stocks")]
                name = st.selectbox("Bundled list", _bundled,
                                     index=pi("p_bundled", _bundled, 0), key="p_bundled")
                syms = uni_mod.BUILTIN_UNIVERSES[name].symbols
                st.caption(f"{len(syms)} symbols.")
            s["symbols"] = uni_mod.clean_symbols(list(syms))

            s["demo"] = st.toggle("Demo mode (fake prices, no internet)",
                                   value=pv("p_demo", False), key="p_demo",
                                   help="Deterministic synthetic data so you can explore the app "
                                        "offline. The numbers mean nothing.")

            # ---------------- breakout ---------------- #
        with card("The breakout"):
            lookback = st.select_slider("New high over", options=[52, 100, 150, 200],
                                         value=pv("p_lookback", 52), key="p_lookback",
                                         format_func=lambda x: f"{x} weeks")
            fresh = st.slider("Must not have made a new high for the previous … weeks", 0, 10,
                              pv("p_fresh", 5), key="p_fresh",
                              help="Your Chartink scan checks the last 5 weeks. This is that number.")
            s["breakout"] = BreakoutConfig(
                lookback_weeks=int(lookback), fresh_weeks=int(fresh),
                ema_fast=st.number_input("Trailing-stop EMA (weekly)", 5, 100,
                                          pv("p_ema_fast", 20), key="p_ema_fast"),
                ema_slow=st.number_input("Final-stop EMA (weekly)", 10, 200,
                                          pv("p_ema_slow", 50), key="p_ema_slow"),
            )
            with st.expander("Technical score — how the qualifiers are ranked"):
                st.caption("Every stock that qualifies is ranked, best to worst. "
                           "These three are percentile-ranked within the week's candidates.")
                w1 = st.slider("Freshness (smaller break = better)", 0.0, 1.0,
                                pv("p_w_fresh", 0.00), 0.05, key="p_w_fresh")
                w2 = st.slider("Volume surge", 0.0, 1.0,
                                pv("p_w_vol", 0.00), 0.05, key="p_w_vol")
                w3 = st.slider("Momentum over the lookback", 0.0, 1.0,
                                pv("p_w_mom", 1.00), 0.05, key="p_w_mom")
                s["breakout"].w_freshness, s["breakout"].w_volume, s["breakout"].w_momentum = w1, w2, w3

            # ---------------- fundamentals ---------------- #
        with card("Fundamentals"):
            s["use_fundamentals"] = st.toggle(
                "Fundamentals analysis", value=pv("p_fund_on", True), key="p_fund_on",
                help="On: the final rank is Technical + Fundamentals. Off: technical only, "
                     "and the combined score is simply the technical score.")
            s["fund_weight"] = 50
            s["fund_in_backtest"] = False
            if s["use_fundamentals"]:
                s["fund_weight"] = st.slider(
                    "How much the business counts", 0, 100, pv("p_fund_weight", 50), 5,
                    key="p_fund_weight",
                    help="0 = chart only. 100 = business only. 50 is a plain average of the two, "
                         "which ranks the same as adding them.")
                s["breakout"].rank_fundamentals = st.toggle(
                    "Rank fundamentals within the week", value=pv("p_fund_rank", True),
                    key="p_fund_rank",
                    help="On (recommended): the fundamentals score is percentile-ranked inside "
                         "the week's candidates before blending, the same way the technical score "
                         "already is. Only then does the slider above mean what it says — a raw "
                         "fundamentals score is far less spread out than a percentile, so at a "
                         "nominal 50/50 the chart would otherwise get about three quarters of the "
                         "say. The absolute score stays visible in the table either way.")
                s["fund_in_backtest"] = st.toggle(
                    "Also use it in the backtest", value=pv("p_fund_bt", False), key="p_fund_bt",
                    help="Off by default on purpose. Yahoo only serves TODAY's statements, so "
                         "ranking a 2021 breakout with them is look-ahead bias — the backtest "
                         "would know things you could not have known. Fine for this week's buys, "
                         "not for history.")
            s["breakout"].w_technical = (100 - s["fund_weight"]) / 100.0 if s["use_fundamentals"] else 1.0
            s["breakout"].w_fundamental = s["fund_weight"] / 100.0 if s["use_fundamentals"] else 0.0
            s["entries_per_week"] = st.number_input("New entries per week", 1, 20,
                                                     pv("p_entries", 5), key="p_entries")
            s["diversify"] = st.toggle(
                "Spread the week's buys across sectors", value=pv("p_div_on", True), key="p_div_on",
                help="A fresh-breakout scan is not sector-neutral: when capital goods run, "
                     "a dozen capital-goods names break out in the same week and a pure score "
                     "ranking buys eight of them. That is one bet held in eight positions.")
            s["max_per_sector"] = 2
            s["max_promote_rank"] = int(s["entries_per_week"]) * 2
            if s["diversify"]:
                s["max_per_sector"] = int(st.number_input(
                    "Max stocks from one sector", 1, 10, pv("p_div_max_sector", 2),
                    key="p_div_max_sector",
                    help="If the cap leaves the list short it is filled by score anyway and "
                         "marked {cap} — a rule that stops you deploying capital costs more "
                         "than the concentration would have."))
                # this box's floor moves with "New entries per week", so a value kept
                # from an earlier setting can fall below it — clamp before the widget
                # is built rather than letting Streamlit raise on a stale state
                _lo = int(s["entries_per_week"])
                _clamp = lambda v: int(min(200, max(_lo, int(v))))          # noqa: E731
                if "p_div_max_rank" in st.session_state:
                    st.session_state["p_div_max_rank"] = _clamp(st.session_state["p_div_max_rank"])
                    _kw = {}
                else:
                    _kw = {"value": _clamp(pv("p_div_max_rank", _lo * 2))}
                s["max_promote_rank"] = int(st.number_input(
                    "Never promote a stock ranked below", _lo, 200,
                    key="p_div_max_rank", **_kw,
                    help="The floor under the sector rule. If 30 names qualify and the 28th has "
                         "a poor chart and poor numbers, buying it to balance a sector is a "
                         "worse decision than holding a third stock from the same sector. "
                         "Diversification never reaches past this rank."))

            # ---------------- screen ---------------- #
        with card("Screen filters"):
            sc = screen_mod.ScreenConfig(enabled=True)

            n_syms = len(s["symbols"])
            _segs = ["Small cap — bottom 50% of your list",
                     "Micro cap — bottom 25% of your list",
                     "Small + Mid — bottom 80% of your list",
                     "Mid cap — 40-70% of your list",
                     "Large cap — top 20% of your list",
                     "AMFI ranks (needs a big list)",
                     "Rupee band",
                     "No cap filter"]
            seg = st.selectbox(
                "Market-cap segment", _segs,
                index=pi("p_mcap_seg", _segs, 0), key="p_mcap_seg",
                help="Recomputed every week from the market caps of the stocks you loaded, so a "
                     "stock that grows out of the band leaves the universe on its own.",
            )
            PCT_SEGMENTS = {
                "Small cap — bottom 50% of your list": (50.0, 100.0),
                "Micro cap — bottom 25% of your list": (75.0, 100.0),
                "Small + Mid — bottom 80% of your list": (20.0, 100.0),
                "Mid cap — 40-70% of your list": (40.0, 70.0),
                "Large cap — top 20% of your list": (0.0, 20.0),
            }
            # picking a different segment must still move the two % boxes with it.
            # They carry a key now, and session state outranks `value=`, so the move
            # has to be made explicitly — but only when the user changed the segment,
            # not on the first run, where a loaded set is the thing that should win.
            if "_last_mcap_seg" in st.session_state and st.session_state["_last_mcap_seg"] != seg:
                if seg in PCT_SEGMENTS:
                    _lo, _hi = PCT_SEGMENTS[seg]
                    st.session_state["p_mcap_pct_min"] = float(_lo)
                    st.session_state["p_mcap_pct_max"] = float(_hi)
            st.session_state["_last_mcap_seg"] = seg

            if seg in PCT_SEGMENTS:
                sc.use_mcap_pct = True
                lo, hi = PCT_SEGMENTS[seg]
                c1, c2 = st.columns(2)
                sc.mcap_pct_min = c1.number_input("From %", 0.0, 100.0, step=5.0,
                                                   key="p_mcap_pct_min",
                                                   **pvkw("p_mcap_pct_min", float(lo)))
                sc.mcap_pct_max = c2.number_input("To %", 0.0, 100.0, step=5.0,
                                                   key="p_mcap_pct_max",
                                                   **pvkw("p_mcap_pct_max", float(hi)))
                keep = int(n_syms * (sc.mcap_pct_max - sc.mcap_pct_min) / 100)
                st.caption(f"0% is the biggest company in your list, 100% the smallest. "
                           f"Of your **{n_syms}** symbols this keeps roughly **{keep}** — "
                           "recomputed every week.")
            elif seg == "AMFI ranks (needs a big list)":
                sc.use_mcap_rank = True
                c1, c2 = st.columns(2)
                sc.mcap_rank_min = int(c1.number_input("From rank", 1, 5000,
                                                        pv("p_mcap_rank_min", 251),
                                                        key="p_mcap_rank_min"))
                sc.mcap_rank_max = int(c2.number_input("To rank", 1, 100_000,
                                                        pv("p_mcap_rank_max", 10_000),
                                                        key="p_mcap_rank_max"))
                if sc.mcap_rank_min > n_syms:
                    st.error(f"**Nothing can qualify.** Your list has {n_syms} symbols, so ranks only "
                              f"run 1–{n_syms} — a band starting at {sc.mcap_rank_min} matches nothing. "
                              "Load Nifty Total Market (750), or use a percentage segment instead.")
                else:
                    st.caption(f"AMFI: 1–100 large, 101–250 mid, 251+ small. Ranked within your "
                               f"{n_syms} symbols, so this only means what AMFI means if you load a "
                               "list of ~750+.")
            elif seg == "Rupee band":
                sc.use_market_cap = True

            if seg in PCT_SEGMENTS or seg == "AMFI ranks (needs a big list)":
                sc.use_market_cap = st.checkbox("Also apply a rupee band",
                                                 value=pv("p_mcap_rupee_on", False),
                                                 key="p_mcap_rupee_on")
            if sc.use_market_cap:
                c1, c2 = st.columns(2)
                sc.mcap_min_cr = c1.number_input("Min (cr)", 0.0, 1e7, pv("p_mcap_min_cr", 500.0),
                                                  step=100.0, key="p_mcap_min_cr")
                sc.mcap_max_cr = c2.number_input("Max (cr)", 0.0, 1e7, pv("p_mcap_max_cr", 50_000.0),
                                                  step=1000.0, key="p_mcap_max_cr")
            sc.use_price = st.checkbox("Price above", value=pv("p_price_on", True), key="p_price_on")
            if sc.use_price:
                sc.price_min = st.number_input("Minimum price", 0.0, 1e6, pv("p_price_min", 30.0),
                                                key="p_price_min")
            sc.use_volume = st.checkbox("Daily volume above", value=pv("p_vol_on", True), key="p_vol_on")
            if sc.use_volume:
                sc.volume_min = st.number_input("Minimum volume", 0.0, 1e9, pv("p_vol_min", 25_000.0),
                                                 step=5000.0, key="p_vol_min")
            sc.use_avg_volume = st.checkbox("SMA(50) of volume above", value=pv("p_avgvol_on", True),
                                             key="p_avgvol_on")
            if sc.use_avg_volume:
                sc.avg_volume_min = st.number_input("Minimum average volume", 0.0, 1e9,
                                                     pv("p_avgvol_min", 50_000.0), step=5000.0,
                                                     key="p_avgvol_min")
            sc.use_trades = st.checkbox("Number of trades (NSE bhavcopy)",
                                         value=pv("p_trades_on", False), key="p_trades_on",
                                         help="The honest stand-in for your buyer/seller-initiated-trades "
                                              "filter — see the note in the Universe tab.")
            if sc.use_trades:
                sc.trades_min = st.number_input("Minimum trades per day", 0.0, 1e7,
                                                 pv("p_trades_min", 400.0), step=50.0,
                                                 key="p_trades_min")
            s["screen"] = sc

            # ---------------- sizing ---------------- #
        with card("Money"):
            cap = st.number_input("Total capital (₹)", 10_000.0, 1e11, pv("p_capital", 5_000_000.0),
                                   step=100_000.0, key="p_capital")
            _modes = ["Fixed ₹ per stock", "Equal capital per stock (%)", "SL-wise (equal risk)"]
            mode = st.radio("Position sizing", _modes,
                             index=pi("p_size_mode", _modes, 0), key="p_size_mode")

            if mode.startswith("Fixed"):
                sz = SizingConfig(capital=float(cap), mode="fixed")
                sz.fixed_amount = st.number_input("Capital per stock (₹)", 1_000.0, 1e9,
                                                   pv("p_fixed_amount", 100_000.0), step=10_000.0,
                                                   key="p_fixed_amount")
                sz.compound = st.toggle("Grow the slice with equity (compound)",
                                         value=pv("p_compound_fixed", True), key="p_compound_fixed")
                n_max = sz.max_concurrent() or 0
                pct = sz.slice_pct()
                if sz.compound:
                    grown_eq = cap * 1.2
                    st.caption(
                        f"₹{cap:,.0f} ÷ ₹{sz.fixed_amount:,.0f} → up to **{n_max} positions** at once. "
                        f"That slice is **{pct:,.2f}% of capital**, and it grows with the book: at "
                        f"₹{grown_eq:,.0f} of equity a new entry becomes "
                        f"₹{sz.fixed_amount * 1.2:,.0f}, still {n_max} slots. It shrinks the same way "
                        "in a drawdown. No caps are applied."
                    )
                else:
                    st.caption(
                        f"₹{cap:,.0f} ÷ ₹{sz.fixed_amount:,.0f} → up to **{n_max} positions** at once. "
                        f"Every entry is exactly ₹{sz.fixed_amount:,.0f} forever, whatever the book "
                        "does. No caps are applied."
                    )
            elif mode.startswith("Equal"):
                sz = SizingConfig(capital=float(cap), mode="equal")
                sz.pct_per_stock = st.slider("Capital per stock (%)", 1.0, 50.0,
                                              pv("p_pct_per_stock", 5.0), 0.5, key="p_pct_per_stock")
                st.caption("A percentage, so the rupee amount moves with your equity if compounding "
                           "is on — two trades years apart will differ in size.")
            else:
                sz = SizingConfig(capital=float(cap), mode="risk")
                sz.risk_pct = st.slider("Risk per trade (% of capital)", 0.1, 10.0,
                                         pv("p_risk_pct", 1.0), 0.1, key="p_risk_pct",
                                         help="Distance from entry to the weekly 20 EMA is the risk.")

            if sz.mode != "fixed":
                sz.max_capital_pct = st.number_input("Max capital in one stock (%)", 1.0, 100.0,
                                                      pv("p_max_cap_pct", 10.0), key="p_max_cap_pct")
                sz.compound = st.toggle("Size off current equity (compound)",
                                         value=pv("p_compound_other", True), key="p_compound_other")

            # applies in every mode — a 30%-away stop is a problem whichever way the
            # position was sized
            if st.toggle("Max SL on one stock", value=pv("cap_sl_on", False), key="cap_sl_on",
                          help="Caps how far below your BUY PRICE the stop may sit. It is a real "
                               "stop: the position exits on a close below that level, alongside "
                               "the weekly 20 EMA rung — whichever is breached first. It also "
                               "sizes the position off whichever stop is tighter."):
                sz.max_stop_distance_pct = st.number_input(
                    "Max SL (% below buy price)", 1.0, 50.0, pv("cap_sl_pct", 20.0), step=0.5,
                    key="cap_sl_pct")
                st.caption(f"Buy at ₹100 → stop never below "
                           f"₹{100 * (1 - sz.max_stop_distance_pct / 100):.0f}. Checked on the "
                           "daily close.")
            s["sizing"] = sz

            # ---------------- ladder ---------------- #
        with card("Exit ladder"):

            def rung_ui(n: int, def_gain: float, def_qty: float):
                c1, c2 = st.columns(2)
                g = c1.number_input(f"Exit {n} — book at gain %", 1.0, 500.0,
                                     pv(f"g{n}", def_gain), key=f"g{n}")
                q = c2.number_input("…this much %", 0.0, 100.0, pv(f"p{n}", def_qty), key=f"p{n}")
                mode, pct = "", 0.0
                if st.checkbox(f"After exit {n}, move the stop on what is left",
                                value=pv(f"tr{n}", False), key=f"tr{n}",
                                help="Applies to the REMAINING quantity, checked on the weekly "
                                     "close like every other rung. A later rung can tighten this "
                                     "floor but never loosen it. A rung set to book 0% with this "
                                     "ticked is a pure stop-move."):
                    _tm = [k for k in TRAIL_MODES if k]
                    mode = st.selectbox("Stop at", _tm,
                                         index=pi(f"trm{n}", _tm, 0), key=f"trm{n}",
                                         format_func=lambda k: TRAIL_MODES[k],
                                         label_visibility="collapsed")
                    if mode == "pct":
                        pct = float(st.number_input("Buy price + %", 0.0, 200.0,
                                                     pv(f"trp{n}", 10.0), step=1.0,
                                                     key=f"trp{n}", label_visibility="collapsed"))
                return float(g), float(q), mode, pct

            g1, p1, m1, l1 = rung_ui(1, 25.0, 20.0)
            g2, p2, m2, l2 = rung_ui(2, 50.0, 20.0)
            g3, p3_q, m3, l3 = rung_ui(3, 75.0, 0.0)
            g4, p4_q, m4, l4 = rung_ui(4, 100.0, 0.0)

            p_ema = st.number_input("Book on close below 20 EMA (%)", 0.0, 100.0,
                                     pv("p_ema_book", 45.0), key="p_ema_book")
            sl_full = st.toggle(
                "20 EMA is a full stop-loss until the first profit books",
                value=pv("p_sl_full", True), key="p_sl_full",
                help="On (your rule): if the breakout fails and the stock closes below the 20 EMA "
                     "before any profit target has been hit, the WHOLE position is sold there — the "
                     "20 EMA is the stop-loss. Once the first target has booked, the same rung goes "
                     "back to trimming the % above. Off: it always trims that %, never a full exit.")

            # the 50 EMA rung is terminal — it sells whatever is left, so its share is
            # computed from the rungs above rather than printed as a fixed number
            _booked = p1 + p2 + p3_q + p4_q + p_ema
            _left = round(100.0 - _booked, 2)
            if _left > 0:
                st.caption(f"50 EMA exits the rest — **{_left:g}%** on a full winner.")
            elif _left == 0:
                st.caption("50 EMA gets 0% on a full winner; it still catches trades that skip a rung.")
            else:
                st.warning(f"These add up to {_booked:g}%. Rungs are capped at what is left, so the "
                           "last to fire sells less than its number says.")

            _tgt = ["Daily close", "Weekly close"]
            s["targets_on_daily_close"] = st.radio(
                "Check profit targets on", _tgt, index=pi("p_targets_on", _tgt, 0),
                key="p_targets_on", horizontal=True,
                help="A stock can trade through +25% on a Wednesday and close the week below it. "
                     "Checked weekly that target never books; checked daily it books on "
                     "Wednesday's close and fills at Thursday's open. The 20/50 EMA stops stay "
                     "on the weekly close either way, and the moved-up trail floor is always "
                     "checked daily.") == "Daily close"

            s["rearm_ema"] = st.toggle(
                "EMA rungs can fire again", value=pv("p_rearm", False), key="p_rearm",
                help="Off: the 20 EMA rung books its 45% once and is spent, exactly as written. "
                     "On: if the stock closes back above the EMA, that rung re-arms, so a later "
                     "break books again — a stop that actually trails. Worth trying both; it "
                     "changes results more than you would expect.")
            s["ladder"] = [
                Rung("profit_1", "gain_pct", g1, p1, trail_to=m1, trail_pct=l1),
                Rung("profit_2", "gain_pct", g2, p2, trail_to=m2, trail_pct=l2),
                Rung("profit_3", "gain_pct", g3, p3_q, trail_to=m3, trail_pct=l3),
                Rung("profit_4", "gain_pct", g4, p4_q, trail_to=m4, trail_pct=l4),
                Rung("trail_ema_fast", "below_ema_fast", 0.0, float(p_ema),
                     full_exit_if_no_profit=bool(sl_full)),
                Rung("trail_ema_slow", "below_ema_slow", 0.0, 15.0, terminal=True),
            ]

            # ---------------- daily confirmation ---------------- #
        with card("Daily timeframe filter"):
            df_ = DailyFilterConfig()
            df_.use_trend = st.toggle("Daily trend stack", value=pv("df_trend_on", False),
                                       key="df_trend_on")
            if df_.use_trend:
                # periods first, so the tick labels below name the numbers actually in use
                c1, c2, c3 = st.columns(3)
                df_.ema_fast = int(c1.number_input("EMA", 2, 400, pv("df_ef", 20), key="df_ef"))
                df_.ema_slow = int(c2.number_input("EMA ", 2, 400, pv("df_es", 50), key="df_es"))
                df_.sma_long = int(c3.number_input("SMA", 2, 500, pv("df_sl", 200), key="df_sl"))
                c1, c2, c3 = st.columns(3)
                df_.above_ema_fast = c1.checkbox(f"Price > {df_.ema_fast} EMA",
                                                  value=pv("df_a20", True), key="df_a20")
                df_.above_ema_mid = c2.checkbox(f"Price > {df_.ema_slow} EMA",
                                                 value=pv("df_a50", False), key="df_a50")
                df_.above_sma_long = c3.checkbox(f"Price > {df_.sma_long} SMA",
                                                  value=pv("df_a200", False), key="df_a200")
                df_.require_full_stack = st.checkbox(
                    f"Price > {df_.ema_fast} EMA > {df_.ema_slow} EMA > {df_.sma_long} SMA",
                    value=pv("df_stack", False), key="df_stack",
                    help="Also requires the averages themselves to be in order, not just price "
                         "above them.")
                if not (df_.above_ema_fast or df_.above_ema_mid or df_.above_sma_long
                        or df_.require_full_stack):
                    st.warning("Nothing is ticked — the trend filter is doing nothing.")

            c1, c2 = st.columns(2)
            df_.use_rsi_above = c1.toggle("Daily RSI above", value=pv("df_rsia_on", False),
                                           key="df_rsia_on")
            df_.use_rsi_below = c2.toggle("Daily RSI below", value=pv("df_rsib_on", False),
                                           key="df_rsib_on")
            if df_.use_rsi_above or df_.use_rsi_below:
                c1, c2, c3 = st.columns(3)
                if df_.use_rsi_above:
                    df_.rsi_above = c1.number_input("Above", 0.0, 100.0, pv("df_rsia", 60.0),
                                                     key="df_rsia")
                if df_.use_rsi_below:
                    df_.rsi_below = c2.number_input("Below", 0.0, 100.0, pv("df_rsib", 80.0),
                                                     key="df_rsib")
                df_.rsi_period = int(c3.number_input("RSI period", 2, 50, pv("df_rsip", 14),
                                                      key="df_rsip"))
                if df_.use_rsi_above and df_.use_rsi_below and df_.rsi_above >= df_.rsi_below:
                    st.error(f"Above {df_.rsi_above:g} and below {df_.rsi_below:g} is an empty band — "
                             "nothing can pass both.")
            s["daily_filter"] = df_

            # ---------------- regime ---------------- #
        with card("Market regime filter"):
            rg = RegimeConfig()
            rg.enabled = st.toggle("Stop new entries in a weak market",
                                    value=pv("p_regime_on", True), key="p_regime_on")
            if rg.enabled:
                _benches = list(uni_mod.BENCHMARK_CANDIDATES.keys())
                bench_name = st.selectbox("Index to judge the market by", _benches,
                                           index=pi("p_bench", _benches, 0), key="p_bench")
                s["bench_candidates"] = tuple(uni_mod.BENCHMARK_CANDIDATES[bench_name])
                s["bench_name"] = bench_name
                rg.use_ema = st.checkbox(f"Block when it is below its weekly {rg.ema_period} EMA",
                                          value=pv("p_regime_ema", True), key="p_regime_ema")
                rg.use_sma = st.checkbox(f"Block when it is below its daily {rg.sma_period} SMA",
                                          value=pv("p_regime_sma", True), key="p_regime_sma")
                _rmode = ["either is true", "both are true"]
                rg.mode = "any" if st.radio("Block if", _rmode, index=pi("p_regime_mode", _rmode, 0),
                                             key="p_regime_mode") == "either is true" else "all"
            else:
                s["bench_candidates"] = tuple(uni_mod.BENCHMARK_CANDIDATES["Nifty 50"])
                s["bench_name"] = "Nifty 50"
            s["regime"] = rg

            # ---------------- costs ---------------- #
            with st.expander("Costs"):
                cst = CostConfig()
                cst.brokerage_pct = st.number_input("Brokerage per side (%)", 0.0, 2.0,
                                                     pv("p_cost_brok", 0.05), 0.01, key="p_cost_brok")
                cst.slippage_pct = st.number_input("Slippage per side (%)", 0.0, 5.0,
                                                    pv("p_cost_slip", 0.15), 0.05, key="p_cost_slip")
                cst.stt_pct = st.number_input("STT etc. on sells (%)", 0.0, 2.0,
                                               pv("p_cost_stt", 0.10), 0.01, key="p_cost_stt")
                s["costs"] = cst

            # ---------------- dates ---------------- #
        with card("Backtest window"):
            today = date.today()
            s["start"] = st.date_input("From", value=pv("p_start", date(today.year - 6, 1, 1)),
                                        key="p_start")
            s["end"] = st.date_input("To", value=pv("p_end", today), key="p_end")
    return s


# --------------------------------------------------------------------------- #
# shared: build everything the tabs need
# --------------------------------------------------------------------------- #
def live_history_weeks(cfg: BreakoutConfig) -> int:
    """How much history a LIVE scan needs — which is far less than a backtest.

    The 52-week high needs its 52 weeks, the freshness test a few more, and the
    50-week EMA wants roughly two of its own periods to stop being a straight
    line. That is about three years. Asking for the backtest's six was quietly
    dropping every company that listed inside them.
    """
    return int(cfg.lookback_weeks + cfg.fresh_weeks + 2 * cfg.ema_slow + 8)


def signal_days(cfg: BreakoutConfig) -> int:
    """Trading days a symbol must have before the scan can say anything about it."""
    return int((cfg.lookback_weeks + cfg.fresh_weeks + 1) * 5)


def build_context(s: dict, for_live: bool = False, restrict_to: list[str] | None = None):
    """Download data, compute signals and the screen. Cached where it matters.

    `restrict_to` narrows the download to a named handful. It is used when an
    imported Chartink list is driving the week: on that path the app's own
    freshness test and screen filters are skipped, and the technical score is
    percentile-ranked *within the week's candidates*, so nothing outside the
    imported list can affect a single number on screen. Loading five hundred
    extra symbols to compute nothing with them is pure waiting.
    """
    today = date.today()
    if for_live:
        # a live scan reads the last few years, never the backtest's window
        start = pd.Timestamp(today) - pd.Timedelta(weeks=live_history_weeks(s["breakout"]))
        end = pd.Timestamp(today)
    else:
        start = pd.Timestamp(s["start"])
        end = pd.Timestamp(s["end"])
    symbols = list(s["symbols"])

    # anything held in the journal must be priced too, even if it left the index
    book = current_book()
    held = set(book.open_symbols()) if book is not None else set()
    # …and so must an imported Chartink list, whose names are mostly NOT in any
    # index file — that is the whole reason for importing one
    imported = st.session_state.get("chartink_symbols") or []

    if restrict_to is not None:
        symbols = sorted(set(restrict_to) | held)
    else:
        symbols = sorted(set(symbols) | held)

    with st.spinner(
        f"Loading prices for {len(symbols)} "
        + ("Chartink + open positions…" if restrict_to is not None else "symbols…")
    ):
        panel, shares = load_panel(tuple(symbols), str(start.date()), str(end.date()), s["demo"])

    close = panel.get("Close", pd.DataFrame())
    if close.empty:
        st.error("No price data came back. Check the internet connection, or tick Demo mode.")
        st.stop()

    # what the download really returned, recorded before anything is filtered —
    # so a symbol that goes missing can say WHY rather than "no price history"
    coverage = data_mod.column_coverage(close)

    need = signal_days(s["breakout"])
    if for_live:
        # judge a live symbol on whether its RECENT history is complete, not on
        # whether it existed six years ago
        panel, dropped = data_mod.align_panel(panel, start, end,
                                               min_history_days=need, tail_days=need)
    else:
        panel, dropped = data_mod.align_panel(panel, start, end)
    mcap = data_mod.market_cap_frame(panel.get("RawClose", close), shares)

    trades_wide = None
    if s["screen"].use_trades and not s["demo"]:
        cal = panel["Close"].index
        with st.spinner("Fetching NSE bhavcopy for trade counts (first run is slow)…"):
            trades_wide, _deliv, rep = nse_mod.fetch_bhavcopy(cal, list(panel["Close"].columns))
        if rep.get("failed", 0) > rep.get("days", 1) * 0.5:
            st.warning("NSE bhavcopy mostly failed — the trades filter is not running. "
                       f"{rep.get('failed')} of {rep.get('days')} days could not be fetched.")

    sig = compute_signals(panel, s["breakout"])
    dfc = s.get("daily_filter")
    daily_ok, daily_info = (pd.DataFrame(), {})
    if dfc is not None and (dfc.use_trend or dfc.use_rsi_above or dfc.use_rsi_below):
        daily_ok, daily_info = daily_filter_mask(panel, sig.weekly["Close"].index, dfc)
    screen_res = screen_mod.build_screen(panel, s["screen"], market_cap=mcap, trades=trades_wide)

    # weekly volume surge, for scoring
    vs_daily = volume_surge_daily(panel, 50)
    vs_weekly = vs_daily.resample("W-FRI").last() if not vs_daily.empty else pd.DataFrame()

    bench, bench_used, bench_tried = load_benchmark(
        s["bench_candidates"], str(start.date()), str(end.date()), s["demo"])
    if bench.empty and s["regime"].enabled:
        st.warning(f"**The regime filter has no index data.** None of these tickers returned "
                   f"anything for {s.get('bench_name', 'the benchmark')}: "
                   f"{', '.join(bench_tried)}. The filter is switched off for this run — "
                   "results below are unfiltered.")

    return {
        "panel": panel, "signals": sig, "screen": screen_res, "mcap": mcap,
        "vol_surge_weekly": vs_weekly, "bench": bench, "dropped": dropped,
        "coverage": coverage, "need_days": need, "start": start, "end": end,
        "shares": shares, "symbols": symbols,
        "bench_used": bench_used, "bench_tried": bench_tried,
        "daily_ok": daily_ok, "daily_info": daily_info,
    }


# --------------------------------------------------------------------------- #
# journal helpers
# --------------------------------------------------------------------------- #
def current_book() -> jn.Book | None:
    path = st.session_state.get("book_path")
    if not path or not os.path.exists(path):
        return None
    if st.session_state.get("_book_cache_path") != path:
        st.session_state["_book"] = jn.load_book(path)
        st.session_state["_book_cache_path"] = path
    return st.session_state.get("_book")


def persist_book(book: jn.Book) -> None:
    book.refresh_compounding()
    path = jn.save_book(JOURNAL_DIR(), book)
    st.session_state["book_path"] = path
    st.session_state["_book"] = book
    st.session_state["_book_cache_path"] = path
    snapshot_books_to_browser()
    url = gv.vault_url(APP_DIR)
    if url:
        try:
            blob = json.loads(_book_json_bytes(book).decode("utf-8"))
            gv.push_book(url, blob)
            st.session_state["_vault_ok"] = True
            st.session_state["_vault_err"] = ""
        except Exception as exc:
            st.session_state["_vault_ok"] = False
            st.session_state["_vault_err"] = f"{type(exc).__name__}: {exc}"


_LS_KEY = "breakout_lab_books"


def _ls_all():
    """Not used. The streamlit_local_storage component broke the web preview."""
    return {}


def snapshot_books_to_browser() -> None:
    return


def hydrate_books_from_browser() -> None:
    """Pull from Google Drive vault if a URL is saved. Journal otherwise stays on disk."""
    if st.session_state.get("_ls_hydrated"):
        return
    st.session_state["_ls_hydrated"] = True
    url = gv.vault_url(APP_DIR)
    if not url:
        return
    if not sg.load_settings(APP_DIR).get("vault_url"):
        gv.set_vault_url(APP_DIR, url)
    if jn.list_books(JOURNAL_DIR()):
        return
    try:
        books = gv.pull_all(url)
        if books:
            gv.write_pulled(JOURNAL_DIR(), books)
            st.session_state["_vault_ok"] = True
            st.session_state["_vault_err"] = ""
    except Exception as exc:
        st.session_state["_vault_ok"] = False
        st.session_state["_vault_err"] = f"{type(exc).__name__}: {exc}"


def _book_json_bytes(book: jn.Book) -> bytes:
    path = st.session_state.get("book_path")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    # fall back to a live dump if the file is not on disk yet
    tmp = jn.save_book(JOURNAL_DIR(), book)
    with open(tmp, "rb") as f:
        return f.read()


def _export_pack(book: jn.Book) -> bytes:
    """One zip: book JSON + open positions CSV + journal/fills CSV + closed trades CSV."""
    led = jn.ledger_frame(book)
    opens = jn.open_positions_frame(book)
    rt = js.round_trips(book)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{book.name}.json", _book_json_bytes(book))
        z.writestr(f"{book.name}_open_positions.csv", opens.to_csv(index=False))
        z.writestr(f"{book.name}_journal.csv", led.to_csv(index=False))
        if len(rt):
            z.writestr(f"{book.name}_closed_trades.csv", rt.to_csv(index=False))
    return buf.getvalue()


def book_tools_ui(book: jn.Book | None, key_prefix: str) -> None:
    """Backup / export / import for BOTH dashboards from one strip.

    Import works even when no book is loaded yet — that's how you restore
    after a Cloud wipe or a fresh browser.
    """
    st.markdown('<div class="book-tools">', unsafe_allow_html=True)
    st.caption("Positions **and** journal together — one JSON.")
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 2])

    if book is not None:
        pack = _export_pack(book)
        raw = _book_json_bytes(book)
        c1.download_button(
            "Backup Book",
            raw,
            file_name=f"{book.name}_backup.json",
            mime="application/json",
            key=f"{key_prefix}_dl_json",
            help="Full book: open positions, closed trades, every fill, cash, drafts.",
        )
        c2.download_button(
            "Export All Data",
            pack,
            file_name=f"{book.name}_export.zip",
            mime="application/zip",
            key=f"{key_prefix}_dl_zip",
            help="JSON + open-positions CSV + journal CSV + closed-trades CSV.",
        )
    else:
        c1.caption("No book yet")
        c2.caption("Import a backup to restore.")

    up = c3.file_uploader(
        "Import / Restore",
        type=["json", "zip"],
        key=f"{key_prefix}_up",
        help="Restore a Backup JSON or an Export ZIP.",
    )
    if up is not None and c3.button("Restore into this app", key=f"{key_prefix}_restore"):
        try:
            payload = up.getvalue()
            name = (up.name or "").lower()
            blob = None
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    jsons = [n for n in z.namelist() if n.lower().endswith(".json")]
                    if not jsons:
                        raise ValueError("ZIP has no book JSON inside.")
                    blob = json.loads(z.read(jsons[0]).decode("utf-8"))
            else:
                blob = json.loads(payload.decode("utf-8"))
            if not isinstance(blob, dict) or "name" not in blob:
                raise ValueError("That file is not a Breakout Lab book.")
            dest = jn.book_path(JOURNAL_DIR(), str(blob.get("name") or "imported"))
            os.makedirs(JOURNAL_DIR(), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2, default=str)
            loaded = jn.load_book(dest)
            persist_book(loaded)
            st.success(f"Restored “{loaded.name}” — {len(loaded.positions)} open, "
                       f"{len(loaded.closed)} closed, {len(loaded.ledger)} fills.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore: {exc}")

    if book is not None and c4.button("Save snapshot to backups folder", key=f"{key_prefix}_snap"):
        files = {
            f"{book.name}.json": raw,
            f"{book.name}_export.zip": pack,
        }
        folder = sg.write_backup(APP_DIR, book.name, files)
        st.success(f"Saved snapshot to {folder}")
    prev = sg.list_backups(APP_DIR)
    if prev:
        c4.caption("Recent: " + " · ".join(f"{n}" for n, _ in prev[:3]))
    st.markdown("</div>", unsafe_allow_html=True)


def book_picker(key_prefix: str = "") -> jn.Book | None:
    books = jn.list_books(JOURNAL_DIR())
    labels = [os.path.basename(b)[:-5] for b in books]
    c1, c2 = st.columns([2, 1])
    with c1:
        if books:
            cur = st.session_state.get("book_path")
            idx = books.index(cur) if cur in books else 0
            pick = st.selectbox("Trading book", labels, index=idx, key=f"{key_prefix}_pick")
            st.session_state["book_path"] = books[labels.index(pick)]
        else:
            st.info("No trading book yet — create one to start journalling.")
    with c2:
        with st.popover("New book", **_WIDE):
            name = st.text_input("Name", value="My breakout book", key=f"{key_prefix}_newname")
            cap = st.number_input("Starting capital (₹)", 10_000.0, 1e10, 1_000_000.0,
                                   step=50_000.0, key=f"{key_prefix}_newcap")
            if st.button("Create", key=f"{key_prefix}_create"):
                b = jn.Book(name=name, capital=float(cap), cash=float(cap))
                persist_book(b)
                st.rerun()
    return current_book()


# --------------------------------------------------------------------------- #
# drafts: intentions, until you confirm the fill
# --------------------------------------------------------------------------- #
def drafts_ui(book: jn.Book, last_px: pd.Series | None) -> None:
    """Confirm, edit or drop the buys queued from the buy list.

    A draft holds no cash and has no P&L. Confirming it is the moment the trade
    becomes real, at the quantity and price you actually got — which is also the
    only moment slippage can be measured.
    """
    if not book.drafts:
        return
    need = sum(d.cost() for d in book.drafts)
    with card(f"Draft buys waiting — {len(book.drafts)}",
              f"{rupees(need)} if all fill · intention, not a position"):
        st.caption("These are decisions, not positions. Nothing is spent and no P&L runs "
                   "until you confirm the fill you actually got.")
        show_df(jn.drafts_frame(book, last_px))

        ed = pd.DataFrame([{"confirm": True, "symbol": d.symbol, "qty": d.qty,
                            "fill price": round(d.decision_price, 2),
                            "stop": round(d.stop, 2)} for d in book.drafts])
        edited = st.data_editor(ed, **_WIDE, key="draft_editor", hide_index=True,
                                 disabled=["symbol"])
        c1, c2, c3 = st.columns([1, 1, 2])
        fill_date = c1.date_input("Fill date", value=date.today(), key="draft_fill_date")
        if c2.button("Confirm ticked", type="primary", key="draft_confirm"):
            n, short = 0, []
            for _, r in edited.iterrows():
                if not bool(r["confirm"]) or int(r["qty"]) <= 0:
                    continue
                cost = int(r["qty"]) * float(r["fill price"])
                if cost > book.cash:
                    short.append(str(r["symbol"]))
                    continue
                if book.confirm_draft(str(r["symbol"]), fill_date, int(r["qty"]),
                                       float(r["fill price"]), float(r["stop"])):
                    n += 1
            persist_book(book)
            if short:
                st.warning("Not enough cash for: " + ", ".join(short)
                           + f" (cash is {rupees(book.cash)}). They are still drafts.")
            st.success(f"Confirmed {n}. P&L starts now.")
            st.rerun()
        with c3.popover("Drop a draft", **_WIDE):
            syms = [d.symbol for d in book.drafts]
            pick = st.multiselect("Not taken", syms, key="draft_drop_pick")
            if st.button("Drop these", key="draft_drop_go") and pick:
                for sym in pick:
                    book.drop_draft(sym)
                persist_book(book)
                st.rerun()


def manage_ui(book: jn.Book) -> None:
    """The housekeeping, kept together at the bottom and out of the way.

    Three things that are not "what am I holding": adding a position the journal
    never saw, adjusting for a corporate action, and removing an entry that
    should not exist. Each is rare and each is destructive in its own way, so
    none of them belongs above the numbers you read every week.
    """
    with card("Manage book", "Capital, restatement, manual add, corporate action, remove"):
        capital_ui(book)
        restated_ui(book)
        manual_add_ui(book)
        corpact_ui(book)
        remove_ui(book)


def capital_ui(book: jn.Book) -> None:
    with st.expander("Add / withdraw capital · compounding"):
        book.refresh_compounding()
        tiles_row([
            ("Book capital", rupees(book.capital), "starting + deposits", ""),
            ("Sizing capital", rupees(book.effective_sizing_capital()),
             "used for this week’s qty", ""),
            ("Realised P&L", rupees(book.realised_pnl()),
             f"next step every {rupees(book.compound_step)}",
             tone_of(book.realised_pnl())),
            ("Compounded steps", str(book.compounded_steps),
             f"{book.compound_buffer_pct:g}% of each step held as DD buffer", ""),
        ])
        st.caption("Rule: every ₹1.20 L of **realised** profit, 20% stays as drawdown "
                   "buffer and 80% is added to sizing capital. 30 L book / ₹60k per stock "
                   "→ after ₹1.20 L booked, sizing 31 L and ₹62k/stock. Cash from sells is "
                   "already in the book; this only changes **how large** the next buy is.")
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Amount (₹)", -1e9, 1e9, 0.0, step=10_000.0, key="cf_amt")
        note = c2.text_input("Note", key="cf_note", placeholder="bank transfer / withdraw")
        if c3.button("Apply to book", key="cf_go"):
            if amt != 0:
                book.add_capital(float(amt), note)
                persist_book(book)
                st.success("Capital updated. This week’s buy qty will use the new sizing capital.")
                st.rerun()
        if book.cash_flows:
            st.markdown("###### Cash-flow log")
            show_df(pd.DataFrame(book.cash_flows))


def restated_ui(book: jn.Book) -> None:
    opens = sorted(book.open_symbols())
    with st.expander("Edit buy price / quantity (after a corporate action)"):
        st.caption("No P&L is booked. Cash does not move.")
        if not opens:
            st.info("No open position to edit.")
            return
        sym = st.selectbox("Stock", opens, key="rs_sym")
        pos = book.find(sym)
        if pos is None:
            return
        c1, c2 = st.columns(2)
        qty = c1.number_input("Quantity", 1, 10_000_000, int(pos.open_qty), key="rs_qty")
        px = c2.number_input("Average buy price (₹)", 0.01, 1e7,
                              float(pos.entry_price), key="rs_px", format="%.4f")
        note = st.text_input("Why", key="rs_note", placeholder="demerger restatement")
        if st.button("Save restatement", type="primary", key="rs_go"):
            book.restate_entry(sym, qty=int(qty), price=float(px), note=note)
            persist_book(book)
            st.success("Updated.")
            st.rerun()


def remove_ui(book: jn.Book) -> None:
    """Erase an entry that should never have existed — a test row, a wrong symbol.

    Deliberately not a sale. Selling records a real exit at a real price and
    books P&L; this puts the cash back and takes the fills with it, so a row you
    typed to see what happened does not end up in your win rate forever.
    """
    live = [(p.symbol, pd.Timestamp(p.entry_date).date(), "open") for p in book.positions]
    done = [(p.symbol, pd.Timestamp(p.entry_date).date(), "closed") for p in book.closed]
    rows = live + done
    with st.expander("Remove an entry (a mistake, or a test row)"):
        st.markdown('<div class="note"><b>This is not a sale.</b> Selling records a real '
                    "exit at a real price and books P&L. This erases the entry as if it had "
                    "never happened: the cash goes back, every fill of it leaves the ledger, "
                    "and nothing is left to distort a win rate or an equity curve. Use it "
                    "for a test row or a wrong symbol — never to tidy away a losing trade, "
                    "which is the one use that would quietly make your record a lie.<br>"
                    "The removal itself is logged, with the rows it took."
                    "</div>", unsafe_allow_html=True)
        if not rows:
            st.caption("Nothing to remove.")
        else:
            labels = [f"{sym} · bought {d} · {what}" for sym, d, what in rows]
            pick = st.multiselect("Entries to erase", labels, key="rm_pick")
            note = st.text_input("Why (kept in the log)", key="rm_note",
                                  placeholder="added by mistake while testing")
            if st.button("Erase these", key="rm_go") and pick:
                n, back = 0, 0.0
                for label in pick:
                    sym, d, _ = rows[labels.index(label)]
                    rec = book.remove_position(sym, d, note=note)
                    if rec:
                        n += 1
                        back += rec["cash_restored"]
                persist_book(book)
                st.success(f"Erased {n} entr{'y' if n == 1 else 'ies'}; "
                           f"{rupees(back)} put back. Cash is now {rupees(book.cash)}.")
                st.rerun()

        if book.corrections:
            st.markdown("###### What has been removed")
            show_df(pd.DataFrame([{
                "when": r.get("when"), "symbol": r.get("symbol"),
                "bought": r.get("entry_date"), "qty": r.get("qty"),
                "price": r.get("entry_price"), "fills removed": r.get("fills_removed"),
                "cash back": r.get("cash_restored"), "why": r.get("note") or "—",
            } for r in reversed(book.corrections)]))


def manual_add_ui(book: jn.Book) -> None:
    """Add a position you already hold at the broker but the journal never saw."""
    with st.expander("Add a position you already hold"):
        st.caption("For stocks bought outside this app. It goes straight in as a confirmed "
                   "position — cash is deducted and the ladder starts watching it.")
        c1, c2, c3 = st.columns(3)
        sym = c1.text_input("Symbol", key="man_add_sym", placeholder="TITAN").strip().upper()
        qty = c2.number_input("Quantity", 1, 10_000_000, 100, key="man_add_qty")
        px = c3.number_input("Buy price (₹)", 0.01, 1e7, 100.0, key="man_add_px")
        c1, c2, c3 = st.columns(3)
        when = c1.date_input("Buy date", value=date.today(), key="man_add_date")
        stop = c2.number_input("Stop (₹)", 0.0, 1e7, 0.0, key="man_add_stop",
                                help="Leave at 0 and the weekly 20 EMA rung still applies; "
                                     "this only seeds the risk figure.")
        c3.write("")
        if c3.button("Add position", key="man_add_go"):
            clean = ck.clean_symbol(sym)
            if not clean:
                st.error("That does not look like an NSE symbol.")
            elif book.find(clean) is not None:
                st.error(f"{clean} is already open in this book. Use a partial exit instead.")
            else:
                book.buy(clean, when, int(qty), float(px), float(stop),
                         note="added by hand")
                persist_book(book)
                st.success(f"Added {qty} x {clean}.")
                st.rerun()


# --------------------------------------------------------------------------- #
# corporate actions
# --------------------------------------------------------------------------- #
def corpact_ui(book: jn.Book) -> None:
    """Splits, bonuses, dividends, rights and demergers, with an audit trail."""
    opens = sorted(book.open_symbols())
    closed = sorted({p.symbol for p in book.closed})
    with st.expander("Adjust for a corporate action"):
        st.markdown(
            '<div class="note"><b>Why this exists.</b> Yahoo back-adjusts its price '
            "history: after a 1:5 split it shows ₹400 for a day the stock really traded "
            "at ₹2,000, all the way back. Your journal holds the broker's ₹2,000, so the "
            "app would compute a gain of <b>−80%</b> on a position that has not lost a "
            "rupee. Recording the action here restates the position to match — 500 shares "
            "at ₹400 — so the stop, the ladder and the gain % all line up again.<br>"
            "<b>Rupees never move.</b> 100 × ₹2,000 and 500 × ₹400 are the same ₹2,00,000, "
            "and a closed trade's recorded P&L is never touched — only the per-share "
            "numbers are restated. Every adjustment is logged with its before and after."
            "</div>", unsafe_allow_html=True)

        if not opens and not closed:
            st.info("Nothing to adjust yet.")
        else:
            c1, c2, c3 = st.columns([2, 2, 2])
            on_closed = c1.toggle("Include closed trades", value=False, key="ca_closed",
                                   help="For an action you only noticed after the trade was "
                                        "over. The rupee P&L stays exactly as recorded.")
            choices = opens + ([s for s in closed if s not in opens] if on_closed else [])
            if not choices:
                st.info("No position to adjust.")
            else:
                sym = c2.selectbox("Stock", choices, key="ca_sym")
                kind = c3.selectbox("Action", list(ca.KINDS), key="ca_kind",
                                     format_func=lambda k: ca.KINDS[k])
                ex = st.date_input("Ex-date", value=date.today(), key="ca_date")

                a = b_ = 1.0
                amount, keep, subscribed = 0.0, 100.0, True
                child_sym, child_ratio = "", 0.0
                if kind == "split":
                    c1, c2 = st.columns(2)
                    a = c1.number_input("Shares before", 1.0, 1e4, 1.0, key="ca_sa")
                    b_ = c2.number_input("become", 1.0, 1e4, 5.0, key="ca_sb")
                    st.caption(f"1:{b_ / a:g} — every share you hold becomes "
                               f"**{b_ / a:g}**, and the price divides by the same.")
                elif kind == "bonus":
                    c1, c2 = st.columns(2)
                    a = c1.number_input("Free shares", 1.0, 1e4, 1.0, key="ca_ba")
                    b_ = c2.number_input("for every", 1.0, 1e4, 1.0, key="ca_bb")
                    st.caption(f"A {a:g}:{b_:g} bonus leaves you holding "
                               f"**{ca.bonus_factor(a, b_):g}x** as many shares. "
                               "(1:1 is 2x, not 1x — that is the usual slip.)")
                elif kind == "dividend":
                    amount = st.number_input("Dividend per share (₹)", 0.0, 1e6, 5.0,
                                              key="ca_div")
                    st.caption("Cash only. The quantity and the cost do not change, and "
                               "this never enters a trade's P&L — it is reported on its "
                               "own line.")
                elif kind == "rights":
                    c1, c2, c3 = st.columns(3)
                    a = c1.number_input("Offered", 1.0, 1e4, 1.0, key="ca_ra")
                    b_ = c2.number_input("for every", 1.0, 1e4, 4.0, key="ca_rb")
                    amount = c3.number_input("Issue price (₹)", 0.01, 1e6, 100.0, key="ca_rp")
                    subscribed = st.checkbox("I took them up", value=True, key="ca_rs")
                    st.caption("Taking them up buys more shares with cash and moves your "
                               "average cost. Not taking them up changes nothing here.")
                else:                                        # demerger
                    keep = st.slider("% of the cost that stays with the parent",
                                      1.0, 99.0, 70.0, 1.0, key="ca_keep")
                    c1, c2 = st.columns(2)
                    child_sym = c1.text_input("Shares received in", key="ca_child",
                                               placeholder="JIOFIN").strip().upper()
                    child_ratio = c2.number_input("…per share held", 0.0, 100.0, 1.0, 0.1,
                                                   key="ca_child_ratio")
                    pos_now = book.find(sym)
                    if pos_now:
                        old_cost = pos_now.qty * pos_now.entry_price
                        cq = int(pos_now.qty * child_ratio)
                        st.caption(
                            f"**No profit and no loss is booked here.** A demerger splits "
                            f"one cost basis across two companies — nothing was bought or "
                            f"sold, so no money was made or lost. Your "
                            f"{rupees(old_cost)} of cost becomes "
                            f"{rupees(old_cost * keep / 100)} in {sym} "
                            f"(₹{pos_now.entry_price * keep / 100:,.2f}/share) and "
                            f"{rupees(old_cost * (100 - keep) / 100)} in "
                            f"{child_sym or 'the new company'}"
                            + (f" ({cq} shares at ₹{old_cost * (100 - keep) / 100 / cq:,.2f})"
                               if child_sym and cq > 0 else "")
                            + ". **The P&L happens when you sell either of them**, and the "
                            "two are linked so the journal can show what the pair did "
                            "together — read alone the parent looks like a sudden loss and "
                            "the new company like a windfall, and neither is true.")
                    if not child_sym:
                        st.caption("Leave the symbol blank and only the parent's cost is "
                                   "cut — add the new company by hand later.")

                note = st.text_input("Note (optional)", key="ca_note")
                if st.button("Apply this action", type="primary", key="ca_go"):
                    act = ca.Action(kind=kind, symbol=sym, ex_date=ex, a=a, b=b_,
                                     amount=amount, keep_pct=keep, subscribed=subscribed,
                                     child_symbol=(child_sym if kind == "demerger" else ""),
                                     child_ratio=(child_ratio if kind == "demerger" else 0.0),
                                     note=note)
                    rec = ca.apply(book, act, on_closed=on_closed)
                    persist_book(book)
                    if rec["ok"]:
                        bf, af = rec["before"] or {}, rec["after"] or {}
                        st.success(
                            f"{act.label()} applied to {sym}: "
                            f"{bf.get('qty')} @ ₹{bf.get('entry_price'):,.2f} → "
                            f"{af.get('qty')} @ ₹{af.get('entry_price'):,.2f}"
                            + (f" · cash {rupees(rec['cash_change'])}"
                               if rec["cash_change"] else ""))
                    else:
                        st.error(rec["problem"])
                    st.rerun()

        audit = ca.audit_frame(book)
        if not audit.empty:
            st.markdown("###### Audit trail")
            st.caption("Append-only. Nothing here is ever removed.")
            show_df(audit, height=240)
        inc = ca.income_frame(book)
        if not inc.empty:
            st.markdown("###### Dividends received")
            show_df(inc)
            st.caption(f"Total **{rupees(ca.total_income(book))}**. Kept out of trade P&L "
                       "on purpose — it is income, not a trading result.")


# --------------------------------------------------------------------------- #
# the week's actions, in one place
# --------------------------------------------------------------------------- #
def action_panel(book: jn.Book | None, plan: pd.DataFrame, pending: pd.DataFrame,
                 week) -> None:
    """One list: what to buy, what to book, what to stop out — or nothing.

    The information was always there, split across two tabs and two buttons.
    Having to assemble Monday's instructions yourself, on a Sunday night, from
    two screens is exactly how a rung gets missed.
    """
    n_buy = len(plan) if plan is not None else 0
    book_rows = pending[pending["rung"].astype(str).str.startswith("profit_")] \
        if pending is not None and not pending.empty else pd.DataFrame()
    stop_rows = pending[~pending["rung"].astype(str).str.startswith("profit_")] \
        if pending is not None and not pending.empty else pd.DataFrame()
    n_draft = len(book.drafts) if book else 0
    total = n_buy + len(book_rows) + len(stop_rows)

    with card("This week’s actions",
              f"Week ending {pd.Timestamp(week).date()} · buy / book / stop in one place"):
        if not total:
            st.success("**No action required.** "
                       + (f"{len(book.open_symbols())} position(s) holding, and nothing hit a "
                          "rung this week." if book else "Nothing qualified and nothing is held."))
        else:
            st.caption(f"**{total} thing(s) to do** this week.")
        if n_draft:
            st.info(f"⏳ **{n_draft} draft buy(s)** still waiting to be confirmed.")

        if n_buy:
            st.markdown("##### Buy")
            cols = [c for c in ["symbol", "rank", "pick", "history", "sector", "index", "qty",
                                "live CMP", "Fri close", "capital", "stop in force"]
                    if c in plan.columns]
            show_df(plan[cols])

        if len(book_rows):
            st.markdown(f"##### Book profit — {len(book_rows)}")
            show_df(book_rows[[c for c in ["symbol", "qty", "why", "gain_%", "last_close",
                                            "est_proceeds", "est_pnl"] if c in book_rows.columns]])

        if len(stop_rows):
            st.markdown(f"##### Exit on the stop — {len(stop_rows)}")
            show_df(stop_rows[[c for c in ["symbol", "qty", "why", "gain_%", "last_close",
                                            "est_proceeds", "est_pnl"] if c in stop_rows.columns]])

        if len(book_rows) or len(stop_rows):
            st.caption("Record these fills on the *Positions & exits* tab — this panel only "
                       "tells you what fired.")


# --------------------------------------------------------------------------- #
# Chartink import
# --------------------------------------------------------------------------- #
def chartink_active() -> list[str]:
    """Imported Chartink names, if the list is in use for this week.

    Default is ON whenever a list exists. A missing toggle key used to look
    like False, which silently loaded the whole Nifty 500 instead of the
    20–30 Chartink names.
    """
    have = list(st.session_state.get("chartink_symbols") or [])
    if not have:
        return []
    if st.session_state.get("chartink_use", True):
        return have
    return []


def _store_chartink(res: "ck.ImportResult", source: str) -> None:
    st.session_state["chartink_symbols"] = list(res.symbols)
    st.session_state["chartink_source"] = source
    st.session_state["chartink_when"] = str(date.today())
    st.session_state["chartink_notes"] = list(res.notes)
    st.session_state["chartink_use"] = True
    st.session_state.pop("buys_ran", None)      # the list changed; the old scan is stale


def chartink_ui() -> None:
    """Upload a Chartink export and use it as this week's candidate list.

    Only the symbols are taken. Chartink's own price, % change and volume are
    ignored on purpose — they are a snapshot of whenever you pressed Download,
    while everything downstream here (rank, quantity, stop) has to be computed
    off the app's weekly close or the buy list stops matching the backtest.
    """
    have = st.session_state.get("chartink_symbols") or []
    title = (f"Chartink list — {len(have)} symbols"
             f" ({st.session_state.get('chartink_when', '')})" if have
             else "Import from Chartink CSV")
    with card(title, "Only symbols are taken. Ranking, qty and stops use this app’s weekly close."):
        up = st.file_uploader("Chartink CSV", type=["csv", "txt"], key="chartink_file",
                               label_visibility="collapsed")
        if up is not None and st.session_state.get("chartink_seen") != up.name + str(up.size):
            res = ck.read_csv(up.getvalue(), up.name)
            st.session_state["chartink_seen"] = up.name + str(up.size)
            if res.ok():
                _store_chartink(res, up.name)
                st.rerun()
            else:
                st.error(res.error)

        with st.popover("…or paste a list", **_WIDE):
            txt = st.text_area("One symbol per line, or comma separated", height=140,
                                key="chartink_paste")
            if st.button("Use this list", key="chartink_paste_go"):
                res = ck.read_text(txt)
                if res.ok():
                    _store_chartink(res, "pasted list")
                    st.rerun()
                else:
                    st.error(res.error)

        if not have:
            return

        if "chartink_use" not in st.session_state:
            st.session_state["chartink_use"] = True
        st.toggle("Use this list as this week's candidates", key="chartink_use",
                   help="On: the app's own breakout scan and screen filters are skipped for "
                        "this week — Chartink already applied them — and these stocks go "
                        "straight into ranking. Off: the app scans your loaded universe as "
                        "usual and this list is just kept around.")
        for n in (st.session_state.get("chartink_notes") or []):
            st.caption(n)
        st.caption(", ".join(have[:40]) + (f" … +{len(have) - 40} more" if len(have) > 40 else ""))
        if st.button("Clear the imported list", key="chartink_clear"):
            for k in ("chartink_symbols", "chartink_source", "chartink_when",
                      "chartink_notes", "chartink_use", "chartink_seen", "buys_ran"):
                st.session_state.pop(k, None)
            st.rerun()


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def equity_figure(equity: pd.Series, bench: pd.Series | None, dark: bool,
                  pct: bool = False) -> go.Figure:
    t = ch.theme(dark)
    fig = go.Figure()
    y = equity.values.astype(float)
    start = float(y[0]) if len(y) else 0.0
    if pct and start != 0:
        y = (y / start - 1.0) * 100.0
        title = "Return from start (%)"
        hover = "%{y:+.2f}%"
        tickprefix = ""
        tickformat = "+,.2f"
        fill = "tozeroy"
        fillcolor = "rgba(5,150,105,0.12)"
    else:
        title = "Portfolio (₹)"
        hover = "₹%{y:,.0f}"
        tickprefix = "₹"
        tickformat = ",.0f"
        fill = "tonexty"
        fillcolor = "rgba(5,150,105,0.10)"
    fig.add_trace(go.Scatter(
        x=equity.index, y=y, name="Strategy",
        line=dict(color="#059669", width=2.4),
        fill="tozeroy" if pct else None,
        fillcolor=fillcolor if pct else None,
        hovertemplate="%{x|%d %b %Y}<br>" + hover + "<extra></extra>",
    ))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max((ymax - ymin) * 0.2, (1.0 if pct else abs(ymin) * 0.002) or 1)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        yaxis=dict(gridcolor="#e5e7eb", title=title,
                   tickprefix=(tickprefix),
                   tickformat=tickformat,
                   range=[ymin - pad, ymax + pad]),
        xaxis=dict(gridcolor="#e5e7eb"),
    )
    return fig


def drawdown_figure(equity: pd.Series, dark: bool) -> go.Figure:
    t = ch.theme(dark)
    dd = M.drawdown_series(equity) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                line=dict(color=t["critical"], width=1.4),
                                fillcolor="rgba(220,38,38,0.12)",
                                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>"))
    ymin, ymax = float(dd.min()), float(dd.max())
    pad = max(0.15, (ymax - ymin) * 0.2 if ymax != ymin else 0.4)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=30, b=10), height=220,
        yaxis=dict(gridcolor="#e5e7eb", title="Drawdown %",
                   range=[ymin - pad, ymax + pad]),
        xaxis=dict(gridcolor="#e5e7eb"),
    )
    return fig


def weekly_equity(eq: pd.Series) -> pd.Series:
    """Friday-week marks so the curve matches how this book is actually traded."""
    if eq is None or eq.empty:
        return eq
    try:
        w = eq.resample("W-FRI").last().dropna()
        return w if len(w) >= 2 else eq
    except Exception:
        return eq


def _day_tone(pnl: float, scale: float) -> str:
    if not np.isfinite(pnl) or abs(pnl) < 1:
        return "be"
    if pnl > 0:
        return "lg" if pnl >= scale else "sg"
    return "ll" if pnl <= -scale else "sl"


def trading_calendar_html(book: jn.Book, year: int, kpis: list | None = None) -> str:
    """Mon–Fri year grid. Colour = that day's realised P&L (exits). Entries marked too."""
    led = pd.DataFrame(book.ledger) if book.ledger else pd.DataFrame()
    pnl_by_day: dict[date, float] = {}
    entry_days: set[date] = set()
    if not led.empty:
        led["date"] = pd.to_datetime(led["date"]).dt.date
        led["pnl"] = pd.to_numeric(led.get("pnl"), errors="coerce").fillna(0.0)
        for d, g in led.groupby("date"):
            sells = g[g["side"].astype(str).str.upper() == "SELL"]
            if len(sells):
                pnl_by_day[d] = float(sells["pnl"].sum())
            if (g["side"].astype(str).str.upper() == "BUY").any():
                entry_days.add(d)
    scale = max(float(book.capital) * 0.005, 10_000.0)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    cal = calmod.Calendar(firstweekday=0)
    cols = []
    for m in range(1, 13):
        weeks = cal.monthdayscalendar(year, m)
        rows = []
        for wi, wd in enumerate(weekdays):
            cells = []
            for week in weeks:
                dayn = week[wi]
                if dayn == 0:
                    cells.append('<span class="cal-empty"></span>')
                    continue
                d = date(year, m, dayn)
                cls = "cal-day"
                title = d.isoformat()
                if d in pnl_by_day:
                    pnl = pnl_by_day[d]
                    cls += " cal-" + _day_tone(pnl, scale)
                    title += f" · P&L ₹{pnl:+,.0f}"
                elif d in entry_days:
                    cls += " cal-en"
                    title += " · entry"
                cells.append(f'<span class="{cls}" title="{title}">{dayn}</span>')
            rows.append(f'<div class="cal-row"><span class="cal-wd">{wd}</span>'
                        + "".join(cells) + "</div>")
        cols.append(f'<div class="cal-month"><div class="cal-mh">{months[m-1]}</div>'
                    + "".join(rows) + "</div>")
    kpi_html = ""
    if kpis:
        bits = []
        for lab, val, sub, tone in kpis:
            bits.append(
                f'<div class="cal-kpi"><div class="k">{lab}</div>'
                f'<div class="v {tone}">{val}</div>'
                f'<div class="s">{sub}</div></div>'
            )
        kpi_html = '<div class="cal-kpis">' + "".join(bits) + "</div>"
    legend = (
        '<div class="cal-leg">'
        '<span class="cal-day cal-lg"></span> Large gain'
        '<span class="cal-day cal-sg"></span> Small gain'
        '<span class="cal-day cal-be"></span> Breakeven'
        '<span class="cal-day cal-sl"></span> Small loss'
        '<span class="cal-day cal-ll"></span> Large loss'
        '<span class="cal-day cal-en"></span> Entry'
        "</div>"
    )
    return (
        '<div class="cal-wrap">'
        + kpi_html
        + '<div class="cal-title">Trading calendar</div>'
        + legend
        + '<div class="cal-grid">' + "".join(cols) + "</div>"
        + '<div class="cal-note">Colour = realised P&L on that day. '
        "Weekly system — most days stay empty, that is normal.</div>"
        "</div>"
    )


def weekly_pnl_figure(rt: pd.DataFrame, dark: bool) -> go.Figure:
    t = ch.theme(dark)
    fig = go.Figure()
    if rt is None or rt.empty:
        fig.update_layout(template="plotly_white", height=220)
        return fig
    d = rt.copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    d["week"] = d["exit_date"].dt.to_period("W-FRI").astype(str)
    g = d.groupby("week", as_index=False)["P&L"].sum()
    colors = [t["good"] if v >= 0 else t["critical"] for v in g["P&L"]]
    fig.add_trace(go.Bar(x=g["week"], y=g["P&L"], marker_color=colors, name="Weekly P&L"))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        margin=dict(l=10, r=10, t=24, b=10), height=260,
        yaxis=dict(gridcolor=t["grid"], title="P&L (₹)"),
        xaxis=dict(gridcolor=t["grid"], title="Week ending Friday"),
        showlegend=False,
    )
    return fig


def donut_figure(title: str, labels, values, colors, dark: bool, center: str) -> go.Figure:
    t = ch.theme(dark)
    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values), hole=0.68,
        marker=dict(colors=list(colors)),
        textinfo="none",
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        margin=dict(l=10, r=10, t=36, b=10), height=220,
        title=dict(text=title, font=dict(size=13, color=t["text"])),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.15),
        annotations=[dict(text=center, x=0.5, y=0.5, font=dict(size=16, color=t["text"]),
                          showarrow=False)],
    )
    return fig


def exit_reason_figure(rb: pd.DataFrame, dark: bool) -> go.Figure:
    t = ch.theme(dark)
    fig = go.Figure()
    if rb is None or rb.empty:
        return fig
    col = "P&L" if "P&L" in rb.columns else rb.columns[-1]
    lab = rb.index.astype(str) if rb.index.name or not isinstance(rb.index, pd.RangeIndex) else rb.iloc[:, 0].astype(str)
    if not isinstance(rb.index, pd.RangeIndex) and rb.index.name:
        y = rb.index.astype(str)
        x = rb[col] if col in rb.columns else rb.iloc[:, -1]
    else:
        y = rb.iloc[:, 0].astype(str)
        x = rb[col] if col in rb.columns else rb.iloc[:, -1]
    colors = [t["good"] if float(v) >= 0 else t["critical"] for v in x]
    fig.add_trace(go.Bar(x=x, y=y, orientation="h", marker_color=colors))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        margin=dict(l=10, r=10, t=10, b=10), height=max(180, 28 * len(y) + 60),
        xaxis=dict(gridcolor=t["grid"], title="P&L (₹)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# TAB 1 — Backtest
# --------------------------------------------------------------------------- #
def tab_backtest(s: dict) -> None:
    page_head("Backtest",
              f"{s['breakout'].label()} · {s['entries_per_week']} new entries every Monday · "
              "Friday close → Monday fill")
    tiles_row([
        ("Universe", f"{len(s['symbols'])} stocks",
         "from the sidebar list", ""),
        ("Lookback", f"{s['breakout'].lookback_weeks} weeks",
         f"fresh {s['breakout'].fresh_weeks}w", ""),
        ("New entries / week", str(s["entries_per_week"]),
         "Monday fills", ""),
        ("Trail / final EMA",
         f"{s['breakout'].ema_fast} / {s['breakout'].ema_slow}",
         "weekly closes", ""),
        ("Capital", rupees(s["sizing"].capital),
         s["sizing"].mode, ""),
    ])

    if st.button("Run backtest", type="primary"):
        ctx = build_context(s)
        cfg = RunConfig(
            start=pd.Timestamp(s["start"]), end=pd.Timestamp(s["end"]),
            entries_per_week=int(s["entries_per_week"]), breakout=s["breakout"],
            sizing=s["sizing"], regime=s["regime"], costs=s["costs"], ladder=s["ladder"],
            rearm_ema=s.get("rearm_ema", False),
            targets_on_daily_close=s.get("targets_on_daily_close", True),
        )
        bt_fund = pd.DataFrame()
        if s.get("use_fundamentals") and s.get("fund_in_backtest"):
            st.warning("**Fundamentals are switched on inside the backtest.** Yahoo only "
                       "serves today's statements, so every week in history is being ranked "
                       "with numbers published after it. The result is optimistic by an "
                       "unknown amount — use it to explore, not to decide.")
            with st.spinner("Reading statements for the universe (first run is slow)…"):
                bt_fund = load_fundamentals(tuple(sorted(s["symbols"])), s["demo"])

        bar = st.progress(0.0, text="Simulating…")
        res = run_backtest(
            ctx["panel"], ctx["signals"], cfg,
            screen_mask=ctx["screen"].mask, bench_daily=ctx["bench"],
            vol_surge_weekly=ctx["vol_surge_weekly"], daily_ok=ctx["daily_ok"],
            fundamentals=bt_fund,
            progress_cb=lambda i, n: bar.progress(i / n, text=f"Week {i} of {n}"),
        )
        bar.empty()
        st.session_state["bt"] = res
        st.session_state["bt_bench"] = ctx["bench"]
        st.session_state["bt_ctx_notes"] = (ctx["screen"].notes + ctx["screen"].missing_inputs
                                             + list(ctx["daily_info"].get("warnings", [])))
        st.session_state["bt_daily_info"] = ctx["daily_info"]

    res = st.session_state.get("bt")
    if res is None:
        st.info("Set the rules in the sidebar, then press **Run backtest**.")
        return
    if res.equity.empty:
        st.error(" ".join(res.notes) or "Nothing to show.")
        return

    eq = res.equity
    rets = M.to_returns(eq)
    start_val, end_val = float(eq.iloc[0]), float(eq.iloc[-1])

    # ---------------- nothing traded: say why, don't draw a flat line -------- #
    if res.trades.empty:
        wl = res.weekly_log
        weeks = len(wl)
        screened = float(wl["passed_screen"].mean()) if weeks else 0
        breaks = int(wl["breakouts"].sum()) if weeks else 0
        cands = int(wl["candidates"].sum()) if weeks else 0
        blocked = int(wl["regime_blocked"].sum()) if weeks else 0
        no_cash = int(wl["skipped_no_cash"].sum()) if weeks else 0

        st.error("**The backtest took no trades.** Here is where the funnel closed:")
        show_df(pd.DataFrame([
            {"Step": "1. Weeks simulated", "Count": weeks, "Meaning": ""},
            {"Step": "2. Stocks passing the screen (avg/week)", "Count": round(screened, 1),
             "Meaning": "market cap, price, volume filters"},
            {"Step": "3. Fresh breakouts fired (total)", "Count": breaks,
             "Meaning": "before the screen is applied"},
            {"Step": "4. Candidates (breakout AND screen)", "Count": cands,
             "Meaning": "what the ranker got to choose from"},
            {"Step": "5. Weeks blocked by the regime filter", "Count": blocked, "Meaning": ""},
            {"Step": "6. Entries skipped for lack of cash", "Count": no_cash, "Meaning": ""},
        ]))

        if screened == 0:
            st.warning("**Step 2 is zero — your screen excludes everything.** The usual cause is a "
                       "market-cap band that cannot match: check the *Market-cap segment* in the "
                       "sidebar. If it is on AMFI ranks, a band starting at rank 251 needs a list "
                       "of at least 251 stocks; use a percentage segment instead, or load a bigger "
                       "list.")
        elif breaks == 0:
            st.warning("**Step 3 is zero — no stock ever made a fresh high.** Try a shorter "
                       "lookback (52 rather than 200 weeks), fewer 'previous weeks' in the "
                       "freshness stack, or a longer date range.")
        elif cands == 0:
            st.warning("**Breakouts happened, but never in a stock that passed the screen.** "
                       "Loosen the screen or widen the market-cap segment.")
        elif blocked >= weeks * 0.9:
            st.warning("**The regime filter blocked almost every week.** Switch it off in the "
                       "sidebar, or require only one of the two conditions.")
        elif no_cash > 0:
            st.warning("**Signals fired but there was never enough cash.** Lower the capital per "
                       "stock, or raise total capital.")

        with st.expander("Week-by-week funnel"):
            v = wl.copy()
            v["signal_week"] = pd.to_datetime(v["signal_week"]).dt.date
            show_df(v[["signal_week", "passed_screen", "breakouts", "candidates",
                        "regime_blocked", "skipped_no_cash", "cash"]], height=400)
        return
    tiles_row([
        ("Final value", rupees(end_val), f"from {rupees(start_val)}", tone_of(end_val - start_val)),
        ("Total return", f"{(end_val/start_val - 1)*100:,.1f}%", "", tone_of(end_val - start_val)),
        ("CAGR", f"{M.cagr(eq)*100:,.1f}%", "", tone_of(M.cagr(eq))),
        ("Max drawdown", f"{M.max_drawdown(eq)*100:,.1f}%", "", "neg"),
        ("Sharpe", _safe_ratio(M.sharpe(rets)), "", ""),
        ("Calmar", _safe_ratio(M.calmar(eq)), "", ""),
    ])

    st.markdown("")
    show_chart(equity_figure(eq, st.session_state.get("bt_bench"), s["dark"]))
    show_chart(drawdown_figure(eq, s["dark"]))

    trades = res.trades
    n_buys = int((trades["side"] == "BUY").sum()) if not trades.empty else 0
    n_sells = int((trades["side"] == "SELL").sum()) if not trades.empty else 0
    tiles_row([
        ("Entries taken", f"{n_buys:,}", "", ""),
        ("Exit fills", f"{n_sells:,}", "partial exits count separately", ""),
        ("Still open at end", f"{len(res.positions):,}", "", ""),
        ("Weeks blocked by regime", f"{len(res.blocked_weeks):,}",
         f"of {len(res.weekly_log):,} weeks", ""),
    ])

    # ---------------- capital usage ---------------- #
    if res.deployed is not None and len(res.deployed):
        peak = float(res.deployed.max())
        peak_day = res.deployed.idxmax()
        avg = float(res.deployed.mean())
        cap = s["sizing"].capital
        tiles_row([
            ("Peak capital deployed", rupees(peak),
             f"{peak / cap * 100:,.0f}% of capital · {pd.Timestamp(peak_day).date()}", ""),
            ("Average deployed", rupees(avg), f"{avg / cap * 100:,.0f}% of capital", ""),
            ("Peak open positions", f"{int(res.n_open.max())}",
             (f"cap allows {s['sizing'].max_concurrent()}"
              if s["sizing"].max_concurrent() else ""), ""),
            ("Idle cash (average)", rupees(float(res.cash.mean())), "", ""),
        ])

    di = st.session_state.get("bt_daily_info") or {}
    if di.get("warnings"):
        st.warning(" ".join(di["warnings"]))

    with card("Where the money came from", "Each rung of the ladder, and what it actually contributed."):
        ls = ladder_summary(trades)
        if not ls.empty:
            show_df(ls)

    with card("Year by year"):
        yb = yearly_breakdown(res)
        if not yb.empty:
            show_chart(ch.yearly_bars(yb.rename(columns={"Return (%)": "Return (%)"}), s["dark"]))
            show_df(yb.style.format({
                "Return (%)": "{:,.2f}", "Start Capital": "₹{:,.0f}", "End Capital": "₹{:,.0f}",
                "Net Profit": "₹{:,.0f}", "Max drawdown %": "{:,.2f}",
                "Max capital used": "₹{:,.0f}", "Max used %": "{:,.1f}",
                "Avg capital used": "₹{:,.0f}",
            }))
            st.download_button("Download yearly (CSV)", yb.to_csv(index=False).encode(),
                                "breakout_yearly.csv", "text/csv")

    with card("Month by month"):
        mt = M.monthly_table(eq)
        if not mt.empty:
            show_chart(ch.monthly_heatmap(mt, s["dark"]))
        mb = monthly_breakdown(res)
        if not mb.empty:
            st.caption("**Max capital used** is the peak cost of open positions during that month — "
                       "how much of your money was actually in the market, not its mark-to-market value.")
            show_df(mb.style.format({
                "Return %": "{:,.2f}", "Equity (end)": "₹{:,.0f}", "Max capital used": "₹{:,.0f}",
                "Max used %": "{:,.1f}", "Avg capital used": "₹{:,.0f}", "Avg open": "{:,.1f}",
                "Booked P&L": "₹{:,.0f}",
            }), height=420)
            st.download_button("Download monthly (CSV)", mb.to_csv(index=False).encode(),
                                "breakout_monthly.csv", "text/csv")

    c1, c2 = st.tabs(["Completed positions", "Every fill"])
    with c1:
        rt = round_trip_trades(trades)
        if not rt.empty:
            show_df(rt, height=380)
            st.download_button("Download positions (CSV)", rt.to_csv(index=False).encode(),
                                "breakout_positions.csv", "text/csv")
    with c2:
        if not trades.empty:
            t2 = trades.copy()
            t2["date"] = pd.to_datetime(t2["date"]).dt.date
            show_df(t2.sort_values("date"), height=380)
            st.download_button("Download fills (CSV)", t2.to_csv(index=False).encode(),
                                "breakout_fills.csv", "text/csv")

    with st.expander("Week-by-week log"):
        wl = res.weekly_log.copy()
        if not wl.empty:
            wl["signal_week"] = pd.to_datetime(wl["signal_week"]).dt.date
            wl["executed"] = pd.to_datetime(wl["executed"]).dt.date
            show_df(wl, height=320)

    notes = st.session_state.get("bt_ctx_notes") or []
    if notes:
        st.markdown('<div class="note">' + "<br>".join(str(n) for n in notes) + "</div>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# TAB 2 — This week's buys
# --------------------------------------------------------------------------- #
def tab_buys(s: dict) -> None:
    page_head("This week’s buys",
              "Friday 2:30–3:00 pm or after close · Chartink rank + actions on open positions")

    book = book_picker("buys")
    # drafts live here, next to the list that creates them — the positions tab
    # is for what you actually hold
    if book is not None:
        drafts_ui(book, st.session_state.get("_last_px"))
    chartink_ui()

    _imported = chartink_active()
    _btn = ("Rank the imported Chartink list" if _imported
            else "Find this week's breakouts")

    # A button is True only on the ONE rerun that follows its click. Gating the
    # whole tab on this button directly meant that clicking any *other* button
    # further down — "Queue as drafts", say — reran the script with this one
    # False, so the code that reads that click was never reached and the click
    # did nothing at all. The scan therefore sets a flag, and the flag is what
    # keeps the results on screen.
    if st.button(_btn, type="primary", key="buys_go"):
        st.session_state["buys_ran"] = True
    if not st.session_state.get("buys_ran"):
        st.info(f"Press the button to rank the {len(_imported)} stocks you imported."
                if _imported else "Press the button to scan the latest completed week.")
        return

    # an imported list drives the whole week, so nothing outside it is used for
    # anything — download those names and stop there
    ctx = build_context(s, for_live=True,
                        restrict_to=_imported if _imported else None)
    sig, panel = ctx["signals"], ctx["panel"]
    wc = sig.weekly.get("Close", pd.DataFrame())
    if wc.empty:
        st.error("No weekly data.")
        return

    week = wc.index[-1]
    # A week is only usable once it has actually finished.
    last_daily = panel["Close"].index[-1]
    if last_daily < week:
        week = wc.index[-2] if len(wc.index) > 1 else week
    st.caption(f"Signal week ending **{pd.Timestamp(week).date()}** · "
               f"latest price bar {pd.Timestamp(last_daily).date()}")

    blocked, detail = regime_blocked(ctx["bench"], week, s["regime"])
    if blocked:
        st.error("**Regime filter says no new entries this week.** "
                 + ", ".join(f"{k}: {v}" for k, v in detail.items() if isinstance(v, bool) and v))
        st.caption("Exits still run as normal — check the Positions & exits tab.")

    # a stock already drafted is already on the list — showing it again would
    # have you queue the same buy twice
    held = (book.open_symbols() | book.draft_symbols()) if book else set()

    if _imported:
        # Chartink already applied the scan and the filters, so neither the
        # freshness test nor the screen runs again here — re-applying them would
        # quietly drop names the source list says qualified. Everything after
        # this point is the app's own work, unchanged.
        priced, missing = ck.split_by_coverage(_imported, list(panel["Close"].columns))
        cands = [c for c in priced if c not in held]
        st.info(f"**Ranking {len(cands)} stocks from “{st.session_state.get('chartink_source')}”.** "
                "The app's own breakout scan and screen filters were skipped — Chartink "
                "already applied them. Ranking, quantity and stops are this app's.")
        n_held = len([c for c in priced if c in held])
        if n_held:
            st.caption(f"Left out: **{n_held} already held or drafted** in this book.")
        if missing:
            gaps = data_mod.explain_gaps(missing, priced, ctx["coverage"], ctx["need_days"])
            counts = gaps["why"].value_counts().to_dict()
            st.warning(f"**{len(missing)} of the imported names could not be ranked** — "
                       + ", ".join(f"{v} {k.lower()}" for k, v in counts.items()) + ".")
            with st.expander(f"Why those {len(missing)} were left out"):
                show_df(gaps)
                st.caption(
                    f"The scan needs about **{ctx['need_days']} trading days** "
                    f"({s['breakout'].lookback_weeks} weeks for the high, plus "
                    f"{s['breakout'].fresh_weeks} for the freshness test). A company "
                    "listed after "
                    f"{pd.Timestamp(ctx['start']).date()} simply has not been trading "
                    "long enough for a 52-week-high rule to mean anything yet — that is "
                    "not the same as having no data, and Chartink will still show it "
                    "because its scan does not need the same history.")
        if not cands:
            st.warning("Nothing left to rank from the imported list.")
            return
    else:
        screen_ok = ctx["screen"].at(week)
        cands = qualifying_at(sig, week, screen_ok, exclude=held)

        if not cands:
            st.warning("No stock passed the full scan this week. That is a normal outcome — "
                       "fresh breakouts are lumpy.")
            return

    fund = pd.DataFrame()
    if s.get("use_fundamentals"):
        with st.spinner(f"Reading the statements of the {len(cands)} names that qualified…"):
            fund = load_fundamentals(tuple(sorted(cands)), s["demo"])
        if not fund.empty:
            n_ok = int(fund["available"].sum())
            if n_ok < len(cands):
                st.caption(f"Fundamentals found for {n_ok} of {len(cands)}. The rest are "
                           "marked *Data unavailable* and ranked on their chart alone.")

    scored = score_week(sig, week, cands, s["breakout"], ctx["vol_surge_weekly"],
                        fundamentals=fund)

    sectors = {}
    with st.spinner("Looking up sectors…"):
        sectors = load_sectors(tuple(sorted(cands)), s["demo"])
    picks = diversify_picks(scored, int(s["entries_per_week"]), sectors,
                            max_per_sector=int(s.get("max_per_sector", 2)),
                            enabled=bool(s.get("diversify")),
                            max_promote_rank=int(s.get("max_promote_rank") or 0) or None)

    # the same NSE size bands the journal reports on, so what you buy and what
    # you later measure are read in the same units
    with st.spinner("Looking up index membership…"):
        ix_buckets, _ix_notes = load_index_buckets(tuple(sorted(picks.index)), s["demo"])

    # Friday 2–3pm Chartink entry AND Monday fill are both valid.
    # Default = live CMP so qty matches the price you actually pay today.
    if "entry_now" not in st.session_state:
        st.session_state["entry_now"] = True
    entry_now = st.radio(
        "Size quantity on",
        options=[True, False],
        format_func=lambda v: (
            "Live CMP — buy today (Friday 2–3pm or Monday)" if v
            else "Friday close — weekend planning"
        ),
        key="entry_now",
        horizontal=True,
        help="Chartink can qualify names by Friday 2pm. If you buy that afternoon or "
             "Monday, size off live CMP. Friday close is the scan reference, kept as "
             "its own column. Slippage = fill minus the price qty was sized on.",
    )

    live_px = pd.Series(dtype=float)
    if not s.get("demo") and len(picks):
        with st.spinner("Fetching live CMP…"):
            live_px = data_mod.live_last_prices(list(picks.index))
    daily_last = panel["Close"].iloc[-1] if "Close" in panel else pd.Series(dtype=float)

    equity = None
    if book:
        last_px = live_px if len(live_px) else daily_last
        st.session_state["_last_px"] = last_px
        book.refresh_compounding()
        equity = book.effective_sizing_capital()

    rows = []
    dist_pct = float(getattr(s["sizing"], "max_stop_distance_pct", 0.0) or 0.0)
    cov = ctx.get("coverage")
    settled = int(s["breakout"].ema_slow * 2 * 5)
    for sym, r in picks.iterrows():
        friday_px = float(r["close"])
        live = np.nan
        if len(live_px) and sym in live_px.index:
            live = float(live_px.get(sym, np.nan))
        if not np.isfinite(live):
            live = float(daily_last.get(sym, np.nan)) if sym in getattr(daily_last, "index", []) else np.nan
        if entry_now and np.isfinite(live) and live > 0:
            ref_px = live
        else:
            ref_px = friday_px
        ema_stop = float(r["ema_fast"]) if pd.notna(r["ema_fast"]) else ref_px * 0.92
        hs = hard_stop_price(ref_px, dist_pct)
        stop = entry_stop(ema_stop, ref_px, dist_pct)
        sized = size_position(ref_px, stop, s["sizing"], equity)
        n_days = int(cov.loc[sym, "days"]) if (cov is not None and sym in cov.index) else 0
        raw_pick = str(r.get("pick") or "")
        pick_label = {"{?}": "no sector yet", "{div}": "sector fill",
                      "{cap}": "over sector cap"}.get(raw_pick, raw_pick)
        hist = f"listed {n_days // 5}w" if 0 < n_days < settled else ""
        rows.append({
            "symbol": sym,
            "rank": int(r["rank"]) if "rank" in r else None,
            "pick": pick_label,
            "history": hist,
            "sector": str(r.get("sector") or "Unknown"),
            "index": ix_buckets.get(sym, "—"),
            "qty": sized.qty,
            "live CMP": (round(live, 2) if np.isfinite(live) else None),
            "Fri close": round(friday_px, 2),
            "decision price": round(ref_px, 2),
            "capital": round(sized.cost, 0),
            "20 EMA": round(ema_stop, 2),
            "20 EMA is away": f"{(ref_px - ema_stop)/ref_px*100:,.1f}%",
            "max SL": round(hs, 2) if hs else None,
            "stop in force": round(stop, 2),
            "which": ("max SL" if hs and hs >= ema_stop else "20 EMA"),
            "stop is away": f"{(ref_px - stop)/ref_px*100:,.1f}%",
            "risk if stopped": round(sized.risk_amount, 0),
            "combined score": round(float(r["combined_score"]), 1),
            "technical": round(float(r["technical_score"]), 1),
            "fundamentals": (round(float(r["fundamentals_score"]), 1)
                             if pd.notna(r.get("fundamentals_score")) else None),
            "what is good": (str(r.get("what is good") or "") or "Data unavailable"),
            "broke above": round(float(r["prior_high"]), 2),
            "extension %": round(float(r["extension_%"]), 2),
            "vol surge": round(float(r["volume_surge"]), 2) if pd.notna(r["volume_surge"]) else None,
            f"{s['breakout'].lookback_weeks}w momentum %": round(float(r["momentum_%"]), 1),
            "capped by": sized.capped_by or "—",
        })
    plan = pd.DataFrame(rows)

    # ---- one list for the whole week ---- #
    pending = pd.DataFrame()
    if book is not None and book.positions:
        last_day = panel["Close"].index[-1]
        trail_levels = {k: (v.loc[last_day] if last_day in v.index else pd.Series(dtype=float))
                        for k, v in sig.trail_emas.items()}
        ef_ = sig.ema_fast.loc[week] if week in sig.ema_fast.index else pd.Series(dtype=float)
        es_ = sig.ema_slow.loc[week] if week in sig.ema_slow.index else pd.Series(dtype=float)
        pending = jn.pending_actions(book, week, wc.loc[week], ef_, es_, s["ladder"],
                                      rearm_ema=s.get("rearm_ema", False),
                                      trail_levels=trail_levels,
                                      daily_close=panel["Close"].iloc[-1],
                                      targets_on_daily_close=s.get("targets_on_daily_close",
                                                                   True))
    action_panel(book, pd.DataFrame() if blocked else plan, pending, week)

    total_cost = float(plan["capital"].sum())
    total_risk = float(plan["risk if stopped"].sum())
    with card("Size this week", "Capital to deploy if you take the ranked list"):
        tiles_row([
            ("Imported and priced" if _imported else "Qualified this week",
             f"{len(cands)}", f"showing top {len(plan)}", ""),
            ("Capital to deploy", rupees(total_cost),
             (f"₹{s['sizing'].amount_for(equity):,.0f} per stock"
              if s["sizing"].mode == "fixed" else ""), ""),
            ("Total risk if all stop out", rupees(total_risk),
             f"{total_risk / s['sizing'].capital * 100:,.2f}% of capital", ""),
            ("Cash in book", rupees(book.cash) if book else "—", "", ""),
        ])

    with card("Ranked buy list",
              "qty uses live CMP by default so Friday 2–3pm and Monday both work"):
        show_df(plan)
        st.caption("**live CMP** = Yahoo last (what you pay if you buy now). "
                   "**Fri close** = weekly signal close (Chartink 52w high). "
                   "**decision price** is whichever the toggle above is on — qty and stop "
                   "are sized on that. Slippage = your fill minus decision price.")
    _ixc = plan["index"].value_counts()
    st.caption("Index bands in this list: "
               + ", ".join(f"{b} x{int(_ixc[b])}"
                           for b in ix_mod.bucket_order(plan["index"]) if b in _ixc)
               + ". Same bands the journal measures returns by, so what you buy and what "
                 "you later read about it are in the same units. NSE publishes only "
                 "today's constituent lists, so these are today's.")

    n_new = int((plan["history"].astype(str).str.startswith("listed")).sum())
    if n_new:
        st.caption(f"**listed Nw** — {n_new} name(s) have been listed for less than "
                   f"{settled // 5} weeks. They qualify, but the "
                   f"{s['breakout'].ema_slow}-week EMA that becomes their final stop is "
                   "still a young average. Nothing is excluded; the number is "
                   "how many weeks of history they actually have.")
    if s.get("diversify"):
        n_div = int((plan["pick"] == "sector fill").sum())
        n_cap = int((plan["pick"] == "over sector cap").sum())
        n_unk = int((plan["pick"] == "no sector yet").sum())
        bits = [f"**sector fill** {n_div} promoted past a higher-ranked name whose sector was full"] if n_div else []
        if n_cap:
            bits.append(f"**over sector cap** {n_cap} taken over the {s['max_per_sector']}-per-sector cap "
                        "to fill the list")
        if n_unk:
            bits.append(f"**no sector yet** {n_unk} with no sector on file, so never capped")
        st.caption(("Sector cap {}/sector · ".format(s["max_per_sector"])
                    + " · ".join(bits)) if bits
                   else f"Sector cap {s['max_per_sector']}/sector — the top {len(plan)} by score "
                        "were already spread out, nothing had to be promoted.")
        st.caption(f"Diversification never reaches past rank {s['max_promote_rank']} — "
                   "below that the stock is not worth owning whatever sector it is in.")
        st.caption("Sectors in this list: "
                   + ", ".join(f"{k} x{v}" for k, v in plan["sector"].value_counts().items()))

    with st.expander(f"All {len(cands)} names "
                      + ("in the imported list" if _imported else "that qualified")
                      + ", ranked"):
        cols = ["combined_score", "technical_score", "fundamentals_score", "what is good",
                "close", "extension_%", "momentum_%", "volume_surge", "ema_fast"]
        show_df(scored[[c for c in cols if c in scored.columns]].round(2))
        st.caption("Combined = "
                   f"{s['breakout'].w_technical:.0%} technical + "
                   f"{s['breakout'].w_fundamental:.0%} fundamentals, "
                   "renormalised for any name whose statements are missing.")

    if s.get("use_fundamentals") and not fund.empty:
        with st.expander("Fundamentals in detail — the numbers behind the score"):
            fcols = ["fundamentals_score", "what is good", "watch", "context applied",
                     "sales CAGR 3y %", "sales growth 1y %", "net margin %",
                     "ROE %", "ROCE %", "CFO / PAT (3y)", "debt / equity", "note"]
            show_df(fund.reindex(scored.index)[[c for c in fcols if c in fund.columns]].round(2))
            st.caption(
                "Cash from operations carries the heaviest weight (35%), then sales growth "
                "and margins (20% each), ROCE (15%) and ROE (10%). **Context applied** lists "
                "every adjustment made to the raw ratios — a low ROE on a debt-free, "
                "cash-generative company is marked *up*; a high ROE that sits far above ROCE "
                "on heavy debt is marked *down*.")

    # ---- record into the journal ---- #
    if book is None:
        st.info("Create a trading book above to record these buys.")
        return
    if blocked:
        return

    with card("Queue these buys",
              "Drafts only — cash and P&L start when you confirm the real fill."):
        editor = plan[["symbol", "qty", "decision price", "stop in force"]].copy()
        editor = editor.rename(columns={"stop in force": "stop"})
        editor.insert(0, "take", True)
        edited = st.data_editor(editor, **_WIDE, key="buy_editor", hide_index=True,
                                 disabled=["decision price"])

        c1, c2 = st.columns([1, 3])
        decided = c1.date_input("Decision date", value=date.today(), key="buy_date")
        if c2.button("Queue as drafts", type="primary"):
            n, skipped = 0, []
            for _, r in edited.iterrows():
                if not bool(r["take"]) or int(r["qty"]) <= 0:
                    continue
                sym = str(r["symbol"])
                row = plan.loc[plan["symbol"] == sym]
                d = book.add_draft(
                    sym, decided, int(r["qty"]), float(r["decision price"]), float(r["stop"]),
                    hard_stop=hard_stop_price(float(r["decision price"]), dist_pct),
                    score=float(row["combined score"].iloc[0]) if len(row) else float("nan"),
                    sector=str(row["sector"].iloc[0]) if len(row) else "")
                if d is None:
                    skipped.append(sym)
                else:
                    n += 1
            persist_book(book)
            if skipped:
                st.warning("Already drafted or already held, so skipped: " + ", ".join(skipped))
            st.success(f"Queued {n} draft(s). Confirm the fills in the panel at the top.")
            if n:
                st.rerun()


# --------------------------------------------------------------------------- #
# TAB 3 — Positions & exits
# --------------------------------------------------------------------------- #
def tab_positions(s: dict) -> None:
    page_head("Positions & exits",
              "Live book · weekly 20/50 EMA · ladder. Refresh to mark-to-market.")
    book = book_picker("pos")
    if book is None:
        return

    # drafts are shown before any download so you can confirm Monday's fills
    # without waiting for prices; the last refresh's marks are reused if there
    # were any, which is what makes the drift column work
    if book.drafts:
        st.info(f"⏳ **{len(book.drafts)} draft buy(s)** waiting — confirm them on the "
                "*This week's buys* tab.")
    if not book.positions:
        st.info("No confirmed positions yet. Queue buys on *This week's buys*, then confirm "
                "the fills there.")
        manage_ui(book)
        return

    # same reason as the buy tab: a button is True for one rerun only, so gating
    # on it directly made every button below this line unclickable
    if st.button("Refresh prices and check the ladder", type="primary"):
        st.session_state["pos_ran"] = True
    if not st.session_state.get("pos_ran"):
        st.caption("Press refresh to price the book and see which rungs have fired.")
        st.markdown(saas_hold_html(jn.open_positions_frame(book, ladder=s["ladder"])),
                    unsafe_allow_html=True)
        return

    ctx = build_context(s, for_live=True, restrict_to=list(book.open_symbols()))
    panel, sig = ctx["panel"], ctx["signals"]
    wc = sig.weekly.get("Close", pd.DataFrame())
    week = wc.index[-1]
    if panel["Close"].index[-1] < week and len(wc.index) > 1:
        week = wc.index[-2]

    last_px = panel["Close"].iloc[-1].copy()
    live_asof = ""
    if not s.get("demo"):
        with st.spinner("Fetching live CMP…"):
            live = data_mod.live_last_prices(list(book.open_symbols()))
        if len(live):
            for sym, px in live.items():
                last_px[sym] = float(px)
            live_asof = _now_ist().strftime("%d %b %Y %H:%M")
    st.session_state["_last_px"] = last_px
    ef = sig.ema_fast.loc[week] if week in sig.ema_fast.index else pd.Series(dtype=float)
    es = sig.ema_slow.loc[week] if week in sig.ema_slow.index else pd.Series(dtype=float)

    summ = jn.summary(book, last_px)
    d = js.open_dashboard(book, last_px, ema_fast=ef)

    open_syms = sorted(book.open_symbols())
    sectors = load_sectors(tuple(open_syms), s["demo"]) if open_syms else {}
    buckets, ix_notes = load_index_buckets(tuple(open_syms), s["demo"])
    hold = jn.open_positions_frame(book, last_px, ef, es, s["ladder"],
                                    sectors=sectors, index_buckets=buckets,
                                    equity=summ["Portfolio value"])
    # Same rupees as the table — one sum, not a second formula.
    if hold is not None and not hold.empty and "open_risk" in hold.columns:
        risk_total = float(pd.to_numeric(hold["open_risk"], errors="coerce").fillna(0).sum())
    else:
        risk_total = float(d.get("risk to stops") or 0)
    cap = float(book.capital or 0)
    risk_pct_cap = (risk_total / cap * 100) if cap else np.nan

    st.markdown("##### Dashboard")
    tiles_row([
        ("Portfolio value", rupees(d["portfolio value"]),
         f"capital {rupees(book.capital)}", ""),
        ("Capital deployed", rupees(d["deployed"]),
         f"{d['deployed %']:,.1f}% working · {d['cash %']:,.1f}% cash", ""),
        ("Unrealised P&L", rupees(d["unrealised"]),
         f"{d['unrealised %']:+,.2f}% on {rupees(d['cost'])} of cost"
         if np.isfinite(d["unrealised %"]) else "", tone_of(d["unrealised"])),
        ("Up / down", f"{d['winners']} / {d['losers']}",
         f"{d['win share %']:,.0f}% of {d['positions']} in profit"
         if np.isfinite(d["win share %"]) else "", ""),
        ("Realised so far", rupees(d["realised so far"]),
         f"{rupees(d['booked in open trades'])} of it from trades still open",
         tone_of(d["realised so far"])),
    ])
    tiles_row([
        ("Best open", (f"{d['best']['symbol']}" if d["best"] else "—"),
         (f"{rupees(d['best']['pnl'])} · {d['best']['pct']:+,.1f}%" if d["best"] else ""),
         tone_of(d["best"]["pnl"] if d["best"] else 0)),
        ("Worst open", (f"{d['worst']['symbol']}" if d["worst"] else "—"),
         (f"{rupees(d['worst']['pnl'])} · {d['worst']['pct']:+,.1f}%" if d["worst"] else ""),
         tone_of(d["worst"]["pnl"] if d["worst"] else 0)),
        ("Biggest position", d["largest"] or "—",
         f"{d['largest %']:,.1f}% of the book" if np.isfinite(d["largest %"]) else "", ""),
        ("Open Risk", rupees(risk_total),
         (f"{risk_pct_cap:,.2f}% of capital ({rupees(cap)})"
          if np.isfinite(risk_pct_cap) else ""),
         "neg"),
        ("Avg weeks held",
         (f"{d['avg days held']/7:,.1f}" if np.isfinite(d["avg days held"]) else "—"),
         "open positions · weekly holds", ""),
    ])

    last_day = panel["Close"].index[-1]
    trail_levels = {k: (v.loc[last_day] if last_day in v.index else pd.Series(dtype=float))
                    for k, v in sig.trail_emas.items()}
    pending = jn.pending_actions(book, week, wc.loc[week], ef, es, s["ladder"],
                                  rearm_ema=s.get("rearm_ema", False),
                                  trail_levels=trail_levels,
                                  daily_close=last_px,
                                  targets_on_daily_close=s.get("targets_on_daily_close", True))
    st.markdown(take_action_html(pending, week), unsafe_allow_html=True)

    wl = js.winners_losers(d["detail"])
    if not wl.empty:
        with card("Up / down"):
            show_money_df(wl, money_cols=("Capital", "Value now", "Unrealised"),
                          pct_cols=("Avg %",))

    open_syms = sorted(book.open_symbols())

    with card("What you hold",
              ("Open positions · live CMP " + (f"as of {live_asof} IST" if live_asof
               else "yesterday’s close — live quote unavailable"))):
        st.markdown(saas_hold_html(hold, capital=cap, risk_total=risk_total),
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with card("By sector"):
            sec = (pd.Series({sym: sectors.get(sym) or "Unknown" for sym in open_syms})
                   .value_counts().rename_axis("Sector").reset_index(name="Positions"))
            st.markdown(saas_simple_html(sec), unsafe_allow_html=True)
    with c2:
        with card("By index"):
            obi = js.open_by_index(book, buckets, last_px, order=ix_mod.bucket_order(buckets.values()))
            view = obi.drop(columns=["Stocks"]) if "Stocks" in obi.columns else obi
            st.markdown(saas_simple_html(view, money=("Value", "Unrealised"),
                                          pct=("% of open value",)),
                        unsafe_allow_html=True)
    st.caption("Index buckets are NSE's own size bands, and they are **today's** lists — "
               "NSE does not publish historical membership. " + " · ".join(ix_notes))

    pairs = js.demerger_pairs(book, last_px)
    if not pairs.empty:
        with card("Demerged holdings, read as a pair"):
            show_df(pairs)
            st.caption("A demerger books no profit and no loss — it splits one cost basis "
                       "across two companies. Read alone the parent looks like a sudden loss "
                       "and the child like a windfall, and neither happened. **Combined P&L** "
                       "is the only number that answers whether the holding was worth it.")

    if not pending.empty:
        with card("Record these fills",
                  f"Week ending {pd.Timestamp(week).date()} · tick what you actually sold"):
            ed = pending[["symbol", "qty", "last_close", "rung", "why"]].copy()
            ed = ed.rename(columns={"last_close": "fill price"})
            ed.insert(0, "sell", True)
            edited = st.data_editor(ed, **_WIDE, key="sell_editor", hide_index=True,
                                     disabled=["rung", "why"])
            c1, c2 = st.columns([1, 3])
            d = c1.date_input("Fill date", value=date.today(), key="sell_date")
            if c2.button("Save these exits to the journal", type="primary"):
                n = 0
                for _, r in edited.iterrows():
                    if not bool(r["sell"]) or int(r["qty"]) <= 0:
                        continue
                    rec = book.sell(str(r["symbol"]), d, int(r["qty"]), float(r["fill price"]),
                                     str(r["rung"]), str(r["why"]))
                    if rec:
                        n += 1
                persist_book(book)
                st.success(f"Recorded {n} exits.")
                st.rerun()

    manage_ui(book)

    with st.expander("Sell something manually"):
        opens = sorted(book.open_symbols())
        if opens:
            c1, c2, c3, c4 = st.columns(4)
            sym = c1.selectbox("Stock", opens, key="man_sym")
            pos = book.find(sym)
            qty = c2.number_input("Qty", 1, int(pos.open_qty), int(pos.open_qty), key="man_qty")
            px = c3.number_input("Price", 0.01, 1e7,
                                  float(last_px.get(sym, pos.entry_price)), key="man_px")
            dd = c4.date_input("Date", value=date.today(), key="man_date")
            if st.button("Record manual exit"):
                book.sell(sym, dd, int(qty), float(px), "manual", "Manual exit")
                persist_book(book)
                st.success("Recorded.")
                st.rerun()


# --------------------------------------------------------------------------- #
# TAB 4 — Trading journal
# --------------------------------------------------------------------------- #
def tab_journal(s: dict) -> None:
    page_head("Trading journal",
              "KPI strip + calendar first · daily equity · weekly P&L")
    book = book_picker("jr")
    if book is None:
        return

    led = jn.ledger_frame(book)
    if led.empty:
        st.info("No fills recorded yet.")
        return

    # ---- prices, so the curve-based numbers can exist at all -------------- #
    close = None
    if st.toggle("Load prices for the full analysis", value=True, key="jr_prices",
                  help="Drawdown, CAGR and the equity curve need a daily mark-to-market of "
                       "everything you held, which needs prices. Without them you still get "
                       "realised P&L, win rate, profit factor and the exit breakdown."):
        names = sorted({p.symbol for p in book.positions + book.closed})
        ctx = build_context(s, for_live=True, restrict_to=names)
        close = ctx["panel"]["Close"].copy()
        if not s.get("demo") and names:
            live = data_mod.live_last_prices(
                [p.symbol for p in book.positions if p.is_open()]
            )
            if len(live):
                last = close.iloc[-1].copy()
                for sym, px in live.items():
                    last[sym] = float(px)
                close.iloc[-1] = last

    st_ = js.stats(book, close)
    rb = js.exit_reasons(book)
    rt = js.round_trips(book)
    eq = js.equity_curve(book, close)

    tiles_row([
        ("Net P&L", rupees(st_["Net P&L"]),
         f"realised {rupees(st_['Realised P&L'])} · unrealised {rupees(st_['Unrealised P&L'])}"
         + (f" · dividends {rupees(st_['Dividends'])}" if st_.get("Dividends") else ""),
         tone_of(st_["Net P&L"])),
        ("Overall ROI", f"{st_['ROI %']:,.1f}%" if np.isfinite(st_["ROI %"]) else "—",
         f"on {rupees(st_['Capital'])}", tone_of(st_["ROI %"])),
        ("Win rate", f"{st_['Win rate %']:,.0f}%" if np.isfinite(st_["Win rate %"]) else "—",
         f"{st_['Wins']}W / {st_['Losses']}L on closed trades", ""),
        # a book with no losing trade yet has an infinite factor, which is a real
        # statement about it — not a missing number
        ("Profit factor",
         "∞" if st_["Profit factor"] == np.inf else _safe_ratio(st_["Profit factor"]),
         "gross win / gross loss", ""),
        ("Max drawdown",
         f"{st_['Max drawdown %']:,.1f}%" if np.isfinite(st_["Max drawdown %"]) else "—",
         "peak-to-trough of daily equity (cash+MTM)" if np.isfinite(st_["Max drawdown %"])
         else "needs prices", "neg"),
    ])

    age = st_.get("Book age (years)", np.nan)
    # Brokers (Zerodha Console etc.) annualise with
    # (end/start)^(365.25/days) − 1 once the book is ~a quarter old.
    cagr_ready = np.isfinite(st_.get("CAGR %", np.nan)) and np.isfinite(age) and age >= 90 / 365.25
    if cagr_ready:
        cagr_sub = f"annualised · {age * 12:,.1f} months"
    elif np.isfinite(age) and age < 90 / 365.25:
        days_so_far = int(round(age * 365.25))
        cagr_sub = f"after 3 months ({days_so_far}d so far)"
    else:
        cagr_sub = "needs prices"
    tiles_row([
        ("CAGR", f"{st_['CAGR %']:,.1f}%" if cagr_ready else "—",
         cagr_sub,
         tone_of(st_.get("CAGR %", 0)) if cagr_ready else ""),
        ("Expectancy", rupees(st_["Expectancy ₹"]) if np.isfinite(st_["Expectancy ₹"]) else "—",
         f"{st_['Expectancy %']:,.1f}% per trade" if np.isfinite(st_["Expectancy %"]) else "",
         tone_of(st_["Expectancy ₹"])),
        ("Avg win / avg loss",
         (f"{st_['Avg win %']:,.1f}% / {st_['Avg loss %']:,.1f}%"
          if np.isfinite(st_["Avg win %"]) and np.isfinite(st_["Avg loss %"]) else "—"),
         f"risk-reward {_safe_ratio(st_['Risk-reward'])}", ""),
        ("Trades closed", f"{st_['Trades closed']}",
         (f"avg {st_['Avg days held']/7:,.1f} weeks held"
          if np.isfinite(st_["Avg days held"]) else ""), ""),
        ("Calmar", _safe_ratio(st_.get("Calmar", np.nan)), "CAGR / max drawdown", ""),
    ])

    if close is None:
        st.caption("**Drawdown, CAGR, Calmar, the equity curve and the year / month tables "
                   "need prices** — a drawdown happens in the weeks when nothing is filled, "
                   "so a list of fills cannot see it. Switch the toggle above on.")

    if book.ledger:
        filled_years = {pd.Timestamp(r.get("date")).year for r in book.ledger if r.get("date")}
        this_year = date.today().year
        start_y = min(filled_years | {this_year, 2026})
        end_y = max(this_year + 10, max(filled_years) if filled_years else this_year)
        years = list(range(int(start_y), int(end_y) + 1))
        prefs_path = os.path.join(APP_DIR, "ui_prefs.json")
        saved_year = None
        try:
            with open(prefs_path, encoding="utf-8") as f:
                saved_year = json.load(f).get("calendar_year")
        except Exception:
            saved_year = st.session_state.get("cal_year")
        default_year = int(saved_year) if saved_year in years else (
            this_year if this_year in years else years[-1])
        year = st.selectbox("Calendar year", years,
                            index=years.index(default_year),
                            key="cal_year",
                            help="Empty years stay empty until you trade in them. "
                                 "Last choice is remembered.")
        try:
            blob = {}
            if os.path.exists(prefs_path):
                with open(prefs_path, encoding="utf-8") as f:
                    blob = json.load(f) or {}
            if blob.get("calendar_year") != year:
                blob["calendar_year"] = int(year)
                with open(prefs_path, "w", encoding="utf-8") as f:
                    json.dump(blob, f)
        except Exception:
            pass
        bm = js.best_month(rt) if len(rt) else None
        last_eq = float(st_["Portfolio value"]) if np.isfinite(st_.get("Portfolio value", np.nan)) else (
            float(eq.iloc[-1]) if len(eq) else book.capital)
        ytd = (last_eq / float(book.capital) - 1) * 100 if book.capital else np.nan
        kpis = [
            ("Starting capital", rupees(book.capital), "book capital", ""),
            ("Final capital", rupees(last_eq),
             f"Net P&L {rupees(st_['Net P&L'])}", tone_of(st_["Net P&L"])),
            ("YTD return",
             f"{ytd:+.2f}%" if np.isfinite(ytd) else "—",
             "vs starting capital", tone_of(ytd if np.isfinite(ytd) else 0)),
            ("Total trades", f"{st_['Trades closed']}",
             f"{st_['Wins']}W / {st_['Losses']}L", ""),
            ("Win rate",
             f"{st_['Win rate %']:,.1f}%" if np.isfinite(st_["Win rate %"]) else "—",
             "closed trades", ""),
            ("Avg R:R", _safe_ratio(st_.get("Risk-reward", np.nan)),
             "among closed trades", ""),
            ("Best month", bm["month"] if bm else "—",
             rupees(bm["pnl"]) if bm else "", "pos" if bm and bm["pnl"] > 0 else ""),
        ]
        st.markdown(trading_calendar_html(book, int(year), kpis), unsafe_allow_html=True)

    # ---- equity curve and drawdown --------------------------------------- #
    if len(eq) > 2:
        with card("Equity & drawdown", "Daily close. ₹ zooms to rupees; % is return from day one."):
            mode = st.radio("Equity curve", ["₹ rupees", "% from start"],
                             horizontal=True, key="eq_mode")
            g1, g2 = st.columns([1.7, 1])
            with g1:
                show_chart(equity_figure(eq, None, s["dark"], pct=mode.startswith("%")))
            with g2:
                show_chart(drawdown_figure(eq, s["dark"]))
                if len(rt):
                    show_chart(weekly_pnl_figure(rt, s["dark"]))

        if len(rt):
            bm = js.best_month(rt)
            sk = js.streaks(rt)
            tiles_row([
                ("Best month", bm["month"] if bm else "—",
                 (rupees(bm["pnl"]) if bm else ""), "pos" if bm and bm["pnl"] > 0 else ""),
                ("Max winning streak", str(sk["max win streak"]),
                 "closed trades in a row", "pos"),
                ("Max losing streak", str(sk["max loss streak"]),
                 "closed trades in a row", "neg"),
                ("Win streak now", str(sk["win streak"]), "", ""),
            ])
            mt = js.monthly_trade_table(rt)
            yt = js.yearly_trade_table(rt)
            with card("Monthly performance"):
                show_money_df(mt, money_cols=("P&L ₹", "Capital"),
                              pct_cols=("Win %", "P&L %", "Avg gain", "Avg loss", "Best", "Worst"))
            with card("Yearly performance"):
                show_money_df(yt, money_cols=("P&L ₹", "Capital"),
                              pct_cols=("Win %", "P&L %", "Avg gain", "Avg loss"))
            if not mt.empty:
                t = ch.theme(False)
                fig = go.Figure(go.Bar(
                    x=mt["Month"], y=mt["P&L %"],
                    marker_color=[t["good"] if v >= 0 else t["critical"] for v in mt["P&L %"]],
                ))
                fig.update_layout(template="plotly_white", height=260,
                                  yaxis_title="P&L %", showlegend=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  paper_bgcolor="#fff", plot_bgcolor="#fff")
                st.markdown("##### P&L vs month")
                show_chart(fig)

        with card("Equity by period", "Full width — year, month or week. No sideways scroll."):
            ydf = js.yearly(eq)
            mdf = js.monthly(eq)
            wdf = js.weekly(eq)
            t_y, t_m, t_w = st.tabs([
                f"Year by year ({len(ydf)})",
                f"Month by month ({len(mdf)})",
                f"Week by week ({len(wdf)})",
            ])
            with t_y:
                st.markdown(period_table_html(ydf, "year"), unsafe_allow_html=True)
            with t_m:
                st.markdown(period_table_html(mdf, "month"), unsafe_allow_html=True)
            with t_w:
                st.markdown(period_table_html(wdf, "week"), unsafe_allow_html=True)

    if len(rt):
        wins = int((rt["P&L"] > 0).sum())
        losses = int((rt["P&L"] <= 0).sum())
        wr = st_["Win rate %"] if np.isfinite(st_["Win rate %"]) else 0
        rr = st_.get("Risk-reward", np.nan)
        t = ch.theme(s["dark"])
        v1, v2 = st.columns(2)
        with v1:
            with card("Strike rate"):
                show_chart(donut_figure(
                    "Strike rate",
                    ["Winners", "Losers"],
                    [max(wins, 0), max(losses, 0)],
                    [t["good"], t["critical"]],
                    s["dark"],
                    f"{wr:,.0f}%" if np.isfinite(wr) else "—",
                ))
        with v2:
            aw = abs(float(st_.get("Avg win %") or 0))
            al = abs(float(st_.get("Avg loss %") or 0))
            if not np.isfinite(aw):
                aw = 0
            if not np.isfinite(al):
                al = 0
            with card("Risk : Reward"):
                show_chart(donut_figure(
                    "Risk : Reward",
                    ["Avg win %", "Avg loss %"],
                    [aw or 0.01, al or 0.01],
                    [t["good"], t["series"][3]],
                    s["dark"],
                    f"1 : {_safe_ratio(rr)}" if np.isfinite(rr) else "—",
                ))

    # ---- where the money came from --------------------------------------- #
    with card("P&L by exit reason"):
        st.caption("Profit targets, the 20 EMA break, the 50 EMA break — what each rung actually "
                   "paid.")
        if not rb.empty:
            show_chart(exit_reason_figure(rb, s["dark"]))
            show_df(rb)

    pb = js.partial_booking(rt)
    if not pb.empty:
        st.markdown("##### Did booking a target help?")
        show_df(pb)
        st.caption("Not a controlled experiment — a trade books a target *because* it went up. "
                   "Read it as a description of the two populations, not proof of cause.")

    # ---- index buckets ---------------------------------------------------- #
    if len(rt):
        syms = tuple(sorted(set(rt["symbol"]) | book.open_symbols()))
        bmap, ix_notes = load_index_buckets(syms, s["demo"])
        bi = js.by_index(rt, bmap, order=ix_mod.bucket_order(bmap.values()))
        if not bi.empty:
            st.markdown("##### Which end of the market pays")
            show_df(bi)
            st.caption("NSE's own size bands. **Drawdown ₹** is the deepest fall in that "
                       "bucket's own cumulative P&L, trade by trade — portfolio drawdown "
                       "cannot be split across buckets because the cash is shared. "
                       "**Buckets are today's lists**: NSE does not publish historical "
                       "membership, and a winner is exactly the stock most likely to have "
                       "moved up a band since you bought it, which flatters the larger ones.")

    # ---- best and worst --------------------------------------------------- #
    if len(rt) >= 2:
        best, worst = js.top_movers(rt, 10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Best trades")
            show_df(best)
        with c2:
            st.markdown("##### Worst trades")
            show_df(worst)
        ch_ = js.characteristics(rt)
        if not ch_.empty:
            st.markdown("##### What the winners had in common")
            show_df(ch_)

    slip = js.slippage_report(book)
    if not slip.empty:
        with st.expander(f"Slippage — buy list vs actual fill ({len(slip)} entries)"):
            show_df(slip)
            st.caption(f"Average **{slip['slippage %'].mean():+.2f}%**, total "
                       f"**{rupees(slip['slippage ₹'].sum())}**. "
                       "**Positive = you filled better than Friday’s close** "
                       "(paid less on a buy). ANTELOPUS 1006.55 → 942.27 is a plus, not a minus.")

    if len(rt):
        with st.expander(f"Every closed trade ({len(rt)})"):
            show_df(rt, height=420)

    with card("Every fill"):
        c1, c2, c3 = st.columns(3)
        syms = ["All"] + sorted(led["symbol"].unique().tolist())
        f_sym = c1.selectbox("Stock", syms)
        f_side = c2.selectbox("Side", ["All", "BUY", "SELL"])
        f_reason = c3.selectbox("Reason", ["All"] + sorted(led["reason"].dropna().unique().tolist()))

        view = led.copy()
        if f_sym != "All":
            view = view[view["symbol"] == f_sym]
        if f_side != "All":
            view = view[view["side"] == f_side]
        if f_reason != "All":
            view = view[view["reason"] == f_reason]
        st.markdown(saas_fills_html(view), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.download_button("Download journal (CSV)", led.to_csv(index=False).encode(),
                        f"{book.name}_journal.csv", "text/csv")
    try:
        import io

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            led.to_excel(xl, sheet_name="Journal", index=False)
            if not rb.empty:
                rb.to_excel(xl, sheet_name="By exit reason")
            if len(rt):
                rt.to_excel(xl, sheet_name="Closed trades", index=False)
            if len(eq) > 2:
                js.yearly(eq).to_excel(xl, sheet_name="Yearly", index=False)
                js.monthly(eq).to_excel(xl, sheet_name="Monthly", index=False)
                eq.to_frame("equity").to_excel(xl, sheet_name="Equity curve")
            if not slip.empty:
                slip.to_excel(xl, sheet_name="Slippage", index=False)
            jn.open_positions_frame(book).to_excel(xl, sheet_name="Open positions", index=False)
            pd.DataFrame([st_]).T.rename(columns={0: "value"}).to_excel(xl, sheet_name="Summary")
        c2.download_button("Download journal (Excel)", buf.getvalue(),
                            f"{book.name}_journal.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        xlsx_blob = buf.getvalue()
    except Exception:
        xlsx_blob = b""

    with st.expander("Backup"):
        st.caption("Writes a dated copy of this book — the raw JSON, the journal as CSV "
                   "and the full workbook — into your data folder. A backup you have to "
                   "remember to take is a backup you will not have; this is one click.")
        c1, c2 = st.columns([1, 2])
        if c1.button("Back up this book now", type="primary", key="jr_backup"):
            files = {f"{book.name}_journal.csv": led.to_csv(index=False).encode()}
            if xlsx_blob:
                files[f"{book.name}_journal.xlsx"] = xlsx_blob
            src = st.session_state.get("book_path")
            if src and os.path.exists(src):
                with open(src, "rb") as f:
                    files[os.path.basename(src)] = f.read()
            folder = sg.write_backup(APP_DIR, book.name, files)
            st.success(f"Saved {len(files)} file(s) to {folder}")
        prev = sg.list_backups(APP_DIR)
        if prev:
            c2.caption("Recent: " + " · ".join(f"{n} ({w})" for n, w in prev[:4]))
        st.caption(f"Backups folder: `{sg.backup_dir(APP_DIR)}`")

    eq = jn.equity_points(book)
    if len(eq) > 1:
        st.markdown("##### Cash after every fill")
        t = ch.theme(s["dark"])
        fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, line=dict(color=t["series"][0], width=2)))
        fig.update_layout(template="plotly_dark" if s["dark"] else "plotly_white",
                          paper_bgcolor=t["surface"], plot_bgcolor=t["surface"], height=260,
                          margin=dict(l=10, r=10, t=20, b=10),
                          yaxis=dict(gridcolor=t["grid"]), xaxis=dict(gridcolor=t["grid"]))
        show_chart(fig)

    with st.expander("Danger zone"):
        st.caption("Deleting a book cannot be undone. The JSON file lives in journal/.")
        if st.button("Delete this book"):
            p = st.session_state.get("book_path")
            if p and os.path.exists(p):
                os.remove(p)
            for k in ("book_path", "_book", "_book_cache_path"):
                st.session_state.pop(k, None)
            st.rerun()


# --------------------------------------------------------------------------- #
# TAB 5 — Universe & data
# --------------------------------------------------------------------------- #
def tab_universe(s: dict) -> None:
    page_head("Universe & data",
              f"{len(s['symbols'])} symbols · prices cached in `{data_mod.DEFAULT_CACHE}`")

    st.markdown('<div class="note"><b>Two honest limits.</b><br>'
                "1. <b>Buyer/seller-initiated trades cannot be sourced.</b> That split needs "
                "tick-level data classified by aggressor; NSE does not publish it free. The "
                "closest honest stand-in is NSE's total <i>number of trades</i> — your "
                "\"200 buyer AND 200 seller\" is roughly \"400 trades\". Switch it on in the "
                "sidebar; it downloads one small CSV per trading day and caches them.<br>"
                "2. <b>Market cap is today's share count × that day's price.</b> Yahoo does not "
                "publish historical share counts, so a company that doubled its shares in 2021 "
                "shows a 2020 market cap about double the real one. Your 500–50,000 cr band is "
                "wide enough that most names land on the right side of it."
                "</div>", unsafe_allow_html=True)

    if not st.button("Check the universe"):
        return

    ctx = build_context(s, for_live=True)
    res = ctx["screen"]
    panel = ctx["panel"]

    tiles_row([
        ("Symbols priced", f"{panel['Close'].shape[1]:,}", "", ""),
        ("Dropped for short history", f"{len(ctx['dropped']):,}", "", ""),
        ("Passing the screen now", f"{int(res.counts.iloc[-1]):,}" if len(res.counts) else "—", "", ""),
        ("Last price bar", str(pd.Timestamp(panel['Close'].index[-1]).date()), "", ""),
    ])

    if res.missing_inputs:
        st.warning("These filters could not run: " + "; ".join(res.missing_inputs))
    for n in res.notes:
        st.caption(n)

    with card("How many stocks pass the screen over time"):
        t = ch.theme(s["dark"])
        fig = go.Figure(go.Scatter(x=res.counts.index, y=res.counts.values,
                                    line=dict(color=t["series"][1], width=1.5)))
        fig.update_layout(template="plotly_white",
                          paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=260,
                          margin=dict(l=10, r=10, t=20, b=10),
                          yaxis=dict(gridcolor="#e5e7eb", title="Qualifying"),
                          xaxis=dict(gridcolor="#e5e7eb"))
        show_chart(fig)

    with card("Why each stock does or doesn't qualify, right now"):
        exp = screen_mod.explain_at(panel, s["screen"], panel["Close"].index[-1],
                                     market_cap=ctx["mcap"])
        show_df(exp, height=420)
        if ctx["dropped"]:
            with st.expander(f"{len(ctx['dropped'])} symbols dropped for insufficient history"):
                st.write(", ".join(ctx["dropped"]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    hydrate_books_from_browser()
    s = sidebar()
    inject_css(s["dark"])

    st.markdown(
        '<div class="app-top"><div><div class="app-name">Breakout Lab</div>'
        '<div class="app-sub">Weekly N-week-high breakouts on NSE · tiered booking · EMA trail · journal</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    with st.popover("Backup / Import"):
        book_tools_ui(current_book(), "hdr")

    if s["demo"]:
        st.warning("**Demo mode is on.** Prices are synthetic. Nothing here means anything "
                   "about real stocks.")

    t1, t2, t3, t4, t5 = st.tabs([
        "Backtest", "This week", "Positions", "Journal", "Universe",
    ])
    with t1:
        tab_backtest(s)
    with t2:
        tab_buys(s)
    with t3:
        tab_positions(s)
    with t4:
        tab_journal(s)
    with t5:
        tab_universe(s)


if __name__ == "__main__":
    main()
