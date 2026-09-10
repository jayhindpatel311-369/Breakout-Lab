"""
Plotly charts.

Colour decisions here follow one rule: the palette is chosen for the *job* the
colour does, and it is colour-vision-safe.

* Portfolio vs benchmark, strategy vs benchmark drawdown — two identities, so
  categorical slots 1 and 2 (blue, orange). Never more than 8 categorical hues;
  anything past that folds into "Other".
* The monthly heatmap is a *diverging* scale — gain vs loss around a zero
  midpoint — so it uses blue↔red with a neutral grey middle, not the
  red-green you usually see. Red-green is the single worst choice for the ~8% of
  men with deuteranomaly: to them a great month and a terrible month are the
  same mud. Red still means loss, so nothing about the intuition is lost.
* One y-axis per chart, always. Two measures on two scales is two charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# theme
# --------------------------------------------------------------------------- #
LIGHT = {
    "surface": "#ffffff",
    "plane": "#f4f6f8",
    "text": "#111827",
    "text2": "#4b5563",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "axis": "#d1d5db",
    "series": ["#2563eb", "#ea580c", "#059669", "#d97706", "#db2777", "#16a34a", "#4f46e5", "#dc2626"],
    "pos": "#059669",
    "neg": "#dc2626",
    "mid": "#f3f4f6",
    "good": "#059669",
    "critical": "#dc2626",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text": "#ffffff",
    "text2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    "pos": "#3987e5",
    "neg": "#d03b3b",
    "mid": "#383835",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}


def theme(dark: bool = True) -> dict:
    return DARK if dark else LIGHT


FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base(fig: go.Figure, t: dict, title: str, height: int = 420, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=t["text"], family=FONT), x=0, xanchor="left"),
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["text2"], size=12),
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], font=dict(family=FONT, color=t["text"], size=12),
                        bordercolor=t["axis"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=t["text2"], size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=t["axis"],
                     tickfont=dict(color=t["muted"], size=11))
    fig.update_yaxes(title=dict(text=ytitle, font=dict(color=t["muted"], size=11)),
                     gridcolor=t["grid"], griddash="solid", zeroline=False,
                     linecolor="rgba(0,0,0,0)", tickfont=dict(color=t["muted"], size=11))
    return fig


def _rupee(v: float) -> str:
    """Indian formatting: 1,05,61,835."""
    if v is None or not np.isfinite(v):
        return "—"
    neg = v < 0
    s = f"{abs(v):.0f}"
    if len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts + [last3])
    return ("-" if neg else "") + "Rs " + s


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def equity_chart(
    equity: pd.Series,
    benchmark: pd.Series | None,
    invested: pd.Series | None,
    dark: bool = True,
    title: str = "Portfolio vs benchmark",
) -> go.Figure:
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Portfolio", mode="lines",
        line=dict(color=t["series"][0], width=2),
        fill="tozeroy", fillcolor=_alpha(t["series"][0], 0.10),
        hovertemplate="%{y:,.0f}<extra>Portfolio</extra>",
    ))
    if benchmark is not None and len(benchmark.dropna()):
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values, name="Benchmark", mode="lines",
            line=dict(color=t["series"][1], width=2),
            hovertemplate="%{y:,.0f}<extra>Benchmark</extra>",
        ))
    if invested is not None and len(invested.dropna()):
        fig.add_trace(go.Scatter(
            x=invested.index, y=invested.values, name="Money put in", mode="lines",
            line=dict(color=t["muted"], width=1.5, dash="dot"),
            hovertemplate="%{y:,.0f}<extra>Invested</extra>",
        ))
    return _base(fig, t, title, 440, "Capital (Rs)")


def growth_chart(twr: pd.Series, bench_twr: pd.Series | None, dark: bool = True) -> go.Figure:
    """Growth of Rs 100 — cashflow effects removed, so this is the honest
    strategy-vs-index comparison."""
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twr.index, y=(twr / twr.iloc[0] * 100).values, name="Strategy", mode="lines",
        line=dict(color=t["series"][0], width=2),
        hovertemplate="%{y:,.1f}<extra>Strategy</extra>",
    ))
    if bench_twr is not None and len(bench_twr.dropna()) > 1:
        fig.add_trace(go.Scatter(
            x=bench_twr.index, y=(bench_twr / bench_twr.iloc[0] * 100).values,
            name="Benchmark", mode="lines", line=dict(color=t["series"][1], width=2),
            hovertemplate="%{y:,.1f}<extra>Benchmark</extra>",
        ))
    fig = _base(fig, t, "Growth of Rs 100 (time-weighted, SIP effect removed)", 400, "Value of Rs 100")
    fig.update_yaxes(type="log")
    return fig


def underwater_chart(dd: pd.Series, bench_dd: pd.Series | None, dark: bool = True) -> go.Figure:
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=(dd * 100).values, name="Strategy", mode="lines",
        line=dict(color=t["series"][0], width=2),
        fill="tozeroy", fillcolor=_alpha(t["series"][0], 0.18),
        hovertemplate="%{y:.2f}%<extra>Strategy</extra>",
    ))
    if bench_dd is not None and len(bench_dd.dropna()):
        fig.add_trace(go.Scatter(
            x=bench_dd.index, y=(bench_dd * 100).values, name="Benchmark", mode="lines",
            line=dict(color=t["series"][1], width=2, dash="dot"),
            hovertemplate="%{y:.2f}%<extra>Benchmark</extra>",
        ))
    return _base(fig, t, "Underwater plot — how far below the previous peak", 340, "Drawdown (%)")


def monthly_heatmap(table: pd.DataFrame, dark: bool = True, title: str = "Monthly returns (%)") -> go.Figure:
    """Diverging blue (gain) ↔ red (loss) with a neutral midpoint at zero."""
    t = theme(dark)
    if table.empty:
        return _base(go.Figure(), t, title, 260)
    z = table.values.astype(float)
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    lim = max(lim, 1.0)
    text = np.where(np.isfinite(z), np.vectorize(lambda v: f"{v:.1f}" if np.isfinite(v) else "")(z), "")
    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(table.columns),
        y=[str(i) for i in table.index],
        colorscale=[[0.0, t["neg"]], [0.5, t["mid"]], [1.0, t["pos"]]],
        zmid=0, zmin=-lim, zmax=lim,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, family=FONT),
        xgap=2, ygap=2,
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(outlinewidth=0, tickfont=dict(color=t["muted"], size=10), thickness=10, len=0.8),
    ))
    fig = _base(fig, t, title, max(240, 42 * len(table) + 100))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed")
    return fig


def yearly_bars(yearly: pd.DataFrame, dark: bool = True) -> go.Figure:
    t = theme(dark)
    if yearly.empty:
        return _base(go.Figure(), t, "Year-wise return", 300)
    vals = yearly["Return (%)"].astype(float)
    colors = [t["pos"] if v >= 0 else t["neg"] for v in vals]
    fig = go.Figure(go.Bar(
        x=yearly["Year"].astype(str), y=vals, marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(color=t["text2"], size=11),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig = _base(fig, t, "Year-wise return (time-weighted)", 320, "Return (%)")
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_traces(marker_cornerradius=4)
    return fig


def allocation_area(weights: pd.DataFrame, labels: dict[str, str] | None, dark: bool = True,
                    max_series: int = 7) -> go.Figure:
    """Stacked exposure through time.

    Only the 7 largest average holdings get their own hue; the rest fold into
    'Other'. Seven plus 'Other' is exactly the 8 categorical slots available —
    an eighth named series would have to reuse a hue, and two identically
    coloured bands in one stack is worse than no breakdown at all. Cash sits on
    the neutral grey, which is not a categorical slot."""
    t = theme(dark)
    if weights.empty:
        return _base(go.Figure(), t, "Allocation over time", 340)
    w = weights.copy() * 100
    avg = w.mean().sort_values(ascending=False)
    top = list(avg.head(max_series).index)
    other = [c for c in w.columns if c not in top]
    plot = w[top].copy()
    if other:
        plot["Other"] = w[other].sum(axis=1)
    cash = (100 - plot.sum(axis=1)).clip(lower=0)

    fig = go.Figure()
    for i, c in enumerate(plot.columns):
        name = "Other" if c == "Other" else (labels or {}).get(c, c)
        fig.add_trace(go.Scatter(
            x=plot.index, y=plot[c].values, name=name, mode="lines", stackgroup="a",
            line=dict(width=0.5, color=t["surface"]),
            fillcolor=_alpha(t["series"][i % len(t["series"])], 0.85),
            hovertemplate="%{y:.1f}%<extra>" + name + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=cash.index, y=cash.values, name="Cash", mode="lines", stackgroup="a",
        line=dict(width=0.5, color=t["surface"]), fillcolor=_alpha(t["muted"], 0.35),
        hovertemplate="%{y:.1f}%<extra>Cash</extra>",
    ))
    return _base(fig, t, "Allocation over time (% of capital)", 380, "Weight (%)")


def rolling_chart(series: pd.Series, title: str, ytitle: str, dark: bool = True,
                  ref: float | None = None) -> go.Figure:
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name=title,
        line=dict(color=t["series"][0], width=2),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=t["muted"], width=1, dash="dot"))
    fig = _base(fig, t, title, 300, ytitle)
    fig.update_layout(showlegend=False)
    return fig


def factor_contribution(scored: pd.DataFrame, weights: dict[str, float], dark: bool = True,
                        top_n: int = 15) -> go.Figure:
    """Which factor pushed which name up the ranking, at one point in time."""
    t = theme(dark)
    if scored is None or scored.empty:
        return _base(go.Figure(), t, "Factor contribution", 340)
    rows = scored.head(top_n)
    fig = go.Figure()
    for i, (k, w) in enumerate(weights.items()):
        col = f"z_{k}"
        if col not in rows.columns:
            continue
        fig.add_trace(go.Bar(
            y=rows.index, x=(rows[col].fillna(0) * w).values, name=k, orientation="h",
            marker=dict(color=_alpha(t["series"][i % len(t["series"])], 0.9), line=dict(width=0)),
            hovertemplate="%{y} · " + k + ": %{x:.3f}<extra></extra>",
        ))
    fig = _base(fig, t, f"Score breakdown — top {len(rows)} by composite", max(320, 26 * len(rows) + 120),
                "")
    fig.update_layout(barmode="relative", hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def scatter_risk_return(points: pd.DataFrame, dark: bool = True) -> go.Figure:
    """points: index=label, columns=['vol','cagr'] as fractions."""
    t = theme(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=points["vol"] * 100, y=points["cagr"] * 100, mode="markers+text",
        text=points.index, textposition="top center",
        textfont=dict(color=t["text2"], size=10),
        marker=dict(size=12, color=t["series"][0], line=dict(width=2, color=t["surface"])),
        hovertemplate="%{text}<br>vol %{x:.1f}% · CAGR %{y:.1f}%<extra></extra>",
    ))
    fig = _base(fig, t, "Risk vs return", 380, "CAGR (%)")
    fig.update_xaxes(title=dict(text="Annualised volatility (%)", font=dict(color=t["muted"], size=11)))
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def universe_size_chart(timeline: pd.DataFrame, dark: bool = True) -> go.Figure:
    """How many names passed the screen at each rebalance.

    A flat line means your filter is really just picking a fixed list. A line
    that swings from 150 to 400 means the screen is doing real work — and that
    the strategy was operating in genuinely different opportunity sets at
    different times, which is the honest picture.
    """
    t = theme(dark)
    if timeline is None or timeline.empty:
        return _base(go.Figure(), t, "Universe size over time", 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["date"], y=timeline["Qualifying"], mode="lines",
        name="Passing the screen", line=dict(color=t["series"][0], width=2),
        fill="tozeroy", fillcolor=_alpha(t["series"][0], 0.12),
        hovertemplate="%{y} stocks<extra></extra>",
    ))
    fig = _base(fig, t, "Universe size over time — how many stocks passed the screen", 320,
                "Stocks qualifying")
    fig.update_layout(showlegend=False)
    return fig


def universe_churn_chart(timeline: pd.DataFrame, dark: bool = True) -> go.Figure:
    """Entries and exits per rebalance — the churn the screen imposes."""
    t = theme(dark)
    if timeline is None or timeline.empty:
        return _base(go.Figure(), t, "Universe churn", 300)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=timeline["date"], y=timeline["Entered"], name="Entered",
        marker=dict(color=t["series"][0], line=dict(width=0)),
        hovertemplate="%{y} entered<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=timeline["date"], y=-timeline["Left"], name="Left",
        marker=dict(color=t["series"][1], line=dict(width=0)),
        hovertemplate="%{y} left<extra></extra>",
    ))
    fig = _base(fig, t, "Universe churn per rebalance", 300, "Stocks")
    fig.update_layout(barmode="relative", bargap=0.15)
    return fig


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"
