"""Plotly figures for the research dashboard: one consistent, restrained style."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

M4A, ELO, NEUTRAL, GRID = "#1f4e79", "#d9822b", "#8a94a6", "#e5e9f0"
COLORS = {"M4A": M4A, "Surface Elo": ELO}
FONT = dict(family="Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=13, color="#1f2933")


def _style(fig: go.Figure, title: str | None = None, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(template="plotly_white", font=FONT, height=height, margin=dict(l=10, r=10, t=50 if title else 20,
                      b=60), title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, title=None),
                      hoverlabel=dict(font_size=12), **layout)  # fmt: skip
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def reliability(curves: dict[str, pl.DataFrame], title: str) -> go.Figure:
    """Calibration curves: mean predicted vs observed frequency per probability bin (marker size ~ matches)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                             name="Perfect calibration", hoverinfo="skip"))  # fmt: skip
    for name, c in curves.items():
        n = c["n"].to_list()
        size = [6 + 14 * (x / max(n)) ** 0.5 for x in n]
        fig.add_trace(go.Scatter(
            x=c["mean_predicted"].to_list(), y=c["observed"].to_list(), mode="lines+markers", name=name,
            line=dict(color=COLORS.get(name, NEUTRAL), width=2), marker=dict(size=size),
            customdata=list(zip(n, c["bin_lower"].to_list(), c["bin_upper"].to_list(), strict=True)),
            hovertemplate="%{customdata[1]:.1f}–%{customdata[2]:.1f} bin<br>predicted %{x:.3f}<br>observed %{y:.3f}"
                          "<br>%{customdata[0]:,} matches<extra>" + name + "</extra>"))  # fmt: skip
    _style(fig, title, height=420)
    fig.update_xaxes(title="Mean predicted P(player A wins)", range=[0, 1])
    fig.update_yaxes(title="Observed frequency", range=[0, 1])
    return fig


def delta_bars(df: pl.DataFrame, label: str, d: str, lo: str | None, hi: str | None, title: str,
               n: str = "n") -> go.Figure:  # fmt: skip
    """Horizontal bars of a difference (M4A − Elo) with optional interval whiskers; negative favours M4A."""
    err = None
    if lo and hi:
        err = dict(type="data", symmetric=False, array=(df[hi] - df[d]).to_list(),
                   arrayminus=(df[d] - df[lo]).to_list(), color=NEUTRAL, thickness=1.2)  # fmt: skip
    colors = [M4A if v < 0 else ELO for v in df[d].to_list()]
    fig = go.Figure(go.Bar(
        y=df[label].to_list(), x=df[d].to_list(), orientation="h", marker_color=colors, error_x=err,
        customdata=df[n].to_list() if n in df.columns else None,
        hovertemplate="%{y}<br>Δ %{x:+.4f}" + ("<br>%{customdata:,} matches" if n in df.columns else "")
                      + "<extra></extra>"))  # fmt: skip
    fig.add_vline(x=0, line=dict(color="#1f2933", width=1))
    _style(fig, title, height=max(260, 46 * df.height + 90))
    fig.update_xaxes(title="Δ log loss, M4A − Elo (negative favours M4A)")
    fig.update_yaxes(autorange="reversed")
    return fig


def metric_bars(rows: list[dict], metric: str, title: str, fmt: str = ".4f") -> go.Figure:
    """Grouped bars for one metric per evaluation period, M4A next to surface Elo."""
    fig = go.Figure()
    for key, name in (("m4a", "M4A"), ("elo", "Surface Elo")):
        fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=[r[key][metric] for r in rows], name=name,
                             marker_color=COLORS[name], texttemplate="%{y:" + fmt + "}", textposition="outside",
                             hovertemplate="%{x}<br>" + name + ": %{y:" + fmt + "}<extra></extra>"))  # fmt: skip
    vals = [r[k][metric] for r in rows for k in ("m4a", "elo")]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.6 or 0.01
    _style(fig, title, height=330, barmode="group")
    fig.update_yaxes(range=[max(0, lo - pad), hi + pad])
    return fig


def confidence_hist(conf: pl.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for name in ("M4A", "Surface Elo"):
        x = conf.filter(pl.col("model") == name)["confidence"].to_list()
        fig.add_trace(
            go.Histogram(
                x=x,
                name=name,
                marker_color=COLORS[name],
                opacity=0.65,
                xbins=dict(start=0.5, end=1.0, size=0.025),
                hovertemplate="%{x}<br>%{y:,} matches<extra>" + name + "</extra>",
            )
        )
    _style(fig, title, barmode="overlay")
    fig.update_xaxes(title="Probability assigned to the favourite")
    fig.update_yaxes(title="Matches")
    return fig


def accuracy_by_confidence(conf: pl.DataFrame, title: str) -> go.Figure:
    b = conf.with_columns(((pl.col("confidence") - 0.5) // 0.05 * 0.05 + 0.5).clip(0.5, 0.95).alias("bucket"))
    g = b.group_by("model", "bucket").agg(pl.len().alias("n"), pl.col("correct").mean().alias("acc")).sort("bucket")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.5, 1], y=[0.5, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                             name="Calibrated", hoverinfo="skip"))  # fmt: skip
    for name in ("M4A", "Surface Elo"):
        s = g.filter(pl.col("model") == name)
        fig.add_trace(go.Scatter(x=(s["bucket"] + 0.025).to_list(), y=s["acc"].to_list(), mode="lines+markers",
                                 name=name, line=dict(color=COLORS[name]), customdata=s["n"].to_list(),
                                 hovertemplate="confidence %{x:.3f}<br>favourite won %{y:.1%}<br>%{customdata:,} "
                                               "matches<extra>" + name + "</extra>"))  # fmt: skip
    _style(fig, title)
    fig.update_xaxes(title="Probability assigned to the favourite (bucket midpoint)", range=[0.5, 1])
    fig.update_yaxes(title="Share of matches the favourite won", range=[0.45, 1.02], tickformat=".0%")
    return fig


def probability_scatter(df: pl.DataFrame, title: str, max_points: int = 6000) -> go.Figure:
    d = df.sample(n=min(max_points, df.height), seed=7) if df.height > max_points else df
    fig = go.Figure()
    for won, name, color in ((True, "Player A won", M4A), (False, "Player B won", ELO)):
        s = d.filter(pl.col("a_won") == won)
        fig.add_trace(go.Scattergl(
            x=s["p_elo"].to_list(), y=s["p_m4a"].to_list(), mode="markers", name=name,
            marker=dict(color=color, size=4, opacity=0.45),
            customdata=list(zip(s["player_a_name"].to_list(), s["player_b_name"].to_list(), s["tournament_name"]
                                .to_list(), s["season"].to_list(), strict=True)),
            hovertemplate="%{customdata[0]} vs %{customdata[1]}<br>%{customdata[2]} %{customdata[3]}<br>"
                          "Elo %{x:.3f} · M4A %{y:.3f}<extra>" + name + "</extra>"))  # fmt: skip
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                             showlegend=False, hoverinfo="skip"))  # fmt: skip
    _style(fig, title, height=420)
    fig.update_xaxes(title="Surface Elo P(player A wins)", range=[0, 1])
    fig.update_yaxes(title="M4A P(player A wins)", range=[0, 1])
    return fig


def cumulative_delta(p: pl.DataFrame, title: str) -> go.Figure:
    s = p.group_by("season").agg(pl.len().alias("n"), pl.col("d_log_loss").sum().alias("sum")).sort("season")
    s = s.with_columns((pl.col("sum").cum_sum() / pl.col("n").cum_sum()).alias("running"),
                       (pl.col("sum") / pl.col("n")).alias("season_mean"))  # fmt: skip
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["season"].to_list(), y=s["season_mean"].to_list(), name="Season mean",
                         marker_color=[M4A if v < 0 else ELO for v in s["season_mean"].to_list()], opacity=0.55,
                         customdata=s["n"].to_list(),
                         hovertemplate="%{x}: %{y:+.4f}<br>%{customdata:,} matches<extra></extra>"))  # fmt: skip
    fig.add_trace(go.Scatter(x=s["season"].to_list(), y=s["running"].to_list(), name="Running mean",
                             mode="lines+markers", line=dict(color="#1f2933", width=2),
                             hovertemplate="%{x}: running %{y:+.4f}<extra></extra>"))  # fmt: skip
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    _style(fig, title)
    fig.update_yaxes(title="Mean log-loss difference (M4A − Elo)")
    return fig


def rating_lines(hist: dict[str, pl.DataFrame], column: str, title: str) -> go.Figure:
    fig = go.Figure()
    palette = [M4A, ELO]
    for (name, h), color in zip(hist.items(), palette, strict=False):
        fig.add_trace(go.Scatter(x=h["tournament_date"].to_list(), y=h[column].to_list(), mode="lines", name=name,
                                 line=dict(color=color, width=2, shape="hv"),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>" + name + "</extra>"))  # fmt: skip
    _style(fig, title)
    fig.update_yaxes(title="Rating entering the match")
    return fig


def coverage_bars(cov: pl.DataFrame, title: str, highlight: tuple[int, int] | None = None) -> go.Figure:
    colors = [M4A if highlight and highlight[0] <= s <= highlight[1] else NEUTRAL for s in cov["season"].to_list()]
    fig = go.Figure(go.Bar(x=cov["season"].to_list(), y=cov["matches"].to_list(), marker_color=colors,
                           hovertemplate="%{x}: %{y:,} matches<extra></extra>"))  # fmt: skip
    _style(fig, title, height=300)
    fig.update_yaxes(title="Tour-level matches")
    return fig
