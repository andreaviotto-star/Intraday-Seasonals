# app.py  —  v3: Market Holidays + Half-Day Sessions filters
import pytz
import os
import threading
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from config import SYMBOLS, PERIODS
from ts_data_fetcher import TradeStationFetcher, fetch_yfinance
from utils.seasonal_stats import (
    compute_avg_path, compute_heatmap_grid,
    apply_event_filter, compute_daily_performance,
    compute_hourly_volatility, compute_edge_score
)

st.set_page_config(layout="wide", page_title="Intraday Seasonals")

CACHE_DIR = "data"
os.makedirs(CACHE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# BACKGROUND DOWNLOADER  (race-condition safe)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def start_background_downloader():
    def fetch_all_silently():
        ts = TradeStationFetcher()
        end_d   = datetime.now()
        start_d = end_d - timedelta(days=365 * 10)
        for sym in SYMBOLS:
            file_path = os.path.join(CACHE_DIR, f"{sym}_local_cache.csv")
            if os.path.exists(file_path):
                continue
            tmp_path = file_path + ".tmp"
            try:
                df = ts.get_intraday_bars(sym, start=start_d, end=end_d, interval="5")
                if not df.empty:
                    df.to_csv(tmp_path)
                    os.replace(tmp_path, file_path)
            except Exception:
                try:
                    df = fetch_yfinance(sym, start=start_d, end=end_d, freq="5m")
                    if not df.empty:
                        df.to_csv(tmp_path)
                        os.replace(tmp_path, file_path)
                except Exception:
                    pass

    thread = threading.Thread(target=fetch_all_silently, daemon=True)
    thread.start()
    return True

start_background_downloader()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("Intraday Seasonals")
symbol       = st.sidebar.selectbox("Symbol", SYMBOLS)
period_label = st.sidebar.radio("Period", list(PERIODS.keys()))
years        = PERIODS[period_label]

st.sidebar.markdown("**Exclude Event Days:**")
event_filters = st.sidebar.multiselect(
    "High-Impact Macro Events",
    ["NFP", "FOMC", "CPI", "PCE", "Triple Witching", "End-of-Month"],
    default=[]
)
calendar_filters = st.sidebar.multiselect(
    "CME Calendar Anomalies",
    ["Market Holidays", "Half-Day Sessions"],
    default=[]
)
all_filters = event_filters + calendar_filters

tz_mode = st.sidebar.radio("Time zone", ["US futures (EST)", "Zürich (CET)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Edge Score Thresholds")
min_win_rate      = st.sidebar.slider("Min Win Rate (%)",    50.0, 70.0, 55.0, 0.5)
min_profit_factor = st.sidebar.slider("Min Profit Factor",    1.0,  3.0,  1.2, 0.1)
max_p_value       = st.sidebar.slider("Max p-value",         0.01, 0.10, 0.05, 0.01)
min_obs           = st.sidebar.slider("Min Observations (N)", 20,  100,   30,    5)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
end_dt   = datetime.now()
start_dt = end_dt - timedelta(days=365 * 10)

@st.cache_data(ttl=timedelta(hours=4), show_spinner=False)
def get_raw_data(sym: str):
    local_file = os.path.join(CACHE_DIR, f"{sym}_local_cache.csv")
    if os.path.exists(local_file):
        df = pd.read_csv(local_file, index_col=0, parse_dates=True)
        return df, "Local Database"
    ts = TradeStationFetcher()
    try:
        df = ts.get_intraday_bars(sym, start=start_dt, end=end_dt, interval="5")
        df.to_csv(local_file)
        return df, "TradeStation"
    except Exception:
        try:
            df = fetch_yfinance(sym, start=start_dt, end=end_dt, freq="5m")
            df.to_csv(local_file)
            return df, "yfinance"
        except Exception as e:
            return pd.DataFrame(), str(e)


@st.cache_data(show_spinner=False)
def calculate_seasonals(df_raw, years_lookback, filters, tz_setting,
                        end_reference, obs_gate, sym):
    cutoff = pd.to_datetime(
        datetime.strptime(end_reference, "%Y-%m-%d") - timedelta(days=365 * years_lookback),
        utc=True
    )
    df = df_raw.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index >= cutoff]

    if "Zürich" in tz_setting:
        df.index = df.index.tz_convert("Europe/Zurich")
    else:
        df.index = df.index.tz_convert("America/New_York")

    for evt in filters:
        df = apply_event_filter(df, evt)

    if df.empty or len(df) < 50:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    path       = compute_avg_path(df, freq="15min", anchor="open", symbol=sym, min_obs=obs_gate)
    heatmap    = compute_heatmap_grid(df, freq="30min", symbol=sym)
    daily_perf = compute_daily_performance(df, symbol=sym)
    hourly_vol = compute_hourly_volatility(df, symbol=sym)

    return path, heatmap, daily_perf, hourly_vol


# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
with st.spinner(f"Loading {symbol}..."):
    raw_df_full, source = get_raw_data(symbol)

if raw_df_full.empty:
    st.error("❌ Failed to fetch data. Background downloader may still be working.")
    st.stop()

with st.spinner("Crunching math..."):
    PathData, HeatmapData, DailyPerf, HourlyVol = calculate_seasonals(
        raw_df_full, years, tuple(all_filters), tz_mode,
        end_dt.strftime("%Y-%m-%d"), min_obs, symbol
    )

if PathData.empty:
    st.warning("Not enough data matches the selected filters.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.write(f"### {symbol} – {period_label} Institutional Analysis")

# Show active filter summary
active = []
if event_filters:   active.append(f"Macro: {', '.join(event_filters)}")
if calendar_filters: active.append(f"Calendar: {', '.join(calendar_filters)}")
filter_str = " | ".join(active) if active else "No filters applied"
st.caption(f"Data source: {source}  •  Filters: {filter_str}")


# ─────────────────────────────────────────────
# SECTION 1: EDGE SCORE TABLE
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Ranked Edge Opportunities")
st.caption(
    f"Win Rate ≥ {min_win_rate}% | "
    f"Profit Factor ≥ {min_profit_factor} | "
    f"p-value ≤ {max_p_value} | N ≥ {min_obs}"
)

edge_df = compute_edge_score(PathData, min_win_rate=min_win_rate,
                              min_profit_factor=min_profit_factor,
                              max_p_value=max_p_value)

if not edge_df.empty:
    display_cols = [c for c in
        ["label","avg","median","win_rate","profit_factor",
         "t_stat","p_value","sharpe","sortino","kelly","count"]
        if c in edge_df.columns]
    fmt = {"avg":"{:.3f}%","median":"{:.3f}%","win_rate":"{:.1f}%",
           "profit_factor":"{:.2f}","t_stat":"{:.2f}","p_value":"{:.4f}",
           "sharpe":"{:.2f}","sortino":"{:.2f}","kelly":"{:.3f}","count":"{:.0f}"}
    st.dataframe(
        edge_df[display_cols].head(10).style.format(
            {k: v for k, v in fmt.items() if k in display_cols}
        ),
        use_container_width=True
    )
else:
    st.warning("No time slots pass the selected thresholds. Try relaxing the filters.")


# ─────────────────────────────────────────────
# SECTION 2: TRADE RECOMMENDATION
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Seasonal Edge Recommendation")

if not edge_df.empty:
    top = edge_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success(f"**Best Entry Slot:**\n### {top.get('label', '—')}")
    with col2:
        st.info(f"**Avg Return:**\n### {top['avg']:.3f}%")
    with col3:
        st.warning(f"**Win Rate (N={int(top['count'])}):**\n### {top['win_rate']:.1f}%")
    with col4:
        kelly_str = f"{top['kelly']:.1%}" if not np.isnan(top.get('kelly', float('nan'))) else "N/A"
        st.info(f"**Kelly Size:**\n### {kelly_str}")
    sharpe = top.get('sharpe', float('nan'))
    if not np.isnan(sharpe):
        st.caption(
            f"Sharpe: **{sharpe:.2f}** | "
            f"Profit Factor: **{top.get('profit_factor', float('nan')):.2f}** | "
            f"t-stat: **{top.get('t_stat', float('nan')):.2f}** | "
            f"p-value: **{top.get('p_value', float('nan')):.4f}**"
        )
else:
    st.info("Refine filters to surface a statistically significant recommendation.")


# ─────────────────────────────────────────────
# SECTION 3: AVERAGE PATH ± PERCENTILE BANDS
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Cumulative Weekly Path — Median & 10th/90th Percentile ({tz_mode})")

fig = go.Figure()

if "p10" in PathData.columns and "p90" in PathData.columns:
    fig.add_trace(go.Scatter(
        x=PathData.index, y=PathData["p90"], mode="lines",
        line=dict(color="rgba(31,119,180,0)", width=0),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=PathData.index, y=PathData["p10"], mode="lines",
        line=dict(color="rgba(31,119,180,0)", width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
        showlegend=True, name="10th–90th pct", hoverinfo="skip"
    ))

fig.add_trace(go.Scatter(
    x=PathData.index, y=PathData["avg"], mode="lines", name="Mean path",
    line=dict(color="#1f77b4", width=2.5),
    text=PathData.get("label", PathData.index),
    customdata=np.stack([
        PathData.get("win_rate",      pd.Series(np.nan, index=PathData.index)),
        PathData.get("t_stat",        pd.Series(np.nan, index=PathData.index)),
        PathData.get("count",         pd.Series(np.nan, index=PathData.index)),
        PathData.get("profit_factor", pd.Series(np.nan, index=PathData.index)),
    ], axis=-1),
    hovertemplate=(
        "<b>%{text}</b><br>Avg: %{y:.3f}%<br>"
        "Win Rate: %{customdata[0]:.1f}%<br>"
        "t-stat: %{customdata[1]:.2f}<br>"
        "N: %{customdata[2]:.0f}<br>"
        "PF: %{customdata[3]:.2f}<extra></extra>"
    )
))

if "median" in PathData.columns:
    fig.add_trace(go.Scatter(
        x=PathData.index, y=PathData["median"], mode="lines",
        name="Median path", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        hoverinfo="skip"
    ))

tick_vals = [24, 48, 72, 96, 120]
tick_text = ["Mon","Tue","Wed","Thu","Fri"]
for x_val in tick_vals:
    fig.add_vline(x=x_val, line_width=1, line_dash="dash", line_color="gray")

fig.update_layout(
    xaxis=dict(title="", tickmode="array", tickvals=tick_vals, ticktext=tick_text),
    yaxis_title="% move vs weekly open",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4: HEATMAP
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Intraday Heatmap — Mean Daily Return % ({tz_mode})")
st.caption("Grey cells = no data (holiday / outside session). Not filled with 0%.")

max_hm_val = HeatmapData.abs().max().max()
fig_hm = px.imshow(
    HeatmapData,
    labels=dict(x="Time", y="Day", color="% move"),
    color_continuous_scale="RdYlGn",
    range_color=[-max_hm_val, max_hm_val],
    aspect="auto",
)
fig_hm.update_xaxes(tickangle=45, nticks=48)
st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 5: DAILY PERFORMANCE + VOLATILITY
# ─────────────────────────────────────────────
st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Daily Probability of Profit (Win Rate %)")
    if "win_rate" in DailyPerf.columns:
        colors = ["#2ca02c" if w >= 55 else "#d62728" for w in DailyPerf["win_rate"]]
        fig_win = go.Figure(data=[
            go.Bar(
                name="Win Rate", x=DailyPerf.index, y=DailyPerf["win_rate"],
                marker_color=colors,
                customdata=DailyPerf.get("count", pd.Series(dtype=float)),
                hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>N: %{customdata:.0f}<extra></extra>"
            )
        ])
        fig_win.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="Breakeven 50%")
        fig_win.add_hline(y=55, line_dash="dot", line_color="orange",
                          annotation_text="Edge threshold 55%")
        fig_win.update_layout(yaxis_title="Win Rate (%)", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_win, use_container_width=True)

    disp_cols = [c for c in
        ["avg","win_rate","profit_factor","sharpe","sortino","kelly","count"]
        if c in DailyPerf.columns]
    if disp_cols:
        st.dataframe(
            DailyPerf[disp_cols].style.format({
                "avg":"{:.3f}%","win_rate":"{:.1f}%","profit_factor":"{:.2f}",
                "sharpe":"{:.2f}","sortino":"{:.2f}","kelly":"{:.3f}","count":"{:.0f}"
            }),
            use_container_width=True
        )

with col_chart2:
    st.subheader(f"Intraday ATR Volatility Profile ({tz_mode})")
    if not HourlyVol.empty:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=HourlyVol["time_label"], y=HourlyVol["vol"],
            name="Mean ATR %", marker_color="#ff7f0e"
        ))
        if "vol_p90" in HourlyVol.columns:
            fig_vol.add_trace(go.Scatter(
                x=HourlyVol["time_label"], y=HourlyVol["vol_p90"],
                mode="lines+markers", name="90th pct ATR",
                line=dict(color="#d62728", width=1.5, dash="dot")
            ))
        fig_vol.update_layout(
            xaxis=dict(title="Hour of Day", tickangle=45, nticks=12),
            yaxis_title="ATR % (True Range / Open)",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_vol, use_container_width=True)