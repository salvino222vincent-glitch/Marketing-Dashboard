"""
Marketing Dashboard — Interactive Streamlit App
Filename: marketing_dashboard_streamlit.py

How to run:
1. pip install streamlit pandas numpy plotly scikit-learn
2. streamlit run marketing_dashboard_streamlit.py

Features:
- Sidebar controls (date range, channel filter, metric selector)
- KPIs (Visitors, Sessions, Conversion Rate, Revenue)
- Visualizations (time series, channel breakdown, conversion funnel)
- Insights box (automatically computed)
- Export filtered data to CSV
- Sample synthetic dataset generator (so app runs out-of-the-box)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Marketing Dashboard", layout="wide")

# ---------------------------
# Utility: generate sample data
# ---------------------------
@st.cache_data
def generate_sample_data(start_date: str = None, days: int = 180, seed: int = 42):
    np.random.seed(seed)
    if start_date is None:
        start = datetime.today() - timedelta(days=days)
    else:
        start = datetime.fromisoformat(start_date)

    dates = [start + timedelta(days=i) for i in range(days)]
    channels = ["Organic", "Paid", "Email", "Social", "Referral"]

    rows = []
    for d in dates:
        base_visitors = 1000 + 5 * (d - dates[0]).days  # small trend
        for ch in channels:
            ch_multiplier = {
                "Organic": 0.45,
                "Paid": 0.18,
                "Email": 0.12,
                "Social": 0.15,
                "Referral": 0.10,
            }[ch]
            visitors = max(5, int(np.random.normal(base_visitors * ch_multiplier, 20)))
            sessions = int(visitors * np.random.uniform(1.0, 1.4))
            conv_rate = {
                "Organic": 0.02,
                "Paid": 0.015,
                "Email": 0.045,
                "Social": 0.01,
                "Referral": 0.03,
            }[ch]
            conversions = np.random.binomial(sessions, conv_rate)
            revenue_per_conv = np.random.normal(70 if ch in ["Paid","Social"] else 60, 8)
            revenue = max(0, conversions * revenue_per_conv)

            rows.append({
                "date": d.date(),
                "channel": ch,
                "visitors": visitors,
                "sessions": sessions,
                "conversions": int(conversions),
                "revenue": round(float(revenue), 2),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ---------------------------
# Load data
# ---------------------------
df = generate_sample_data(days=240)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Filters & Controls")
min_date = df["date"].min().date()
max_date = df["date"].max().date()
start, end = st.sidebar.date_input("Date range", [min_date, max_date])

channels = st.sidebar.multiselect("Channels", options=sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))

metric = st.sidebar.selectbox("Main metric", ["Visitors", "Sessions", "Conversions", "Revenue"]) 

show_forecast = st.sidebar.checkbox("Show linear trend forecast (7 days)", value=False)

# ---------------------------
# Filter data
# ---------------------------
filtered = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end)) & (df["channel"].isin(channels))]

# Aggregate daily totals
agg = filtered.groupby("date").sum().reset_index().sort_values("date")
agg["conversion_rate"] = agg.apply(lambda r: (r["conversions"]/r["sessions"] if r["sessions"]>0 else 0), axis=1)

# ---------------------------
# Top row: Title and Highlights
# ---------------------------
st.title("📊 Marketing Dashboard — Interactive (Streamlit)")
st.markdown("**Highlights** — quick snapshot of performance for selected range and channels")

col1, col2, col3, col4 = st.columns(4)
with col1:
    tot_visitors = int(agg["visitors"].sum())
    st.metric("Visitors", f"{tot_visitors:,}")
with col2:
    tot_sessions = int(agg["sessions"].sum())
    st.metric("Sessions", f"{tot_sessions:,}")
with col3:
    tot_conversions = int(agg["conversions"].sum())
    conv_rate = (tot_conversions / tot_sessions) if tot_sessions > 0 else 0
    st.metric("Conversions", f"{tot_conversions:,}", f"{conv_rate:.2%} conv rate")
with col4:
    tot_revenue = agg["revenue"].sum()
    avg_order = (tot_revenue / tot_conversions) if tot_conversions > 0 else 0
    st.metric("Revenue (USD)", f"${tot_revenue:,.2f}", f"Avg order ${avg_order:,.2f}")

# ---------------------------
# Visualizations
# ---------------------------
st.markdown("---")
st.header("Visualization")

left, right = st.columns((2,1))

with left:
    st.subheader(f"{metric} over time")
    y_col = metric.lower()
    if y_col == "conversions":
        y_col = "conversions"
    elif y_col == "revenue":
        y_col = "revenue"
    elif y_col == "sessions":
        y_col = "sessions"
    else:
        y_col = "visitors"

    ts = filtered.groupby(["date"])[y_col].sum().reset_index()
    fig_ts = px.line(ts, x="date", y=y_col, title=f"{metric} (daily)", markers=True)

    # optionally add forecast
    if show_forecast and len(ts) > 7:
        lr = LinearRegression()
        X = np.arange(len(ts)).reshape(-1,1)
        y = ts[y_col].values
        lr.fit(X, y)
        # predict next 7 days
        future_X = np.arange(len(ts)+7).reshape(-1,1)
        preds = lr.predict(future_X)
        fig_ts.add_traces(go.Scatter(x=pd.date_range(ts["date"].min(), periods=len(preds)).date, y=preds, mode='lines', name='Trend (linear)', line=dict(dash='dash')))

    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Channel breakdown over time")
    ch_ts = filtered.groupby(["date","channel"]).sum().reset_index()
    fig_area = px.area(ch_ts, x="date", y="visitors", color="channel", title="Visitors by Channel (stacked)")
    st.plotly_chart(fig_area, use_container_width=True)

with right:
    st.subheader("Channel distribution")
    channel_sum = (
    filtered.groupby("channel")[["visitors", "sessions", "conversions", "revenue"]]
    .sum()
    .reset_index()
    )
    fig_pie = px.pie(channel_sum, names="channel", values="visitors", title="Visitors share by channel")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Top channels by conversions")
    n_rows = 5
    top_conv = channel_sum.sort_values("conversions", ascending=False).head(10)
    fig_bar = px.bar(top_conv, x="channel", y="conversions", title="Conversions by Channel")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# KPI Insights
# ---------------------------
st.markdown("---")
st.header("KPIs & Insights")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Conversion Funnel (aggregate)")
    funnel = agg[["visitors", "sessions", "conversions", "revenue"]].sum()
    funnel_stages = [
        {"stage": "Visitors", "value": int(funnel["visitors"])},
        {"stage": "Sessions", "value": int(funnel["sessions"])},
        {"stage": "Conversions", "value": int(funnel["conversions"])},
    ]
    f_fig = go.Figure(go.Funnel(y=[s['stage'] for s in funnel_stages], x=[s['value'] for s in funnel_stages]))
    f_fig.update_layout(title_text="Aggregate Funnel")
    st.plotly_chart(f_fig, use_container_width=True)

with col_b:
    st.subheader("Automated insights")
    insights = []

    # Insight 1: best performing channel by ROI-ish (revenue per visitor)
    channel_sum["rev_per_visitor"] = channel_sum.apply(lambda r: r["revenue"]/r["visitors"] if r["visitors"]>0 else 0, axis=1)
    best_rev_channel = channel_sum.loc[channel_sum["rev_per_visitor"].idxmax()]
    insights.append(f"Best revenue-per-visitor channel: {best_rev_channel['channel']} (${best_rev_channel['rev_per_visitor']:.2f} per visitor)")

    # Insight 2: conversion rate trend (last 14 days vs previous 14 days)
    recent = agg.sort_values('date')
    if len(recent) >= 28:
        last14 = recent.tail(14)
        prev14 = recent.tail(28).head(14)
        last_rate = (last14.conversions.sum()/last14.sessions.sum()) if last14.sessions.sum()>0 else 0
        prev_rate = (prev14.conversions.sum()/prev14.sessions.sum()) if prev14.sessions.sum()>0 else 0
        delta = last_rate - prev_rate
        sign = "up" if delta>0 else "down"
        insights.append(f"Conversion rate {sign} {abs(delta):.2%} in the last 14 days compared to the prior 14 days.")
    else:
        insights.append("Not enough data to compute short-term conversion trend (need 28 days).")

    # Insight 3: largest growth channel (visitors)
    ch_growth = filtered.groupby('channel').apply(lambda g: (g.groupby('date').sum()['visitors'].diff().mean())).fillna(0)
    if len(ch_growth)>0:
        best_growth_ch = ch_growth.idxmax()
        insights.append(f"Channel with highest average daily visitor growth: {best_growth_ch} (avg daily change {ch_growth.max():.1f})")

    for s in insights:
        st.write("- ", s)

# ---------------------------
# Data and export
# ---------------------------
st.markdown("---")
st.header("Data & Export")
with st.expander("Show filtered data (first 200 rows)"):
    st.dataframe(filtered.sort_values('date').reset_index(drop=True).head(200))

csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered data as CSV", csv, "marketing_filtered.csv", "text/csv")

# ---------------------------
# Footer: quick how-to and tech stack
# ---------------------------
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Tech stack**\n- Python\n- Streamlit\n- Pandas, NumPy\n- Plotly (interactive charts)\n- scikit-learn (simple trend)")
with col2:
    st.markdown("**What to explore next**\n- Add channel-level ROAS if ad spend is available\n- Add cohort analyses (LTV)\n- Add anomaly detection for spikes/dips\n- Add segmentation by campaign or landing page")

st.caption("Generated sample data — replace with your real analytics export (UTM/channel, date, visitors/sessions/conversions/revenue).")
