import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Energy Dashboard MVP", layout="wide")

# ---------------------------------------------------------
# GROUPED UNIVERSE DEFINITIONS
# ---------------------------------------------------------

futures_groups = {
    "Crude Oil": ["CL=F", "BZ=F"],
    "Refined Products": ["RB=F", "HO=F"],
    "Natural Gas": ["NG=F"],
    "Other": ["EH=F"]
}

equity_groups = {
    "Integrated Majors": ["XOM", "CVX", "BP", "SHEL", "TTE"],
    "Shale Producers": ["PXD", "EOG", "CLR", "DVN"],
    "Oilfield Services": ["SLB", "HAL", "BKR"],
    "Refiners": ["VLO", "MPC", "PSX"],
    "Midstream": ["ENB", "KMI", "WMB"]
}

etf_groups = {
    "Upstream": ["XOP", "IEO"],
    "Integrated/Broad Energy": ["XLE", "VDE"],
    "Oilfield Services": ["OIH"],
    "Midstream": ["AMLP"]
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def normalize(df):
    return df / df.iloc[0] * 100

def performance_table(tickers):
    today = date.today()

    lookbacks = {
        "1D": today - timedelta(days=1),
        "1M": today - timedelta(days=30),
        "3M": today - timedelta(days=90),
        "12M": today - timedelta(days=365),
    }

    df = yf.download(tickers, period="1y", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()

    results = []

    for ticker in tickers:
        series = df[ticker].dropna()
        if series.empty:
            results.append([ticker, None, None, None, None])
            continue

        last_price = series.iloc[-1]
        row = [ticker]

        for _, start_date in lookbacks.items():
            past = series[series.index >= pd.to_datetime(start_date)]
            if past.empty:
                row.append(None)
            else:
                pct = (last_price - past.iloc[0]) / past.iloc[0] * 100
                row.append(round(pct, 2))

        results.append(row)

    return pd.DataFrame(results, columns=["Ticker", "1D %", "1M %", "3M %", "12M %"])

def grouped_performance_table(groups):
    frames = []
    for group_name, tickers in groups.items():
        table = performance_table(tickers)
        table.insert(0, "Group", group_name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Energy Dashboard MVP")
section = st.sidebar.radio(
    "Select Section",
    ["Futures", "Equities", "ETFs"]
)

st.sidebar.subheader("Date Range")
default_start = date.today() - timedelta(days=365)
default_end = date.today()

start_date = st.sidebar.date_input("Start", default_start)
end_date = st.sidebar.date_input("End", default_end)

# ---------------------------------------------------------
# FUTURES PAGE
# ---------------------------------------------------------

if section == "Futures":
    st.title("🛢️ Futures Prices")

    tickers = sum(futures_groups.values(), [])
    df = load_prices(tickers, start_date, end_date)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy Futures Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance")
        fig2 = px.line(normalize(df), title="Futures Performance")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("Performance Table")
        table = grouped_performance_table(futures_groups)
        st.dataframe(table, use_container_width=True)

# ---------------------------------------------------------
# EQUITIES PAGE
# ---------------------------------------------------------

elif section == "Equities":
    st.title("📈 Energy Equities")

    tickers = sum(equity_groups.values(), [])
    df = load_prices(tickers, start_date, end_date)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy Equities Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance")
        fig2 = px.line(normalize(df), title="Equities Performance")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("Performance Table")
        table = grouped_performance_table(equity_groups)
        st.dataframe(table, use_container_width=True)

# ---------------------------------------------------------
# ETFs PAGE
# ---------------------------------------------------------

elif section == "ETFs":
    st.title("📊 Energy ETFs")

    tickers = sum(etf_groups.values(), [])
    df = load_prices(tickers, start_date, end_date)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy ETFs Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance")
        fig2 = px.line(normalize(df), title="ETF Performance")
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.subheader("Performance Table")
        table = grouped_performance_table(etf_groups)
        st.dataframe(table, use_container_width=True)
