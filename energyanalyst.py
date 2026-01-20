import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Energy Dashboard MVP", layout="wide")

# --- 1. Define your universe ---
futures = [
    "CL=F", "BZ=F", "RB=F", "HO=F", "NG=F", "EH=F"
]

equities = [
    "XOM", "CVX", "BP", "SHEL", "TTE",
    "COP", "EOG", "FANG", "DVN",
    "SLB", "HAL", "BKR",
    "VLO", "MPC", "PSX",
    "ENB", "KMI", "WMB"
]

etfs = [
    "XOP", "IEO",
    "XLE", "VDE",
    "OIH",
    "AMLP"
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def load_prices(tickers, start, end):
    """Download Yahoo Finance data safely."""
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def normalize(df):
    """Normalize prices to 100 = start."""
    return df / df.iloc[0] * 100

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Energy Dashboard MVP")
section = st.sidebar.radio(
    "Select Section",
    ["Futures", "Equities", "ETFs"]
)

# Date range selector
st.sidebar.subheader("Date Range")
default_start = date.today() - timedelta(days=365)
default_end = date.today()

start_date = st.sidebar.date_input("Start", default_start)
end_date = st.sidebar.date_input("End", default_end)

# ---------------------------------------------------------
# FUTURES
# ---------------------------------------------------------
if section == "Futures":
    st.title("🛢️ Futures Prices")

    df = load_prices(futures, start_date, end_date)

    if df.empty:
        st.error("Yahoo Finance returned no futures data.")
    else:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy Futures Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance (100 = start date)")
        fig2 = px.line(normalize(df), title="Futures Performance")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# EQUITIES
# ---------------------------------------------------------
elif section == "Equities":
    st.title("📈 Energy Equities")

    df = load_prices(equities, start_date, end_date)

    if df.empty:
        st.error("Yahoo Finance returned no equity data.")
    else:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy Equities Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance (100 = start date)")
        fig2 = px.line(normalize(df), title="Equities Performance")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# ETFs
# ---------------------------------------------------------
elif section == "ETFs":
    st.title("📊 Energy ETFs")

    df = load_prices(etfs, start_date, end_date)

    if df.empty:
        st.error("Yahoo Finance returned no ETF data.")
    else:
        st.subheader("Price Levels")
        fig = px.line(df, title="Energy ETFs Prices")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Normalized Performance (100 = start date)")
        fig2 = px.line(normalize(df), title="ETF Performance")
        st.plotly_chart(fig2, use_container_width=True)
