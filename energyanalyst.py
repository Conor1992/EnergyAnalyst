import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Energy Dashboard MVP", layout="wide")

# ---------------------------------------------------------
# HUMAN‑READABLE LABELS
# ---------------------------------------------------------
label_map = {
    # Futures
    "CL=F": "WTI Crude",
    "BZ=F": "Brent Crude",
    "RB=F": "RBOB Gasoline",
    "HO=F": "Heating Oil",
    "NG=F": "Henry Hub NatGas",

    # Equities
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "BP": "BP",
    "SHEL": "Shell",
    "TTE": "TotalEnergies",
    "EOG": "EOG Resources",
    "DVN": "Devon Energy",
    "SLB": "Schlumberger",
    "HAL": "Halliburton",
    "BKR": "Baker Hughes",
    "VLO": "Valero",
    "MPC": "Marathon Petroleum",
    "PSX": "Phillips 66",
    "ENB": "Enbridge",
    "KMI": "Kinder Morgan",
    "WMB": "Williams",

    # ETFs
    "XOP": "S&P Oil & Gas E&P ETF",
    "IEO": "iShares Oil & Gas E&P",
    "XLE": "Energy Select Sector SPDR",
    "VDE": "Vanguard Energy ETF",
    "OIH": "Oil Services ETF",
    "AMLP": "Alerian MLP ETF"
}

# ---------------------------------------------------------
# GROUPED UNIVERSE (cleaned)
# ---------------------------------------------------------
futures_groups = {
    "Crude Oil": ["CL=F", "BZ=F"],
    "Refined Products": ["RB=F", "HO=F"],
    "Natural Gas": ["NG=F"]
}

equity_groups = {
    "Integrated Majors": ["XOM", "CVX", "BP", "SHEL", "TTE"],
    "Shale Producers": ["EOG", "DVN"],
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
    df = df.rename(columns=label_map)
    return df.dropna(how="all")

def normalize(df, norm_date):
    if norm_date not in df.index:
        norm_date = df.index[0]
    return df / df.loc[norm_date] * 100, norm_date

def performance_table(tickers):
    df = yf.download(tickers, period="1y", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.rename(columns=label_map)

    results = []
    for ticker in tickers:
        name = label_map[ticker]
        series = df[name].dropna()
        if series.empty:
            continue

        last_price = series.iloc[-1]

        # Compute returns
        today = series.index[-1]
        lookbacks = {
            "1D %": today - timedelta(days=1),
            "1M %": today - timedelta(days=30),
            "3M %": today - timedelta(days=90),
            "12M %": today - timedelta(days=365),
        }

        row = [name, round(last_price, 2)]

        for _, start_date in lookbacks.items():
            past = series[series.index >= start_date]
            if past.empty:
                row.append(None)
            else:
                pct = (last_price - past.iloc[0]) / past.iloc[0] * 100
                row.append(round(pct, 2))

        results.append(row)

    return pd.DataFrame(results, columns=["Name", "Price", "1D %", "1M %", "3M %", "12M %"])

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
section = st.sidebar.radio("Select Section", ["Futures", "Equities", "ETFs"])

st.sidebar.subheader("Date Range")
default_start = date.today() - timedelta(days=365)
default_end = date.today()

start_date = st.sidebar.date_input("Start", default_start)
end_date = st.sidebar.date_input("End", default_end)

st.sidebar.subheader("Normalization Date")
norm_input = st.sidebar.date_input("Normalize from", date(2022, 3, 1))

# ---------------------------------------------------------
# PAGE TEMPLATE
# ---------------------------------------------------------
def render_page(title, groups):
    st.title(title)

    tickers = sum(groups.values(), [])
    df = load_prices(tickers, start_date, end_date)

    # ------------------ PRICE LEVELS ------------------
    st.subheader("Price Levels")
    fig = px.line(df, title=f"{title} — Price Levels", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ PERFORMANCE TABLE UNDER PRICE ------------------
    st.subheader("Performance Table")
    table = grouped_performance_table(groups)
    st.dataframe(table, use_container_width=True)

    # ------------------ NORMALIZED PERFORMANCE ------------------
    st.subheader(f"Normalized Performance (100 = {norm_input})")
    df_norm, actual_norm_date = normalize(df, pd.to_datetime(norm_input))
    fig2 = px.line(df_norm, title=f"{title} — Normalized Since {actual_norm_date.date()}", color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ CORRELATION HEATMAP ------------------
    st.subheader("Correlation Heatmap")
    corr = df.pct_change().corr()

    fig3, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig3)

# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------
if section == "Futures":
    render_page("🛢️ Futures", futures_groups)

elif section == "Equities":
    render_page("📈 Energy Equities", equity_groups)

elif section == "ETFs":
    render_page("📊 Energy ETFs", etf_groups)
