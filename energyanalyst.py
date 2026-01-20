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
# GROUPED UNIVERSE (PRICES)
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
# MACRO UNIVERSE
# ---------------------------------------------------------
macro_yields = {
    "US 2Y": "^UST2Y",
    "US 10Y": "^TNX",      # note: ^TNX is 10x yield
    "US 30Y": "^TYX",      # 10x yield
    "UK 10Y": "^GUKG10",
    "Germany 10Y": "^DE10Y",
    "France 10Y": "^FR10Y",
    "Japan 10Y": "^JP10Y"
}

macro_fx = {
    "DXY": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "CNY=X",
    "GBP/USD": "GBPUSD=X"
}

macro_policy = {
    "Fed Funds Futures (proxy)": "ZQ=F"
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

def load_macro_series(series_dict, start, end):
    data = {}
    for name, ticker in series_dict.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)["Close"]
            if df.empty:
                continue
            # Adjust for ^TNX/^TYX being 10x yield
            if ticker in ["^TNX", "^TYX"]:
                df = df / 10.0
            data[name] = df
        except Exception:
            continue
    return pd.DataFrame(data).dropna(how="all")

def macro_context_table(df):
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        current = series.iloc[-1]
        min_1y = series.min()
        max_1y = series.max()
        avg_1y = series.mean()
        rows.append([
            col,
            round(current, 3),
            round(min_1y, 3),
            round(max_1y, 3),
            round(avg_1y, 3)
        ])
    return pd.DataFrame(rows, columns=["Series", "Current", "1Y Min", "1Y Max", "1Y Avg"])

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Energy Dashboard MVP")
section = st.sidebar.radio("Select Section", ["Prices", "Macro Drivers"])

st.sidebar.subheader("Date Range")
default_start = date.today() - timedelta(days=365)
default_end = date.today()

start_date = st.sidebar.date_input("Start", default_start)
end_date = st.sidebar.date_input("End", default_end)

st.sidebar.subheader("Normalization Date (Prices)")
norm_input = st.sidebar.date_input("Normalize from", date(2022, 3, 1))

# ---------------------------------------------------------
# PRICES PAGE
# ---------------------------------------------------------
if section == "Prices":
    st.title("💰 Prices")

    tabs = st.tabs(["Futures", "Equities", "ETFs"])

    # ---------- FUTURES ----------
    with tabs[0]:
        st.subheader("Futures — Price Levels")
        futures_tickers = sum(futures_groups.values(), [])
        df_fut = load_prices(futures_tickers, start_date, end_date)

        fig_fut = px.line(
            df_fut,
            title="Futures — Price Levels",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_fut, use_container_width=True)

        st.subheader("Futures — Performance Table")
        fut_table = grouped_performance_table(futures_groups)
        st.dataframe(fut_table, use_container_width=True)

        st.subheader(f"Futures — Normalized Performance (100 = {norm_input})")
        df_fut_norm, fut_norm_date = normalize(df_fut, pd.to_datetime(norm_input))
        fig_fut_norm = px.line(
            df_fut_norm,
            title=f"Futures — Normalized Since {fut_norm_date.date()}",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_fut_norm, use_container_width=True)

        st.subheader("Futures — Correlation Heatmap")
        corr_fut = df_fut.pct_change().corr()
        fig_cf, axf = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_fut, annot=True, cmap="coolwarm", linewidths=0.5, ax=axf)
        st.pyplot(fig_cf)

    # ---------- EQUITIES ----------
    with tabs[1]:
        st.subheader("Equities — Price Levels")
        eq_tickers = sum(equity_groups.values(), [])
        df_eq = load_prices(eq_tickers, start_date, end_date)

        fig_eq = px.line(
            df_eq,
            title="Equities — Price Levels",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Equities — Performance Table")
        eq_table = grouped_performance_table(equity_groups)
        st.dataframe(eq_table, use_container_width=True)

        st.subheader(f"Equities — Normalized Performance (100 = {norm_input})")
        df_eq_norm, eq_norm_date = normalize(df_eq, pd.to_datetime(norm_input))
        fig_eq_norm = px.line(
            df_eq_norm,
            title=f"Equities — Normalized Since {eq_norm_date.date()}",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_eq_norm, use_container_width=True)

        st.subheader("Equities — Correlation Heatmap")
        corr_eq = df_eq.pct_change().corr()
        fig_ce, axe = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_eq, annot=True, cmap="coolwarm", linewidths=0.5, ax=axe)
        st.pyplot(fig_ce)

    # ---------- ETFS ----------
    with tabs[2]:
        st.subheader("ETFs — Price Levels")
        etf_tickers = sum(etf_groups.values(), [])
        df_etf = load_prices(etf_tickers, start_date, end_date)

        fig_etf = px.line(
            df_etf,
            title="ETFs — Price Levels",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_etf, use_container_width=True)

        st.subheader("ETFs — Performance Table")
        etf_table = grouped_performance_table(etf_groups)
        st.dataframe(etf_table, use_container_width=True)

        st.subheader(f"ETFs — Normalized Performance (100 = {norm_input})")
        df_etf_norm, etf_norm_date = normalize(df_etf, pd.to_datetime(norm_input))
        fig_etf_norm = px.line(
            df_etf_norm,
            title=f"ETFs — Normalized Since {etf_norm_date.date()}",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_etf_norm, use_container_width=True)

        st.subheader("ETFs — Correlation Heatmap")
        corr_etf = df_etf.pct_change().corr()
        fig_ct, axt = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_etf, annot=True, cmap="coolwarm", linewidths=0.5, ax=axt)
        st.pyplot(fig_ct)

# ---------------------------------------------------------
# MACRO DRIVERS PAGE
# ---------------------------------------------------------
elif section == "Macro Drivers":
    st.title("🌍 Macro Drivers")

    # ---------- LOAD MACRO DATA ----------
    df_yields = load_macro_series(macro_yields, start_date, end_date)
    df_fx = load_macro_series(macro_fx, start_date, end_date)
    df_policy = load_macro_series(macro_policy, start_date, end_date)

    # ---------- RATES & YIELDS ----------
    st.subheader("Rates & Yields (US, UK, Germany, France, Japan)")
    if not df_yields.empty:
        fig_yields = px.line(
            df_yields,
            title="Sovereign Yields",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_yields, use_container_width=True)

        st.subheader("Yields — Historical Context (1Y)")
        yields_context = macro_context_table(df_yields)
        st.dataframe(yields_context, use_container_width=True)
    else:
        st.warning("No yield data available for the selected period.")

    # ---------- YIELD CURVE SPREADS (US) ----------
    if all(col in df_yields.columns for col in ["US 2Y", "US 10Y", "US 30Y"]):
        st.subheader("US Yield Curve Spreads")
        spreads = pd.DataFrame({
            "10Y–2Y": df_yields["US 10Y"] - df_yields["US 2Y"],
            "30Y–10Y": df_yields["US 30Y"] - df_yields["US 10Y"]
        }).dropna()

        fig_spreads = px.line(
            spreads,
            title="US Yield Curve Spreads",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_spreads, use_container_width=True)

    # ---------- FX ----------
    st.subheader("FX — USD Majors")
    if not df_fx.empty:
        fig_fx = px.line(
            df_fx,
            title="FX Rates",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_fx, use_container_width=True)

        st.subheader("FX — Historical Context (1Y)")
        fx_context = macro_context_table(df_fx)
        st.dataframe(fx_context, use_container_width=True)
    else:
        st.warning("No FX data available for the selected period.")

    # ---------- POLICY (PROXY) ----------
    if not df_policy.empty:
        st.subheader("Policy Rate Proxy (Fed Funds Futures)")
        fig_pol = px.line(
            df_policy,
            title="Fed Funds Futures (Proxy)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pol, use_container_width=True)

    # ---------- MACRO CORRELATION HEATMAP ----------
    st.subheader("Macro Correlation Heatmap")
    macro_combined = pd.concat(
        [df_yields.pct_change(), df_fx.pct_change(), df_policy.pct_change()],
        axis=1
    ).dropna(how="all")

    if not macro_combined.empty:
        corr_macro = macro_combined.corr()
        fig_mc, axm = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_macro, annot=True, cmap="coolwarm", linewidths=0.5, ax=axm)
        st.pyplot(fig_mc)
    else:
        st.warning("Not enough macro data to compute correlations.")

    macro_fx = {
        "DXY": "DX-Y.NYB",
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CNY": "CNY=X"
    }

    # ------------------ FETCH FRED DATA ------------------
    fred_data = {}
    for name, series in macro_fred.items():
        try:
            df = yf.download(series, start=start_date, end=end_date, progress=False)["Close"]
            fred_data[name] = df
        except:
            pass

    # ------------------ FETCH FX DATA ------------------
    fx_data = {}
    for name, ticker in macro_fx.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)["Close"]
            fx_data[name] = df
        except:
            pass

    # ------------------ COMBINE INTO DATAFRAMES ------------------
    df_rates = pd.DataFrame(fred_data).dropna(how="all")
    df_fx = pd.DataFrame(fx_data).dropna(how="all")

    # ------------------ RATES & YIELDS ------------------
    st.subheader("Interest Rates & Yields")
    fig_rates = px.line(df_rates, title="US Treasury Yields & Fed Funds", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_rates, use_container_width=True)

    # ------------------ YIELD CURVE SPREADS ------------------
    st.subheader("Yield Curve Spreads")
    spreads = pd.DataFrame({
        "10Y–2Y": df_rates["US 10Y Yield"] - df_rates["US 2Y Yield"],
        "30Y–10Y": df_rates["US 30Y Yield"] - df_rates["US 10Y Yield"]
    }).dropna()

    fig_spreads = px.line(spreads, title="Yield Curve Spreads", color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_spreads, use_container_width=True)

    # ------------------ FX ------------------
    st.subheader("Foreign Exchange (USD Majors)")
    fig_fx = px.line(df_fx, title="FX Rates", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_fx, use_container_width=True)

    # ------------------ CORRELATION HEATMAP ------------------
    st.subheader("Macro Correlation Heatmap")
    corr = pd.concat([df_rates.pct_change(), df_fx.pct_change()], axis=1).corr()

    fig_corr, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig_corr)

