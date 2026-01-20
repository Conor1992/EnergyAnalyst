import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, timedelta
from fredapi import Fred
from prophet import Prophet

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
# MACRO UNIVERSE (FRED CODES — CORE SET)
# ---------------------------------------------------------
macro_us = {
    "US 2Y": "DGS2",
    "US 10Y": "DGS10",
    "US 30Y": "DGS30",
    "Fed Funds": "FEDFUNDS",
    "CPI (Index)": "CPIAUCSL"
}

macro_global_yields = {
    "Germany 10Y": "IRLTLT01DEM156N",
    "France 10Y": "IRLTLT01FRM156N",
    "UK 10Y": "IRLTLT01GBM156N",
    "Japan 10Y": "IRLTLT01JPM156N"
}

macro_fx = {
    "EUR/USD": "DEXUSEU",
    "GBP/USD": "DEXUSUK",
    "USD/JPY": "DEXJPUS",
    "USD/CNY": "DEXCHUS"
}

# ---------------------------------------------------------
# HELPERS — PRICES
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
        if name not in df.columns:
            continue
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
        if table.empty:
            continue
        table.insert(0, "Group", group_name)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------
# HELPERS — MACRO (FRED)
# ---------------------------------------------------------
def load_fred_series(fred, series_dict, start, end):
    data = {}
    for name, code in series_dict.items():
        try:
            s = fred.get_series(code)
            s = s[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]
            if not s.empty:
                data[name] = s
        except Exception:
            continue
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

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
# HELPERS — TECHNICALS / PROPHET
# ---------------------------------------------------------
def compute_technical_indicators(df):
    out = df.copy()

    out["SMA_20"] = out["Close"].rolling(window=20).mean()
    out["SMA_50"] = out["Close"].rolling(window=50).mean()

    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    out["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    return out

def run_prophet_forecast(df, periods=30):
    df_prophet = df.reset_index()[["Date", "Close"]]
    df_prophet.columns = ["ds", "y"]

    model = Prophet(daily_seasonality=False, weekly_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Energy Dashboard MVP")
section = st.sidebar.radio(
    "Select Section",
    ["Prices", "Macro Drivers", "Supply & Demand (EIA)"]
)

st.sidebar.subheader("Date Range")
default_start = date.today() - timedelta(days=365)
default_end = date.today()
start_date = st.sidebar.date_input("Start", default_start)
end_date = st.sidebar.date_input("End", default_end)

st.sidebar.subheader("Normalization Date (Prices)")
norm_input = st.sidebar.date_input("Normalize from", date(2022, 3, 1))

# ---------------------------------------------------------
# PRICES PAGE (YAHOO + TECHNICALS)
# ---------------------------------------------------------
if section == "Prices":
    st.title("💰 Prices")

    tabs = st.tabs(["Futures", "Equities", "ETFs", "Historical Price & Technicals"])

    # ---------- FUTURES ----------
    with tabs[0]:
        st.subheader("Futures — Price Levels")
        futures_tickers = sum(futures_groups.values(), [])
        df_fut = load_prices(futures_tickers, start_date, end_date)

        if df_fut.empty:
            st.warning("No futures data available for the selected period.")
        else:
            fig_fut = px.line(
                df_fut,
                title="Futures — Price Levels",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_fut, use_container_width=True)

            st.subheader("Futures — Performance Table")
            fut_table = grouped_performance_table(futures_groups)
            if fut_table.empty:
                st.warning("No futures performance data available.")
            else:
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

        if df_eq.empty:
            st.warning("No equity data available for the selected period.")
        else:
            fig_eq = px.line(
                df_eq,
                title="Equities — Price Levels",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Equities — Performance Table")
            eq_table = grouped_performance_table(equity_groups)
            if eq_table.empty:
                st.warning("No equity performance data available.")
            else:
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

        if df_etf.empty:
            st.warning("No ETF data available for the selected period.")
        else:
            fig_etf = px.line(
                df_etf,
                title="ETFs — Price Levels",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_etf, use_container_width=True)

            st.subheader("ETFs — Performance Table")
            etf_table = grouped_performance_table(etf_groups)
            if etf_table.empty:
                st.warning("No ETF performance data available.")
            else:
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

    # ---------- HISTORICAL PRICE & TECHNICALS ----------
    with tabs[3]:
        st.subheader("Historical Price & Technical Analysis")

        ticker = st.selectbox(
            "Select a ticker for technical analysis",
            options=list(label_map.keys()),
            format_func=lambda x: label_map[x]
        )

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            st.warning("No data available for this ticker.")
        else:
            df = df.reset_index()
            df_ta = compute_technical_indicators(df)

            st.markdown("### Price with Moving Averages")
            fig_ma = px.line(
                df_ta,
                x="Date",
                y=["Close", "SMA_20", "SMA_50", "EMA_20", "EMA_50"],
                title=f"{label_map[ticker]} — Price & Moving Averages"
            )
            st.plotly_chart(fig_ma, use_container_width=True)

            st.markdown("### Relative Strength Index (RSI)")
            fig_rsi = px.line(
                df_ta,
                x="Date",
                y="RSI_14",
                title=f"{label_map[ticker]} — RSI (14)"
            )
            fig_rsi.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2)
            st.plotly_chart(fig_rsi, use_container_width=True)

            st.markdown("### MACD")
            fig_macd = px.line(
                df_ta,
                x="Date",
                y=["MACD", "MACD_signal"],
                title=f"{label_map[ticker]} — MACD (12–26–9)"
            )
            st.plotly_chart(fig_macd, use_container_width=True)

            st.markdown("### Forecast (Prophet)")
            forecast = run_prophet_forecast(df)
            fig_fc = px.line(
                forecast,
                x="ds",
                y=["yhat", "yhat_lower", "yhat_upper"],
                title=f"{label_map[ticker]} — 30‑Day Forecast (Prophet)"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown("### Term Structure (Placeholder)")
            if ticker in ["CL=F", "BZ=F", "NG=F", "RB=F", "HO=F"]:
                st.info("Term structure for individual futures curves will be added here (per‑contract strip) in a later iteration.")
            else:
                st.info("Term structure only applies to futures contracts.")

# ---------------------------------------------------------
# MACRO DRIVERS PAGE (FRED)
# ---------------------------------------------------------
elif section == "Macro Drivers":
    st.title("🌍 Macro Drivers")

    st.subheader("FRED API Key")
    fred_key = st.text_input("Enter your FRED API Key", type="password")

    if not fred_key:
        st.warning("Enter your FRED API key to load macro data.")
        st.stop()

    try:
        fred = Fred(api_key=fred_key)
    except Exception:
        st.error("Invalid FRED API key. Please check and try again.")
        st.stop()

    macro_tabs = st.tabs(["Core Macro (Option A)", "Extended Macro (Add‑On)"])

    # ---------- CORE MACRO TAB ----------
    with macro_tabs[0]:
        st.subheader("Core Macro — Yields, FX, Policy, Inflation")

        df_us = load_fred_series(fred, macro_us, start_date, end_date)
        df_global = load_fred_series(fred, macro_global_yields, start_date, end_date)
        df_fx = load_fred_series(fred, macro_fx, start_date, end_date)

        st.markdown("### US Rates & Yields")
        if not df_us.empty:
            fig_us = px.line(
                df_us[["US 2Y", "US 10Y", "US 30Y", "Fed Funds"]],
                title="US Yields & Fed Funds",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_us, use_container_width=True)

            st.markdown("#### US Rates — Historical Context (1Y)")
            us_context = macro_context_table(df_us[["US 2Y", "US 10Y", "US 30Y", "Fed Funds"]])
            st.dataframe(us_context, use_container_width=True)
        else:
            st.warning("No US macro data available for the selected period.")

        st.markdown("### Global 10Y Yields (Germany, France, UK, Japan)")
        if not df_global.empty:
            fig_global = px.line(
                df_global,
                title="Global 10Y Yields",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_global, use_container_width=True)

            st.markdown("#### Global Yields — Historical Context (1Y)")
            global_context = macro_context_table(df_global)
            st.dataframe(global_context, use_container_width=True)
        else:
            st.warning("No global yield data available for the selected period.")

        st.markdown("### FX — USD Majors (FRED DEX Series)")
        if not df_fx.empty:
            fig_fx = px.line(
                df_fx,
                title="FX Rates (FRED DEX)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_fx, use_container_width=True)

            st.markdown("#### FX — Historical Context (1Y)")
            fx_context = macro_context_table(df_fx)
            st.dataframe(fx_context, use_container_width=True)
        else:
            st.warning("No FX data available for the selected period.")

        if all(col in df_us.columns for col in ["US 2Y", "US 10Y", "US 30Y"]):
            st.markdown("### US Yield Curve Spreads")
            spreads = pd.DataFrame({
                "10Y–2Y": df_us["US 10Y"] - df_us["US 2Y"],
                "30Y–10Y": df_us["US 30Y"] - df_us["US 10Y"]
            }).dropna()

            fig_spreads = px.line(
                spreads,
                title="US Yield Curve Spreads",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_spreads, use_container_width=True)

        st.markdown("### Macro Correlation Heatmap")
        macro_combined = pd.concat(
            [
                df_us[["US 2Y", "US 10Y", "US 30Y", "Fed Funds"]].pct_change(),
                df_global.pct_change(),
                df_fx.pct_change()
            ],
            axis=1
        ).dropna(how="all")

        if not macro_combined.empty:
            corr_macro = macro_combined.corr()
            fig_mc, axm = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_macro, annot=True, cmap="coolwarm", linewidths=0.5, ax=axm)
            st.pyplot(fig_mc)
        else:
            st.warning("Not enough macro data to compute correlations.")

    # ---------- EXTENDED MACRO TAB (ADD‑ON)
    with macro_tabs[1]:
        st.subheader("Extended Macro Indicators")

        st.markdown(
            "These indicators are optional add‑ons powered by FRED. "
            "They expand the macro view into credit, growth, labour markets, PMIs, and inflation expectations."
        )

        extended_macro = {
            "HY OAS": "BAMLH0A0HYM2EY",
            "IG OAS": "BAMLC0A0CM",
            "Real GDP": "GDPC1",
            "Industrial Production": "INDPRO",
            "Retail Sales": "RSAFS",
            "Unemployment Rate": "UNRATE",
            "Job Openings (JOLTS)": "JTSJOL",
            "ISM Manufacturing PMI": "NAPM",
            "ISM Services PMI": "NAPMS",
            "5Y5Y Inflation Expectations": "T5YIFR",
            "10Y Breakeven Inflation": "T10YIE"
        }

        df_ext = load_fred_series(fred, extended_macro, start_date, end_date)

        if df_ext.empty:
            st.warning("No extended macro data available for the selected period.")
        else:
            st.markdown("### Extended Macro — Time Series")
            fig_ext = px.line(
                df_ext,
                title="Extended Macro Indicators",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_ext, use_container_width=True)

            st.markdown("### Extended Macro — Historical Context (1Y)")
            ext_context = macro_context_table(df_ext)
            st.dataframe(ext_context, use_container_width=True)

            st.markdown("### Extended Macro — Correlation Heatmap")
            corr_ext = df_ext.pct_change().corr()
            fig_ex, ax_ex = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_ext, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_ex)
            st.pyplot(fig_ex)

# ---------------------------------------------------------
# SUPPLY & DEMAND PAGE (PLACEHOLDER)
# ---------------------------------------------------------
elif section == "Supply & Demand (EIA)":
    st.title("📦 US Supply & Demand — EIA")

    st.subheader("EIA API Key (Placeholder)")
    st.text_input("Enter your EIA API Key (not yet used)", type="password")

    st.markdown(
        """
        ### Supply & Demand Module — Coming Soon

        This section will host a US crude & products supply/demand model built on the EIA v2 API.

        Planned features:

        - Weekly crude production, net imports, refinery runs  
        - Total products supplied (demand) and total stocks  
        - Simple supply vs demand balance  
        - Stocks vs implied stock change  
        - Correlation heatmaps across S&D components  

        For now, this page is a placeholder while the EIA v2 integration is finalized.
        """
    )

