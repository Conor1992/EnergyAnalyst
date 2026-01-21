import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests

st.set_page_config(page_title="Energy & Macro Dashboard", layout="wide")

# ---------------------------------------------------------
# 1. ENERGY TICKERS & NAMES
# ---------------------------------------------------------

futures = ["CL=F", "BZ=F", "RB=F", "HO=F", "NG=F", "EH=F"]

equities = [
    "XOM", "CVX", "BP", "SHEL", "TTE",
    "PXD", "EOG", "CLR", "DVN",
    "SLB", "HAL", "BKR",
    "VLO", "MPC", "PSX",
    "ENB", "KMI", "WMB"
]

etfs = ["XOP", "IEO", "XLE", "VDE", "OIH", "AMLP"]

indexes = ["^GSPE", "^DJUSEN", "^FTNMX0530"]

all_tickers = futures + equities + etfs + indexes

ticker_names = {
    "CL=F": "WTI Crude Oil",
    "BZ=F": "Brent Crude Oil",
    "RB=F": "RBOB Gasoline",
    "HO=F": "Heating Oil",
    "NG=F": "Henry Hub Natural Gas",
    "EH=F": "Ethanol",

    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "BP": "BP plc",
    "SHEL": "Shell plc",
    "TTE": "TotalEnergies",

    "PXD": "Pioneer Natural Resources",
    "EOG": "EOG Resources",
    "CLR": "Continental Resources",
    "DVN": "Devon Energy",

    "SLB": "Schlumberger",
    "HAL": "Halliburton",
    "BKR": "Baker Hughes",

    "VLO": "Valero Energy",
    "MPC": "Marathon Petroleum",
    "PSX": "Phillips 66",

    "ENB": "Enbridge",
    "KMI": "Kinder Morgan",
    "WMB": "Williams Companies",

    "XOP": "S&P Oil & Gas E&P ETF",
    "IEO": "iShares U.S. Oil & Gas E&P ETF",
    "XLE": "Energy Select Sector SPDR",
    "VDE": "Vanguard Energy ETF",
    "OIH": "VanEck Oil Services ETF",
    "AMLP": "Alerian MLP ETF",
}

# ---------------------------------------------------------
# 2. ENERGY PRICE DATA
# ---------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_energy_data():
    return yf.download(
        all_tickers,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        start="2000-01-01"
    )

data = load_energy_data()

min_date = data.index.min()
max_date = data.index.max()

min_dt = pd.to_datetime(min_date).to_pydatetime()
max_dt = pd.to_datetime(max_date).to_pydatetime()

# ---------------------------------------------------------
# 3. MACRO EXPLANATIONS
# ---------------------------------------------------------

explanations = {
    "US 10Y Yield (DGS10)": [
        "Benchmark long-term interest rate.",
        "Higher yields strengthen USD and weaken oil demand expectations.",
        "Usually bearish for crude when rising."
    ],
    "Fed Funds Rate (EFFR)": [
        "Short-term interest rate set by the Fed.",
        "Higher rates tighten liquidity and slow demand.",
        "Rate cuts are typically bullish for crude."
    ],
    "USD Index (DTWEXBGS)": [
        "Broad measure of USD strength.",
        "Oil priced in USD → stronger USD makes oil more expensive globally.",
        "Strong inverse correlation with crude."
    ],
    "CNY per USD (DEXCHUS)": [
        "China’s currency vs USD.",
        "China is the largest marginal buyer of crude.",
        "Weak CNY reduces Chinese import demand."
    ],
    "INR per USD (DEXINUS)": [
        "India’s currency vs USD.",
        "India is a major importer of crude.",
        "Weak INR reduces affordability of oil."
    ],
    "S&P 500 (SP500)": [
        "Broad US equity index.",
        "Proxy for global growth sentiment.",
        "Strong equities → bullish crude."
    ],
    "Nasdaq Composite (NASDAQCOM)": [
        "Tech-heavy US equity index.",
        "Risk-on indicator.",
        "Weak Nasdaq often signals macro slowdown → bearish crude."
    ],
    "VIX Index (VIXCLS)": [
        "Market volatility index.",
        "High VIX = risk-off → crude sells off.",
        "Low VIX = stable carry → bullish."
    ],
    "High Yield OAS (BAMLH0A0HYM2)": [
        "Credit stress indicator.",
        "Wider spreads = recession risk.",
        "Bearish for crude and energy equities."
    ],
    "Bloomberg Commodity Index (^BCOM)": [
        "Broad commodity index.",
        "Tracks cross-commodity flows.",
        "Strong BCOM often coincides with bullish crude."
    ],
    "S&P GSCI (^SPGSCI)": [
        "Energy-heavy commodity index.",
        "Tracks systematic commodity allocation.",
        "Strong GSCI = strong energy complex."
    ]
}

explanation_df = pd.DataFrame.from_dict(
    explanations, orient="index",
    columns=["What It Measures", "Why It Matters", "Typical Impact"]
)

# ---------------------------------------------------------
# 4. FRED MACRO DATA HELPERS
# ---------------------------------------------------------

macro_series = {
    "US 10Y Yield (DGS10)": "DGS10",
    "Fed Funds Rate (EFFR)": "EFFR",
    "USD Index (DTWEXBGS)": "DTWEXBGS",
    "CNY per USD (DEXCHUS)": "DEXCHUS",
    "INR per USD (DEXINUS)": "DEXINUS",
    "S&P 500 (SP500)": "SP500",
    "Nasdaq Composite (NASDAQCOM)": "NASDAQCOM",
    "VIX Index (VIXCLS)": "VIXCLS",
    "High Yield OAS (BAMLH0A0HYM2)": "BAMLH0A0HYM2",
    "Bloomberg Commodity Index (^BCOM)": "BCOM",
    "S&P GSCI (^SPGSCI)": "SPGSCI"
}

def get_fred_series(series_id, api_key, start="1990-01-01"):
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}"
    )
    r = requests.get(url)
    data = r.json()
    if "observations" not in data:
        return pd.Series(dtype=float)
    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")["value"]
    return df

def build_macro_df(api_key):
    macro_df = pd.DataFrame()
    for name, code in macro_series.items():
        s = get_fred_series(code, api_key)
        if not s.empty:
            macro_df[name] = s
    return macro_df

# ---------------------------------------------------------
# 5. ENERGY HELPERS (FILTER, NORMALIZE, PLOTS)
# ---------------------------------------------------------

def filter_by_date(series, user_start_dt, date_range):
    s = series.loc[user_start_dt:]
    return s.loc[date_range[0]:date_range[1]]

def extract_close_series(ticker, user_start_dt, date_range):
    try:
        s = data[ticker]["Close"].dropna()
        return filter_by_date(s, user_start_dt, date_range)
    except Exception:
        return pd.Series(dtype=float)

def normalize(series, method="first", base_value=100):
    if series.empty:
        return series
    if method == "first":
        return series / series.iloc[0]
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    elif method == "zscore":
        return (series - series.mean()) / series.std()
    elif method == "custom_base":
        return (series / series.iloc[0]) * base_value

def normalize_to_date(series, base_date, base_value=100):
    if series.empty:
        return series
    if base_date not in series.index:
        base_date = series.index[0]
    return (series / series.loc[base_date]) * base_value

def plot_price_group(tickers, title, user_start_dt, date_range):
    df = pd.DataFrame({
        ticker_names.get(t, t): extract_close_series(t, user_start_dt, date_range)
        for t in tickers
    })
    df = df.dropna(how="all")
    fig = px.line(df, title=title)
    fig.update_layout(hovermode="x unified")
    return fig

def plot_normalized_group(tickers, title, method, user_start_dt, date_range, base_value=100):
    df_dict = {}
    for t in tickers:
        s = extract_close_series(t, user_start_dt, date_range)
        if not s.empty:
            df_dict[ticker_names.get(t, t)] = normalize(s, method=method, base_value=base_value)
    df = pd.DataFrame(df_dict).dropna(how="all")
    fig = px.line(df, title=f"{title} (Normalized: {method})")
    fig.update_layout(hovermode="x unified")
    return fig

def plot_indexed_to_100(tickers, title, base_date, user_start_dt, date_range):
    df_dict = {}
    for t in tickers:
        s = extract_close_series(t, user_start_dt, date_range)
        if not s.empty:
            df_dict[ticker_names.get(t, t)] = normalize_to_date(s, base_date)
    df = pd.DataFrame(df_dict).dropna(how="all")
    fig = px.line(df, title=f"{title} (Indexed to 100 at {base_date.date()})")
    fig.update_layout(hovermode="x unified")
    return fig

def plot_category_heatmap(tickers, title, user_start_dt, date_range):
    df = pd.DataFrame({
        ticker_names.get(t, t): extract_close_series(t, user_start_dt, date_range)
        for t in tickers
    })
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return px.imshow([[0]], text_auto=True, title=f"{title} — Correlation Heatmap")
    corr = df.pct_change().corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"{title} — Correlation Heatmap"
    )
    fig.update_layout(height=600)
    return fig

def create_custom_index(tickers, user_start_dt, date_range, base_value=100):
    df = pd.DataFrame({
        t: extract_close_series(t, user_start_dt, date_range)
        for t in tickers
    })
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return pd.Series(dtype=float), None
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float), None
    base_date = df.index.min()
    base_prices = df.loc[base_date]
    index_series = (df / base_prices) * base_value
    index_series["Custom_Index"] = index_series.mean(axis=1)
    return index_series["Custom_Index"], base_date

def plot_custom_index(index_series, base_date):
    fig = px.line(index_series, title=f"Custom ETF Index (Base Date: {base_date.date()})")
    fig.update_layout(hovermode="x unified")
    return fig

def compute_return_table(tickers, user_start_dt, date_range):
    rows = []
    for t in tickers:
        s = extract_close_series(t, user_start_dt, date_range)
        if s.empty:
            continue
        current = s.iloc[-1]
        def safe_ret(series, periods):
            if len(series) > periods:
                return (series.iloc[-1] / series.iloc[-periods] - 1) * 100
            return np.nan
        rows.append({
            "Name": ticker_names.get(t, t),
            "Current Price": round(current, 2),
            "Weekly %": safe_ret(s, 5),
            "Monthly %": safe_ret(s, 21),
            "Yearly %": safe_ret(s, 252)
        })
    if not rows:
        return pd.DataFrame(columns=["Current Price", "Weekly %", "Monthly %", "Yearly %"])
    return pd.DataFrame(rows).set_index("Name")

# ---------------------------------------------------------
# 6. TOP-LEVEL TABS
# ---------------------------------------------------------

tab_overview, tab_macro_data, tab_macro_analysis = st.tabs(
    ["Overview", "Macro Data", "Macro Analysis"]
)

# ---------------------------------------------------------
# 7. OVERVIEW TAB
# ---------------------------------------------------------

with tab_overview:
    st.title("Energy Market Overview")

    user_start_date = st.sidebar.date_input(
        "Select start date",
        value=min_dt.date(),
        min_value=min_dt.date(),
        max_value=max_dt.date()
    )
    user_start_dt = pd.to_datetime(user_start_date).to_pydatetime()

    date_range = st.sidebar.slider(
        "Select date range",
        min_value=user_start_dt,
        max_value=max_dt,
        value=(user_start_dt, max_dt)
    )

    norm_choice = st.sidebar.selectbox(
        "Price view",
        ["Raw Prices", "Normalized (first)", "Normalized (minmax)", "Normalized (zscore)", "Indexed to 100 (custom date)"]
    )

    index_start_date = st.sidebar.slider(
        "Index to 100 starting at:",
        min_value=user_start_dt,
        max_value=max_dt,
        value=user_start_dt
    )

    show_heatmap = st.sidebar.checkbox("Show correlation heatmap", True)
    show_returns = st.sidebar.checkbox("Show return table", True)
    show_custom_index = st.sidebar.checkbox("Show custom ETF index", True)

    sub_fut, sub_etf, sub_eq = st.tabs(["Futures", "ETFs", "Equities"])

    def render_tab_group(tickers, title, show_index=False):
        st.subheader(f"{title} — Price View")

        if norm_choice == "Raw Prices":
            fig = plot_price_group(tickers, f"{title} Prices", user_start_dt, date_range)
        elif norm_choice == "Normalized (first)":
            fig = plot_normalized_group(tickers, title, "first", user_start_dt, date_range)
        elif norm_choice == "Normalized (minmax)":
            fig = plot_normalized_group(tickers, title, "minmax", user_start_dt, date_range)
        elif norm_choice == "Normalized (zscore)":
            fig = plot_normalized_group(tickers, title, "zscore", user_start_dt, date_range)
        elif norm_choice == "Indexed to 100 (custom date)":
            fig = plot_indexed_to_100(tickers, title, index_start_date, user_start_dt, date_range)

        st.plotly_chart(fig, use_container_width=True)

        if show_index and show_custom_index:
            st.subheader("Custom ETF Index")
            idx, base_date = create_custom_index(tickers, user_start_dt, date_range)
            if base_date is None or idx.empty:
                st.info("Not enough overlapping data to compute a custom ETF index for the selected date range.")
            else:
                fig_idx = plot_custom_index(idx, base_date)
                st.plotly_chart(fig_idx, use_container_width=True)

        if show_heatmap:
            st.subheader(f"{title} — Correlation Heatmap")
            fig_hm = plot_category_heatmap(tickers, title, user_start_dt, date_range)
            st.plotly_chart(fig_hm, use_container_width=True)

        if show_returns:
            st.subheader(f"{title} — Return Table")
            df_ret = compute_return_table(tickers, user_start_dt, date_range)
            if df_ret.empty:
                st.info("No data available for return calculations in the selected date range.")
            else:
                st.dataframe(
                    df_ret.style.format("{:.2f}").background_gradient(
                        subset=["Weekly %", "Monthly %", "Yearly %"],
                        cmap="RdYlGn"
                    )
                )

    with sub_fut:
        render_tab_group(futures, "Energy Futures")
    with sub_etf:
        render_tab_group(etfs, "Energy ETFs", show_index=True)
    with sub_eq:
        render_tab_group(equities, "Energy Equities")

# ---------------------------------------------------------
# 8. MACRO DATA TAB (FRED API + LOADER)
# ---------------------------------------------------------

with tab_macro_data:
    st.title("Macro Data Loader (FRED)")

    fred_api_key = st.text_input(
        "Enter your FRED API Key:",
        type="password",
        help="Required to download macroeconomic data from FRED."
    )

    if "macro_df" not in st.session_state:
        st.session_state["macro_df"] = pd.DataFrame()

    if st.button("Load Macro Data") and fred_api_key:
        with st.spinner("Downloading macro series from FRED..."):
            macro_df = build_macro_df(fred_api_key)
            st.session_state["macro_df"] = macro_df

    macro_df = st.session_state["macro_df"]

    if macro_df.empty:
        st.info("No macro data loaded yet. Enter your FRED API key and click 'Load Macro Data'.")
    else:
        st.success("Macro data loaded.")
        st.write("Preview of macro data:")
        st.dataframe(macro_df.tail())

# ---------------------------------------------------------
# 9. MACRO ANALYSIS TAB
# ---------------------------------------------------------

with tab_macro_analysis:
    st.title("Macro–Energy Relationship Analysis")

    macro_df = st.session_state.get("macro_df", pd.DataFrame())
    if macro_df.empty:
        st.warning("No macro data available. Go to the 'Macro Data' tab, enter your FRED API key, and load data.")
    else:
        energy = yf.download(
            ["CL=F", "BZ=F"],
            start="1990-01-01",
            auto_adjust=True,
            progress=False
        )["Close"]
        energy.columns = ["WTI", "Brent"]

        macro_list = list(macro_df.columns)
        selected_macro = st.selectbox("Select a macro variable:", macro_list)

        series = macro_df[selected_macro].dropna()
        aligned = pd.concat([series, energy["WTI"]], axis=1).dropna()
        aligned.columns = [selected_macro, "WTI"]

        st.subheader("Time-Series Relationship")
        fig_ts = px.line(
            aligned,
            labels={"value": "Value", "index": "Date"},
            title=f"{selected_macro} vs WTI Crude"
        )
        fig_ts.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("90‑Day Rolling Correlation")
        rolling_corr = (
            aligned[selected_macro]
            .pct_change()
            .rolling(90)
            .corr(aligned["WTI"].pct_change())
        )
        fig_corr = px.line(
            rolling_corr,
            title=f"{selected_macro} vs WTI — 90‑Day Rolling Correlation",
            labels={"value": "Correlation", "index": "Date"}
        )
        fig_corr.add_hline(y=0, line_width=1, line_color="black")
        fig_corr.update_layout(hovermode="x unified")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Historical Context")

        def historical_context(series_in):
            s = series_in.dropna()
            latest = s.iloc[-1]
            pct = s.rank(pct=True).iloc[-1] * 100
            if pct >= 90:
                pos = "near the highest levels in its history"
            elif pct >= 70:
                pos = "well above its long‑term average"
            elif pct >= 40:
                pos = "around its long‑term average"
            elif pct >= 10:
                pos = "well below its long‑term average"
            else:
                pos = "near the lowest levels in its history"
            return latest, pct, pos

        latest, pct, pos = historical_context(series)

        st.markdown(f"""
        **Latest Value:** {latest:.3f}  
        **Historical Percentile:** {pct:.1f}%  
        **Interpretation:** This series is **{pos}**.
        """)

        st.subheader("Macro Explanation")
        if selected_macro in explanation_df.index:
            st.dataframe(explanation_df.loc[[selected_macro]])
        else:
            st.info("No explanation available for this macro series.")


