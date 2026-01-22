import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Try to import arch for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

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
# 6. REGRESSION HELPERS
# ---------------------------------------------------------

def run_macro_regressions(macro_series, wti_series, horizons=(1, 5, 21)):
    results = []

    aligned = pd.concat([macro_series, wti_series], axis=1).dropna()
    aligned.columns = ["macro", "wti"]

    aligned["macro_z"] = (aligned["macro"] - aligned["macro"].mean()) / aligned["macro"].std()
    wti_log = np.log(aligned["wti"])

    for h in horizons:
        future_ret = (wti_log.shift(-h) - wti_log).rename("future_ret")
        df_reg = pd.concat([aligned["macro_z"], future_ret], axis=1).dropna()

        if df_reg.empty:
            continue

        X = sm.add_constant(df_reg["macro_z"])
        y = df_reg["future_ret"]

        model = sm.OLS(y, X).fit()

        results.append({
            "Horizon (days)": h,
            "Coefficient": model.params["macro_z"],
            "p-value": model.pvalues["macro_z"],
            "R-squared": model.rsquared
        })

    return pd.DataFrame(results)

def interpret_regression_row(row, macro_name):
    h = int(row["Horizon (days)"])
    coef = row["Coefficient"]
    p = row["p-value"]
    r2 = row["R-squared"]

    if p < 0.05:
        sig = "statistically significant"
    else:
        sig = "not statistically significant"

    if coef > 0:
        effect = "higher"
    else:
        effect = "lower"

    return (
        f"Over a {h}-day horizon: coefficient = {coef:.4f}, p={p:.3f}, R²={r2:.3f}. "
        f"This suggests {sig} increases in **{macro_name}** tend to be associated with "
        f"**{effect} future WTI returns**."
    )

def run_multifactor_regression(macro_df, wti_series, selected_macros, horizon=5):
    if not selected_macros:
        return None

    macro_sub = macro_df[selected_macros].dropna(how="all")
    aligned = macro_sub.join(wti_series.rename("WTI"), how="inner").dropna()
    if aligned.empty:
        return None

    wti_log = np.log(aligned["WTI"])
    future_ret = (wti_log.shift(-horizon) - wti_log).rename("future_ret")
    df_reg = pd.concat([aligned.drop(columns=["WTI"]), future_ret], axis=1).dropna()
    if df_reg.empty:
        return None

    X_raw = df_reg.drop(columns=["future_ret"])
    X = (X_raw - X_raw.mean()) / X_raw.std()
    X = sm.add_constant(X)
    y = df_reg["future_ret"]
    model = sm.OLS(y, X).fit()
    return model

def run_rolling_regression(macro_series, wti_series, window=252, horizon=5):
    aligned = pd.concat([macro_series, wti_series], axis=1).dropna()
    aligned.columns = ["macro", "wti"]
    wti_log = np.log(aligned["wti"])
    future_ret = (wti_log.shift(-horizon) - wti_log).rename("future_ret")
    df = pd.concat([aligned["macro"], future_ret], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    betas = []
    index = []
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        X = sm.add_constant(window_df["macro"])
        y = window_df["future_ret"]
        model = sm.OLS(y, X).fit()
        betas.append(model.params["macro"])
        index.append(df.index[i])
    return pd.Series(betas, index=index, name="Rolling Beta")

# ---------------------------------------------------------
# 7. TECHNICAL INDICATORS & MODELING HELPERS
# ---------------------------------------------------------

def compute_rsi(series, window=14):
    delta = series.diff()
    delta_np = np.asarray(delta).reshape(-1)
    gain = np.where(delta_np > 0, delta_np, 0)
    loss = np.where(delta_np < 0, -delta_np, 0)
    gain_s = pd.Series(gain, index=series.index)
    loss_s = pd.Series(loss, index=series.index)
    roll_up = gain_s.rolling(window).mean()
    roll_down = loss_s.rolling(window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def fit_arimax(price_series, exog_df, order=(1,1,1), steps=10):
    s = price_series.dropna()
    if s.empty:
        return None, None
    exog = exog_df.reindex(s.index).fillna(method="ffill").dropna()
    s = s.loc[exog.index]
    if len(s) < 50:
        return None, None
    model = ARIMA(s, order=order, exog=exog)
    res = model.fit()
    last_exog = exog.iloc[-1:]
    exog_fc = pd.concat([last_exog] * steps)
    fc = res.get_forecast(steps=steps, exog=exog_fc)
    fc_mean = fc.predicted_mean
    conf = fc.conf_int()
    return fc_mean, conf

def fit_garch_x(return_series, exog_df, p=1, o=0, q=1, dist="normal", steps=10):
    if not ARCH_AVAILABLE:
        return None, None
    r = return_series.dropna()
    if r.empty:
        return None, None
    exog = exog_df.reindex(r.index).fillna(method="ffill").dropna()
    r = r.loc[exog.index]
    if len(r) < 200:
        return None, None
    am = arch_model(
        r * 100,
        mean="Constant",
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,
        x=exog
    )
    res = am.fit(disp="off")
    fc = res.forecast(horizon=steps)
    vol_fc = np.sqrt(fc.variance.values[-1, :])
    return vol_fc, res

def compute_rolling_stats(series, window=21):
    returns = series.pct_change()
    roll_vol = returns.rolling(window).std()
    roll_mean = returns.rolling(window).mean()
    return roll_vol, roll_mean

# ---------------------------------------------------------
# 8. TOP-LEVEL TABS (M3)
# ---------------------------------------------------------

tab_overview, tab_macro_data, tab_macro_analysis, tab_technicals, tab_arimax, tab_garchx, tab_regressions = st.tabs(
    ["Overview", "Macro Data", "Macro Analysis", "Technicals", "ARIMAX", "GARCH‑X", "Regressions"]
)

# ---------------------------------------------------------
# 9. OVERVIEW TAB
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
# 10. MACRO DATA TAB (FRED API + LOADER + CHARTS)
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
        st.success("Macro data loaded successfully.")

        st.subheader("Preview of Macro Data")
        st.dataframe(macro_df.tail())

        st.subheader("Macro Series Preview (First 5)")
        preview_cols = macro_df.columns[:5]
        fig_preview = px.line(
            macro_df[preview_cols],
            title="Macro Series Preview",
            labels={"value": "Value", "index": "Date"}
        )
        fig_preview.update_layout(hovermode="x unified")
        st.plotly_chart(fig_preview, use_container_width=True)

        st.subheader("Macro Correlation Heatmap")
        corr = macro_df.pct_change().corr().round(2)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Macro Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Macro Data Availability")
        counts = macro_df.notna().sum().sort_values(ascending=False)
        fig_counts = px.bar(
            counts,
            title="Number of Observations per Macro Series",
            labels={"value": "Observations", "index": "Series"}
        )
        st.plotly_chart(fig_counts, use_container_width=True)

# ---------------------------------------------------------
# 11. MACRO ANALYSIS TAB (SINGLE-FACTOR REGRESSIONS)
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

        st.subheader("Regression: Macro vs Future WTI Returns")
        reg_df = run_macro_regressions(series, energy["WTI"], horizons=(1, 5, 21))
        if reg_df.empty:
            st.info("Not enough data to run regressions for this macro series.")
        else:
            st.dataframe(
                reg_df.style.format({
                    "Coefficient": "{:.4f}",
                    "p-value": "{:.3f}",
                    "R-squared": "{:.3f}"
                })
            )
            st.markdown("**Interpretation:**")
            for _, row in reg_df.iterrows():
                explanation = interpret_regression_row(row, selected_macro)
                st.markdown(f"- {explanation}")

# ---------------------------------------------------------
# 12. TECHNICALS TAB (MACD, RSI, ROLLING STATS)
# ---------------------------------------------------------

with tab_technicals:
    st.title("Technical Indicators & Rolling Stats")

    modeling_ticker = st.selectbox(
        "Select ticker for technicals:",
        ["CL=F"] + equities + etfs
    )

    @st.cache_data(show_spinner=True)
    def load_single_ticker(ticker):
        return yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)["Close"].dropna()

    price_series = load_single_ticker(modeling_ticker)

    st.subheader(f"Price Series — {ticker_names.get(modeling_ticker, modeling_ticker)}")
    fig_price = px.line(price_series, title="Price History", labels={"value": "Price", "index": "Date"})
    fig_price.update_layout(hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    # MACD
    st.subheader("MACD")
    macd_raw, signal_raw, hist_raw = compute_macd(price_series)
    macd = pd.Series(np.asarray(macd_raw).reshape(-1), index=price_series.index)
    signal_line = pd.Series(np.asarray(signal_raw).reshape(-1), index=price_series.index)
    hist = pd.Series(np.asarray(hist_raw).reshape(-1), index=price_series.index)
    macd_df = pd.DataFrame({
        "MACD": macd,
        "Signal": signal_line,
        "Histogram": hist
    }).dropna()
    fig_macd = px.line(macd_df[["MACD", "Signal"]], title="MACD & Signal")
    fig_macd.update_layout(hovermode="x unified")
    st.plotly_chart(fig_macd, use_container_width=True)

    # RSI
    st.subheader("RSI (14)")
    rsi = compute_rsi(price_series, window=14)
    rsi = pd.Series(np.asarray(rsi).reshape(-1), index=price_series.index)
    fig_rsi = px.line(rsi, title="RSI (14)", labels={"value": "RSI", "index": "Date"})
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(hovermode="x unified")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Rolling stats (FIXED)
    st.subheader("Rolling Volatility & Mean (Returns)")
    window_rs = st.slider("Rolling window (days)", 5, 126, 21)
    roll_vol, roll_mean = compute_rolling_stats(price_series, window=window_rs)

    roll_vol = pd.Series(np.asarray(roll_vol).reshape(-1), index=price_series.index)
    roll_mean = pd.Series(np.asarray(roll_mean).reshape(-1), index=price_series.index)

    roll_df = pd.DataFrame({
        "Rolling Volatility": roll_vol,
        "Rolling Mean": roll_mean
    }).dropna()

    fig_roll = px.line(roll_df, title=f"Rolling {window_rs}-Day Volatility & Mean")
    fig_roll.update_layout(hovermode="x unified")
    st.plotly_chart(fig_roll, use_container_width=True)

# ---------------------------------------------------------
# 13. ARIMAX TAB (USER-SELECTED MACROS)
# ---------------------------------------------------------

with tab_arimax:
    st.title("ARIMAX Modeling (Price with Macro Exogenous)")

    macro_df = st.session_state.get("macro_df", pd.DataFrame())
    if macro_df.empty:
        st.warning("No macro data available. Load macro data first.")
    else:
        arimax_ticker = st.selectbox(
            "Select ticker for ARIMAX:",
            ["CL=F", "BZ=F"] + futures + equities + etfs,
            key="arimax_ticker"
        )

        price_series = load_single_ticker(arimax_ticker)
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.squeeze()

        st.subheader("ARIMA(p, d, q) Model Structure")
        st.markdown("""
        **ARIMA(p, d, q)** decomposes price behaviour into:
        - **p (autoregressive lags):** how many past prices influence the next price  
        - **d (differencing):** how many times the series is differenced to remove trend  
        - **q (moving-average lags):** how many past shocks influence the next price  

        This structure captures **trend**, **persistence**, and **shock propagation**.
        """)

        st.subheader("Select Macro Exogenous Variables")
        macro_list = list(macro_df.columns)
        selected_macros_arimax = st.multiselect(
            "Macro variables to include as exogenous:",
            macro_list,
            default=macro_list[:3],
            key="arimax_macro_select"
        )

        if not selected_macros_arimax:
            st.info("Select at least one macro variable to run ARIMAX.")
        else:
            exog = macro_df[selected_macros_arimax].dropna(how="all")
            price_df = price_series.rename("Price").to_frame()
            aligned = exog.join(price_df, how="inner").dropna()

            if aligned.empty:
                st.info("No overlapping data between selected macro series and price.")
            else:
                price_aligned = aligned["Price"]
                exog_aligned = aligned.drop(columns=["Price"])

                steps = st.slider("Forecast horizon (days)", 5, 60, 20)
                order_p = st.number_input("AR order (p)", 0, 5, 1)
                order_d = st.number_input("Differencing (d)", 0, 2, 1)
                order_q = st.number_input("MA order (q)", 0, 5, 1)

                if st.button("Run ARIMAX"):
                    with st.spinner("Fitting ARIMAX model..."):
                        fc_mean, conf = fit_arimax(
                            price_aligned,
                            exog_aligned,
                            order=(order_p, order_d, order_q),
                            steps=steps
                        )

                    if fc_mean is None:
                        st.info("Not enough data to fit ARIMAX.")
                    else:
                        st.subheader("ARIMAX Forecast")

                        fc_index = pd.date_range(
                            start=price_aligned.index[-1] + pd.Timedelta(days=1),
                            periods=steps,
                            freq="B"
                        )
                        fc_series = pd.Series(fc_mean.values, index=fc_index, name="Forecast")

                        hist_df = price_aligned.rename("Price").to_frame()
                        fc_df = fc_series.to_frame()

                        fig_fc = px.line(hist_df, title=f"{ticker_names.get(arimax_ticker, arimax_ticker)} — Price & ARIMAX Forecast")
                        fig_fc.add_scatter(
                            x=fc_df.index,
                            y=fc_df["Forecast"],
                            mode="lines",
                            name="Forecast"
                        )

                        if conf is not None:
                            conf = conf.rename(columns={conf.columns[0]: "lower", conf.columns[1]: "upper"})
                            conf.index = fc_index
                            fig_fc.add_scatter(
                                x=conf.index, y=conf["lower"],
                                mode="lines", line=dict(width=0), showlegend=False
                            )
                            fig_fc.add_scatter(
                                x=conf.index, y=conf["upper"],
                                mode="lines", line=dict(width=0), fill="tonexty",
                                name="Confidence Interval", opacity=0.2
                            )

                        fig_fc.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_fc, use_container_width=True)

                        # -----------------------------
                        # Forecast Percentile Analysis
                        # -----------------------------
                        st.subheader("Forecast Interpretation")

                        hist = price_aligned.dropna()
                        combined = hist.append(pd.Series([fc_series.iloc[-1]], index=[fc_index[-1]]))
                        rank_pct = combined.rank(pct=True).iloc[-1] * 100

                        st.markdown(
                            f"Last forecast value sits at the **{rank_pct:.1f}th percentile** "
                            f"of historical prices."
                        )

                        if rank_pct > 70:
                            st.markdown("→ Model expects **relative strength** vs history.")
                        elif rank_pct < 30:
                            st.markdown("→ Model expects **relative weakness** vs history.")
                        else:
                            st.markdown("→ Model expects **neutral / mid‑range** pricing.")

# ---------------------------------------------------------
# 14. GARCH‑X TAB
# ---------------------------------------------------------

with tab_garchx:
    st.title("GARCH‑X Volatility Modeling")

    if not ARCH_AVAILABLE:
        st.warning("The 'arch' package is not installed.")
    else:
        macro_df = st.session_state.get("macro_df", pd.DataFrame())
        if macro_df.empty:
            st.warning("Load macro data first.")
        else:
            garch_ticker = st.selectbox(
                "Select ticker for GARCH‑X:",
                ["CL=F", "BZ=F"] + futures + equities + etfs,
                key="garch_ticker"
            )

            price_series = load_single_ticker(garch_ticker)
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.squeeze()

            returns = price_series.pct_change().dropna()
            if isinstance(returns, pd.DataFrame):
                returns = returns.squeeze()

            st.subheader("GARCH‑X Model Interpretation")
            st.markdown("""
            **GARCH‑X** models volatility, not direction.

            - **ω (omega):** baseline volatility  
            - **α (alpha):** sensitivity to shocks (yesterday’s surprise)  
            - **β (beta):** persistence of volatility (how long shocks last)  
            - **X:** macro factors that shift volatility regimes  

            High α → markets react strongly to shocks  
            High β → volatility is persistent  
            """)

            st.subheader("Select Macro Exogenous Variables")
            macro_list = list(macro_df.columns)
            selected_macros_garch = st.multiselect(
                "Macro variables to include as exogenous:",
                macro_list,
                default=macro_list[:3],
                key="garch_macro_select"
            )

            if not selected_macros_garch:
                st.info("Select at least one macro variable.")
            else:
                exog = macro_df[selected_macros_garch].dropna(how="all")
                returns_df = returns.rename("ret").to_frame()
                aligned = exog.join(returns_df, how="inner").dropna()

                if aligned.empty:
                    st.info("No overlapping data between macro series and returns.")
                else:
                    ret_aligned = aligned["ret"]
                    exog_aligned = aligned.drop(columns=["ret"])

                    steps = st.slider("Volatility forecast horizon (days)", 5, 60, 20)

                    if st.button("Run GARCH‑X"):
                        with st.spinner("Fitting GARCH‑X model..."):
                            vol_fc, res = fit_garch_x(ret_aligned, exog_aligned, steps=steps)

                        if vol_fc is None:
                            st.info("Not enough data to fit GARCH‑X.")
                        else:
                            st.subheader("GARCH‑X Volatility Forecast")

                            fc_index = pd.date_range(
                                start=ret_aligned.index[-1] + pd.Timedelta(days=1),
                                periods=steps,
                                freq="B"
                            )
                            vol_series = pd.Series(vol_fc, index=fc_index, name="Forecast Volatility")

                            fig_vol = px.line(
                                vol_series,
                                title=f"{ticker_names.get(garch_ticker, garch_ticker)} — Forecast Volatility (GARCH‑X)",
                                labels={"value": "Volatility", "index": "Date"}
                            )
                            fig_vol.update_layout(hovermode="x unified")
                            st.plotly_chart(fig_vol, use_container_width=True)

                            # -----------------------------
                            # Volatility Regime Analysis
                            # -----------------------------
                            st.subheader("Volatility Regime Interpretation")

                            hist_vol = ret_aligned.rolling(21).std().dropna()
                            if not hist_vol.empty:
                                combined = hist_vol.append(pd.Series([vol_series.iloc[0]], index=[fc_index[0]]))
                                rank_pct = combined.rank(pct=True).iloc[-1] * 100

                                st.markdown(
                                    f"Near‑term forecast volatility is at the **{rank_pct:.1f}th percentile** "
                                    f"of recent 21‑day volatility."
                                )

                                if rank_pct > 70:
                                    st.markdown("→ Indicates a **high‑volatility regime** (risk‑off / stress).")
                                elif rank_pct < 30:
                                    st.markdown("→ Indicates a **low‑volatility regime** (calm conditions).")
                                else:
                                    st.markdown("→ Indicates a **mid‑range volatility regime**.")
                            else:
                                st.markdown("Not enough history to benchmark volatility regimes.")

                            # -----------------------------
                            # Trend & Directional Context
                            # -----------------------------
                            st.subheader("Trend & Directional Context")
                            last_ret = ret_aligned.iloc[-1] * 100
                            st.markdown(
                                f"Last daily return was **{last_ret:.2f}%**. "
                                "GARCH‑X does **not** forecast direction, but high volatility combined with "
                                "negative recent returns often signals stressed downside regimes."
                            )

                            # -----------------------------
                            # GARCH Parameter Snapshot
                            # -----------------------------
                            if res is not None:
                                st.subheader("GARCH Parameter Snapshot")
                                st.text(res.summary())

# ---------------------------------------------------------
# 15. REGRESSIONS TAB (MULTIFACTOR & ROLLING BETA)
# ---------------------------------------------------------

with tab_regressions:
    st.title("Multifactor & Rolling Regression")

    macro_df = st.session_state.get("macro_df", pd.DataFrame())
    if macro_df.empty:
        st.warning("Load macro data first.")
    else:

        # ---------------------------------------------------------
        # MULTIFACTOR REGRESSION (MACRO CHANGES → FUTURE RETURNS)
        # ---------------------------------------------------------
        st.subheader("Multifactor Regression vs WTI (Macro Changes → Future Returns)")

        # Load WTI
        energy = yf.download(
            ["CL=F"],
            start="1990-01-01",
            auto_adjust=True,
            progress=False
        )["Close"]
        if isinstance(energy, pd.DataFrame):
            energy = energy.squeeze()
        energy.name = "WTI"

        # Compute WTI returns
        wti_ret = energy.pct_change()

        macro_list = list(macro_df.columns)
        selected_macros_multi = st.multiselect(
            "Select macro variables for multifactor regression:",
            macro_list,
            key="multi_macro_select"
        )

        horizon_multi = st.slider("Horizon (days) for future returns", 1, 60, 5)

        if selected_macros_multi:

            # Prepare macro changes
            macro_changes = macro_df[selected_macros_multi].pct_change()

            # Align everything
            df = pd.concat([macro_changes, wti_ret], axis=1).dropna()
            df.columns = selected_macros_multi + ["wti_ret"]

            # Compute future returns
            df["future_ret"] = df["wti_ret"].shift(-horizon_multi)
            df = df.dropna()

            if df.empty:
                st.info("Not enough overlapping data.")
            else:
                # Standardize predictors
                X = (df[selected_macros_multi] - df[selected_macros_multi].mean()) / df[selected_macros_multi].std()
                X = sm.add_constant(X)
                y = df["future_ret"]

                model = sm.OLS(y, X).fit()

                st.subheader("Regression Summary")
                st.text(model.summary())

                # ---------------------------------------------------------
                # EXPLAINABILITY BLOCK
                # ---------------------------------------------------------
                st.subheader("Economic Interpretation")

                params = model.params
                pvals = model.pvalues
                r2 = model.rsquared

                st.markdown(f"### Model Fit")
                st.markdown(f"**R² = {r2:.3f}** — typical macro–crude models range **0.05–0.30** for daily data.")

                st.markdown("### Coefficient Signs & Economic Meaning")
                for name in selected_macros_multi:
                    if name in params.index:
                        coef = params[name]
                        sign = "positive" if coef > 0 else "negative"
                        st.markdown(f"- **{name}**: {sign} coefficient ({coef:.4f})")

                st.markdown("### Statistical Significance (p-values)")
                for name in selected_macros_multi:
                    if name in pvals.index:
                        p = pvals[name]
                        signif = "SIGNIFICANT" if p < 0.05 else "not significant"
                        st.markdown(f"- **{name}**: p = {p:.4f} → {signif}")

                st.markdown("### Economic Interpretation Summary")
                st.markdown("""
                - **Positive coefficient** → when this macro factor rises, crude tends to rise  
                - **Negative coefficient** → when this macro factor rises, crude tends to fall  
                - **Significant variables** → real explanatory power  
                - **Non‑significant variables** → weak or unstable relationship  
                """)

                st.markdown("### Macro‑Economic Consistency Check")
                st.markdown("""
                - **USD Index** → usually negative (strong USD pressures crude lower)  
                - **10Y Yield** → often negative (tighter financial conditions)  
                - **VIX** → typically negative (risk‑off reduces demand expectations)  
                - **Equity indices** → usually positive (risk‑on supports crude)  
                """)

        # ---------------------------------------------------------
        # ROLLING BETA (MACRO CHANGES → FUTURE RETURNS)
        # ---------------------------------------------------------
        st.subheader("Rolling Single-Factor Beta vs WTI")

        selected_macro_rb = st.selectbox(
            "Select macro variable:",
            macro_list,
            key="rolling_macro_select"
        )

        window_rb = st.slider("Rolling window (days)", 60, 504, 252)
        horizon_rb = st.slider("Horizon (days)", 1, 60, 5, key="rolling_horizon_slider")

        # Prepare macro changes
        macro_chg_rb = macro_df[selected_macro_rb].pct_change()
        wti_ret_rb = energy.pct_change()

        df_rb = pd.concat([macro_chg_rb, wti_ret_rb], axis=1).dropna()
        df_rb.columns = ["macro_chg", "wti_ret"]
        df_rb["future_ret"] = df_rb["wti_ret"].shift(-horizon_rb)
        df_rb = df_rb.dropna()

        if df_rb.empty:
            st.info("Not enough overlapping data.")
        else:
            betas = []
            idx = []

            for i in range(window_rb, len(df_rb)):
                window_df = df_rb.iloc[i-window_rb:i]
                X = sm.add_constant(window_df["macro_chg"])
                y = window_df["future_ret"]
                model = sm.OLS(y, X).fit()
                betas.append(model.params["macro_chg"])
                idx.append(df_rb.index[i])

            beta_series = pd.Series(betas, index=idx, name="Rolling Beta")

            if beta_series.empty:
                st.info("Not enough data to compute rolling betas.")
            else:
                fig_beta = px.line(
                    beta_series,
                    title=f"Rolling {window_rb}-Day Beta of {selected_macro_rb} vs WTI",
                    labels={"value": "Beta", "index": "Date"}
                )
                fig_beta.add_hline(y=0, line_width=1, line_color="black")
                fig_beta.update_layout(hovermode="x unified")
                st.plotly_chart(fig_beta, use_container_width=True)

                # ---------------------------------------------------------
                # EXPLAINABILITY FOR ROLLING BETA
                # ---------------------------------------------------------
                st.subheader("Rolling Beta Interpretation")

                latest_beta = beta_series.iloc[-1]

                st.markdown(f"**Latest beta:** {latest_beta:.3f}")

                if latest_beta > 0:
                    st.markdown(
                        f"- When **{selected_macro_rb}** rises, WTI tends to rise (positive sensitivity)."
                    )
                else:
                    st.markdown(
                        f"- When **{selected_macro_rb}** rises, WTI tends to fall (negative sensitivity)."
                    )

                st.markdown("""
                - Rising beta → macro factor is becoming more influential  
                - Falling beta → macro factor influence is weakening  
                - Beta crossing zero → regime shift in macro–crude relationship  
                """)
