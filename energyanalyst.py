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
    fc = res.forecast(horizon=steps, x=exog.iloc[[-1]].values)
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
        st.warning("No macro data available. Go to the 'Macro Data' tab, enter your FRED API key, and load data.")
    else:
        arimax_ticker = st.selectbox(
            "Select ticker for ARIMAX:",
            ["CL=F"] + equities + etfs,
            key="arimax_ticker"
        )

        price_series = load_single_ticker(arimax_ticker)

        st.subheader("Select Macro Exogenous Variables")
        macro_list = list(macro_df.columns)
        selected_macros_arimax = st.multiselect(
            "Macro variables to include as exogenous:",
            macro_list,
            default=macro_list[:3]
        )

        if not selected_macros_arimax:
            st.info("Select at least one macro variable to run ARIMAX.")
        else:
            exog_df = macro_df[selected_macros_arimax]

            st.subheader("ARIMA Parameters (p, d, q)")
            col_p, col_d, col_q = st.columns(3)
            with col_p:
                p = st.number_input("p (AR order)", min_value=0, max_value=5, value=1, step=1)
            with col_d:
                d = st.number_input("d (diff order)", min_value=0, max_value=2, value=1, step=1)
            with col_q:
                q = st.number_input("q (MA order)", min_value=0, max_value=5, value=1, step=1)

            steps = st.slider("Forecast horizon (days)", 5, 60, 20)

            if st.button("Run ARIMAX"):
                with st.spinner("Fitting ARIMAX model..."):
                    fc_mean, conf = fit_arimax(price_series, exog_df, order=(p, d, q), steps=steps)

                if fc_mean is None:
                    st.info("Not enough data or alignment issues to fit ARIMAX.")
                else:
                    fc_index = pd.date_range(start=price_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
                    fc_series = pd.Series(fc_mean.values, index=fc_index, name="Forecast")
                    conf_df = conf.copy()
                    conf_df.index = fc_index

                    hist_df = price_series.rename("Price")
                    fig_arima = px.line(hist_df, title="ARIMAX Forecast")
                    fig_arima.add_scatter(x=fc_series.index, y=fc_series.values, mode="lines", name="Forecast")
                    fig_arima.add_scatter(
                        x=conf_df.index, y=conf_df.iloc[:,0], mode="lines",
                        line=dict(dash="dash", color="gray"), name="Lower CI"
                    )
                    fig_arima.add_scatter(
                        x=conf_df.index, y=conf_df.iloc[:,1], mode="lines",
                        line=dict(dash="dash", color="gray"), name="Upper CI"
                    )
                    fig_arima.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_arima, use_container_width=True)

# ---------------------------------------------------------
# 14. GARCH-X TAB (USER-SELECTED MACROS, p-o-q)
# ---------------------------------------------------------

with tab_garchx:
    st.title("GARCH‑X Volatility Modeling (Returns with Macro Exogenous)")

    if not ARCH_AVAILABLE:
        st.warning("The 'arch' library is not installed. Install it with 'pip install arch' to use GARCH‑X.")
    else:
        macro_df = st.session_state.get("macro_df", pd.DataFrame())
        if macro_df.empty:
            st.warning("No macro data available. Go to the 'Macro Data' tab, enter your FRED API key, and load data.")
        else:
            garch_ticker = st.selectbox(
                "Select ticker for GARCH‑X:",
                ["CL=F"] + equities + etfs,
                key="garch_ticker"
            )

            price_series = load_single_ticker(garch_ticker)
            returns = price_series.pct_change().dropna()

            st.subheader("Select Macro Exogenous Variables")
            macro_list = list(macro_df.columns)
            selected_macros_garch = st.multiselect(
                "Macro variables to include as exogenous:",
                macro_list,
                default=macro_list[:3]
            )

            if not selected_macros_garch:
                st.info("Select at least one macro variable to run GARCH‑X.")
            else:
                exog_df = macro_df[selected_macros_garch]

                st.subheader("GARCH Parameters (p, o, q)")
                col_p, col_o, col_q = st.columns(3)
                with col_p:
                    p_g = st.number_input("p (ARCH order)", min_value=0, max_value=5, value=1, step=1)
                with col_o:
                    o_g = st.number_input("o (asymmetry)", min_value=0, max_value=5, value=0, step=1)
                with col_q:
                    q_g = st.number_input("q (GARCH order)", min_value=0, max_value=5, value=1, step=1)

                dist = st.selectbox("Distribution", ["normal", "t", "skewt"], index=0)
                steps_g = st.slider("Volatility forecast horizon (days)", 5, 60, 20)

                if st.button("Run GARCH‑X"):
                    with st.spinner("Fitting GARCH‑X model..."):
                        vol_fc, res_garch = fit_garch_x(returns, exog_df, p=p_g, o=o_g, q=q_g, dist=dist, steps=steps_g)

                    if vol_fc is None:
                        st.info("Not enough data or alignment issues to fit GARCH‑X.")
                    else:
                        fc_index = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=steps_g, freq="B")
                        vol_series = pd.Series(vol_fc, index=fc_index, name="Forecast Volatility")

                        st.subheader("Forecast Volatility (Annualized Approx.)")
                        fig_vol = px.line(vol_series * np.sqrt(252), title="GARCH‑X Volatility Forecast (Annualized)")
                        fig_vol.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_vol, use_container_width=True)

                        st.subheader("Last In-Sample Conditional Volatility")
                        cond_vol = res_garch.conditional_volatility
                        fig_cv = px.line(cond_vol, title="In-Sample Conditional Volatility")
                        fig_cv.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_cv, use_container_width=True)

# ---------------------------------------------------------
# 15. REGRESSIONS TAB (MULTI-FACTOR & ROLLING)
# ---------------------------------------------------------

with tab_regressions:
    st.title("Advanced Regression Analysis")

    macro_df = st.session_state.get("macro_df", pd.DataFrame())
    if macro_df.empty:
        st.warning("No macro data available. Go to the 'Macro Data' tab, enter your FRED API key, and load data.")
    else:
        energy = yf.download(
            ["CL=F"],
            start="1990-01-01",
            auto_adjust=True,
            progress=False
        )["Close"]
        wti = energy["CL=F"].rename("WTI").dropna()

        st.subheader("Multi-Factor Regression (User-Selected Macro Basket → Future WTI Returns)")
        macro_list = list(macro_df.columns)
        selected_macros_mf = st.multiselect(
            "Select macro variables for multi-factor regression:",
            macro_list,
            default=macro_list[:4]
        )
        horizon = st.selectbox("Horizon (days)", [5, 21, 63], index=1)
        if selected_macros_mf:
            model = run_multifactor_regression(macro_df, wti, selected_macros_mf, horizon=horizon)
            if model is None:
                st.info("Not enough data to run multi-factor regression.")
            else:
                st.markdown(f"**R-squared:** {model.rsquared:.3f}")
                coef_df = model.params.to_frame("Coefficient")
                coef_df["p-value"] = model.pvalues
                st.dataframe(coef_df.style.format({"Coefficient": "{:.4f}", "p-value": "{:.3f}"}))
        else:
            st.info("Select at least one macro variable for multi-factor regression.")

        st.subheader("Rolling Regression (Single Macro → WTI Returns)")
        selected_macro_roll = st.selectbox("Select macro variable for rolling regression:", macro_list)
        window = st.slider("Rolling window (days)", 60, 504, 252)
        horizon_roll = st.selectbox("Horizon (days) for rolling regression", [5, 21], index=0)

        roll_series = run_rolling_regression(macro_df[selected_macro_roll], wti, window=window, horizon=horizon_roll)
        if roll_series.empty:
            st.info("Not enough data to compute rolling regression.")
        else:
            fig_roll = px.line(
                roll_series,
                title=f"Rolling {window}-Day Beta: {selected_macro_roll} → WTI ({horizon_roll}-day returns)",
                labels={"value": "Beta", "index": "Date"}
            )
            fig_roll.add_hline(y=0, line_width=1, line_color="black")
            fig_roll.update_layout(hovermode="x unified")
            st.plotly_chart(fig_roll, use_container_width=True)
