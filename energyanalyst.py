import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------
# 1. Define Ticker Groups
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

# ---------------------------------------------------------
# 2. Ticker → Full Name Mapping
# ---------------------------------------------------------

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
# 3. Download Data (cached)
# ---------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_data():
    return yf.download(
        all_tickers,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        start="2000-01-01"
    )

data = load_data()

# ---------------------------------------------------------
# 4. User-Defined Start Date
# ---------------------------------------------------------

min_date = data.index.min()
max_date = data.index.max()

min_dt = pd.to_datetime(min_date).to_pydatetime()
max_dt = pd.to_datetime(max_date).to_pydatetime()

user_start_date = st.sidebar.date_input(
    "Select start date",
    value=min_dt.date(),
    min_value=min_dt.date(),
    max_value=max_dt.date()
)

user_start_dt = pd.to_datetime(user_start_date).to_pydatetime()

# ---------------------------------------------------------
# 5. Date Range Slider
# ---------------------------------------------------------

date_range = st.sidebar.slider(
    "Select date range",
    min_value=user_start_dt,
    max_value=max_dt,
    value=(user_start_dt, max_dt)
)

def filter_by_date(series):
    s = series.loc[user_start_dt:]
    return s.loc[date_range[0]:date_range[1]]

# ---------------------------------------------------------
# 6. Extract Close Series
# ---------------------------------------------------------

def extract_close_series(ticker):
    try:
        s = data[ticker]["Close"].dropna()
        return filter_by_date(s)
    except:
        return pd.Series(dtype=float)

# ---------------------------------------------------------
# 7. Normalization
# ---------------------------------------------------------

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
    if base_date not in series.index:
        base_date = series.index[0]
    return (series / series.loc[base_date]) * base_value

# ---------------------------------------------------------
# 8. Plotly Chart Functions
# ---------------------------------------------------------

def plot_price_group(tickers, title):
    df = pd.DataFrame({ticker_names.get(t, t): extract_close_series(t) for t in tickers})
    df = df.dropna(how="all")

    fig = px.line(df, title=title)
    fig.update_layout(hovermode="x unified")
    return fig

def plot_normalized_group(tickers, title, method="first", base_value=100):
    df = {}
    for t in tickers:
        s = extract_close_series(t)
        if not s.empty:
            df[ticker_names.get(t, t)] = normalize(s, method=method, base_value=base_value)

    df = pd.DataFrame(df).dropna(how="all")

    fig = px.line(df, title=f"{title} (Normalized: {method})")
    fig.update_layout(hovermode="x unified")
    return fig

def plot_indexed_to_100(tickers, title, base_date):
    df = {}
    for t in tickers:
        s = extract_close_series(t)
        if not s.empty:
            df[ticker_names.get(t, t)] = normalize_to_date(s, base_date)

    df = pd.DataFrame(df).dropna(how="all")

    fig = px.line(df, title=f"{title} (Indexed to 100 at {base_date.date()})")
    fig.update_layout(hovermode="x unified")
    return fig

def plot_custom_index(index_series, base_date):
    fig = px.line(
        index_series,
        title=f"Custom ETF Index (Base Date: {base_date.date()})"
    )
    fig.update_layout(hovermode="x unified")
    return fig

def plot_category_heatmap(tickers, title):
    df = pd.DataFrame({ticker_names.get(t, t): extract_close_series(t) for t in tickers})
    df = df.dropna(axis=1, how="all")
    corr = df.pct_change().corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"{title} — Correlation Heatmap"
    )
    fig.update_layout(height=600)
    return fig

# ---------------------------------------------------------
# 9. Return Table
# ---------------------------------------------------------

def compute_return_table(tickers):
    rows = []
    for t in tickers:
        s = extract_close_series(t)
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
    return pd.DataFrame(rows).set_index("Name")

# ---------------------------------------------------------
# 10. Sidebar Controls
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# 11. Tabs
# ---------------------------------------------------------

tab_futures, tab_etfs, tab_equities = st.tabs(["Futures", "ETFs", "Equities"])

# ---------------------------------------------------------
# 12. Tab Renderer
# ---------------------------------------------------------

def render_tab(tickers, title, show_index=False):

    st.subheader(f"{title} — Price View")

    if norm_choice == "Raw Prices":
        fig = plot_price_group(tickers, f"{title} Prices")

    elif norm_choice == "Normalized (first)":
        fig = plot_normalized_group(tickers, title, method="first")

    elif norm_choice == "Normalized (minmax)":
        fig = plot_normalized_group(tickers, title, method="minmax")

    elif norm_choice == "Normalized (zscore)":
        fig = plot_normalized_group(tickers, title, method="zscore")

    elif norm_choice == "Indexed to 100 (custom date)":
        fig = plot_indexed_to_100(tickers, title, index_start_date)

    st.plotly_chart(fig, use_container_width=True)

    if show_index:
        st.subheader("Custom ETF Index")
        idx, base_date = create_custom_index(tickers)
        fig_idx = plot_custom_index(idx, base_date)
        st.plotly_chart(fig_idx, use_container_width=True)

    if show_heatmap:
        st.subheader(f"{title} — Correlation Heatmap")
        fig_hm = plot_category_heatmap(tickers, title)
        st.plotly_chart(fig_hm, use_container_width=True)

    if show_returns:
        st.subheader(f"{title} — Return Table")
        df = compute_return_table(tickers)
        st.dataframe(df.style.format("{:.2f}").background_gradient(
            subset=["Weekly %", "Monthly %", "Yearly %"],
            cmap="RdYlGn"
        ))

# ---------------------------------------------------------
# 13. Render Tabs
# ---------------------------------------------------------

with tab_futures:
    render_tab(futures, "Energy Futures")

with tab_etfs:
    render_tab(etfs, "Energy ETFs", show_index=True)

with tab_equities:
    render_tab(equities, "Energy Equities")
