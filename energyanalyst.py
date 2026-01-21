import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use("seaborn-v0_8")

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
    data = yf.download(
        all_tickers,
        auto_adjust=True,
        progress=False,
        group_by="ticker"
    )
    return data

data = load_data()

# ---------------------------------------------------------
# 4. Extract Close Series
# ---------------------------------------------------------

def extract_close_series(ticker):
    try:
        return data[ticker]["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

# ---------------------------------------------------------
# 5. Normalization Methods
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
    else:
        raise ValueError("Unknown normalization method")

# ---------------------------------------------------------
# 6. Plotting Functions
# ---------------------------------------------------------

def plot_price_group(tickers, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in tickers:
        s = extract_close_series(t)
        if not s.empty:
            ax.plot(s, label=ticker_names.get(t, t))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_normalized_group(tickers, title, norm_method="first", base_value=100):
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in tickers:
        s = extract_close_series(t)
        if not s.empty:
            s_norm = normalize(s, method=norm_method, base_value=base_value)
            ax.plot(s_norm, label=ticker_names.get(t, t))
    ax.set_title(f"{title} (Normalized: {norm_method})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

# ---------------------------------------------------------
# 7. Index Creator (Auto Base Date)
# ---------------------------------------------------------

def create_custom_index(tickers, base_value=100):
    df = pd.DataFrame({t: extract_close_series(t) for t in tickers})
    df = df.dropna(how="all")

    df_common = df.dropna()
    if df_common.empty:
        return pd.Series(dtype=float)

    common_start = df_common.index.min()
    base_prices = df.loc[common_start]

    index_series = (df / base_prices) * base_value
    index_series["Custom_Index"] = index_series.mean(axis=1)

    return index_series["Custom_Index"], common_start

# ---------------------------------------------------------
# 8. Category-Specific Correlation Heatmaps
# ---------------------------------------------------------

def plot_category_heatmap(tickers, title):
    df = pd.DataFrame({ticker_names.get(t, t): extract_close_series(t) for t in tickers})
    df = df.dropna(axis=1, how="all")

    corr = df.pct_change().corr().round(1)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        ax=ax
    )
    ax.set_title(f"{title} — Correlation Heatmap")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------
# 9. Return Tables
# ---------------------------------------------------------

def compute_return_table(tickers):
    rows = []

    for t in tickers:
        s = extract_close_series(t)
        if s.empty:
            continue

        current = s.iloc[-1]

        def safe_return(series, periods):
            if len(series) > periods:
                return (series.iloc[-1] / series.iloc[-periods] - 1) * 100
            return np.nan

        weekly = safe_return(s, 5)
        monthly = safe_return(s, 21)
        yearly = safe_return(s, 252)

        rows.append({
            "Name": ticker_names.get(t, t),
            "Current Price": round(current, 2),
            "Weekly % Change": weekly,
            "Monthly % Change": monthly,
            "Yearly % Change": yearly
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Name")
    return df

# ---------------------------------------------------------
# 10. Streamlit App Layout
# ---------------------------------------------------------

st.title("Energy Market Overview Dashboard")

group = st.sidebar.selectbox(
    "Select asset group",
    ["Futures", "ETFs", "Equities"]
)

norm_choice = st.sidebar.selectbox(
    "Price view",
    ["Raw Prices", "Normalized (first)", "Normalized (minmax)", "Normalized (zscore)", "Normalized (base=100)"]
)

show_heatmap = st.sidebar.checkbox("Show correlation heatmap", value=True)
show_returns = st.sidebar.checkbox("Show return table", value=True)
show_custom_index = st.sidebar.checkbox("Show custom index (ETFs only)", value=True)

if group == "Futures":
    tickers = futures
    title = "Energy Futures"
elif group == "ETFs":
    tickers = etfs
    title = "Energy ETFs"
else:
    tickers = equities
    title = "Energy Equities"

st.subheader(f"{title} — Price View")

if norm_choice == "Raw Prices":
    fig = plot_price_group(tickers, f"{title} Prices")
else:
    if norm_choice == "Normalized (first)":
        method = "first"
        base = 100
    elif norm_choice == "Normalized (minmax)":
        method = "minmax"
        base = 100
    elif norm_choice == "Normalized (zscore)":
        method = "zscore"
        base = 100
    else:
        method = "custom_base"
        base = 100
    fig = plot_normalized_group(tickers, title, norm_method=method, base_value=base)

st.pyplot(fig)

if group == "ETFs" and show_custom_index:
    st.subheader("Custom ETF Index")
    custom_index, base_date = create_custom_index(etfs)
    if not custom_index.empty:
        fig_idx, ax = plt.subplots(figsize=(12, 4))
        ax.plot(custom_index, label="Custom ETF Index")
        ax.set_title(f"Custom ETF Index (Base Date: {base_date.date()})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_idx.tight_layout()
        st.pyplot(fig_idx)

if show_heatmap:
    st.subheader(f"{title} — Correlation Heatmap")
    fig_hm = plot_category_heatmap(tickers, title)
    st.pyplot(fig_hm)

if show_returns:
    st.subheader(f"{title} — Return Table")
    returns_df = compute_return_table(tickers)
    st.dataframe(
        returns_df.style.format("{:.2f}").background_gradient(
            subset=["Weekly % Change", "Monthly % Change", "Yearly % Change"],
            cmap="RdYlGn"
        )
    )
