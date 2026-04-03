from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf

# ============================================================
# Configuration
# ============================================================
APP_TITLE = "XTB Open Positions vs S&P 500"
DEFAULT_EXCEL_PATH = "xtb_transactions.xlsx"

CACHE_DIR = Path("cache_price_data")
CACHE_DIR.mkdir(exist_ok=True)

XTB_TO_YAHOO_SUFFIX = {
    "BE": "BR",   # Euronext Brussels
    "DE": "DE",   # Xetra
    "NL": "AS",   # Amsterdam
    "PL": "WA",   # Warsaw
    "ES": "MC",   # Madrid
    "PT": "LS",   # Lisbon
    "FR": "PA",   # Paris
    "UK": "L",    # London
    "CH": "SW",   # SIX Swiss Exchange
    "SE": "ST",   # Stockholm
    "FI": "HE",   # Helsinki
    "DK": "CO",   # Copenhagen
    "NO": "OL",   # Oslo
    "IT": "MI",   # Milan
    "AT": "VI",   # Vienna
    "CZ": "PR",   # Prague
}

SP500_TICKER = "^GSPC"
NY_TZ = ZoneInfo("America/New_York")


# ============================================================
# Helpers
# ============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def find_open_positions_sheet(excel_path: str) -> str:
    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        if str(sheet).strip().upper().startswith("OPEN POSITION"):
            return sheet
    raise ValueError("No sheet starting with 'OPEN POSITION' was found.")


def xtb_to_yahoo_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    suffix_map = dict(XTB_TO_YAHOO_SUFFIX)

    if "." not in symbol:
        return symbol

    base, suffix = symbol.rsplit(".", 1)
    yahoo_suffix = suffix_map.get(suffix.upper(), suffix.upper())
    return f"{base}.{yahoo_suffix}"


@st.cache_data(show_spinner=False)
def load_positions(excel_path: str) -> pd.DataFrame:
    sheet_name = find_open_positions_sheet(excel_path)

    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=10)
    raw = normalize_columns(raw)

    required = [
        "Position", "Symbol", "Type", "Volume",
        "Open time", "Open price", "Purchase value",
    ]

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw[required].copy()
    df = df[df["Symbol"].notna()]
    df = df[df["Volume"].notna()]

    df["Type"] = df["Type"].astype(str).str.upper().str.strip()
    df = df[df["Type"].isin(["BUY", "LONG"])]

    df["Open time"] = pd.to_datetime(df["Open time"], dayfirst=True, errors="coerce")

    for col in ["Volume", "Open price", "Purchase value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open time", "Volume", "Open price", "Purchase value"])
    df = df[df["Volume"] > 0]

    df["Yahoo Symbol"] = df["Symbol"].apply(xtb_to_yahoo_symbol)
    df["Open date"] = df["Open time"].dt.normalize()

    return df.reset_index(drop=True)


def cache_file_path(ticker: str) -> Path:
    safe = ticker.replace("^", "__index__").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"


def read_cached_close_frame(ticker: str) -> Optional[pd.DataFrame]:
    cache_path = cache_file_path(ticker)
    if not cache_path.exists():
        return None

    try:
        df = pd.read_parquet(cache_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        if "Close" not in df.columns:
            return None

        df = df[["Close"]].copy()
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])

        return df

    except Exception:
        return None


def get_incremental_fetch_start(existing, requested_start, overlap_days=7):
    requested_start = pd.Timestamp(requested_start).normalize()

    if existing is None or existing.empty:
        return requested_start

    last_cached = pd.Timestamp(existing.index.max()).normalize()
    return min(requested_start, last_cached - pd.Timedelta(days=overlap_days))


def fetch_tickers_history_batch(tickers, start, end):
    tickers = sorted(set(tickers))
    if not tickers:
        return {}

    hist = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
    )

    result = {}

    if isinstance(hist.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in hist.columns.get_level_values(0):
                continue

            df = hist[ticker][["Close"]].dropna()
            result[ticker] = df

    else:
        ticker = tickers[0]
        result[ticker] = hist[["Close"]].dropna()

    return result


# ============================================================
# Core
# ============================================================
def build_portfolio_and_benchmark(positions_df):
    overall_start = positions_df["Open date"].min()

    tickers = list(set(positions_df["Yahoo Symbol"].tolist() + [SP500_TICKER]))

    price_map = fetch_tickers_history_batch(
        tickers,
        start=overall_start,
        end=pd.Timestamp.today(),
    )

    calendar = price_map[SP500_TICKER].index

    portfolio = pd.Series(0.0, index=calendar)
    benchmark = pd.Series(0.0, index=calendar)

    for _, row in positions_df.iterrows():
        ticker = row["Yahoo Symbol"]
        open_date = row["Open date"]
        volume = row["Volume"]
        purchase = row["Purchase value"]

        prices = price_map[ticker]["Close"].reindex(calendar).ffill()
        component = prices * volume
        component[component.index < open_date] = 0

        portfolio += component

        spx = price_map[SP500_TICKER]["Close"]
        alloc_date = spx.index[spx.index >= open_date][0]
        units = purchase / spx.loc[alloc_date]

        bench = spx.reindex(calendar).ffill() * units
        bench[bench.index < alloc_date] = 0

        benchmark += bench

    df = pd.DataFrame({
        "Portfolio Value": portfolio,
        "S&P 500 Equivalent": benchmark,
    })

    return df


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

excel_file = Path(DEFAULT_EXCEL_PATH)

if not excel_file.exists():
    st.error("Excel file not found")
    st.stop()

positions_df = load_positions(str(excel_file))

history_df = build_portfolio_and_benchmark(positions_df)

sns.set_theme(style="white")

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

sns.lineplot(x=history_df.index, y=history_df["Portfolio Value"], ax=ax, label="Portfolio")
sns.lineplot(x=history_df.index, y=history_df["S&P 500 Equivalent"], ax=ax, label="S&P 500")

ax.set_title(APP_TITLE)
ax.set_xlabel("Date")
ax.set_ylabel("Value")

st.pyplot(fig, transparent=True)