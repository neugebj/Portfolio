from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# Configuration
# ============================================================
APP_TITLE = "XTB Open Positions vs S&P 500"
DEFAULT_EXCEL_PATH = "xtb_transactions.xlsx"
CACHE_DIR = Path("cache_price_data")
CACHE_DIR.mkdir(exist_ok=True)

# XTB suffix -> Yahoo Finance suffix
# Extend this dictionary if your broker uses other market suffixes.
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
    "CZ": "PR",   # Prague (common Yahoo suffix)
}

SP500_TICKER = "^GSPC"
NY_TZ = ZoneInfo("America/New_York")
PRAGUE_TZ = ZoneInfo("Europe/Prague")


# ============================================================
# Data classes
# ============================================================
@dataclass
class Position:
    position_id: str
    xtb_symbol: str
    yahoo_symbol: str
    side: str
    volume: float
    open_time: pd.Timestamp
    open_price: float
    purchase_value: float


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
    raise ValueError(
        "No sheet starting with 'OPEN POSITION' was found in the workbook."
    )


def xtb_to_yahoo_symbol(symbol: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    symbol = str(symbol).strip().upper()
    suffix_map = dict(XTB_TO_YAHOO_SUFFIX)
    if custom_map:
        suffix_map.update({str(k).upper(): str(v).upper() for k, v in custom_map.items()})

    if "." not in symbol:
        return symbol

    base, suffix = symbol.rsplit(".", 1)
    yahoo_suffix = suffix_map.get(suffix.upper(), suffix.upper())
    return f"{base}.{yahoo_suffix}"


@st.cache_data(show_spinner=False)
def load_positions(excel_path: str, custom_map_json: str) -> pd.DataFrame:
    sheet_name = find_open_positions_sheet(excel_path)
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=10)
    raw = normalize_columns(raw)

    required = [
        "Position",
        "Symbol",
        "Type",
        "Volume",
        "Open time",
        "Open price",
        "Purchase value",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Excel sheet: {missing}")

    df = raw[required].copy()
    df = df[df["Symbol"].notna()].copy()
    df = df[df["Volume"].notna()].copy()

    df["Type"] = df["Type"].astype(str).str.upper().str.strip()
    df = df[df["Type"].isin(["BUY", "LONG"])]

    df["Open time"] = pd.to_datetime(df["Open time"], dayfirst=True, errors="coerce")
    numeric_cols = ["Volume", "Open price", "Purchase value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open time", "Volume", "Open price", "Purchase value"])
    df = df[df["Volume"] > 0].copy()

    try:
        custom_map = json.loads(custom_map_json) if custom_map_json.strip() else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Suffix mapping JSON is invalid: {e}") from e

    df["Yahoo Symbol"] = df["Symbol"].apply(lambda x: xtb_to_yahoo_symbol(x, custom_map))
    df["Open date"] = df["Open time"].dt.normalize()

    return df.reset_index(drop=True)


def cache_file_path(ticker: str) -> Path:
    safe = ticker.replace("^", "__index__").replace("/", "_")
    return CACHE_DIR / f"{safe}.parquet"



def get_incremental_download_start(existing: Optional[pd.DataFrame], requested_start: pd.Timestamp) -> pd.Timestamp:
    if existing is None or existing.empty:
        return requested_start

    last_cached = pd.Timestamp(existing.index.max()).normalize()
    overlap_start = last_cached - pd.Timedelta(days=7)
    return min(overlap_start, requested_start) if overlap_start < requested_start else overlap_start



def download_daily_close_history(ticker: str, start_date: pd.Timestamp) -> pd.Series:
    cache_path = cache_file_path(ticker)
    existing: Optional[pd.DataFrame] = None

    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            if not isinstance(existing.index, pd.DatetimeIndex):
                existing.index = pd.to_datetime(existing.index)
            existing.index = existing.index.tz_localize(None)
            existing = existing.sort_index()
        except Exception:
            existing = None

    fetch_start = get_incremental_download_start(existing, start_date.normalize())

    # A small forward buffer is safe. yfinance will only return available daily bars.
    fetch_end = (pd.Timestamp.now(tz=NY_TZ).tz_localize(None) + pd.Timedelta(days=2)).date()

    downloaded = yf.download(
        ticker,
        start=fetch_start.date().isoformat(),
        end=(fetch_end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )

    if downloaded.empty and existing is None:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)

    if not downloaded.empty:
        downloaded = downloaded.copy()
        if isinstance(downloaded.columns, pd.MultiIndex):
            downloaded.columns = downloaded.columns.get_level_values(0)
        downloaded.index = pd.to_datetime(downloaded.index).tz_localize(None)
        downloaded = downloaded.sort_index()
        if "Close" not in downloaded.columns:
            raise ValueError(f"Downloaded data for '{ticker}' does not contain a Close column.")
        frames.append(downloaded[["Close"]])

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[merged.index >= start_date.normalize()].copy()

    if merged.empty:
        raise ValueError(f"No usable price history available for '{ticker}'.")

    merged.to_parquet(cache_path)
    return merged["Close"].astype(float)


@st.cache_data(show_spinner=False)
def get_price_history_cached(ticker: str, start_date_str: str) -> pd.Series:
    return download_daily_close_history(ticker, pd.Timestamp(start_date_str))



def align_on_or_after(series: pd.Series, target_date: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    target_date = pd.Timestamp(target_date).normalize()
    eligible = series[series.index >= target_date]
    if eligible.empty:
        raise ValueError(
            f"No benchmark price available on or after {target_date.date()} for allocation."
        )
    alloc_date = pd.Timestamp(eligible.index[0]).normalize()
    alloc_price = float(eligible.iloc[0])
    return alloc_date, alloc_price



def build_portfolio_and_benchmark(positions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_start = positions_df["Open date"].min().normalize()

    all_tickers = sorted(set(positions_df["Yahoo Symbol"].tolist() + [SP500_TICKER]))
    price_map: Dict[str, pd.Series] = {}

    for ticker in all_tickers:
        price_map[ticker] = get_price_history_cached(ticker, overall_start.date().isoformat())

    calendar_index = price_map[SP500_TICKER].index
    for ticker, series in price_map.items():
        calendar_index = calendar_index.union(series.index)
    calendar_index = calendar_index.sort_values()

    portfolio_value = pd.Series(0.0, index=calendar_index)
    benchmark_value = pd.Series(0.0, index=calendar_index)

    allocation_rows: List[dict] = []

    for _, row in positions_df.iterrows():
        ticker = row["Yahoo Symbol"]
        open_date = pd.Timestamp(row["Open date"]).normalize()
        volume = float(row["Volume"])
        purchase_value = float(row["Purchase value"])

        asset_prices = price_map[ticker].reindex(calendar_index).ffill()
        asset_component = asset_prices * volume
        asset_component.loc[asset_component.index < open_date] = 0.0
        portfolio_value = portfolio_value.add(asset_component, fill_value=0.0)

        bench_alloc_date, bench_alloc_price = align_on_or_after(price_map[SP500_TICKER], open_date)
        bench_units = purchase_value / bench_alloc_price
        bench_prices = price_map[SP500_TICKER].reindex(calendar_index).ffill()
        bench_component = bench_prices * bench_units
        bench_component.loc[bench_component.index < bench_alloc_date] = 0.0
        benchmark_value = benchmark_value.add(bench_component, fill_value=0.0)

        allocation_rows.append(
            {
                "Position": row["Position"],
                "XTB Symbol": row["Symbol"],
                "Yahoo Symbol": ticker,
                "Open date": open_date.date().isoformat(),
                "Purchase value": purchase_value,
                "Benchmark allocation date": bench_alloc_date.date().isoformat(),
                "Benchmark allocation close": round(bench_alloc_price, 6),
                "Benchmark units": bench_units,
            }
        )

    result = pd.DataFrame(
        {
            "Portfolio Value": portfolio_value,
            "S&P 500 Equivalent": benchmark_value,
        }
    ).sort_index()

    result = result.dropna(how="all")
    result["Portfolio Return"] = result["Portfolio Value"] - result["Portfolio Value"].replace(0, pd.NA).ffill().fillna(0).iloc[0]
    result["S&P 500 Return"] = result["S&P 500 Equivalent"] - result["S&P 500 Equivalent"].replace(0, pd.NA).ffill().fillna(0).iloc[0]

    allocation_df = pd.DataFrame(allocation_rows)
    return result, allocation_df



def first_non_zero(series: pd.Series) -> float:
    non_zero = series[series != 0]
    if non_zero.empty:
        return 0.0
    return float(non_zero.iloc[0])



def add_trace(fig: go.Figure, x, y, name: str, visible: bool = True) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            visible=True if visible else "legendonly",
        )
    )


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        [data-testid="stSidebar"] {
            background-color: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(
    "Reads currently open positions from the XTB export, downloads daily Yahoo Finance closes, "
    "caches them locally, and compares the portfolio with an S&P 500 investment made on each position's open date."
)

with st.sidebar:
    st.header("Settings")
    excel_path = st.text_input("Excel file path", value=DEFAULT_EXCEL_PATH)
    sp500_ticker = st.text_input("Benchmark ticker", value=SP500_TICKER, disabled=True)
    metric_mode = st.radio(
        "Display mode",
        options=["Portfolio value", "Money return"],
        index=0,
    )
    show_portfolio = st.checkbox("Show portfolio", value=True)
    show_benchmark = st.checkbox("Show S&P 500 equivalent", value=True)
    custom_map_text = st.text_area(
        "Optional suffix mapping overrides (JSON)",
        value='{"BE": "BR"}',
        height=120,
        help="Example: {\n  \"BE\": \"BR\",\n  \"DE\": \"DE\"\n}",
    )
    refresh_note = st.caption(
        "Historical prices are cached in ./cache_price_data. "
        "Each run downloads only a short overlap window after the last cached date and then merges it."
    )

excel_file = Path(excel_path)
if not excel_file.exists():
    st.error(f"Excel file was not found: {excel_file.resolve() if excel_file.is_absolute() else excel_file}")
    st.stop()

try:
    positions_df = load_positions(str(excel_file), custom_map_text)
except Exception as e:
    st.exception(e)
    st.stop()

if positions_df.empty:
    st.warning("No BUY/LONG open positions were found in the selected workbook.")
    st.stop()

with st.spinner("Downloading and updating cached prices..."):
    try:
        history_df, allocation_df = build_portfolio_and_benchmark(positions_df)
    except Exception as e:
        st.exception(e)
        st.stop()

if history_df.empty:
    st.warning("No time series could be built from the available positions and market data.")
    st.stop()

value_mode = metric_mode == "Portfolio value"
portfolio_series = history_df["Portfolio Value"] if value_mode else history_df["Portfolio Return"]
benchmark_series = history_df["S&P 500 Equivalent"] if value_mode else history_df["S&P 500 Return"]

y_axis_title = "Value" if value_mode else "Money return"

fig = go.Figure()
if show_portfolio:
    add_trace(fig, history_df.index, portfolio_series, "Portfolio", visible=True)
if show_benchmark:
    add_trace(fig, history_df.index, benchmark_series, "S&P 500 equivalent", visible=True)

fig.update_layout(
    template="plotly_dark",
    height=650,
    title=f"{APP_TITLE} — {metric_mode}",
    xaxis_title="Date",
    yaxis_title=y_axis_title,
    hovermode="x unified",
    legend_title="Series",
)

st.plotly_chart(fig, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
portfolio_start = first_non_zero(history_df["Portfolio Value"])
benchmark_start = first_non_zero(history_df["S&P 500 Equivalent"])
portfolio_last = float(history_df["Portfolio Value"].iloc[-1])
benchmark_last = float(history_df["S&P 500 Equivalent"].iloc[-1])

with c1:
    st.metric("Portfolio start value", f"{portfolio_start:,.2f}")
with c2:
    st.metric("Portfolio latest value", f"{portfolio_last:,.2f}", delta=f"{portfolio_last - portfolio_start:,.2f}")
with c3:
    st.metric("S&P 500 start value", f"{benchmark_start:,.2f}")
with c4:
    st.metric("S&P 500 latest value", f"{benchmark_last:,.2f}", delta=f"{benchmark_last - benchmark_start:,.2f}")

st.subheader("Open positions loaded from XTB")
display_positions = positions_df[[
    "Position",
    "Symbol",
    "Yahoo Symbol",
    "Type",
    "Volume",
    "Open time",
    "Open price",
    "Purchase value",
]].copy()
st.dataframe(display_positions, use_container_width=True)

st.subheader("Benchmark allocation details")
st.dataframe(allocation_df, use_container_width=True)

st.info(
    "Notes: Portfolio history is reconstructed from the currently open positions only. "
    "The benchmark invests the same purchase value for each position into the S&P 500 at that day's close. "
    "If that date is not a US trading day, the next available S&P 500 close is used."
)
