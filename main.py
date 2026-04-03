from __future__ import annotations

import time
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
    raise ValueError("No sheet starting with 'OPEN POSITION' was found in the workbook.")


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

    for col in ["Volume", "Open price", "Purchase value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open time", "Volume", "Open price", "Purchase value"])
    df = df[df["Volume"] > 0].copy()

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


def get_incremental_fetch_start(
    existing: Optional[pd.DataFrame],
    requested_start: pd.Timestamp,
    overlap_days: int = 7,
) -> pd.Timestamp:
    requested_start = pd.Timestamp(requested_start).normalize()

    if existing is None or existing.empty:
        return requested_start

    last_cached = pd.Timestamp(existing.index.max()).normalize()
    return min(requested_start, last_cached - pd.Timedelta(days=overlap_days))


def fetch_ticker_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pause_seconds: float = 0.6,
) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    time.sleep(pause_seconds)

    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    hist = hist.sort_index()

    if "Close" not in hist.columns:
        return pd.DataFrame()

    hist = hist[["Close"]].copy()
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist = hist.dropna(subset=["Close"])
    return hist


def download_and_update_cache_for_ticker(
    ticker: str,
    requested_start: pd.Timestamp,
    allow_download: bool = True,
) -> pd.Series:
    existing = read_cached_close_frame(ticker)
    requested_start = pd.Timestamp(requested_start).normalize()

    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)

    if allow_download:
        fetch_start = get_incremental_fetch_start(existing, requested_start, overlap_days=7)
        fetch_end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None).normalize() + pd.Timedelta(days=2)

        downloaded = fetch_ticker_history(
            ticker=ticker,
            start=fetch_start,
            end=fetch_end,
            pause_seconds=1,
        )

        if not downloaded.empty:
            frames.append(downloaded)

    if not frames:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged[merged.index >= requested_start].copy()

    if merged.empty:
        raise ValueError(f"No usable price history available for '{ticker}'.")

    merged.to_parquet(cache_file_path(ticker))
    return merged["Close"].astype(float)


def get_price_histories(
    tickers: List[str],
    requested_start: pd.Timestamp,
    allow_download: bool = True,
) -> Dict[str, pd.Series]:
    tickers = sorted(set(str(t).strip() for t in tickers if str(t).strip()))
    result: Dict[str, pd.Series] = {}

    for ticker in tickers:
        result[ticker] = download_and_update_cache_for_ticker(
            ticker=ticker,
            requested_start=requested_start,
            allow_download=allow_download,
        )

    return result


def align_on_or_after(series: pd.Series, target_date: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    target_date = pd.Timestamp(target_date).normalize()
    eligible = series[series.index >= target_date]
    if eligible.empty:
        raise ValueError(f"No benchmark price available on or after {target_date.date()} for allocation.")
    alloc_date = pd.Timestamp(eligible.index[0]).normalize()
    alloc_price = float(eligible.iloc[0])
    return alloc_date, alloc_price


def compute_relative_return_percent(value_series: pd.Series, invested_series: pd.Series) -> pd.Series:
    invested = invested_series.astype(float)
    value = value_series.astype(float)

    result = pd.Series(0.0, index=value.index, dtype=float)
    mask = invested > 0
    result.loc[mask] = (value.loc[mask] / invested.loc[mask] - 1.0) * 100.0
    return result


def format_metric_with_value(relative_pct: float, latest_value: float) -> Tuple[str, str]:
    return f"{relative_pct:,.2f}%", f"{latest_value:,.2f}"


# ============================================================
# Core build
# ============================================================
def build_portfolio_and_benchmark(
    positions_df: pd.DataFrame,
    allow_download: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_start = positions_df["Open date"].min().normalize()
    all_tickers = sorted(set(positions_df["Yahoo Symbol"].tolist() + [SP500_TICKER]))

    price_map = get_price_histories(
        tickers=all_tickers,
        requested_start=overall_start,
        allow_download=allow_download,
    )

    calendar_index = price_map[SP500_TICKER].index
    for series in price_map.values():
        calendar_index = calendar_index.union(series.index)
    calendar_index = calendar_index.sort_values()

    portfolio_value = pd.Series(0.0, index=calendar_index, dtype=float)
    benchmark_value = pd.Series(0.0, index=calendar_index, dtype=float)
    invested_capital = pd.Series(0.0, index=calendar_index, dtype=float)
    benchmark_invested_capital = pd.Series(0.0, index=calendar_index, dtype=float)

    instrument_traces = pd.DataFrame(index=calendar_index)
    allocation_rows: List[dict] = []

    spx_prices_full = price_map[SP500_TICKER].reindex(calendar_index).ffill()

    for _, row in positions_df.iterrows():
        position_id = str(row["Position"])
        xtb_symbol = str(row["Symbol"])
        ticker = str(row["Yahoo Symbol"])
        open_date = pd.Timestamp(row["Open date"]).normalize()
        volume = float(row["Volume"])
        purchase_value = float(row["Purchase value"])

        asset_prices = price_map[ticker].reindex(calendar_index).ffill()
        asset_component = asset_prices * volume
        asset_component.loc[asset_component.index < open_date] = 0.0
        portfolio_value = portfolio_value.add(asset_component, fill_value=0.0)

        invested_component = pd.Series(0.0, index=calendar_index, dtype=float)
        invested_component.loc[invested_component.index >= open_date] = purchase_value
        invested_capital = invested_capital.add(invested_component, fill_value=0.0)

        bench_alloc_date, bench_alloc_price = align_on_or_after(price_map[SP500_TICKER], open_date)
        bench_units = purchase_value / bench_alloc_price

        bench_component = spx_prices_full * bench_units
        bench_component.loc[bench_component.index < bench_alloc_date] = 0.0
        benchmark_value = benchmark_value.add(bench_component, fill_value=0.0)

        bench_invested_component = pd.Series(0.0, index=calendar_index, dtype=float)
        bench_invested_component.loc[bench_invested_component.index >= bench_alloc_date] = purchase_value
        benchmark_invested_capital = benchmark_invested_capital.add(bench_invested_component, fill_value=0.0)

        instrument_traces[f"{xtb_symbol} ({position_id})"] = asset_component
        instrument_traces[f"S&P for {xtb_symbol} ({position_id})"] = bench_component

        allocation_rows.append(
            {
                "Position": position_id,
                "XTB Symbol": xtb_symbol,
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
            "Portfolio Invested": invested_capital,
            "S&P 500 Invested": benchmark_invested_capital,
        }
    ).sort_index()

    result["Portfolio Return (%)"] = compute_relative_return_percent(
        result["Portfolio Value"],
        result["Portfolio Invested"],
    )
    result["S&P 500 Return (%)"] = compute_relative_return_percent(
        result["S&P 500 Equivalent"],
        result["S&P 500 Invested"],
    )

    allocation_df = pd.DataFrame(allocation_rows)
    return result, instrument_traces


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
    "Reads currently open positions from the XTB export, downloads daily Yahoo Finance closes "
    "ticker-by-ticker, caches them locally, and compares the portfolio with an S&P 500 investment "
    "made on each position's open date."
)

with st.sidebar:
    metric_mode = st.radio(
        "Display mode",
        options=["Portfolio value", "Return (%)"],
        index=0,
    )

    show_portfolio = st.checkbox("Show portfolio", value=True)
    show_benchmark = st.checkbox("Show S&P 500 equivalent", value=True)

    refresh_prices = st.button("Refresh market data")

excel_file = Path(DEFAULT_EXCEL_PATH)
if not excel_file.exists():
    st.error(f"Excel file was not found: {excel_file.resolve() if excel_file.is_absolute() else excel_file}")
    st.stop()

try:
    positions_df = load_positions(str(excel_file))
except Exception as e:
    st.exception(e)
    st.stop()

if positions_df.empty:
    st.warning("No BUY/LONG open positions were found in the selected workbook.")
    st.stop()

history_df = None
instrument_df = None

if refresh_prices:
    try:
        with st.spinner("Downloading and updating cached prices...", show_time=True):
            history_df, instrument_df = build_portfolio_and_benchmark(
                positions_df,
                allow_download=True,
            )
        st.session_state["history_df"] = history_df
        st.session_state["instrument_df"] = instrument_df
        st.success("Market data refreshed.")
    except Exception as e:
        st.exception(e)
        st.stop()
else:
    if "history_df" in st.session_state and "instrument_df" in st.session_state:
        history_df = st.session_state["history_df"]
        instrument_df = st.session_state["instrument_df"]
    else:
        st.info("Click 'Refresh market data' in the sidebar to load Yahoo prices.")
        st.stop()

if history_df is None or history_df.empty:
    st.warning("No time series could be built from the available positions and market data.")
    st.stop()

value_mode = metric_mode == "Portfolio value"

if value_mode:
    portfolio_series = history_df["Portfolio Value"]
    benchmark_series = history_df["S&P 500 Equivalent"]
    y_axis_title = "Value"
else:
    portfolio_series = history_df["Portfolio Return (%)"]
    benchmark_series = history_df["S&P 500 Return (%)"]
    y_axis_title = "Return (%)"

sns.set_theme(style="darkgrid")

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

if show_portfolio:
    sns.lineplot(
        x=history_df.index,
        y=portfolio_series,
        ax=ax,
        label="Portfolio",
    )

if show_benchmark:
    sns.lineplot(
        x=history_df.index,
        y=benchmark_series,
        ax=ax,
        label="S&P 500 equivalent",
    )

ax.set_title(f"{APP_TITLE} — {metric_mode}")
ax.set_xlabel("Date")
ax.set_ylabel(y_axis_title)

legend = ax.legend(title="Series")
if legend is not None:
    legend.get_frame().set_alpha(0.0)

fig.autofmt_xdate()
plt.tight_layout()

st.pyplot(fig, transparent=True)

# ============================================================
# Summary metrics
# ============================================================
c1, c2, c3, c4 = st.columns(4)

portfolio_invested = float(history_df["Portfolio Invested"].iloc[-1])
benchmark_invested = float(history_df["S&P 500 Invested"].iloc[-1])
portfolio_last = float(history_df["Portfolio Value"].iloc[-1])
benchmark_last = float(history_df["S&P 500 Equivalent"].iloc[-1])

portfolio_return_pct = float(history_df["Portfolio Return (%)"].iloc[-1]) if portfolio_invested > 0 else 0.0
benchmark_return_pct = float(history_df["S&P 500 Return (%)"].iloc[-1]) if benchmark_invested > 0 else 0.0

with c1:
    st.metric("Portfolio total invested", f"{portfolio_invested:,.2f}")

with c2:
    st.metric(
        "Portfolio latest value",
        f"{portfolio_last:,.2f}",
        delta=f"{portfolio_return_pct:,.2f}%",
    )

with c3:
    st.metric("S&P 500 total invested", f"{benchmark_invested:,.2f}")

with c4:
    st.metric(
        "S&P 500 latest value",
        f"{benchmark_last:,.2f}",
        delta=f"{benchmark_return_pct:,.2f}%",
    )



st.info(
    "Notes: Portfolio history is reconstructed from the currently open positions only. "
    "Return (%) is calculated relative to cumulative invested capital available at each date. "
    "The benchmark invests the same purchase value for each position into the S&P 500 at that day's close. "
    "If that date is not a US trading day, the next available S&P 500 close is used. "
    "In the instrument-level chart, click legend items to show or hide individual lines."
)