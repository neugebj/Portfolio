from __future__ import annotations

import io
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
def load_excel_bytes_from_path(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False)
def find_open_positions_sheet(file_bytes: bytes) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
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
def load_positions(file_bytes: bytes) -> pd.DataFrame:
    sheet_name = find_open_positions_sheet(file_bytes)
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=10)
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


def fetch_tickers_history_batch(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    tickers = sorted(set(str(t).strip() for t in tickers if str(t).strip()))
    if not tickers:
        return {}

    hist = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if hist is None or hist.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if isinstance(hist.columns, pd.MultiIndex):
        first_level = hist.columns.get_level_values(0)

        for ticker in tickers:
            if ticker not in first_level:
                continue

            ticker_df = hist[ticker].copy()
            ticker_df.index = pd.to_datetime(ticker_df.index)

            if getattr(ticker_df.index, "tz", None) is not None:
                ticker_df.index = ticker_df.index.tz_localize(None)

            ticker_df = ticker_df.sort_index()

            if "Close" not in ticker_df.columns:
                continue

            ticker_df = ticker_df[["Close"]].copy()
            ticker_df["Close"] = pd.to_numeric(ticker_df["Close"], errors="coerce")
            ticker_df = ticker_df.dropna(subset=["Close"])

            if not ticker_df.empty:
                result[ticker] = ticker_df
    else:
        ticker = tickers[0]
        ticker_df = hist.copy()
        ticker_df.index = pd.to_datetime(ticker_df.index)

        if getattr(ticker_df.index, "tz", None) is not None:
            ticker_df.index = ticker_df.index.tz_localize(None)

        ticker_df = ticker_df.sort_index()

        if "Close" in ticker_df.columns:
            ticker_df = ticker_df[["Close"]].copy()
            ticker_df["Close"] = pd.to_numeric(ticker_df["Close"], errors="coerce")
            ticker_df = ticker_df.dropna(subset=["Close"])

            if not ticker_df.empty:
                result[ticker] = ticker_df

    return result


def get_price_histories(
    tickers: List[str],
    requested_start: pd.Timestamp,
    allow_download: bool = True,
) -> Dict[str, pd.Series]:
    tickers = sorted(set(str(t).strip() for t in tickers if str(t).strip()))
    requested_start = pd.Timestamp(requested_start).normalize()

    result: Dict[str, pd.Series] = {}
    cached_data: Dict[str, Optional[pd.DataFrame]] = {}

    for ticker in tickers:
        cached_data[ticker] = read_cached_close_frame(ticker)

    downloaded_map: Dict[str, pd.DataFrame] = {}

    if allow_download:
        fetch_starts = [
            get_incremental_fetch_start(cached_data[ticker], requested_start, overlap_days=7)
            for ticker in tickers
        ]
        batch_fetch_start = min(fetch_starts) if fetch_starts else requested_start
        batch_fetch_end = pd.Timestamp.now(tz=NY_TZ).tz_localize(None).normalize() + pd.Timedelta(days=2)

        downloaded_map = fetch_tickers_history_batch(
            tickers=tickers,
            start=batch_fetch_start,
            end=batch_fetch_end,
        )

    for ticker in tickers:
        frames = []

        existing = cached_data[ticker]
        if existing is not None and not existing.empty:
            frames.append(existing)

        downloaded = downloaded_map.get(ticker)
        if downloaded is not None and not downloaded.empty:
            frames.append(downloaded)

        if not frames:
            raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged[merged.index >= requested_start].copy()

        if merged.empty:
            raise ValueError(f"No usable price history available for '{ticker}'.")

        merged.to_parquet(cache_file_path(ticker))
        result[ticker] = merged["Close"].astype(float)

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


@st.cache_data(show_spinner=False)
def build_portfolio_and_benchmark_cached(
    positions_df: pd.DataFrame,
    allow_download: bool,
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
    instrument_series = []
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

        asset_component.name = f"{xtb_symbol} ({position_id})"
        bench_component.name = f"S&P for {xtb_symbol} ({position_id})"

        instrument_series.append(asset_component)
        instrument_series.append(bench_component)

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

    instrument_traces = pd.concat(instrument_series, axis=1)

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


@st.cache_data(show_spinner=False)
def render_plot_png(
    history_df: pd.DataFrame,
    metric_mode: str,
    show_portfolio: bool,
    show_benchmark: bool,
    text_color: str,
) -> bytes:
    value_mode = metric_mode == "Portfolio value"

    if value_mode:
        portfolio_series = history_df["Portfolio Value"]
        benchmark_series = history_df["S&P 500 Equivalent"]
        y_axis_title = "Value"
    else:
        portfolio_series = history_df["Portfolio Return (%)"]
        benchmark_series = history_df["S&P 500 Return (%)"]
        y_axis_title = "Return (%)"

    sns.set_theme(style="white")

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

    ax.set_title(f"{APP_TITLE} — {metric_mode}", color=text_color)
    ax.set_xlabel("Date", color=text_color)
    ax.set_ylabel(y_axis_title, color=text_color)
    ax.tick_params(colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(text_color)

    legend = ax.legend(title="Series")
    if legend is not None:
        legend.get_frame().set_alpha(0.0)
        plt.setp(legend.get_texts(), color=text_color)
        plt.setp(legend.get_title(), color=text_color)

    fig.autofmt_xdate()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


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
    "in batch with yf.download, caches them locally, and compares the portfolio with an S&P 500 investment "
    "made on each position's open date."
)

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload XTB Excel file",
        type=["xlsx", "xls"],
        help="Upload the XTB export containing a sheet whose name starts with 'OPEN POSITION'.",
    )

    metric_mode = st.radio(
        "Display mode",
        options=["Portfolio value", "Return (%)"],
        index=0,
    )

    show_portfolio = st.checkbox("Show portfolio", value=True)
    show_benchmark = st.checkbox("Show S&P 500 equivalent", value=True)

    refresh_prices = st.button("Refresh market data")
    clear_cache = st.button("Clear all caches")

if clear_cache:
    st.cache_data.clear()
    st.success("Streamlit cache cleared.")

file_bytes: Optional[bytes] = None
file_name_to_show = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name_to_show = uploaded_file.name
else:
    excel_file = Path(DEFAULT_EXCEL_PATH)
    if excel_file.exists():
        file_bytes = load_excel_bytes_from_path(str(excel_file))
        file_name_to_show = str(excel_file)
    else:
        st.error(
            "No file uploaded and default Excel file was not found. "
            f"Expected default file: {excel_file.resolve() if excel_file.is_absolute() else excel_file}"
        )
        st.stop()

st.sidebar.caption(f"Using file: {file_name_to_show}")

try:
    positions_df = load_positions(file_bytes)
except Exception as e:
    st.exception(e)
    st.stop()

if positions_df.empty:
    st.warning("No BUY/LONG open positions were found in the selected workbook.")
    st.stop()

try:
    with st.spinner("Building portfolio history...", show_time=True):
        history_df, instrument_df = build_portfolio_and_benchmark_cached(
            positions_df=positions_df,
            allow_download=refresh_prices,
        )
except Exception as e:
    st.exception(e)
    st.stop()

if history_df is None or history_df.empty:
    st.warning("No time series could be built from the available positions and market data.")
    st.stop()


# -------------------------------------------------
# Plot
# -------------------------------------------------
text_color = "white"

plot_png = render_plot_png(
    history_df=history_df,
    metric_mode=metric_mode,
    show_portfolio=show_portfolio,
    show_benchmark=show_benchmark,
    text_color=text_color,
)

st.image(plot_png, use_container_width=True)


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
    "If that date is not a US trading day, the next available S&P 500 close is used."
)