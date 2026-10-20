"""Shared data utilities used by training and inference."""

from pathlib import Path

import pandas as pd


_OHLCV_COLUMN_NAMES = {
    "timestamp": "Timestamp",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names and sort rows by timestamp.

    If the timestamp is stored in the index rather than a column, move it into
    a column before applying the canonical names used by feature processing.
    """
    result = df.copy()
    has_timestamp_column = any(
        str(column).casefold() == "timestamp" for column in result.columns
    )
    if not has_timestamp_column:
        result = result.reset_index()

    renamed_columns = {
        column: _OHLCV_COLUMN_NAMES.get(str(column).casefold(), str(column))
        for column in result.columns
    }
    result = result.rename(columns=renamed_columns)
    if "Timestamp" in result.columns:
        result = result.sort_values(by="Timestamp")
    return result.reset_index(drop=True)


def load_cached_ohlcv(
    csv_path: str | Path, standardize: bool = True
) -> pd.DataFrame | None:
    """Load an OHLCV CSV cache, standardizing its columns by default.

    Return ``None`` when the cache file does not exist. Set ``standardize`` to
    ``False`` to return the CSV data with its original column names and order.
    """
    cache_path = Path(csv_path)
    if not cache_path.is_file():
        return None

    df = pd.read_csv(cache_path)
    return standardize_ohlcv_columns(df) if standardize else df
