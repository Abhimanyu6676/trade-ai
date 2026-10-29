"""Shared data utilities used by training and inference."""

from pathlib import Path
from typing import Any

import pandas as pd

from trade_ai.logger import logger

_OHLCV_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def lowercase_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename timestamp and OHLCV columns to lowercase; leave other columns intact."""
    column_map = {
        column: str(column).casefold()
        for column in df.columns
        if str(column).casefold() in _OHLCV_COLUMNS
    }
    return df.rename(columns=column_map)


def merge_ohlcv_history(
    history_df: pd.DataFrame,
    latest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge historical candles followed by newer candles in timestamp order.

    If both inputs contain the same timestamp, the later input wins. With one
    input, existing row order breaks timestamp ties, so appended/live rows win.
    """
    frames = [
        frame
        for frame in (history_df, latest_df)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return lowercase_ohlcv_columns(history_df.iloc[0:0])

    combined = lowercase_ohlcv_columns(pd.concat(frames))
    if "timestamp" not in combined.columns:
        index_name = str(combined.index.name).casefold() if combined.index.name else ""
        if isinstance(combined.index, pd.DatetimeIndex) or index_name in {
            "timestamp",
            "datetime",
            "date",
            "time",
        }:
            combined = combined.rename_axis("timestamp").reset_index()
        else:
            raise ValueError("OHLCV data must include a timestamp column or index")

    timestamps = pd.to_datetime(combined["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("OHLCV data contains invalid timestamps")
    combined["timestamp"] = timestamps
    return (
        combined.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )


def candles_to_dataframe(candles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert candle mappings to normalized, chronological OHLCV rows."""
    df = pd.DataFrame(candles)
    timestamps = []
    for timestamp in df["timestamp"]:
        if isinstance(timestamp, (int, float)):
            unit = "ms" if abs(timestamp) >= 100_000_000_000 else "s"
            timestamps.append(pd.to_datetime(timestamp, unit=unit, errors="raise"))
        else:
            timestamps.append(pd.to_datetime(timestamp, errors="raise"))
    df["timestamp"] = timestamps
    if (
        not df["timestamp"].is_monotonic_increasing
        or df["timestamp"].duplicated().any()
    ):
        logger.warning(
            "Supplied candles are not chronologically valid "
            "(first=%s, last=%s, rows=%s); sorting and duplicate removal "
            "will be applied",
            df["timestamp"].iloc[0],
            df["timestamp"].iloc[-1],
            len(df),
        )
    return merge_ohlcv_history(df)


def load_cached_ohlcv(csv_path: str | Path) -> pd.DataFrame | None:
    """Load an OHLCV CSV cache with lowercase timestamp and OHLCV columns."""
    cache_path = Path(csv_path)
    if not cache_path.is_file():
        return None

    df = pd.read_csv(cache_path)
    return lowercase_ohlcv_columns(df)


def _normalize_resample_rule(rule: str) -> str:
    """Use pandas' unambiguous ``min`` alias for numeric minute rules."""
    if rule.endswith("m") and rule[:-1].isdecimal():
        return f"{rule[:-1]}min"
    return rule


def resample_ohlcv(
    df: pd.DataFrame, rule: str, offset: str | None = None
) -> pd.DataFrame:
    """Aggregate candle-open OHLCV rows into completed bars labeled at close."""

    rule = _normalize_resample_rule(rule)
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp["timestamp"]):
        df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
    if not df_temp["timestamp"].is_monotonic_increasing:
        df_temp = df_temp.sort_values("timestamp")
    df_temp = df_temp.set_index("timestamp")

    if offset is not None:
        return pd.DataFrame(
            {
                "open": df_temp["open"]
                .resample(rule, label="right", closed="left", offset=offset)
                .first(),
                "high": df_temp["high"]
                .resample(rule, label="right", closed="left", offset=offset)
                .max(),
                "low": df_temp["low"]
                .resample(rule, label="right", closed="left", offset=offset)
                .min(),
                "close": df_temp["close"]
                .resample(rule, label="right", closed="left", offset=offset)
                .last(),
                "volume": df_temp["volume"]
                .resample(rule, label="right", closed="left", offset=offset)
                .sum(),
            }
        ).dropna().reset_index()

    return pd.DataFrame(
        {
            "open": df_temp["open"]
            .resample(rule, label="right", closed="left")
            .first(),
            "high": df_temp["high"].resample(rule, label="right", closed="left").max(),
            "low": df_temp["low"].resample(rule, label="right", closed="left").min(),
            "close": df_temp["close"]
            .resample(rule, label="right", closed="left")
            .last(),
            "volume": df_temp["volume"]
            .resample(rule, label="right", closed="left")
            .sum(),
        }
    ).dropna().reset_index()


def resample_close(series: pd.Series, rule: str, offset: str | None = None):
    """Return completed-bar closes, optionally aligned to a session offset."""
    rule = _normalize_resample_rule(rule)
    if offset is None:
        return series.resample(rule, label="right", closed="left").last().dropna()
    return (
        series.resample(rule, label="right", closed="left", offset=offset)
        .last()
        .dropna()
    )
