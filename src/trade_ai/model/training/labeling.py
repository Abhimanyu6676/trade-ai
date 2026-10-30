"""Shared direction-aware Triple Barrier label generation."""

import numpy as np
import pandas as pd

from trade_ai.config.trading_config import (
    ENTRY_END_HOUR,
    ENTRY_END_MINUTE,
    ENTRY_START_HOUR,
    ENTRY_START_MINUTE,
    LONG_STOP_LOSS_PCT,
    LONG_TAKE_PROFIT_PCT,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MAX_HOLDING_PERIOD,
    SHORT_STOP_LOSS_PCT,
    SHORT_TAKE_PROFIT_PCT,
    Direction,
)
from trade_ai.logger import logger

LABEL_SCHEMA_VERSION = 4


def apply_triple_barrier(
    df: pd.DataFrame,
    direction: Direction = "long",
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    max_holding_period: int = MAX_HOLDING_PERIOD,
    start_hour: int = ENTRY_START_HOUR,
    start_minute: int = ENTRY_START_MINUTE,
    end_hour: int = ENTRY_END_HOUR,
    end_minute: int = ENTRY_END_MINUTE,
    market_close_hour: int = MARKET_CLOSE_HOUR,
    market_close_minute: int = MARKET_CLOSE_MINUTE,
) -> pd.Series:
    """Label whether directional TP or SL is reached first within a session.

    Classes remain trade-relative for both directions: ``1`` means take profit,
    ``-1`` means stop loss, and ``0`` means timeout. Candle timestamps are
    assumed to be at candle open, and future closes are checked only up to the
    configured holding period and market close.
    """
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    if max_holding_period < 1:
        raise ValueError("max_holding_period must be at least 1")

    if direction == "long":
        tp_distance = LONG_TAKE_PROFIT_PCT if take_profit_pct is None else take_profit_pct
        sl_distance = LONG_STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
    else:
        tp_distance = (
            abs(SHORT_TAKE_PROFIT_PCT) if take_profit_pct is None else abs(take_profit_pct)
        )
        sl_distance = SHORT_STOP_LOSS_PCT if stop_loss_pct is None else abs(stop_loss_pct)
    if (
        not np.isfinite(tp_distance)
        or not np.isfinite(sl_distance)
        or tp_distance <= 0
        or not 0 < sl_distance < 1
    ):
        raise ValueError("Barrier percentages must be finite and within valid ranges")

    df = df.copy()
    closes = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamps contain values that cannot be parsed")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("df must be sorted chronologically by timestamp")

    logger.info(
        "Starting %s Triple Barrier labels on %d rows (TP %.2f%%, SL %.2f%%).",
        direction,
        len(df),
        tp_distance * 100,
        sl_distance * 100,
    )
    n = len(df)
    labels = np.full(n, np.nan)
    dates = timestamps.dt.date.to_numpy()
    seconds_in_day = (
        timestamps.dt.hour.to_numpy(dtype=np.int64, na_value=0) * 3600
        + timestamps.dt.minute.to_numpy(dtype=np.int64, na_value=0) * 60
        + timestamps.dt.second.to_numpy(dtype=np.int64, na_value=0)
    )

    start_seconds = start_hour * 3600 + start_minute * 60
    end_seconds = end_hour * 3600 + end_minute * 60
    close_seconds = market_close_hour * 3600 + market_close_minute * 60
    entry_indices = np.flatnonzero(
        (seconds_in_day >= start_seconds) & (seconds_in_day <= end_seconds)
    )
    invalid_entry_prices = 0

    for processed_entries, row_index in enumerate(entry_indices, start=1):
        if processed_entries % 50_000 == 0:
            logger.info(
                "%s label progress: %d/%d eligible rows processed",
                direction,
                processed_entries,
                len(entry_indices),
            )
        i = int(row_index)
        entry = closes[i]
        if not np.isfinite(entry) or entry <= 0:
            labels[i] = 0
            invalid_entry_prices += 1
            continue

        if direction == "long":
            take_profit = entry * (1 + tp_distance)
            stop_loss = entry * (1 - sl_distance)
        else:
            take_profit = entry * (1 - tp_distance)
            stop_loss = entry * (1 + sl_distance)

        end = min(i + max_holding_period, n - 1) + 1
        while end > i + 1 and (
            dates[end - 1] != dates[i] or seconds_in_day[end - 1] >= close_seconds
        ):
            end -= 1
        future = closes[i + 1 : end]
        if direction == "long":
            tp_hits = np.flatnonzero(future >= take_profit)
            sl_hits = np.flatnonzero(future <= stop_loss)
        else:
            tp_hits = np.flatnonzero(future <= take_profit)
            sl_hits = np.flatnonzero(future >= stop_loss)
        first_tp = int(tp_hits[0]) if len(tp_hits) else np.iinfo(np.int64).max
        first_sl = int(sl_hits[0]) if len(sl_hits) else np.iinfo(np.int64).max
        if first_tp < first_sl:
            labels[i] = 1
        elif first_sl < first_tp:
            labels[i] = -1
        else:
            labels[i] = 0

    finite_labels = labels[np.isfinite(labels)]
    counts = {int(value): int((finite_labels == value).sum()) for value in (-1, 0, 1)}
    logger.info(
        "%s labels complete: count=%d classes=%s invalid_entries=%d",
        direction,
        len(finite_labels),
        counts,
        invalid_entry_prices,
    )
    return pd.Series(labels, index=df.index, name="target")
