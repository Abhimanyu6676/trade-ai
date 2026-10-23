import logging
from datetime import time

import numpy as np
import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def apply_triple_barrier(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    atr_col: str = "ATR_14_1m",
    tp_multiplier: float = 2.0,
    sl_multiplier: float = 1.0,
    max_holding_period: int = 60,
    start_hour: int = 10,
    start_minute: int = 0,
    end_hour: int = 14,
    end_minute: int = 30,
    market_close_hour: int = 15,
    market_close_minute: int = 15,
) -> pd.Series:
    """Generates Triple Barrier Method labels (1: Win, -1: Loss, 0: Timeout)

    Entries are restricted to the configured window. Each forward barrier search
    stops at the earlier of the holding limit or that trading session's close.
    Candle timestamps are assumed to be candle-open timestamps.
    """
    df = df.copy()
    logger.info(
        "Starting Triple Barrier generation on %d rows. Time window: %02d:%02d to"
        " %02d:%02d",
        len(df),
        start_hour,
        start_minute,
        end_hour,
        end_minute,
    )

    # Bulletproof type coercion
    closes = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float)
    atrs = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=float)

    # Parse timestamps to extract time of day
    timestamps = pd.to_datetime(df[timestamp_col])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("df must be sorted chronologically by timestamp")
    times = timestamps.dt.time

    # Define trading window boundaries using standard library 'time'
    trade_start = time(start_hour, start_minute)
    trade_end = time(end_hour, end_minute)
    market_close = time(market_close_hour, market_close_minute)

    n = len(df)
    labels = np.full(n, np.nan)

    skipped_time = 0
    skipped_nan = 0
    valid_labels = 0

    if max_holding_period < 1:
        raise ValueError("max_holding_period must be at least 1")

    for i in range(n):
        # 1. Enforce time-of-day filter
        current_time = times.iloc[i]
        if not (trade_start <= current_time <= trade_end):
            skipped_time += 1
            continue

        entry_price = closes[i]
        atr = atrs[i]

        # Handle cases where ATR, price, or coerced value is NaN
        if np.isnan(atr) or np.isnan(entry_price) or atr == 0:
            skipped_nan += 1
            labels[i] = 0
            continue

        upper_barrier = entry_price + (atr * tp_multiplier)
        lower_barrier = entry_price - (atr * sl_multiplier)

        # Bound the vertical barrier by both the holding period and this session.
        # The first candle at/after market close is not a valid intraday outcome.
        entry_date = timestamps.iloc[i].date()
        future_end = i + 1
        max_future_idx = min(i + max_holding_period, n - 1)
        while future_end <= max_future_idx:
            future_timestamp = timestamps.iloc[future_end]
            if (
                future_timestamp.date() != entry_date
                or future_timestamp.time() >= market_close
            ):
                break
            future_end += 1

        future_window = closes[i + 1 : future_end]

        # Find first touch
        touched_tp = np.where(future_window >= upper_barrier)[0]
        touched_sl = np.where(future_window <= lower_barrier)[0]

        first_tp = touched_tp[0] if len(touched_tp) > 0 else float("inf")
        first_sl = touched_sl[0] if len(touched_sl) > 0 else float("inf")

        if first_tp < first_sl:
            labels[i] = 1  # Take profit hit first
        elif first_sl < first_tp:
            labels[i] = -1  # Stop loss hit first
        else:
            labels[i] = 0  # Time expired (vertical barrier)

        valid_labels += 1

    # Count label distribution for summary logs
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    label_counts = dict(zip(unique, counts))

    logger.info("Triple Barrier generation complete.")
    logger.info(
        " - Valid labels generated: %d (Wins [1]: %d, Losses [-1]: %d,"
        " Timeouts [0]: %d)",
        valid_labels,
        label_counts.get(1, 0),
        label_counts.get(-1, 0),
        label_counts.get(0, 0),
    )
    logger.info(
        " - Skipped (outside time window): %d rows",
        skipped_time,
    )
    logger.info(
        " - Skipped (NaN/Zero ATR or Price): %d rows",
        skipped_nan,
    )

    return pd.Series(labels, index=df.index, name="target")
