import numpy as np
import pandas as pd

from trade_ai.indicator.atr import calculate_atr
from trade_ai.logger import logger

MAX_HOLDING_PERIOD = 60
# v3 replaces ATR barriers with fixed percentage barriers.
LABEL_SCHEMA_VERSION = 3
TAKE_PROFIT_PCT = 0.01
STOP_LOSS_PCT = 0.005


def add_absolute_label_atr(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Attach prior completed-bar ATR for future ATR-based label experiments.

    This helper is retained for optional experiments. The active trainers use
    fixed percentage barriers and do not call it.
    """
    label_atr = calculate_atr(source_df, period=period, relative=False).shift(1)
    atr_by_timestamp = pd.DataFrame(
        {
            "timestamp": source_df["timestamp"].to_numpy(),
            "ATR_14_1m_absolute": label_atr.to_numpy(),
        }
    )
    return feature_df.merge(
        atr_by_timestamp,
        on="timestamp",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def apply_triple_barrier(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    take_profit_pct: float = TAKE_PROFIT_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    max_holding_period: int = MAX_HOLDING_PERIOD,
    start_hour: int = 10,
    start_minute: int = 0,
    end_hour: int = 14,
    end_minute: int = 30,
    market_close_hour: int = 15,
    market_close_minute: int = 15,
) -> pd.Series:
    """Generates Triple Barrier Method labels (1: Win, -1: Loss, 0: Timeout)

    Entries are restricted to the configured window. The take-profit and
    stop-loss barriers are fixed percentages above and below the entry close.
    Each forward barrier search stops at the earlier of the holding limit or
    that trading session's close. Candle timestamps are assumed to be at the
    candle open.
    """
    df = df.copy()
    logger.info(
        "Starting Triple Barrier generation on %d rows. Time window: %02d:%02d to"
        " %02d:%02d. Barriers: TP +%.2f%%, SL -%.2f%%",
        len(df),
        start_hour,
        start_minute,
        end_hour,
        end_minute,
        take_profit_pct * 100,
        stop_loss_pct * 100,
    )

    # Bulletproof type coercion
    closes = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float)

    # Parse timestamps to extract time of day
    timestamps = pd.to_datetime(df[timestamp_col])
    if not timestamps.is_monotonic_increasing:
        raise ValueError("df must be sorted chronologically by timestamp")
    n = len(df)
    labels = np.full(n, np.nan)

    # Convert times once. Repeated Series.iloc and Timestamp property access in
    # the row loop dominates runtime for long histories.
    timestamp_dates = timestamps.dt.date.to_numpy()
    nanoseconds_per_second = 1_000_000_000
    time_of_day_ns = (
        timestamps.dt.hour.to_numpy(dtype=np.int64, na_value=0)
        * 3_600
        * nanoseconds_per_second
        + timestamps.dt.minute.to_numpy(dtype=np.int64, na_value=0)
        * 60
        * nanoseconds_per_second
        + timestamps.dt.second.to_numpy(dtype=np.int64, na_value=0)
        * nanoseconds_per_second
        + timestamps.dt.microsecond.to_numpy(dtype=np.int64, na_value=0) * 1_000
        + timestamps.dt.nanosecond.to_numpy(dtype=np.int64, na_value=0)
    )

    def time_to_ns(hour: int, minute: int) -> int:
        return (hour * 60 + minute) * 60 * nanoseconds_per_second

    trade_start_ns = time_to_ns(start_hour, start_minute)
    trade_end_ns = time_to_ns(end_hour, end_minute)
    market_close_ns = time_to_ns(market_close_hour, market_close_minute)
    entry_indices = np.flatnonzero(
        (time_of_day_ns >= trade_start_ns) & (time_of_day_ns <= trade_end_ns)
    )

    skipped_time = n - len(entry_indices)
    invalid_entry_prices = 0
    valid_labels = 0

    if max_holding_period < 1:
        raise ValueError("max_holding_period must be at least 1")
    if (
        not np.isfinite(take_profit_pct)
        or not np.isfinite(stop_loss_pct)
        or take_profit_pct <= 0
        or not 0 < stop_loss_pct < 1
    ):
        raise ValueError(
            "Barrier percentages must be finite and positive; "
            "stop loss must be below 100%"
        )

    for processed_entries, row_index in enumerate(entry_indices, start=1):
        if processed_entries % 50_000 == 0:
            logger.info(
                "Triple Barrier progress: %d/%d eligible rows processed",
                processed_entries,
                len(entry_indices),
            )

        i = int(row_index)
        entry_price = closes[i]

        # A non-positive or invalid entry cannot form percentage barriers.
        if not np.isfinite(entry_price) or entry_price <= 0:
            invalid_entry_prices += 1
            labels[i] = 0
            continue

        upper_barrier = entry_price * (1 + take_profit_pct)
        lower_barrier = entry_price * (1 - stop_loss_pct)

        # Bound the vertical barrier by both the holding period and this session.
        # The first candle at/after market close is not a valid intraday outcome.
        entry_date = timestamp_dates[i]
        future_end = i + 1
        max_future_idx = min(i + max_holding_period, n - 1)
        while future_end <= max_future_idx:
            if (
                timestamp_dates[future_end] != entry_date
                or time_of_day_ns[future_end] >= market_close_ns
            ):
                break
            future_end += 1

        future_window = closes[i + 1 : future_end]

        # Find first touch
        touched_tp = np.flatnonzero(future_window >= upper_barrier)
        touched_sl = np.flatnonzero(future_window <= lower_barrier)

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
        " - Invalid entry prices assigned timeout labels: %d rows",
        invalid_entry_prices,
    )

    return pd.Series(labels, index=df.index, name="target")
