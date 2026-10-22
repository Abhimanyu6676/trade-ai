"""Opening Range Breakout features for intraday candle data."""

from datetime import date, time

import numpy as np
import pandas as pd


def calculate_orb(
    bars: pd.DataFrame,
    opening_minutes: int,
    *,
    window_label: str | None = None,
    session_open: str = "09:15",
    source_timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Calculate session opening-range features without using unfinished ranges.

    ``bars`` must contain one-minute OHLC candles indexed by candle-open time.
    The opening range uses bars from the session open up to (but excluding) the
    window end. Its values become available only at or after that end time.
    """
    if opening_minutes <= 0:
        raise ValueError("opening_minutes must be greater than zero")

    timestamps = pd.DatetimeIndex(pd.to_datetime(bars.index))
    if timestamps.tz is None:
        local_timestamps = timestamps.tz_localize(source_timezone)
    else:
        local_timestamps = timestamps.tz_convert(source_timezone)

    label = window_label or f"{opening_minutes}m"
    prefix = f"ORB_{label}"
    feature_names = (
        f"{prefix}_Range_Pct",
        f"{prefix}_High_Distance_Pct",
        f"{prefix}_Low_Distance_Pct",
        f"{prefix}_Midpoint_Distance_Pct",
        f"{prefix}_Breakout_Direction",
    )
    output = {name: np.zeros(len(bars), dtype=float) for name in feature_names}
    if bars.empty:
        return pd.DataFrame(output, index=bars.index)

    open_time = time.fromisoformat(session_open)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    session_positions: dict[date, list[int]] = {}
    for index, session_date in enumerate(local_timestamps.date):
        session_positions.setdefault(session_date, []).append(index)

    for session_date, positions_list in session_positions.items():
        positions = np.asarray(positions_list, dtype=int)
        session_timestamps = local_timestamps[positions]
        session_start = pd.timestamp.combine(session_date, open_time).tz_localize(
            source_timezone
        )
        range_end = session_start + pd.Timedelta(minutes=opening_minutes)
        opening_positions = positions[
            (session_timestamps >= session_start) & (session_timestamps < range_end)
        ]
        after_positions = positions[session_timestamps >= range_end]
        expected_opening_times = pd.date_range(
            start=session_start, periods=opening_minutes, freq="min"
        )
        if (
            len(opening_positions) != opening_minutes
            or not local_timestamps[opening_positions].equals(expected_opening_times)
            or len(after_positions) == 0
        ):
            continue

        range_high = float(high[opening_positions].max())
        range_low = float(low[opening_positions].min())
        range_midpoint = (range_high + range_low) / 2.0
        if range_midpoint <= 0:
            continue

        output[f"{prefix}_Range_Pct"][after_positions] = (
            range_high - range_low
        ) / range_midpoint
        valid_positions = after_positions[close[after_positions] > 0]
        valid_close = close[valid_positions]
        output[f"{prefix}_High_Distance_Pct"][valid_positions] = (
            range_high - valid_close
        ) / valid_close
        output[f"{prefix}_Low_Distance_Pct"][valid_positions] = (
            range_low - valid_close
        ) / valid_close
        output[f"{prefix}_Midpoint_Distance_Pct"][valid_positions] = (
            valid_close - range_midpoint
        ) / valid_close
        breakout_direction = output[f"{prefix}_Breakout_Direction"]
        breakout_direction[valid_positions[valid_close > range_high]] = 1.0
        breakout_direction[valid_positions[valid_close < range_low]] = -1.0

    return pd.DataFrame(output, index=bars.index)
