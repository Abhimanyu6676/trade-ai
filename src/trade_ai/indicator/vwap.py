"""Session VWAP calculations for intraday OHLCV candles."""

from datetime import time

import numpy as np
import pandas as pd


def calculate_vwap(
    bars: pd.DataFrame,
    *,
    roc_period: int = 5,
    session_open: str = "09:15",
    source_timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Return price-relative VWAP distance and its session-local rate of change.

    ``bars`` must be indexed by candle-open timestamp and contain High, Low,
    Close, and Volume columns. VWAP resets for each local calendar date. The
    rate of change compares VWAP with its value ``roc_period`` candles earlier
    in the same session.
    """
    if roc_period <= 0:
        raise ValueError("roc_period must be greater than zero")

    timestamps = pd.DatetimeIndex(pd.to_datetime(bars.index))
    if timestamps.tz is None:
        local_timestamps = timestamps.tz_localize(source_timezone)
    else:
        local_timestamps = timestamps.tz_convert(source_timezone)

    typical_price = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    volume = pd.to_numeric(bars["Volume"], errors="coerce").clip(lower=0)
    market_open = time.fromisoformat(session_open)
    during_session = np.asarray(
        [timestamp.time() >= market_open for timestamp in local_timestamps],
        dtype=bool,
    )
    volume = volume.where(during_session, 0.0)
    weighted_price = typical_price * volume
    session_dates = pd.Index(local_timestamps.date, name="session_date")

    cumulative_value = weighted_price.groupby(session_dates, sort=False).cumsum()
    cumulative_volume = volume.groupby(session_dates, sort=False).cumsum()
    vwap = cumulative_value.div(cumulative_volume.where(cumulative_volume > 0))
    vwap_previous = vwap.groupby(session_dates, sort=False).shift(roc_period)

    close = pd.to_numeric(bars["Close"], errors="coerce")
    output = pd.DataFrame(index=bars.index)
    output["VWAP_Dist"] = close.div(vwap).sub(1.0)
    output["VWAP_ROC_5"] = vwap.div(vwap_previous).sub(1.0)
    output = output.replace([np.inf, -np.inf], np.nan)
    return output
