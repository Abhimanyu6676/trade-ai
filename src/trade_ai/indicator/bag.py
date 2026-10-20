"""Breakaway Gap (BAG) features for OHLC candles."""

import numpy as np
import pandas as pd


def calculate_bag(bars: pd.DataFrame, breakout_period: int = 20) -> pd.DataFrame:
    """Return the latest unfilled breakout gap state for each bar.

    A bullish breakaway gap requires the current low to be above the previous
    bar's high and the current close to exceed the prior rolling high. The
    bearish case is symmetric. A gap remains active until price fills its full
    zone. Width and signed distance from close to the midpoint are close ratios.
    """
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    prior_high = (
        bars["High"].shift(1).rolling(breakout_period).max().to_numpy(dtype=float)
    )
    prior_low = (
        bars["Low"].shift(1).rolling(breakout_period).min().to_numpy(dtype=float)
    )

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    active_direction = 0
    zone_lower = 0.0
    zone_upper = 0.0

    for index in range(len(bars)):
        current_close = close[index]

        if active_direction == 1 and low[index] <= zone_lower:
            active_direction = 0
        elif active_direction == -1 and high[index] >= zone_upper:
            active_direction = 0

        if index >= 1:
            if low[index] > high[index - 1] and current_close > prior_high[index]:
                active_direction = 1
                zone_lower = high[index - 1]
                zone_upper = low[index]
            elif high[index] < low[index - 1] and current_close < prior_low[index]:
                active_direction = -1
                zone_lower = high[index]
                zone_upper = low[index - 1]

        if active_direction and current_close != 0:
            midpoint = (zone_lower + zone_upper) / 2
            direction[index] = active_direction
            size_pct[index] = (zone_upper - zone_lower) / current_close
            distance_pct[index] = (midpoint - current_close) / current_close

    return pd.DataFrame(
        {
            "BAG_Direction": direction,
            "BAG_Size_Pct": size_pct,
            "BAG_Distance_Pct": distance_pct,
        },
        index=bars.index,
    )
