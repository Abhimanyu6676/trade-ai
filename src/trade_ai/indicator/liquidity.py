"""Buy-side/sell-side liquidity levels and sweep signals."""

import numpy as np
import pandas as pd


def calculate_liquidity(
    bars: pd.DataFrame, pivot_bars: int = 2
) -> pd.DataFrame:
    """Calculate nearest unswept liquidity and sweep events for each bar.

    Buy-side liquidity is tracked at confirmed swing highs; sell-side liquidity
    is tracked at confirmed swing lows. A pivot is confirmed after
    ``pivot_bars`` bars on its right. A sweep is a wick through an existing
    level followed by a close back across it. Distances are signed ratios to
    close: buy-side levels are above close and sell-side levels below it.
    """
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    length = len(bars)

    bsl_distance_pct = np.zeros(length, dtype=float)
    ssl_distance_pct = np.zeros(length, dtype=float)
    bsl_sweep = np.zeros(length, dtype=float)
    ssl_sweep = np.zeros(length, dtype=float)
    buy_levels: list[float] = []
    sell_levels: list[float] = []

    for index in range(length):
        current_close = close[index]

        # Only levels known before this candle can be swept by it.
        remaining_buy_levels = []
        for level in buy_levels:
            if high[index] > level:
                if current_close < level:
                    bsl_sweep[index] = 1.0
            else:
                remaining_buy_levels.append(level)
        buy_levels = remaining_buy_levels

        remaining_sell_levels = []
        for level in sell_levels:
            if low[index] < level:
                if current_close > level:
                    ssl_sweep[index] = 1.0
            else:
                remaining_sell_levels.append(level)
        sell_levels = remaining_sell_levels

        # Confirm the candidate pivot now, after its right-side bars have closed.
        pivot_index = index - pivot_bars
        if pivot_index >= pivot_bars:
            left_slice = slice(pivot_index - pivot_bars, pivot_index)
            right_slice = slice(pivot_index + 1, index + 1)

            if high[pivot_index] > np.max(high[left_slice]) and high[
                pivot_index
            ] >= np.max(high[right_slice]):
                buy_levels.append(high[pivot_index])

            if low[pivot_index] < np.min(low[left_slice]) and low[
                pivot_index
            ] <= np.min(low[right_slice]):
                sell_levels.append(low[pivot_index])

        if current_close > 0:
            overhead_levels = [level for level in buy_levels if level > current_close]
            support_levels = [level for level in sell_levels if level < current_close]
            if overhead_levels:
                bsl_distance_pct[index] = (
                    min(overhead_levels) - current_close
                ) / current_close
            if support_levels:
                ssl_distance_pct[index] = (
                    max(support_levels) - current_close
                ) / current_close

    return pd.DataFrame(
        {
            "BSL_Distance_Pct": bsl_distance_pct,
            "SSL_Distance_Pct": ssl_distance_pct,
            "BSL_Sweep": bsl_sweep,
            "SSL_Sweep": ssl_sweep,
        },
        index=bars.index,
    )
