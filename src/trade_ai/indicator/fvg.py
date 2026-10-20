"""Fair Value Gap (FVG) features for OHLC candles."""

import numpy as np
import pandas as pd

from trade_ai.indicator.liquidity import calculate_liquidity


def calculate_fvg(
    bars: pd.DataFrame, liquidity_values: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the latest unfilled three-candle FVG state for each bar.

    A bullish FVG requires a sell-side sweep on the middle candle and a gap
    between the first and third candles. A bearish FVG requires a buy-side
    sweep on the middle candle and the inverse gap. Gaps stay active through
    partial mitigation, clear after a full wick fill, and flip to an inverse
    gap when a candle closes through the far edge. Direction is +1 bullish,
    -1 bearish, and 0 inactive. Width and signed midpoint distance are relative
    to close; the midpoint distance represents consequent encroachment.
    """
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    if liquidity_values is None:
        liquidity_values = calculate_liquidity(bars)
    ssl_sweep = liquidity_values["SSL_Sweep"].to_numpy(dtype=bool)
    bsl_sweep = liquidity_values["BSL_Sweep"].to_numpy(dtype=bool)

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    is_inverse = np.zeros(len(bars), dtype=float)

    active_direction = 0
    active_is_inverse = False
    zone_lower = 0.0
    zone_upper = 0.0

    for index in range(len(bars)):
        current_close = close[index]

        # A full wick fill retires a normal FVG; a close through it flips it
        # into an inverse FVG, which stays active until reclaimed by close.
        if active_direction == 1:
            if active_is_inverse and current_close < zone_lower:
                active_direction = 0
                active_is_inverse = False
            elif not active_is_inverse and low[index] <= zone_lower:
                if current_close <= zone_lower:
                    active_direction = -1
                    active_is_inverse = True
                else:
                    active_direction = 0
        elif active_direction == -1:
            if active_is_inverse and current_close > zone_upper:
                active_direction = 0
                active_is_inverse = False
            elif not active_is_inverse and high[index] >= zone_upper:
                if current_close >= zone_upper:
                    active_direction = 1
                    active_is_inverse = True
                else:
                    active_direction = 0

        # The middle candle must sweep opposing liquidity before displacement
        # completes the three-candle gap on the current candle.
        if index >= 2:
            if low[index] > high[index - 2] and ssl_sweep[index - 1]:
                active_direction = 1
                active_is_inverse = False
                zone_lower = high[index - 2]
                zone_upper = low[index]
            elif high[index] < low[index - 2] and bsl_sweep[index - 1]:
                active_direction = -1
                active_is_inverse = False
                zone_lower = high[index]
                zone_upper = low[index - 2]

        if active_direction and current_close != 0:
            midpoint = (zone_lower + zone_upper) / 2
            direction[index] = active_direction
            size_pct[index] = (zone_upper - zone_lower) / current_close
            distance_pct[index] = (midpoint - current_close) / current_close
            is_inverse[index] = float(active_is_inverse)

    return pd.DataFrame(
        {
            "FVG_Direction": direction,
            "FVG_Size_Pct": size_pct,
            "FVG_Distance_Pct": distance_pct,
            "FVG_Is_Inverse": is_inverse,
        },
        index=bars.index,
    )
