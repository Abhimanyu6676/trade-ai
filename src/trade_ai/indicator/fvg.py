"""Fair Value Gap (FVG) detection and state features for OHLC candles."""

import numpy as np
import pandas as pd

from trade_ai.indicator.liquidity import calculate_liquidity


def detect_fvg_events(
    bars: pd.DataFrame, liquidity_values: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Detect three-candle gaps qualified by an opposing liquidity sweep."""
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    if liquidity_values is None:
        liquidity_values = calculate_liquidity(bars)
    ssl_sweep = liquidity_values["SSL_Sweep"].to_numpy(dtype=bool)
    bsl_sweep = liquidity_values["BSL_Sweep"].to_numpy(dtype=bool)

    direction = np.zeros(len(bars), dtype=float)
    zone_lower = np.full(len(bars), np.nan)
    zone_upper = np.full(len(bars), np.nan)
    for index in range(2, len(bars)):
        if low[index] > high[index - 2] and ssl_sweep[index - 1]:
            direction[index] = 1.0
            zone_lower[index] = high[index - 2]
            zone_upper[index] = low[index]
        elif high[index] < low[index - 2] and bsl_sweep[index - 1]:
            direction[index] = -1.0
            zone_lower[index] = high[index]
            zone_upper[index] = low[index - 2]

    return pd.DataFrame(
        {
            "FVG_Event_Direction": direction,
            "FVG_Zone_Lower": zone_lower,
            "FVG_Zone_Upper": zone_upper,
        },
        index=bars.index,
    )


def calculate_fvg(
    bars: pd.DataFrame,
    liquidity_values: pd.DataFrame | None = None,
    fvg_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Track the latest sweep-qualified FVG and its inverse state.

    Partial mitigation retains the gap; a full wick fill without a close
    through the far edge retires it. A close through that edge flips it into an
    inverse FVG. The midpoint distance is consequent encroachment, and all
    price distances are ratios to close.
    """
    low = bars["Low"].to_numpy(dtype=float)
    high = bars["High"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    if fvg_events is None:
        fvg_events = detect_fvg_events(bars, liquidity_values=liquidity_values)
    event_direction = fvg_events["FVG_Event_Direction"].to_numpy()
    event_lower = fvg_events["FVG_Zone_Lower"].to_numpy()
    event_upper = fvg_events["FVG_Zone_Upper"].to_numpy()

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    is_inverse = np.zeros(len(bars), dtype=float)
    active_direction = 0
    active_is_inverse = False
    zone_lower = 0.0
    zone_upper = 0.0

    for index, current_close in enumerate(close):
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

        if event_direction[index]:
            active_direction = int(event_direction[index])
            active_is_inverse = False
            zone_lower = event_lower[index]
            zone_upper = event_upper[index]

        if active_direction and current_close > 0:
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
