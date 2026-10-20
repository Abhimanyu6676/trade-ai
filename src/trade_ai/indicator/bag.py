"""Balanced Price Gap (BAG/BPR) features for OHLC candles."""

import numpy as np
import pandas as pd

from trade_ai.indicator.fvg import detect_fvg_events
from trade_ai.indicator.liquidity import calculate_liquidity


def calculate_bag(
    bars: pd.DataFrame,
    fvg_events: pd.DataFrame | None = None,
    liquidity_values: pd.DataFrame | None = None,
    pairing_window: int = 5,
) -> pd.DataFrame:
    """Find overlaps between recent opposing, sweep-qualified FVGs.

    The overlap is the balanced zone. Direction follows the newer reversal
    FVG. Bullish zones use their upper edge as support; bearish zones use the
    lower edge as resistance. A BAG remains active until a close breaks its far
    edge. Width and distance to the relevant edge are relative to close.
    """
    high = bars["High"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    close = bars["Close"].to_numpy(dtype=float)
    if fvg_events is None:
        if liquidity_values is None:
            liquidity_values = calculate_liquidity(bars)
        fvg_events = detect_fvg_events(bars, liquidity_values=liquidity_values)
    event_direction = fvg_events["FVG_Event_Direction"].to_numpy()
    event_lower = fvg_events["FVG_Zone_Lower"].to_numpy()
    event_upper = fvg_events["FVG_Zone_Upper"].to_numpy()

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    rejection = np.zeros(len(bars), dtype=float)
    active_direction = 0
    zone_lower = 0.0
    zone_upper = 0.0
    recent_fvgs: list[tuple[int, int, float, float]] = []

    for index, current_close in enumerate(close):
        if active_direction == 1 and low[index] <= zone_upper:
            rejection[index] = float(current_close > zone_upper)
        elif active_direction == -1 and high[index] >= zone_lower:
            rejection[index] = float(current_close < zone_lower)

        if active_direction == 1 and current_close < zone_lower:
            active_direction = 0
        elif active_direction == -1 and current_close > zone_upper:
            active_direction = 0

        recent_fvgs = [
            event for event in recent_fvgs if index - event[0] <= pairing_window
        ]
        current_event_direction = int(event_direction[index])
        if current_event_direction:
            current_lower = event_lower[index]
            current_upper = event_upper[index]
            for _, prior_direction, prior_lower, prior_upper in recent_fvgs:
                if prior_direction == current_event_direction:
                    continue
                overlap_lower = max(current_lower, prior_lower)
                overlap_upper = min(current_upper, prior_upper)
                if overlap_lower < overlap_upper:
                    active_direction = current_event_direction
                    zone_lower = overlap_lower
                    zone_upper = overlap_upper

            recent_fvgs.append(
                (index, current_event_direction, current_lower, current_upper)
            )

        if active_direction and current_close > 0:
            rejection_edge = zone_upper if active_direction == 1 else zone_lower
            direction[index] = active_direction
            size_pct[index] = (zone_upper - zone_lower) / current_close
            distance_pct[index] = (rejection_edge - current_close) / current_close

    return pd.DataFrame(
        {
            "BAG_Direction": direction,
            "BAG_Size_Pct": size_pct,
            "BAG_Distance_Pct": distance_pct,
            "BAG_Rejection": rejection,
        },
        index=bars.index,
    )
