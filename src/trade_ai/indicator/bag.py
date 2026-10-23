"""Balanced Price Gap (BAG/BPR) features for OHLC candles."""

import numpy as np
import pandas as pd

from trade_ai.indicator.fvg import detect_fvg_events
from trade_ai.indicator.liquidity import calculate_liquidity_extended


def calculate_bag(
    bars: pd.DataFrame,
    fvg_events: pd.DataFrame | None = None,
    liquidity_values: pd.DataFrame | None = None,
    pairing_window: int = 2,
) -> pd.DataFrame:
    """Find immediate, qualified reversal overlaps and track their retests.

    Both FVG legs must pass the displacement and structure-break filters. The
    newer leg must close completely beyond the prior opposing FVG and overlap
    it within ``pairing_window`` bars. Bullish zones use their upper edge as
    support; bearish zones use the lower edge as resistance. The BAG expires
    when a close breaks its far edge.
    """
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    if liquidity_values is None:
        liquidity_values = calculate_liquidity_extended(bars)
    if fvg_events is None:
        fvg_events = detect_fvg_events(bars, liquidity_values=liquidity_values)
    event_direction = fvg_events["FVG_Event_Direction"].to_numpy()
    event_lower = fvg_events["FVG_Zone_Lower"].to_numpy()
    event_upper = fvg_events["FVG_Zone_Upper"].to_numpy()
    event_qualified = fvg_events["FVG_Event_Qualified"].to_numpy(dtype=bool)
    event_displacement = fvg_events["FVG_Event_Displacement_ATR"].to_numpy(dtype=float)
    bsl_distance = liquidity_values["BSL_Distance_Pct"].to_numpy(dtype=float)
    ssl_distance = liquidity_values["SSL_Distance_Pct"].to_numpy(dtype=float)

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    rejection = np.zeros(len(bars), dtype=float)
    retest_depth = np.zeros(len(bars), dtype=float)
    stop_distance = np.zeros(len(bars), dtype=float)
    target_distance = np.zeros(len(bars), dtype=float)
    reversal_displacement = np.zeros(len(bars), dtype=float)
    active_direction = 0
    zone_lower = 0.0
    zone_upper = 0.0
    deepest_retest = 0.0
    active_displacement = 0.0
    recent_fvgs: list[tuple[int, int, float, float, bool, float]] = []

    for index, current_close in enumerate(close):
        if active_direction == 1 and low[index] <= zone_upper:
            rejection[index] = float(current_close > zone_upper)
        elif active_direction == -1 and high[index] >= zone_lower:
            rejection[index] = float(current_close < zone_lower)

        if (
            active_direction == 1
            and current_close < zone_lower
            or active_direction == -1
            and current_close > zone_upper
        ):
            active_direction = 0

        recent_fvgs = [
            event for event in recent_fvgs if index - event[0] <= pairing_window
        ]
        current_event_direction = int(event_direction[index])
        if current_event_direction:
            current_lower = event_lower[index]
            current_upper = event_upper[index]
            current_qualified = bool(event_qualified[index])
            for (
                prior_index,
                prior_direction,
                prior_lower,
                prior_upper,
                prior_qualified,
                _,
            ) in recent_fvgs:
                if (
                    prior_direction == current_event_direction
                    or not prior_qualified
                    or not current_qualified
                    or index - prior_index > pairing_window
                ):
                    continue
                overlap_lower = max(current_lower, prior_lower)
                overlap_upper = min(current_upper, prior_upper)
                closes_past_prior = (
                    current_close > prior_upper
                    if current_event_direction == 1
                    else current_close < prior_lower
                )
                if overlap_lower < overlap_upper and closes_past_prior:
                    active_direction = current_event_direction
                    zone_lower = overlap_lower
                    zone_upper = overlap_upper
                    deepest_retest = 0.0
                    active_displacement = event_displacement[index]

            recent_fvgs.append(
                (
                    index,
                    current_event_direction,
                    current_lower,
                    current_upper,
                    current_qualified,
                    event_displacement[index],
                )
            )

        width = zone_upper - zone_lower
        if active_direction and current_close > 0 and width > 0:
            if active_direction == 1:
                rejection_edge = zone_upper
                penetration = (zone_upper - low[index]) / width
                risk = (current_close - zone_lower) / current_close
                target = bsl_distance[index]
            else:
                rejection_edge = zone_lower
                penetration = (high[index] - zone_lower) / width
                risk = (zone_upper - current_close) / current_close
                target = abs(ssl_distance[index])
            if penetration > 0:
                deepest_retest = max(deepest_retest, min(1.0, penetration))
            direction[index] = active_direction
            size_pct[index] = width / current_close
            distance_pct[index] = (rejection_edge - current_close) / current_close
            retest_depth[index] = deepest_retest
            stop_distance[index] = max(0.0, risk)
            target_distance[index] = max(0.0, target)
            reversal_displacement[index] = active_displacement

    return pd.DataFrame(
        {
            "BAG_Direction": direction,
            "BAG_Size_Pct": size_pct,
            "BAG_Distance_Pct": distance_pct,
            "BAG_Rejection": rejection,
            "BAG_Retest_Depth": retest_depth,
            "BAG_Stop_Distance_Pct": stop_distance,
            "BAG_Target_Liquidity_Distance_Pct": target_distance,
            "BAG_Reversal_Displacement_ATR": reversal_displacement,
        },
        index=bars.index,
    )
