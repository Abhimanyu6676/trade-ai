"""Fair Value Gap (FVG) detection and playbook state features."""

import numpy as np
import pandas as pd

from trade_ai.indicator.liquidity import calculate_liquidity_extended

LONDON_KILLZONE = (7, 10)
NEW_YORK_KILLZONE = (7, 10)


def _killzone_mask(index: pd.Index, source_timezone: str) -> np.ndarray:
    """Mark London 07:00-10:00 or New York 07:00-10:00 local time.

    Naive candle timestamps are interpreted in ``source_timezone``. For the
    project's NSE/OpenAlgo candles this defaults to Asia/Kolkata; timezone-aware
    inputs are converted from their existing timezone.
    """
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(index))
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize(source_timezone)
        london = timestamps.tz_convert("Europe/London")
        new_york = timestamps.tz_convert("America/New_York")
        london_minutes = london.hour * 60 + london.minute
        new_york_minutes = new_york.hour * 60 + new_york.minute
        london_start, london_end = (hour * 60 for hour in LONDON_KILLZONE)
        ny_start, ny_end = (hour * 60 for hour in NEW_YORK_KILLZONE)
        return ((london_minutes >= london_start) & (london_minutes < london_end)) | (
            (new_york_minutes >= ny_start) & (new_york_minutes < ny_end)
        )
    except TypeError, ValueError:
        return np.zeros(len(index), dtype=bool)


def detect_fvg_events(
    bars: pd.DataFrame,
    liquidity_values: pd.DataFrame | None = None,
    *,
    structure_lookback: int = 20,
    displacement_atr_multiple: float = 1.0,
    source_timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Detect sweep-qualified three-candle gaps and measure playbook context.

    A playbook-qualified event also has a displacement candle with a body of at
    least ``displacement_atr_multiple`` times the rolling 14-bar true range and
    closes beyond the preceding ``structure_lookback`` bars' structure. The
    killzone flag is recorded separately because some instruments do not trade
    during London/New York killzones.
    """
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    if liquidity_values is None:
        liquidity_values = calculate_liquidity(bars)
    if len(bars) == 0:
        return pd.DataFrame(
            {
                "FVG_Event_Direction": np.zeros(0),
                "FVG_Zone_Lower": np.zeros(0),
                "FVG_Zone_Upper": np.zeros(0),
                "FVG_Event_Displacement_ATR": np.zeros(0),
                "FVG_Event_Structure_Break": np.zeros(0),
                "FVG_Event_Killzone": np.zeros(0),
                "FVG_Event_Qualified": np.zeros(0),
            },
            index=bars.index,
        )
    ssl_sweep = liquidity_values["SSL_Sweep"].to_numpy(dtype=bool)
    bsl_sweep = liquidity_values["BSL_Sweep"].to_numpy(dtype=bool)

    prior_high = (
        bars["high"]
        .shift(2)
        .rolling(structure_lookback, min_periods=structure_lookback)
        .max()
        .to_numpy(dtype=float)
    )
    prior_low = (
        bars["low"]
        .shift(2)
        .rolling(structure_lookback, min_periods=structure_lookback)
        .min()
        .to_numpy(dtype=float)
    )
    previous_close = bars["close"].shift(1).to_numpy(dtype=float)
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)),
    )
    true_range[0] = high[0] - low[0]
    atr = (
        pd.Series(true_range, index=bars.index)
        .rolling(14, min_periods=1)
        .mean()
        .to_numpy()
    )
    displacement_atr = np.divide(
        np.abs(close - open_), atr, out=np.zeros(len(bars)), where=atr > 0
    )
    bullish_structure_break = close > prior_high
    bearish_structure_break = close < prior_low
    killzone = _killzone_mask(bars.index, source_timezone)

    direction = np.zeros(len(bars), dtype=float)
    zone_lower = np.full(len(bars), np.nan)
    zone_upper = np.full(len(bars), np.nan)
    event_displacement_atr = np.zeros(len(bars), dtype=float)
    event_structure_break = np.zeros(len(bars), dtype=float)
    event_killzone = np.zeros(len(bars), dtype=float)
    event_qualified = np.zeros(len(bars), dtype=float)

    for index in range(2, len(bars)):
        if low[index] > high[index - 2] and ssl_sweep[index - 1]:
            direction[index] = 1.0
            zone_lower[index] = high[index - 2]
            zone_upper[index] = low[index]
            structure_break = bullish_structure_break[index]
        elif high[index] < low[index - 2] and bsl_sweep[index - 1]:
            direction[index] = -1.0
            zone_lower[index] = high[index]
            zone_upper[index] = low[index - 2]
            structure_break = bearish_structure_break[index]
        else:
            continue

        event_displacement_atr[index] = displacement_atr[index]
        event_structure_break[index] = float(structure_break)
        event_killzone[index] = float(killzone[index])
        event_qualified[index] = float(
            displacement_atr[index] >= displacement_atr_multiple and structure_break
        )

    return pd.DataFrame(
        {
            "FVG_Event_Direction": direction,
            "FVG_Zone_Lower": zone_lower,
            "FVG_Zone_Upper": zone_upper,
            "FVG_Event_Displacement_ATR": event_displacement_atr,
            "FVG_Event_Structure_Break": event_structure_break,
            "FVG_Event_Killzone": event_killzone,
            "FVG_Event_Qualified": event_qualified,
        },
        index=bars.index,
    )


def calculate_fvg_extended(
    bars: pd.DataFrame,
    liquidity_values: pd.DataFrame | None = None,
    fvg_events: pd.DataFrame | None = None,
    *,
    source_timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Track the latest sweep-qualified FVG and its playbook state.

    Partial mitigation retains the gap; a full wick fill without a close
    through the far edge retires it. A close through that edge flips it into an
    inverse FVG. Retest depth is the deepest penetration so far, measured as a
    fraction of the zone. Stop distance uses candle 1's extreme, and target
    distance uses the nearest remaining liquidity level in the trade direction.
    """
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    if liquidity_values is None:
        liquidity_values = calculate_liquidity_extended(bars)
    if fvg_events is None:
        fvg_events = detect_fvg_events(
            bars, liquidity_values=liquidity_values, source_timezone=source_timezone
        )
    event_direction = fvg_events["FVG_Event_Direction"].to_numpy()
    event_lower = fvg_events["FVG_Zone_Lower"].to_numpy()
    event_upper = fvg_events["FVG_Zone_Upper"].to_numpy()
    event_displacement = fvg_events["FVG_Event_Displacement_ATR"].to_numpy()
    event_structure = fvg_events["FVG_Event_Structure_Break"].to_numpy()
    event_killzone = fvg_events["FVG_Event_Killzone"].to_numpy()
    event_qualified = fvg_events["FVG_Event_Qualified"].to_numpy()
    bsl_distance = liquidity_values["BSL_Distance_Pct"].to_numpy(dtype=float)
    ssl_distance = liquidity_values["SSL_Distance_Pct"].to_numpy(dtype=float)

    direction = np.zeros(len(bars), dtype=float)
    size_pct = np.zeros(len(bars), dtype=float)
    distance_pct = np.zeros(len(bars), dtype=float)
    is_inverse = np.zeros(len(bars), dtype=float)
    displacement_atr = np.zeros(len(bars), dtype=float)
    structure_break = np.zeros(len(bars), dtype=float)
    in_killzone = np.zeros(len(bars), dtype=float)
    setup_qualified = np.zeros(len(bars), dtype=float)
    retest_depth = np.zeros(len(bars), dtype=float)
    ce_touched = np.zeros(len(bars), dtype=float)
    entry_edge_distance = np.zeros(len(bars), dtype=float)
    stop_distance = np.zeros(len(bars), dtype=float)
    target_distance = np.zeros(len(bars), dtype=float)
    active_direction = 0
    active_is_inverse = False
    zone_lower = 0.0
    zone_upper = 0.0
    anchor_stop = 0.0
    active_displacement = 0.0
    active_structure = 0.0
    active_killzone = 0.0
    active_qualified = 0.0
    deepest_retest = 0.0
    midpoint_touched = False

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
            anchor_index = max(0, index - 2)
            anchor_stop = (
                low[anchor_index] if active_direction == 1 else high[anchor_index]
            )
            active_displacement = event_displacement[index]
            active_structure = event_structure[index]
            active_killzone = event_killzone[index]
            active_qualified = event_qualified[index]
            deepest_retest = 0.0
            midpoint_touched = False

        width = zone_upper - zone_lower
        if active_direction and current_close > 0 and width > 0:
            if active_direction == 1:
                penetration = (zone_upper - low[index]) / width
                target = bsl_distance[index]
                risk = (current_close - anchor_stop) / current_close
            else:
                penetration = (high[index] - zone_lower) / width
                target = abs(ssl_distance[index])
                risk = (anchor_stop - current_close) / current_close
            if penetration > 0:
                deepest_retest = max(deepest_retest, min(1.0, penetration))
            midpoint = (zone_lower + zone_upper) / 2
            entry_edge = zone_upper if active_direction == 1 else zone_lower
            if low[index] <= midpoint <= high[index]:
                midpoint_touched = True
            direction[index] = active_direction
            size_pct[index] = width / current_close
            distance_pct[index] = (midpoint - current_close) / current_close
            entry_edge_distance[index] = (entry_edge - current_close) / current_close
            is_inverse[index] = float(active_is_inverse)
            displacement_atr[index] = active_displacement
            structure_break[index] = active_structure
            in_killzone[index] = active_killzone
            setup_qualified[index] = float(active_qualified and active_killzone)
            retest_depth[index] = deepest_retest
            ce_touched[index] = float(midpoint_touched)
            stop_distance[index] = max(0.0, risk)
            target_distance[index] = max(0.0, target)

    return pd.DataFrame(
        {
            "FVG_Direction": direction,
            "FVG_Size_Pct": size_pct,
            "FVG_Distance_Pct": distance_pct,
            "FVG_Is_Inverse": is_inverse,
            "FVG_Displacement_ATR": displacement_atr,
            "FVG_Structure_Break": structure_break,
            "FVG_In_Killzone": in_killzone,
            "FVG_Setup_Qualified": setup_qualified,
            "FVG_Retest_Depth": retest_depth,
            "FVG_CE_Touched": ce_touched,
            "FVG_Entry_Edge_Distance_Pct": entry_edge_distance,
            "FVG_Stop_Distance_Pct": stop_distance,
            "FVG_Target_Liquidity_Distance_Pct": target_distance,
        },
        index=bars.index,
    )


def calculate_fvg(df: pd.DataFrame) -> pd.Series:
    """Detects Bullish Fair Value Gaps (FVG)."""
    fvg = pd.Series(0, index=df.index)
    fvg_condition = df["high"].shift(2) < df["low"]
    fvg.loc[fvg_condition] = 1
    return fvg
