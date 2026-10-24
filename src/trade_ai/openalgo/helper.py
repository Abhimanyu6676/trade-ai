"""Data conversion helpers for OpenAlgo responses."""

import math
from typing import cast

import pandas as pd

from trade_ai.openalgo.openalgoTypes import OHLCVCandleDataType


def convert_OHLCV_dfToArray(
    history: pd.DataFrame,
) -> list[OHLCVCandleDataType]:
    """Convert an OpenAlgo history DataFrame into serializable candle objects."""
    candles: list[OHLCVCandleDataType] = []
    for timestamp, row in history.iterrows():
        candle_timestamp = pd.Timestamp(cast(pd.Timestamp, timestamp))
        if candle_timestamp.tzinfo is None:
            # The SDK builds daily indexes from epoch seconds without attaching
            # a timezone; those values originate in UTC.
            candle_timestamp = candle_timestamp.tz_localize("UTC")
        candle_values = {
            name: float(row[name])
            for name in ("open", "high", "low", "close", "volume")
        }
        oi_value = row.get("oi")
        oi = (
            float(oi_value)
            if oi_value is not None and math.isfinite(float(oi_value))
            else None
        )
        candles.append(
            OHLCVCandleDataType(
                timestamp=int(candle_timestamp.timestamp()),
                **candle_values,
                oi=oi,
            )
        )
    return candles
