"""Daily macro trend gate calculations."""

import pandas as pd


def calculate_macro_trend_gate(
    daily_close: pd.Series, period: int = 10
) -> pd.Series:
    """Return 1 for aligned bullish trend, -1 bearish, and 0 otherwise."""
    daily_sma = daily_close.rolling(window=period, min_periods=period).mean()
    sma_slope = daily_sma.diff()
    bullish = (daily_close > daily_sma) & (sma_slope > 0)
    bearish = (daily_close < daily_sma) & (sma_slope < 0)

    gate = pd.Series(0.0, index=daily_close.index, dtype="float64")
    gate = gate.where(~bullish, 1.0)
    gate = gate.where(~bearish, -1.0)
    gate.index.name = daily_close.index.name or "Timestamp"
    return gate
