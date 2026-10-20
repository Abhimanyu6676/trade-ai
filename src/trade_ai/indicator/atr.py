"""Average True Range calculations."""

import pandas as pd


def calculate_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR as a fraction of close for OHLC candles."""
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat(
        [high_low, high_prev_close, low_prev_close], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr / df["Close"]
