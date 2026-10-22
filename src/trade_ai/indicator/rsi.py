"""Relative Strength Index calculations."""

import pandas as pd


def old_calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI normalized to the 0.0–1.0 range."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    relative_strength = gain / loss
    return (100 - (100 / (1 + relative_strength))) / 100.0


def calculate_rsi(df: pd.DataFrame, period: int = 14, scale_0_1=False) -> pd.Series:
    """Computes RSI on a given OHLCV dataframe (bounded 0-100 scale).

    scale_0_1: boolean (default: False ) => If true instead of return 0-100 default scale the method will return 0-1 scale
    """
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if scale_0_1:
        return rsi / 100
    return rsi
