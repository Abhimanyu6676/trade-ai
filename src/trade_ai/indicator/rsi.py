"""Relative Strength Index calculations."""

import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI normalized to the 0.0–1.0 range."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    relative_strength = gain / loss
    return (100 - (100 / (1 + relative_strength))) / 100.0
