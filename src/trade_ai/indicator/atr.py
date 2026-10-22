"""Average True Range calculations."""

import pandas as pd


def calculate_atr(
    df: pd.DataFrame, period: int = 14, relative: bool = True
) -> pd.Series:
    """Computes ATR.

    If relative=True, returns percentage of close price.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    if relative:
        return atr / df["close"]
    return atr
