"""Exponential moving average calculations."""

import pandas as pd


def calculate_ema(df: pd.DataFrame, span: int) -> pd.Series:
    """Calculate an exponential moving average of the DataFrame's close prices."""
    return df["close"].ewm(span=span, adjust=False).mean()


def calculate_ema_distance(
    df: pd.DataFrame, span: int = 20, relative: bool = True
) -> pd.Series:
    """Computes distance from EMA.

    If relative=True, returns percentage distance.
    """
    close = df["close"]
    ema = calculate_ema(df, span)
    if relative:
        return (close - ema) / ema
    return close - ema
