"""Exponential moving average calculations."""

import pandas as pd


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average for a price series."""
    return series.ewm(span=span, adjust=False).mean()
