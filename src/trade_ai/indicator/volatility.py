"""Rolling return-volatility calculations."""

import pandas as pd


def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling standard deviation of percentage returns."""
    returns = series.pct_change(1)
    return returns.rolling(window=period).std()
