"""Rolling return-volatility calculations."""

import pandas as pd


def calculate_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling standard deviation of percentage returns."""
    returns = series.pct_change(1)
    return returns.rolling(window=period).std()


def calculate_volatility_ratio(
    series: pd.Series, period1: int = 10, period2: int = 30
) -> pd.Series:
    """Calculate Volatility Expansion/Contraction Ratio over 2 period"""
    returns = series.pct_change(1)
    return returns.rolling(window=period1).std() / returns.rolling(window=period2).std()
