"""Exponential moving average calculations."""

import pandas as pd


def calculate_ema(close: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average from a price series."""
    return close.ewm(span=span, adjust=False).mean()


def calculate_ema_distance(
    close: pd.Series, span: int = 20, relative: bool = True
) -> pd.Series:
    """Compute the close-to-EMA distance.

    If relative=True, return the distance as a ratio of close; otherwise return
    the difference in price units.
    """
    # Calculate the EMA using the same series that supplies the current price.
    ema = calculate_ema(close, span)
    # Divide by close for a scale-independent ratio when requested.
    if relative:
        return (close - ema) / close
    # Keep the signed price-unit difference for raw-distance callers.
    return close - ema


def calculate_ema_spread(
    close: pd.Series, span1: int, span2: int, relative: bool = True
) -> pd.Series:
    """Computes the spread/distance between two EMAs (EMA(span1) - EMA(span2)).

    If relative=True, returns the difference normalized by the close price.
    """
    # Calculate both moving averages from the same input price series.
    ema1 = calculate_ema(close, span1)
    ema2 = calculate_ema(close, span2)

    spread = ema1 - ema2

    # Normalize by close by default so the spread is comparable across prices.
    if relative:
        return spread / close
    # Preserve the absolute EMA difference when relative scaling is disabled.
    return spread
