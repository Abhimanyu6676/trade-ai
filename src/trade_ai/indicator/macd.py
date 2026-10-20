"""Relative Moving Average Convergence Divergence calculations."""

import pandas as pd


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD line, signal, and histogram relative to the close.

    Returned values are decimal ratios rather than values multiplied by 100.
    """
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    return pd.DataFrame(
        {
            "MACD_Line_Pct": macd_line / close,
            "MACD_Signal_Pct": signal_line / close,
            "MACD_Hist_Pct": (macd_line - signal_line) / close,
        },
        index=close.index,
    )
