"""Relative Moving Average Convergence Divergence calculations."""

import pandas as pd
from trade_ai.indicator.ema import calculate_ema


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD line, signal, and histogram relative to the close.

    Returned values are decimal ratios rather than values multiplied by 100.
    """
    # Reuse the shared EMA implementation for each MACD smoothing stage.
    fast_ema = calculate_ema(close, span=fast_period)
    slow_ema = calculate_ema(close, span=slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, span=signal_period)

    return pd.DataFrame(
        {
            "MACD_Line_Pct": macd_line / close,
            "MACD_Signal_Pct": signal_line / close,
            "MACD_Hist_Pct": (macd_line - signal_line) / close,
        },
        index=close.index,
    )
