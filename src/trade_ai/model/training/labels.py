"""Backward-compatible imports for the original long-only label API."""

import pandas as pd

from trade_ai.indicator.atr import calculate_atr
from trade_ai.config.trading_config import (
    ENTRY_END_HOUR,
    ENTRY_END_MINUTE,
    ENTRY_START_HOUR,
    ENTRY_START_MINUTE,
    LONG_STOP_LOSS_PCT,
    LONG_TAKE_PROFIT_PCT,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    MAX_HOLDING_PERIOD,
)

from .labeling import LABEL_SCHEMA_VERSION, apply_triple_barrier as _label

TAKE_PROFIT_PCT = LONG_TAKE_PROFIT_PCT
STOP_LOSS_PCT = LONG_STOP_LOSS_PCT


def add_absolute_label_atr(
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Attach prior completed-bar ATR for optional label experiments."""
    label_atr = calculate_atr(source_df, period=period, relative=False).shift(1)
    atr_by_timestamp = pd.DataFrame(
        {
            "timestamp": source_df["timestamp"].to_numpy(),
            "ATR_14_1m_absolute": label_atr.to_numpy(),
        }
    )
    return feature_df.merge(
        atr_by_timestamp,
        on="timestamp",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def apply_triple_barrier(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    take_profit_pct: float = TAKE_PROFIT_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    max_holding_period: int = MAX_HOLDING_PERIOD,
    start_hour: int = ENTRY_START_HOUR,
    start_minute: int = ENTRY_START_MINUTE,
    end_hour: int = ENTRY_END_HOUR,
    end_minute: int = ENTRY_END_MINUTE,
    market_close_hour: int = MARKET_CLOSE_HOUR,
    market_close_minute: int = MARKET_CLOSE_MINUTE,
) -> pd.Series:
    """Apply the long barriers while preserving the previous call signature."""
    if take_profit_pct != TAKE_PROFIT_PCT or stop_loss_pct != STOP_LOSS_PCT:
        raise ValueError("Barrier percentages are centralized in trading_config.py")
    return _label(
        df,
        direction="long",
        timestamp_col=timestamp_col,
        close_col=close_col,
        max_holding_period=max_holding_period,
        start_hour=start_hour,
        start_minute=start_minute,
        end_hour=end_hour,
        end_minute=end_minute,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
