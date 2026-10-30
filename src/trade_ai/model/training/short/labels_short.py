"""Short-specialist labels with inverted price barriers."""

from typing import Any

import pandas as pd

from trade_ai.config.trading_config import SHORT_STOP_LOSS_PCT, SHORT_TAKE_PROFIT_PCT
from trade_ai.model.training.labeling import LABEL_SCHEMA_VERSION, apply_triple_barrier


def apply_short_labels(df: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """Return short-relative stop, timeout, and take-profit labels."""
    return apply_triple_barrier(
        df,
        direction="short",
        take_profit_pct=SHORT_TAKE_PROFIT_PCT,
        stop_loss_pct=SHORT_STOP_LOSS_PCT,
        **kwargs,
    )


__all__ = ["LABEL_SCHEMA_VERSION", "apply_short_labels"]
