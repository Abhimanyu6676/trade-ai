"""Entry point for training the long GRU specialist."""

from typing import Any

from trade_ai.model.training.train_gru import train_gru_model


def train_long_gru(**kwargs: Any) -> None:
    """Train the long-side GRU checkpoint."""
    train_gru_model(direction="long", **kwargs)


__all__ = ["train_long_gru"]
