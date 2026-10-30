"""Entry point for training the short GRU specialist."""

from typing import Any

from trade_ai.model.training.train_gru import train_gru_model


def train_short_gru(**kwargs: Any) -> None:
    """Train the short-side GRU checkpoint."""
    train_gru_model(direction="short", **kwargs)


__all__ = ["train_short_gru"]
