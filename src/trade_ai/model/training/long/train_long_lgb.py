"""Entry point for training the long LightGBM specialist."""

from trade_ai.model.training.train_lgb import (
    prepare_training_data as prepare_long_training_data,
    train_lightgbm_model,
)


def train_long_lgb() -> bool:
    """Train the long-side LightGBM checkpoint."""
    return train_lightgbm_model(direction="long")


__all__ = ["prepare_long_training_data", "train_long_lgb"]
