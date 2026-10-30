"""Entry point for training the short LightGBM specialist."""

from trade_ai.model.training.train_lgb import train_lightgbm_model


def train_short_lgb() -> bool:
    """Train the short-side LightGBM checkpoint."""
    return train_lightgbm_model(direction="short")


__all__ = ["train_short_lgb"]
