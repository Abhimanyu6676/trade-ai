"""Short-restricted simulator for directional RL experiments."""

import pandas as pd

from trade_ai.model.training.trading_env import MetaTradingEnv


class ShortTradingEnv(MetaTradingEnv):
    """Expose only flat and short position-size actions."""

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__(frame, allowed_direction="short")
