"""Historical portfolio simulator and state contract for the PPO meta-agent."""

from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trade_ai.config.trading_config import (
    BARRIER_MULTIPLIERS,
    DISAGREEMENT_PENALTY,
    DRAWDOWN_PENALTY,
    MAX_HOLDING_PERIOD,
    POSITION_SIZE_LEVELS,
    TRANSACTION_COST_PCT,
    barrier_percentages,
)

BASE_PROBABILITY_COLUMNS = tuple(
    f"{direction}_{family}_{probability}"
    for direction in ("long", "short")
    for family in ("lgb", "gru")
    for probability in ("p_stop", "p_timeout", "p_take_profit")
)
OBSERVATION_SIZE = 20
OBSERVATION_FEATURES = (
    *BASE_PROBABILITY_COLUMNS,
    "long_model_agreement",
    "short_model_agreement",
    "atr_14_1m_scaled",
    "vol_ratio_10_30_1m_scaled",
    "portfolio_position_scaled",
    "unrealized_return_scaled",
    "bars_in_position_scaled",
    "active_barrier_multiplier_scaled",
)
_SIZE_LEVELS = tuple(POSITION_SIZE_LEVELS[1:])
_BARRIER_SCALE = max(
    1.0 - BARRIER_MULTIPLIERS[0], BARRIER_MULTIPLIERS[-1] - 1.0
)
_DIRECTIONAL_ACTIONS = tuple(
    (position, multiplier)
    for position in (*_SIZE_LEVELS, *(-size for size in _SIZE_LEVELS))
    for multiplier in BARRIER_MULTIPLIERS
)
UNIFIED_ACTIONS = ((0.0, 1.0), *_DIRECTIONAL_ACTIONS)
UNIFIED_ACTION_CONTRACT = tuple(
    {"position": position, "barrier_multiplier": multiplier}
    for position, multiplier in UNIFIED_ACTIONS
)
UNIFIED_ACTION_POSITIONS = tuple(position for position, _ in UNIFIED_ACTIONS)


def build_meta_observation(
    probabilities: dict[str, np.ndarray | list[float]],
    atr_relative: float,
    volatility_ratio: float,
    portfolio_position: float = 0.0,
    unrealized_return: float = 0.0,
    bars_in_position: int = 0,
    barrier_multiplier: float = 1.0,
) -> np.ndarray:
    """Build the fixed 20-value, schema-ordered observation vector."""
    flattened: list[float] = []
    for direction in ("long", "short"):
        for family in ("lgb", "gru"):
            key = f"{direction}_{family}"
            values = np.asarray(probabilities[key], dtype=np.float64).reshape(-1)
            if (
                values.shape != (3,)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError(f"{key} probabilities must contain 3 finite values")
            total = float(values.sum())
            if total <= 0:
                raise ValueError(f"{key} probabilities must sum to a positive value")
            flattened.extend((values / total).tolist())

    agreements: list[float] = []
    for direction in ("long", "short"):
        left = np.asarray(probabilities[f"{direction}_lgb"], dtype=np.float64)
        right = np.asarray(probabilities[f"{direction}_gru"], dtype=np.float64)
        agreements.append(float(np.clip(1.0 - 0.5 * np.abs(left - right).sum(), 0, 1)))

    scaled_state = [
        float(np.clip(atr_relative, 0.0, 0.20) / 0.20),
        float(np.clip(volatility_ratio, 0.0, 5.0) / 5.0),
        float(np.clip(portfolio_position / max(POSITION_SIZE_LEVELS[-1], 1.0), -1, 1)),
        float(np.clip(unrealized_return / 0.03, -1, 1)),
        float(np.clip(bars_in_position / MAX_HOLDING_PERIOD, 0, 1)),
        (
            float(
                np.clip(
                    (barrier_multiplier - 1.0) / _BARRIER_SCALE,
                    -1,
                    1,
                )
            )
            if portfolio_position
            else 0.0
        ),
    ]
    observation = np.asarray([*flattened, *agreements, *scaled_state], dtype=np.float32)
    if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
        raise ValueError("Meta-agent observation does not match the current schema")
    return observation


class MetaTradingEnv(gym.Env[np.ndarray, int]):
    """Simulate one position over a sampled ticker/session from OOS predictions.

    Actions choose flat or a directional size and bracket multiplier. The
    selected multiplier scales the baseline 1.0% target and 0.5% stop together
    and stays fixed for the life of the position. The simulator uses
    close-to-close returns, a 60-bar holding cap, estimated transaction costs,
    disagreement penalties, and an incremental drawdown penalty. It does not
    model order books, partial fills, or broker execution.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        allowed_direction: Literal["both", "long", "short"] = "both",
    ) -> None:
        super().__init__()
        self.allowed_direction = allowed_direction
        self.frame = self._validate_frame(frame)
        self.episodes = self._make_episodes(self.frame)
        if not self.episodes:
            raise ValueError("RL data must contain at least one session with 2 rows")

        if allowed_direction == "both":
            self._actions = UNIFIED_ACTIONS
        elif allowed_direction == "long":
            self._actions = (
                (0.0, 1.0),
                *(
                    (size, multiplier)
                    for size in _SIZE_LEVELS
                    for multiplier in BARRIER_MULTIPLIERS
                ),
            )
        else:
            self._actions = (
                (0.0, 1.0),
                *(
                    (-size, multiplier)
                    for size in _SIZE_LEVELS
                    for multiplier in BARRIER_MULTIPLIERS
                ),
            )
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._episode: pd.DataFrame | None = None
        self._index = 0
        self._position = 0.0
        self._entry_price: float | None = None
        self._bars_held = 0
        self._barrier_multiplier = 1.0
        self._equity = 1.0
        self._peak_equity = 1.0

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Validate columns and chronology before the simulator accepts data."""
        required = {
            "ticker",
            "timestamp",
            "close",
            "ATR_14_1m",
            "Vol_Ratio_10_30_1m",
            *BASE_PROBABILITY_COLUMNS,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"RL frame is missing columns: {sorted(missing)}")
        result = frame.copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
        if result["timestamp"].isna().any():
            raise ValueError("RL frame contains invalid timestamps")
        numeric_columns = ["close", "ATR_14_1m", "Vol_Ratio_10_30_1m"]
        result[numeric_columns] = result[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        if (
            not np.isfinite(result[numeric_columns].to_numpy(dtype=np.float64)).all()
            or (result["close"] <= 0).any()
        ):
            raise ValueError(
                "RL market state contains non-finite values or a non-positive close"
            )
        values = result[list(BASE_PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("RL frame contains invalid model probabilities")
        if not np.allclose(values.reshape(-1, 4, 3).sum(axis=2), 1.0, atol=1e-4):
            raise ValueError("Each base-model probability vector must sum to 1")
        result = result.sort_values(["ticker", "timestamp"], kind="stable")
        result["session_date"] = result["timestamp"].dt.date.astype(str)
        return result.reset_index(drop=True)

    @staticmethod
    def _make_episodes(frame: pd.DataFrame) -> list[pd.DataFrame]:
        """Split observations at ticker and calendar-session boundaries."""
        return [
            group.reset_index(drop=True)
            for _, group in frame.groupby(["ticker", "session_date"], sort=True)
            if len(group) >= 2
        ]

    def _observation(self) -> np.ndarray:
        assert self._episode is not None
        row = self._episode.iloc[self._index]
        probabilities = {
            f"{direction}_{family}": row[
                [
                    f"{direction}_{family}_p_stop",
                    f"{direction}_{family}_p_timeout",
                    f"{direction}_{family}_p_take_profit",
                ]
            ].to_numpy(dtype=np.float64)
            for direction in ("long", "short")
            for family in ("lgb", "gru")
        }
        unrealized = 0.0
        if self._entry_price and self._position:
            unrealized = self._position * (float(row["close"]) / self._entry_price - 1)
        return build_meta_observation(
            probabilities,
            float(row["ATR_14_1m"]),
            float(row["Vol_Ratio_10_30_1m"]),
            self._position,
            unrealized,
            self._bars_held,
            self._barrier_multiplier,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Start from a random ticker/session or a requested episode index."""
        super().reset(seed=seed)
        requested_index = (options or {}).get("episode_index")
        if requested_index is None:
            episode_index = int(self.np_random.integers(0, len(self.episodes)))
        else:
            episode_index = int(requested_index)
            if not 0 <= episode_index < len(self.episodes):
                raise ValueError("episode_index is outside the available data")
        self._episode = self.episodes[episode_index]
        self._index = 0
        self._position = 0.0
        self._entry_price = None
        self._bars_held = 0
        self._barrier_multiplier = 1.0
        self._equity = 1.0
        self._peak_equity = 1.0
        return self._observation(), {"episode_index": episode_index}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Apply a position target and reward its next close-to-close result."""
        if self._episode is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")

        current = self._episode.iloc[self._index]
        next_index = self._index + 1
        next_row = self._episode.iloc[next_index]
        current_close = float(current["close"])
        next_close = float(next_row["close"])
        target_position, requested_barrier_multiplier = self._actions[int(action)]
        target_position = float(target_position)
        prior_position = self._position
        turnover = abs(target_position - prior_position)
        transaction_cost = TRANSACTION_COST_PCT * turnover

        if target_position == 0 or prior_position == 0 or np.sign(target_position) != np.sign(prior_position):
            self._entry_price = current_close if target_position else None
            self._bars_held = 0
            self._barrier_multiplier = (
                requested_barrier_multiplier if target_position else 1.0
            )
        self._position = target_position

        gross_return = target_position * (next_close / current_close - 1.0)
        net_return = gross_return - transaction_cost
        exit_reason: str | None = None
        take_profit: float | None = None
        stop_loss: float | None = None
        self._bars_held = self._bars_held + 1 if target_position else 0

        trade_return = 0.0
        if target_position and self._entry_price:
            trade_return = float(np.sign(target_position)) * (
                next_close / self._entry_price - 1.0
            )
            if target_position > 0:
                take_profit, stop_loss = barrier_percentages(
                    "long", self._barrier_multiplier
                )
            else:
                take_profit, stop_loss = barrier_percentages(
                    "short", self._barrier_multiplier
                )
            if trade_return >= take_profit:
                exit_reason = "take_profit"
            elif trade_return <= -stop_loss:
                exit_reason = "stop_loss"
            elif self._bars_held >= MAX_HOLDING_PERIOD:
                exit_reason = "max_holding_period"
            elif next_index == len(self._episode) - 1:
                exit_reason = "session_end"

        disagreement = 0.0
        # Calculate agreement from persisted base probabilities to keep OOS data
        # compact and use the same formula in training and inference states.
        if target_position:
            side = "long" if target_position > 0 else "short"
            lgb_probs = current[
                [f"{side}_lgb_p_stop", f"{side}_lgb_p_timeout", f"{side}_lgb_p_take_profit"]
            ].to_numpy(dtype=np.float64)
            gru_probs = current[
                [f"{side}_gru_p_stop", f"{side}_gru_p_timeout", f"{side}_gru_p_take_profit"]
            ].to_numpy(dtype=np.float64)
            agreement = float(np.clip(1 - 0.5 * np.abs(lgb_probs - gru_probs).sum(), 0, 1))
            disagreement = 1 - agreement

        if exit_reason:
            net_return -= TRANSACTION_COST_PCT * abs(target_position)
            self._position = 0.0
            self._entry_price = None
            self._bars_held = 0
            self._barrier_multiplier = 1.0

        previous_drawdown = 1.0 - self._equity / self._peak_equity
        self._equity *= max(0.0, 1.0 + net_return)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = 1.0 - self._equity / self._peak_equity
        incremental_drawdown = max(0.0, drawdown - previous_drawdown)
        disagreement_cost = (
            DISAGREEMENT_PENALTY * disagreement * abs(target_position)
            if target_position and prior_position == 0
            else 0.0
        )
        reward = 100.0 * (
            net_return
            - disagreement_cost
            - DRAWDOWN_PENALTY * incremental_drawdown
        )

        self._index = next_index
        terminated = self._index >= len(self._episode) - 1
        return self._observation(), float(reward), terminated, False, {
            "equity": self._equity,
            "position": self._position,
            "trade_return": trade_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "exit_reason": exit_reason,
            "take_profit_pct": take_profit,
            "stop_loss_pct": stop_loss,
            "barrier_multiplier": self._barrier_multiplier,
        }
