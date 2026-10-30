"""Generate predictions using model instances managed by ``models.py``."""

import asyncio
import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from trade_ai.config.trading_config import barrier_percentages
from trade_ai.logger import logger
from trade_ai.model.dataStore import dataStore
from trade_ai.openalgo.openalgo_client import TYPE_OPENALGO_EXCHANGE, openalgo
from trade_ai.util import candles_to_dataframe, merge_ohlcv_history

from .features import (
    GRU_MODEL_INPUT_COLUMNS,
    LGB_MODEL_INPUT_COLUMNS,
    compute_normalized_features,
)
from .models import GRUTradingBrain, get_model_manager
from .training.gru_sequence_builder import (
    DEFAULT_SEQUENCE_LENGTH,
    latest_session_sequence,
)
from .training.trading_env import UNIFIED_ACTIONS, build_meta_observation

MAX_PREDICTION_LOOKBACK_SESSIONS = 90


class PredictionError(Exception):
    """Prediction failure with an HTTP status suitable for the route."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def predict(
    raw_df: pd.DataFrame,
    ticker: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    portfolio_position: float = 0.0,
    unrealized_return: float = 0.0,
    bars_in_position: int = 0,
    barrier_multiplier: float = 1.0,
) -> tuple[int, float, dict]:
    """Generate an ensemble prediction without changing loaded model state."""
    normalized_df = merge_ohlcv_history(raw_df)
    return _predict_normalized(
        normalized_df,
        ticker,
        sequence_length,
        portfolio_position,
        unrealized_return,
        bars_in_position,
        barrier_multiplier,
    )


def _predict_normalized(
    raw_df: pd.DataFrame,
    ticker: str,
    sequence_length: int,
    portfolio_position: float = 0.0,
    unrealized_return: float = 0.0,
    bars_in_position: int = 0,
    barrier_multiplier: float = 1.0,
) -> tuple[int, float, dict]:
    """Run all four frozen specialists and the policy on chronological candles."""
    model_manager = get_model_manager()
    model_keys = ("long_lgb", "long_gru", "short_lgb", "short_gru")
    models = {key: model_manager.get_model(key) for key in model_keys}
    missing_models = [key for key, model in models.items() if model is None]
    if missing_models:
        logger.warning("Base models are not loaded: %s", ", ".join(missing_models))
        return 0, 0.0, {"error": "Base models unavailable: " + ", ".join(missing_models)}
    meta_agent = model_manager.get_meta_agent()
    if meta_agent is None:
        return 0, 0.0, {"error": "Validated RL meta-agent is not available"}

    df_lgb_feat = compute_normalized_features(raw_df, ticker, model_type="lgb")
    df_gru_feat = compute_normalized_features(raw_df, ticker, model_type="gru")
    if df_lgb_feat.empty or df_gru_feat.empty:
        return 0, 0.0, {"error": "Insufficient data after feature warmup"}

    latest_lgb = df_lgb_feat[list(LGB_MODEL_INPUT_COLUMNS)].iloc[[-1]]
    sequence_features = latest_session_sequence(
        df_gru_feat,
        list(GRU_MODEL_INPUT_COLUMNS),
        sequence_length=sequence_length,
    )
    if sequence_features is None:
        return (
            0,
            0.0,
            {"error": f"Newest session has fewer than {sequence_length} valid GRU rows"},
        )

    probabilities: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for direction in ("long", "short"):
            lgb_model = models[f"{direction}_lgb"]
            gru_model = models[f"{direction}_gru"]
            if not isinstance(lgb_model, lgb.Booster) or not isinstance(
                gru_model, GRUTradingBrain
            ):
                return 0, 0.0, {"error": f"{direction} model types are invalid"}
            lgb_values = np.asarray(lgb_model.predict(latest_lgb), dtype=np.float64)
            lgb_values = lgb_values[0] if lgb_values.ndim > 1 else lgb_values
            x_tensor = (
                torch.tensor(sequence_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(model_manager.device)
            )
            gru_values = torch.softmax(gru_model(x_tensor), dim=1).cpu().numpy()[0]
            probabilities[f"{direction}_lgb"] = np.asarray(lgb_values, dtype=np.float64)
            probabilities[f"{direction}_gru"] = np.asarray(gru_values, dtype=np.float64)

    latest = df_lgb_feat.iloc[-1]
    try:
        observation = build_meta_observation(
            probabilities,
            float(latest["ATR_14_1m"]),
            float(latest["Vol_Ratio_10_30_1m"]),
            portfolio_position,
            unrealized_return,
            bars_in_position,
            barrier_multiplier,
        )
        raw_action, _ = meta_agent.predict(observation, deterministic=True)
        action_index = int(np.asarray(raw_action).reshape(-1)[0])
        if not 0 <= action_index < len(UNIFIED_ACTIONS):
            raise ValueError("RL agent returned an action outside its schema")
    except Exception as error:
        logger.exception("RL meta-agent inference failed for %s", ticker)
        return 0, 0.0, {"error": f"RL meta-agent inference failed: {error!s}"}

    position, selected_barrier_multiplier = UNIFIED_ACTIONS[action_index]
    position = float(position)
    effective_barrier_multiplier: float | None = None
    if position:
        effective_barrier_multiplier = selected_barrier_multiplier
        if portfolio_position and np.sign(position) == np.sign(portfolio_position):
            effective_barrier_multiplier = barrier_multiplier
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    if position:
        assert effective_barrier_multiplier is not None
        take_profit_pct, stop_loss_pct = barrier_percentages(
            "long" if position > 0 else "short",
            effective_barrier_multiplier,
        )
    signal = 1 if position > 0 else -1 if position < 0 else 0
    long_probs = 0.5 * (probabilities["long_lgb"] + probabilities["long_gru"])
    short_probs = 0.5 * (probabilities["short_lgb"] + probabilities["short_gru"])
    confidence = (
        float(long_probs[2])
        if signal > 0
        else float(short_probs[2])
        if signal < 0
        else 0.0
    )
    breakdown = {
        # Keep the previous response keys as aliases for the long specialists.
        "lgb_probabilities": probabilities["long_lgb"].tolist(),
        "gru_probabilities": probabilities["long_gru"].tolist(),
        "ensemble_probabilities": long_probs.tolist(),
        "long_lgb_probabilities": probabilities["long_lgb"].tolist(),
        "long_gru_probabilities": probabilities["long_gru"].tolist(),
        "short_lgb_probabilities": probabilities["short_lgb"].tolist(),
        "short_gru_probabilities": probabilities["short_gru"].tolist(),
        "long_ensemble_probabilities": long_probs.tolist(),
        "short_ensemble_probabilities": short_probs.tolist(),
        "meta_action": action_index,
        "position_size": abs(position),
        "barrier_multiplier": effective_barrier_multiplier,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
    }
    return signal, confidence, breakdown


async def predict_with_history(
    ticker: str,
    exchange: TYPE_OPENALGO_EXCHANGE,
    supplied_candles: list[dict[str, Any]] | None,
    sessions: int = 30,
    portfolio_position: float = 0.0,
    unrealized_return: float = 0.0,
    bars_in_position: int = 0,
    barrier_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, int, float, dict[str, Any]]:
    """Fetch prior history, combine target-day candles, and run inference."""
    try:
        target_date = datetime.date.today()
        supplied_frame = None
        if supplied_candles is not None:
            try:
                supplied_frame = await asyncio.to_thread(
                    candles_to_dataframe, supplied_candles
                )
            except Exception as error:
                raise PredictionError(
                    f"Invalid supplied candle data: {error!s}", 422
                ) from error
            if supplied_frame.empty:
                raise PredictionError("No candles supplied", 422)
            target_date = pd.to_datetime(supplied_frame["timestamp"]).max().date()

        history = await dataStore.get_history_before(
            ticker,
            exchange,
            target_date,
            sessions=min(
                MAX_PREDICTION_LOOKBACK_SESSIONS, max(30, sessions)
            ),
        )
        if history is None or history.empty:
            raise PredictionError(
                "Unable to retrieve required trading-session history", 502
            )

        if supplied_frame is None:
            current = await openalgo.get_history(
                symbol=ticker,
                exchange=exchange,
                interval="1m",
                start_date=target_date,
                end_date=target_date,
            )
            if not isinstance(current, pd.DataFrame) or current.empty:
                raise PredictionError(
                    "No candles are available for the prediction date", 502
                )
            supplied_frame = await asyncio.to_thread(merge_ohlcv_history, current)

        # Prediction-day candles are transient; dataStore retains prior sessions only.
        combined = await asyncio.to_thread(merge_ohlcv_history, history, supplied_frame)
        return await asyncio.to_thread(
            predict_from_dataframe,
            ticker,
            combined,
            portfolio_position,
            unrealized_return,
            bars_in_position,
            barrier_multiplier,
        )
    except PredictionError:
        raise
    except Exception as error:
        logger.exception("Prediction history preparation failed for %s", ticker)
        raise PredictionError(f"Execution Failure: {error!s}", 500) from error


def predict_from_dataframe(
    ticker: str,
    df: pd.DataFrame,
    portfolio_position: float = 0.0,
    unrealized_return: float = 0.0,
    bars_in_position: int = 0,
    barrier_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, int, float, dict[str, Any]]:
    """Run inference on a pre-merged chronological candle frame."""
    signal, confidence, breakdown = _predict_normalized(
        df,
        ticker,
        DEFAULT_SEQUENCE_LENGTH,
        portfolio_position,
        unrealized_return,
        bars_in_position,
        barrier_multiplier,
    )
    if "error" in breakdown:
        message = str(breakdown["error"])
        status_code = (
            503
            if "unavailable" in message or "meta-agent" in message
            else 422
            if "Insufficient" in message or "fewer than" in message
            else 500
        )
        raise PredictionError(message, status_code)
    logger.info(
        "Prediction result for %s: signal=%s, confidence=%.4f, "
        "candles_processed=%s, latest_candle=%s, breakdown=%s",
        ticker,
        signal,
        confidence,
        len(df),
        df["timestamp"].iloc[-1],
        breakdown,
    )
    return df, signal, confidence, breakdown
