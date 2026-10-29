"""Generate predictions using model instances managed by ``models.py``."""

import asyncio
import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

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
) -> tuple[int, float, dict]:
    """Generate an ensemble prediction without changing loaded model state."""
    model_manager = get_model_manager()
    lgb_model = model_manager.get_model("lgb")
    gru_model = model_manager.get_model("gru")

    if lgb_model is None and gru_model is None:
        logger.warning("No trained models are loaded.")
        return 0, 0.0, {"error": "Models not loaded"}

    raw_df = merge_ohlcv_history(raw_df)
    df_lgb_feat = compute_normalized_features(raw_df, ticker, model_type="lgb")
    df_gru_feat = compute_normalized_features(raw_df, ticker, model_type="gru")

    lgb_feature_cols = list(LGB_MODEL_INPUT_COLUMNS)
    gru_feature_cols = list(GRU_MODEL_INPUT_COLUMNS)

    if df_lgb_feat.empty:
        return 0, 0.0, {"error": "Insufficient data after feature warmup"}

    latest_snapshot = df_lgb_feat[lgb_feature_cols].iloc[[-1]]
    sequence_features = latest_session_sequence(
        df_gru_feat,
        gru_feature_cols,
        sequence_length=sequence_length,
    )

    lgb_probs = None
    gru_probs = None

    if isinstance(lgb_model, lgb.Booster):
        raw_lgb_pred = lgb_model.predict(latest_snapshot)
        pred_arr = np.asarray(raw_lgb_pred)
        if pred_arr.ndim > 0 and len(pred_arr) > 0:
            lgb_probs = pred_arr[0] if pred_arr.ndim > 1 else pred_arr

    if isinstance(gru_model, GRUTradingBrain) and sequence_features is not None:
        with torch.inference_mode():
            x_tensor = (
                torch.tensor(sequence_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(model_manager.device)
            )
            logits = gru_model(x_tensor)
            gru_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    elif isinstance(gru_model, GRUTradingBrain):
        logger.warning(
            "Newest session has fewer than %d valid rows for GRU inference.",
            sequence_length,
        )

    if lgb_probs is not None and gru_probs is not None:
        final_probs = (0.5 * lgb_probs) + (0.5 * gru_probs)
    elif lgb_probs is not None:
        final_probs = lgb_probs
    elif gru_probs is not None:
        final_probs = gru_probs
    else:
        return (
            0,
            0.0,
            {
                "error": "No inference result; newest session lacks a complete GRU sequence"
            },
        )

    class_mapping = {0: -1, 1: 0, 2: 1}
    best_class_idx = int(np.argmax(final_probs))
    signal = class_mapping[best_class_idx]
    confidence = float(final_probs[best_class_idx])
    breakdown = {
        "lgb_probabilities": lgb_probs.tolist() if lgb_probs is not None else None,
        "gru_probabilities": gru_probs.tolist() if gru_probs is not None else None,
        "ensemble_probabilities": final_probs.tolist(),
    }
    return signal, confidence, breakdown


async def predict_with_history(
    ticker: str,
    exchange: TYPE_OPENALGO_EXCHANGE,
    supplied_candles: list[dict[str, Any]] | None,
    sessions: int = 30,
) -> tuple[pd.DataFrame, int, float]:
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
        return await asyncio.to_thread(predict_from_dataframe, ticker, combined)
    except PredictionError:
        raise
    except Exception as error:
        logger.exception("Prediction history preparation failed for %s", ticker)
        raise PredictionError(f"Execution Failure: {error!s}", 500) from error


def predict_from_dataframe(ticker: str, df: pd.DataFrame):
    """Run inference against an already merged chronological candle frame."""
    df = merge_ohlcv_history(df)
    signal, confidence, breakdown = predict(
        df,
        ticker,
        DEFAULT_SEQUENCE_LENGTH,
    )
    if "error" in breakdown:
        message = str(breakdown["error"])
        status_code = 503 if message == "Models not loaded" else 422
        raise PredictionError(message, status_code)
    return df, signal, confidence
