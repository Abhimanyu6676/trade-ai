"""Generate predictions using model instances managed by ``models.py``."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from trade_ai.util import merge_ohlcv_history

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

logger = logging.getLogger(__name__)


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


def run_ai_prediction_inference(
    raw_df: pd.DataFrame,
    ticker: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[int, float]:
    """Wrapper function matching data engine calls."""
    signal, confidence, _ = predict(raw_df, ticker, sequence_length)
    return signal, confidence
