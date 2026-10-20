import datetime
import gc
import os
import shutil
import threading

import pandas as pd
import torch

from trade_ai.features import FEATURE_COLUMNS
from trade_ai.model import StockPredictionBrain

# Path configuration for the trained model weights.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")
MODEL_LOCK = threading.RLock()

# Initialize the model dynamically to match the current feature schema.
prediction_ai = StockPredictionBrain(
    input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2
)
MODEL_READY = False

# Automatically load trained checkpoint weights if available.
if os.path.exists(MODEL_PATH):
    try:
        prediction_ai.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )
        MODEL_READY = True
        print(
            f"✅ [Prediction Engine] Successfully loaded trained weights from: {MODEL_PATH}"
        )
    except Exception as error:
        print(
            f"⚠️ [Prediction Engine] Failed to load model weights: {error}. Inference is"
            " disabled until a compatible checkpoint is trained."
        )
else:
    print(
        f"⚠️ [Prediction Engine] Warning: No trained weights found at {MODEL_PATH}."
        " Inference is disabled until training completes."
    )

prediction_ai.eval()


def reset_prediction_brain() -> str:
    """Back up and remove the checkpoint, then unload trained weights."""
    global prediction_ai, MODEL_READY

    backup_path = None
    backup_error = None
    delete_error = None
    with MODEL_LOCK:
        if os.path.isfile(MODEL_PATH):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(
                CACHE_DIR, f"gru_trading_model_backup_{timestamp}.pth"
            )
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                shutil.copy2(MODEL_PATH, backup_path)
            except OSError as error:
                backup_error = error
                backup_path = None

            if backup_path is not None:
                try:
                    os.remove(MODEL_PATH)
                except OSError as error:
                    delete_error = error

        prediction_ai = None
        MODEL_READY = False
        gc.collect()

    if backup_error is not None:
        return (
            "Model reset in memory, but the checkpoint was retained because its "
            f"backup failed: {backup_error}"
        )
    if delete_error is not None:
        return (
            f"Checkpoint backup created at {backup_path}, but the original "
            f"checkpoint could not be deleted: {delete_error}. A restart may "
            "load the original checkpoint."
        )
    if backup_path is None:
        return "Model reset. No checkpoint file existed to back up. Inference is disabled until retraining."
    return (
        f"Model reset. Checkpoint backed up to {backup_path} and removed. "
        "Inference is disabled until retraining."
    )


def unload_prediction_brain() -> None:
    """Disable inference and release the in-memory model before training."""
    global prediction_ai, MODEL_READY

    with MODEL_LOCK:
        prediction_ai = None
        MODEL_READY = False
    gc.collect()


def reload_prediction_brain() -> bool:
    """Load the current checkpoint into the inference model, if compatible."""
    global prediction_ai, MODEL_READY

    loaded_model = None
    if os.path.isfile(MODEL_PATH):
        try:
            loaded_model = StockPredictionBrain(
                input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2
            )
            loaded_model.load_state_dict(
                torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            )
            loaded_model.eval()
        except Exception as error:
            print(f"⚠️ [Prediction Engine] Failed to load checkpoint: {error}")
            loaded_model = None

    with MODEL_LOCK:
        prediction_ai = loaded_model
        MODEL_READY = loaded_model is not None

    return MODEL_READY


def run_ai_prediction_inference(df: pd.DataFrame, sequence_length: int = 5):
    """Serialize inference with reset so trained weights cannot remain in use."""
    with MODEL_LOCK:
        return _run_ai_prediction_inference(df, sequence_length)


def _run_ai_prediction_inference(df: pd.DataFrame, sequence_length: int = 5):
    """Compute a directional probability from normalized feature data."""
    model = prediction_ai
    if not MODEL_READY or model is None:
        return "Model unavailable: retrain with the current feature schema", 0.0
    if len(df) < sequence_length:
        return "Warmup Active", 0.0

    target_slice = df.tail(sequence_length)

    for col in FEATURE_COLUMNS:
        if col not in target_slice.columns:
            return f"Missing Feature: {col}", 0.0

    features_matrix = target_slice[FEATURE_COLUMNS].values
    input_tensor = torch.tensor(features_matrix, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        raw_output_probability = model(input_tensor).item()

    predicted_direction = 1 if raw_output_probability >= 0.50 else -1
    macro_gate = int(target_slice["Macro_Trend_Gate"].iloc[-1])
    if macro_gate != predicted_direction:
        return "⏸️ NO TRADE (Macro trend gate is neutral or opposing)", round(
            max(raw_output_probability, 1 - raw_output_probability) * 100, 2
        )

    if predicted_direction > 0:
        signal = "📈 BULLISH FORECAST (Buy/Long Signal)"
    else:
        signal = "📉 BEARISH FORECAST (Sell/Short Signal)"

    direction_confidence = (
        raw_output_probability
        if predicted_direction > 0
        else 1 - raw_output_probability
    )
    return signal, round(direction_confidence * 100, 2)
