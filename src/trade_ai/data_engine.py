import datetime
import gc
import os
import shutil
import threading

import pandas as pd
import torch

from trade_ai.features import FEATURE_COLUMNS, compute_normalized_features
from trade_ai.model import StockPredictionBrain
from trade_ai.openalgo import openalgo

# Path configuration for the trained model weights
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")
MODEL_LOCK = threading.RLock()

# Initialize model dynamically to match the current feature schema.
prediction_ai = StockPredictionBrain(
    input_dim=len(FEATURE_COLUMNS), hidden_dim=32, num_layers=2
)
MODEL_READY = False

# Automatically load trained checkpoint weights if available
if os.path.exists(MODEL_PATH):
    try:
        prediction_ai.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )
        MODEL_READY = True
        print(
            f"✅ [Data Engine] Successfully loaded trained weights from: {MODEL_PATH}"
        )
    except Exception as e:
        print(
            f"⚠️ [Data Engine] Failed to load model weights: {e}. Inference is"
            " disabled until a compatible checkpoint is trained."
        )
else:
    print(
        f"⚠️ [Data Engine] Warning: No trained weights found at {MODEL_PATH}."
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


def unload_prediction_brain():
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
            print(f"⚠️ [Data Engine] Failed to load checkpoint: {error}")
            loaded_model = None

    with MODEL_LOCK:
        prediction_ai = loaded_model
        MODEL_READY = loaded_model is not None

    return MODEL_READY


def run_ai_prediction_inference(df: pd.DataFrame, sequence_length=5):
    """Serialize inference with reset so trained weights cannot remain in use."""
    with MODEL_LOCK:
        return _run_ai_prediction_inference(df, sequence_length)


def _run_ai_prediction_inference(df: pd.DataFrame, sequence_length=5):
    """Ingests normalized feature dataframes and computes directional probability."""
    if not MODEL_READY:
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
        raw_output_probability = prediction_ai(input_tensor).item()

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
        raw_output_probability if predicted_direction > 0 else 1 - raw_output_probability
    )
    return signal, round(direction_confidence * 100, 2)


def fetch_and_clean_candles(ticker: str, days_back: int, interval: str, exchange: str):
    """Queries OpenAlgo, processes multi-timeframe normalized features,

    and executes inference.
    """
    if interval != "1m":
        return (
            None,
            "This feature pipeline requires 1-minute source candles; select 1m.",
            0.0,
        )

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(days_back))
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        print(
            f"[SDK Call] Requesting {exchange}:{ticker} ({interval}) via Data Engine..."
        )
        df = openalgo.get_history(
            symbol=ticker,
            exchange=exchange,
            interval=interval,
            start_date=start_str,
            end_date=end_str,
            source="api",
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, "API Empty Payload Error", 0.0

        MAX_ALLOWED_ROWS = 5000
        if len(df) > MAX_ALLOWED_ROWS:
            df = df.tail(MAX_ALLOWED_ROWS).reset_index(drop=True)

        # Compute multi-timeframe normalized features for the current schema.
        processed_df = compute_normalized_features(df, ticker)

        if processed_df.empty or len(processed_df) < 5:
            return None, "Insufficient Data After Feature Warmup", 0.0

        ai_signal, confidence = run_ai_prediction_inference(
            processed_df, sequence_length=5
        )

        display_layout = [
            col
            for col in [
                "Timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Return_1m",
                "ATR_Pct_5m",
                "Volatility_20_30m",
                "Macro_Trend_Gate",
                "Volume",
            ]
            if col in processed_df.columns
        ]

        return processed_df[display_layout], ai_signal, confidence

    except Exception as e:
        print(f"[Data Engine Exception]: {e}")
        return None, f"Execution Failure: {e!s}", 0.0


def get_openalgo_status_message():
    try:
        res = openalgo.get_analyzer_status()
        if res.get("status") == "success":
            data = res.get("data", {})
            if not isinstance(data, dict):
                data = {}
            mode = data.get("mode", "unknown")
            analyze_mode = data.get("analyze_mode", True)
            return (
                f"🟢 **OpenAlgo Status: Online** | Mode: `{mode}` | Sandbox Analyze"
                f" Mode: `{analyze_mode}`"
            )
        return "⚠️ OpenAlgo responded, but status check was unsuccessful."
    except Exception as e:
        return f"❌ **OpenAlgo Connection Error:** {e!s}"
