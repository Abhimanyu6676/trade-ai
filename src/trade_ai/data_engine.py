import datetime
import os

import pandas as pd
import torch
from openalgo import api

from trade_ai.features import FEATURE_COLUMNS, compute_normalized_features
from trade_ai.model import StockPredictionBrain

OPENALGO_HOST = os.environ.get(
    "OPENALGO_HOST", "https://openalgo-1.sternetsmart.com"
)
_openalgo_client = None


def get_openalgo_client():
    """Create the SDK client on first use, without storing credentials in source."""
    global _openalgo_client
    api_key = os.environ.get("OPENALGO_API_KEY")
    if not api_key:
        raise RuntimeError("OPENALGO_API_KEY is required to connect to OpenAlgo")
    if _openalgo_client is None:
        _openalgo_client = api(api_key=api_key, host=OPENALGO_HOST)
    return _openalgo_client

# Path configuration for the trained model weights
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(BASE_DIR, ".run_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "gru_trading_model.pth")

# Initialize model dynamically matching the 78-column feature schema.
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


def run_ai_prediction_inference(df: pd.DataFrame, sequence_length=5):
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
        client = get_openalgo_client()
        df = client.history(
            symbol=ticker,
            exchange=exchange,
            interval=interval,
            start_date=start_str,
            end_date=end_str,
            source="api",
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, "API Empty Payload Error", 0.0

        df = df.copy()
        if df.index.name == "timestamp" or "timestamp" not in df.columns:
            df = df.reset_index()

        clean_cols = {}
        for col in df.columns:
            c_low = str(col).lower()
            if c_low in ["timestamp", "open", "high", "low", "close", "volume"]:
                clean_cols[col] = c_low.capitalize()
            else:
                clean_cols[col] = str(col)
        df = df.rename(columns=clean_cols)
        df = df.sort_values(by="Timestamp").reset_index(drop=True)

        MAX_ALLOWED_ROWS = 5000
        if len(df) > MAX_ALLOWED_ROWS:
            df = df.tail(MAX_ALLOWED_ROWS).reset_index(drop=True)

        # Compute multi-timeframe normalized features (78 model features)
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
        client = get_openalgo_client()
        res = client.analyzerstatus()
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
