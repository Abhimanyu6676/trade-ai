import datetime

import pandas as pd

from trade_ai.brain.features import compute_normalized_features
from trade_ai.brain.prediction_engine import run_ai_prediction_inference
from trade_ai.brain.sequence_loader import DEFAULT_SEQUENCE_LENGTH
from trade_ai.openalgo_client import openalgo
from trade_ai.util import merge_ohlcv_history

FEATURE_LOOKBACK_CALENDAR_DAYS = 30


def fetch_and_clean_candles(ticker: str, days_back: int, interval: str, exchange: str):
    """Queries OpenAlgo, processes multi-timeframe normalized features,
    and executes inference with a safe historical window.
    """
    if interval != "1m":
        return (
            None,
            "This feature pipeline requires 1-minute source candles; select 1m.",
            0.0,
        )

    end_date = datetime.date.today()
    # Add a small buffer multiplier to account for weekends and holidays
    calendar_buffer_days = int(days_back * 1.5) + 5
    display_start_date = end_date - datetime.timedelta(days=calendar_buffer_days)
    # Keep the requested display window intact while fetching enough prior
    # sessions to warm up higher-timeframe indicators such as the 1h EMA 50.
    history_start_date = display_start_date - datetime.timedelta(
        days=FEATURE_LOOKBACK_CALENDAR_DAYS
    )

    start_str = history_start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        print(
            f"[SDK Call] Requesting {exchange}:{ticker} ({interval}) from {start_str} to {end_str} via Data Engine..."
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

        # Normalize order before trimming so the retained tail is the newest data.
        df = merge_ohlcv_history(df)

        # Increased row cap to safely allow ~60 days of 1m history (~22,500 rows max)
        # without overflowing your 4GB VPS RAM during feature generation.
        MAX_ALLOWED_ROWS = 25000
        if len(df) > MAX_ALLOWED_ROWS:
            print(
                f"⚠️ Trimming historical rows from {len(df)} down to {MAX_ALLOWED_ROWS} max limit."
            )
            df = df.tail(MAX_ALLOWED_ROWS).reset_index(drop=True)

        # Engineer on the entire chronological history so higher-timeframe EMAs
        # are warmed up before the requested/display window begins.
        processed_df = compute_normalized_features(df, ticker, model_type="gru")

        if processed_df.empty:
            return None, "Insufficient Data After Feature Warmup", 0.0

        ai_signal, confidence = run_ai_prediction_inference(
            df,
            ticker,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
        )

        display_timestamps = pd.to_datetime(processed_df["timestamp"])
        processed_df = processed_df.loc[
            display_timestamps.dt.date >= display_start_date
        ].copy()
        if processed_df.empty:
            return None, "No candles found in the requested display window", 0.0

        display_layout = [
            col
            for col in [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "Return_1m",
                "ATR_Pct_5m",
                "Volatility_20_30m",
                "Macro_Trend_Gate",
                "volume",
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
