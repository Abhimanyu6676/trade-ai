import asyncio
import datetime
from typing import Any

import pandas as pd

from trade_ai.logger import logger
from trade_ai.model.features import compute_normalized_features
from trade_ai.model.prediction_engine import run_ai_prediction_inference
from trade_ai.model.training.gru_sequence_builder import DEFAULT_SEQUENCE_LENGTH
from trade_ai.openalgo.openalgo_client import TYPE_OPENALGO_EXCHANGE, openalgo
from trade_ai.util import merge_ohlcv_history

FEATURE_LOOKBACK_CALENDAR_DAYS = 30
MAX_ALLOWED_ROWS = 25000


async def fetch_and_clean_candles(
    ticker: str, days_back: int, interval: str, exchange: TYPE_OPENALGO_EXCHANGE
):
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

    try:
        logger.info(
            "[SDK Call] Requesting %s:%s (%s) from %s to %s via Data Engine...",
            exchange,
            ticker,
            interval,
            history_start_date,
            end_date,
        )
        df = await openalgo.get_history(
            symbol=ticker,
            exchange=exchange,
            interval=interval,
            start_date=history_start_date,
            end_date=end_date,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, "API Empty Payload Error", 0.0

        return await asyncio.to_thread(
            _process_fetched_candles, df, ticker, display_start_date
        )

    except Exception as e:
        logger.error("[Data Engine Exception]: %s", e)
        return None, f"Execution Failure: {e!s}", 0.0


def _process_fetched_candles(
    df: pd.DataFrame, ticker: str, display_start_date: datetime.date
) -> tuple[pd.DataFrame | None, int | str, float]:
    """Normalize, engineer, infer, and select display rows off the event loop."""
    # Normalize order before trimming so the retained tail is the newest data.
    df = merge_ohlcv_history(df)

    # Keep enough rows for warm-up without unbounded feature-generation cost.
    if len(df) > MAX_ALLOWED_ROWS:
        logger.warning(
            "⚠️ Trimming historical rows from %s down to %s max limit.",
            len(df),
            MAX_ALLOWED_ROWS,
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
            "volume",
        ]
        if col in processed_df.columns
    ]
    return processed_df[display_layout], ai_signal, confidence


def predict_from_candles(ticker: str, candles: list[dict[str, Any]]):
    """Run inference against caller-supplied OHLCV candles."""
    df = pd.DataFrame(candles)
    timestamps = []
    for timestamp in df["timestamp"]:
        if isinstance(timestamp, (int, float)):
            unit = "ms" if abs(timestamp) >= 100_000_000_000 else "s"
            timestamps.append(pd.to_datetime(timestamp, unit=unit, errors="raise"))
        else:
            timestamps.append(pd.to_datetime(timestamp, errors="raise"))
    df["timestamp"] = timestamps
    df = merge_ohlcv_history(df)
    signal, confidence = run_ai_prediction_inference(
        df,
        ticker,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
    )
    return df, signal, confidence
