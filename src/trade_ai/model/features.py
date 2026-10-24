from typing import Literal

import pandas as pd

from trade_ai.indicator import (
    calculate_atr,
    calculate_ema_distance,
    calculate_fvg,
    calculate_liquidity_sweep,
    calculate_rsi,
)
from trade_ai.model.stock_metadata import (
    METADATA_FEATURE_COLUMNS,
    encode_categorical_metadata,
)
from trade_ai.util import lowercase_ohlcv_columns, resample_ohlcv

DEFAULT_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h")
DEFAULT_EMA_SPANS = (9, 21, 50)
# Increment when the model input feature names, order, or meanings change.
FEATURE_SCHEMA_VERSION = 2


def _add_multi_timeframe_features(
    df: pd.DataFrame,
    model_type: Literal["gru", "lgb"],
    timeframes=None,
    ema_spans=None,
    relative: bool = True,
) -> pd.DataFrame:
    """Orchestrates multi-timeframe feature calculations, safely merging

    them back to the 1-minute base timeline without lookahead bias.
    """

    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES

    if ema_spans is None:
        ema_spans = DEFAULT_EMA_SPANS

    # Section 1: Prepare sorted one-minute candles and their timeframe closes.
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for tf in timeframes:
        # 1. Resample base data to target timeframe
        tf_df = resample_ohlcv(df, tf)

        # 2. Compute indicators cleanly on that timeframe's dataframe
        features_tf = pd.DataFrame()
        features_tf["timestamp"] = tf_df["timestamp"]
        features_tf[f"RSI_14_{tf}"] = calculate_rsi(
            tf_df, period=14, scale_0_1=model_type == "gru"
        )
        features_tf[f"ATR_14_{tf}"] = calculate_atr(tf_df, period=14, relative=relative)

        # Compute EMA distances for 9, 21, and 50 spans dynamically
        for span in ema_spans:
            features_tf[f"EMA{span}_Dist_{tf}"] = calculate_ema_distance(
                tf_df, span=span, relative=relative
            )

        features_tf[f"FVG_{tf}"] = calculate_fvg(tf_df)
        features_tf[f"Sweep_{tf}"] = calculate_liquidity_sweep(tf_df, window=20)

        # 3. Safely map back to the 1-minute timeline using merge_asof (backward)
        df = pd.merge_asof(
            df,
            features_tf,
            on="timestamp",
            direction="backward",
        )

    return df


def compute_normalized_features(
    df: pd.DataFrame,
    ticker: str,
    model_type: Literal["gru", "lgb"],
    timeframes=None,
    ema_spans=None,
    relative: bool = True,
    include_targets: bool = False,
) -> pd.DataFrame:
    """Build return features, technical indicators, metadata, and optional labels."""
    df = lowercase_ohlcv_columns(df)

    # Section 1: 1m, 5m, 15m Rate of Change features.
    df["ROC_1m"] = df["close"].pct_change(1)
    df["ROC_5m"] = df["close"].pct_change(5)
    df["ROC_15m"] = df["close"].pct_change(15)

    # Section 2: EMA stacks, timeframe-specific indicators, and daily gate.
    df = _add_multi_timeframe_features(df, model_type, timeframes, ema_spans, relative)

    # Section 3: Ticker-level market-cap and sector flags.
    df = encode_categorical_metadata(df, ticker)

    # Section 4: Add forward labels only for training; keep the latest row for inference.
    required_columns = FEATURE_COLUMNS
    if include_targets:
        df["Future_ROC_5m"] = df["close"].shift(-5).pct_change(5)
        df["Target_Direction"] = (df["Future_ROC_5m"] > 0).astype(float)
        required_columns = [*FEATURE_COLUMNS, "Future_ROC_5m", "Target_Direction"]

    return df.dropna(subset=required_columns)


# Model input schema for the default timeframes and EMA spans above.
FEATURE_COLUMNS = [
    "ROC_1m",
    "ROC_5m",
    "ROC_15m",
    *[
        feature_name
        for timeframe in DEFAULT_TIMEFRAMES
        for feature_name in (
            f"RSI_14_{timeframe}",
            f"ATR_14_{timeframe}",
            *(f"EMA{span}_Dist_{timeframe}" for span in DEFAULT_EMA_SPANS),
            f"FVG_{timeframe}",
            f"Sweep_{timeframe}",
        )
    ],
    *METADATA_FEATURE_COLUMNS,
]

# Keep training and inference input order fixed, independent of source column order.
# Raw OHLCV remains in the frame for feature calculations, but is not model input.
MODEL_INPUT_COLUMNS = tuple(FEATURE_COLUMNS)
