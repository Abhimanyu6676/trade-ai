from typing import Literal

import pandas as pd

from trade_ai.indicator import (
    calculate_atr,
    calculate_ema_distance,
    calculate_fvg,
    calculate_liquidity_sweep,
    calculate_rsi,
    calculate_volatility_ratio,
)
from trade_ai.model.stock_metadata import (
    METADATA_FEATURE_COLUMNS,
    encode_categorical_metadata,
)
from trade_ai.util import lowercase_ohlcv_columns, resample_ohlcv

DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "1h")
DEFAULT_EMA_SPANS = (9, 21, 50)
# Increment when the model input feature names, order, or meanings change.
FEATURE_SCHEMA_VERSION = 7


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
        features_tf[f"Vol_Ratio_10_30_{tf}"] = calculate_volatility_ratio(
            tf_df["close"], period1=10, period2=30
        )

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

    # Section 1: Prepare sorted one-minute candles and their timeframe closes.
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Section 1: 1m, 5m, 15m Rate of Change features.
    df["ROC_1m"] = df["close"].pct_change(1)
    df["ROC_5m"] = df["close"].pct_change(5)
    df["ROC_15m"] = df["close"].pct_change(15)

    # Section 2: EMA stacks, timeframe-specific indicators, and daily gate.
    df = _add_multi_timeframe_features(df, model_type, timeframes, ema_spans, relative)

    # Section 3: Static ticker metadata is an input to LightGBM only.
    if model_type == "lgb":
        df = encode_categorical_metadata(df, ticker)
        active_feature_columns = LGB_FEATURE_COLUMNS
    else:
        df = df.drop(
            columns=[
                column for column in METADATA_FEATURE_COLUMNS if column in df.columns
            ]
        )
        active_feature_columns = GRU_FEATURE_COLUMNS

    # Section 4: Add forward labels only for training; keep the latest row for inference.
    required_columns = active_feature_columns
    if include_targets:
        df["Future_ROC_5m"] = df["close"].shift(-5).pct_change(5)
        df["Target_Direction"] = (df["Future_ROC_5m"] > 0).astype(float)
        required_columns = [
            *active_feature_columns,
            "Future_ROC_5m",
            "Target_Direction",
        ]

    return df.dropna(subset=required_columns)


# Shared technical features for the default timeframes and EMA spans above.
BASE_FEATURE_COLUMNS = [
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
            f"Vol_Ratio_10_30_{timeframe}",
        )
    ],
]

# Keep model-specific input order fixed, independent of source column order.
# Raw OHLCV remains in the frame for feature calculations, but is not model input.
GRU_FEATURE_COLUMNS = tuple(BASE_FEATURE_COLUMNS)
LGB_FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *METADATA_FEATURE_COLUMNS]
GRU_MODEL_INPUT_COLUMNS = GRU_FEATURE_COLUMNS
LGB_MODEL_INPUT_COLUMNS = tuple(LGB_FEATURE_COLUMNS)
